"""v2 experiment: leakage-free stellar flare *forecasting* on TESS 2-min light curves.

Task: from the last L cadences, predict whether a catalogued flare (Guenther et al. 2020)
begins within the next H cadences. See src/task.py for the exact definition and the
leakage fixes relative to v1 (main.py).

Compared models (all scored on the same test samples):
  Hybrid-ETSformer / Hybrid-LSTM / Hybrid-Naive
      forecast the horizon, concatenate it with the lookback and classify the (L+H) window
      with a Stella CNN trained on true (lookback + horizon) windows -- the v1 idea.
  Oracle         the same classifier on the *true* horizon (upper bound; not a forecast).
  Direct-CNN     a Stella CNN on the lookback alone, trained directly on the label.
  Climatology    the star's positive rate in the training data.
  Recency        1 / (1 + cadences since the last catalogued flare).

Thresholds are chosen on the validation split (max TSS) and applied unchanged to test.

Usage:
    python run_forecasting.py --split star --horizons 5 15 30 60 --seeds 0 1 2 3 4
    python run_forecasting.py --smoke   # tiny CPU run to check the pipeline end to end
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import climatology_scores, recency_scores
from src.data import DEFAULT_TARGETS, fetch_all
from src.forecaster import ETSformerForecaster, LSTMForecaster, NaivePersistenceForecaster
from src.metrics import evaluate_scores, select_threshold
from src.stella_model import StellaCNN
from src.task import (
    ForecastSamples,
    assert_no_leakage,
    build_dataset,
    star_folds,
    star_split_masks,
    time_split_masks,
)
from src.train import train_classifier, train_forecaster

# The 8 stars used as the v1 held-out confirmation set (scripts/holdout_confirm.py).
HOLDOUT_TARGETS = [f"TIC {t}" for t in (441398770, 201883033, 364588501, 220536421,
                                        425938795, 220556639, 220433364, 220433363)]

logger = logging.getLogger("run_forecasting")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS + HOLDOUT_TARGETS)
    p.add_argument("--sectors", nargs="+", type=int, default=[1, 2],
                   help="Guenther et al. 2020 labels only cover sectors 1-2.")
    p.add_argument("--cache-dir", default="./tess_cache")
    p.add_argument("--lookback", type=int, default=150)
    p.add_argument("--horizons", nargs="+", type=int, default=[5, 15, 30, 60])
    p.add_argument("--stride", type=int, default=5, help="Cadences between consecutive anchors.")
    p.add_argument("--clip", type=float, default=30.0, help="Clip normalised flux to +-clip robust sigmas.")
    p.add_argument("--split", choices=["star", "time"], default="star")
    p.add_argument("--folds", type=int, default=4, help="Star split: number of star folds.")
    p.add_argument("--val-stars", type=int, default=2, help="Star split: validation stars per fold.")
    p.add_argument("--split-seed", type=int, default=0, help="Seed for assigning stars to folds (fixed across --seeds).")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--models", nargs="+",
                   default=["ETSformer", "LSTM", "Naive", "Oracle", "Direct-CNN", "Climatology", "Recency"])
    p.add_argument("--forecaster-epochs", type=int, default=15)
    p.add_argument("--classifier-epochs", type=int, default=25)
    p.add_argument("--forecaster-lr", type=float, default=3e-4)
    p.add_argument("--classifier-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--fc-pos-repeat", type=int, default=4,
                   help="Repeat positive (flare-in-horizon) samples this many times in forecaster training.")
    p.add_argument("--fc-max-train", type=int, default=20000,
                   help="Cap on forecaster training samples: all positives are kept, negatives are randomly "
                        "subsampled. Anchors overlap heavily (stride << L), so this costs little information.")
    p.add_argument("--fc-max-val", type=int, default=5000, help="Cap on forecaster validation samples.")
    p.add_argument("--patience", type=int, default=3, help="Forecaster early-stopping patience (epochs).")
    p.add_argument("--resume", action="store_true",
                   help="Keep existing results in --output-dir and only run (horizon, seed, fold) combinations "
                        "that are missing (e.g. after a reboot). Every run re-seeds itself, so resumed results "
                        "are identical to an uninterrupted run.")
    p.add_argument("--allow-missing", action="store_true",
                   help="Continue if some targets fail to load (default: abort, so runs stay comparable).")
    p.add_argument("--threshold-criterion", choices=["tss", "f1"], default="tss")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="./results_v2")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny configuration (4 stars, 1 horizon, 1 seed, 1 epoch) to test the pipeline.")
    args = p.parse_args()
    if args.smoke:
        args.targets = args.targets[:4]
        args.horizons, args.seeds = [30], [0]
        args.folds, args.val_stars = 2, 1
        args.forecaster_epochs = args.classifier_epochs = 1
        args.stride = 25
        args.output_dir = os.path.join(args.output_dir, "smoke")
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(*arrays, batch_size, shuffle, seed=0):
    ds = TensorDataset(*(torch.from_numpy(np.ascontiguousarray(a)) for a in arrays))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)


def _cap(y: np.ndarray, max_n: int, rng) -> np.ndarray:
    """Indices keeping every positive and a random subset of negatives, at most max_n in total."""
    pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y == 0)
    n_neg = max(min(len(neg), max_n - len(pos)), 0)
    return np.sort(np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)]))


def forecast_loaders(train: ForecastSamples, val: ForecastSamples, args, seed):
    rng = np.random.default_rng(seed)
    tr_idx = _cap(train.y, args.fc_max_train, rng)
    rep = np.where(train.y[tr_idx] > 0, max(args.fc_pos_repeat, 1), 1)
    idx = np.repeat(tr_idx, rep)
    va_idx = _cap(val.y, args.fc_max_val, rng)
    tr = _loader(train.lookback[idx, :, None], train.horizon[idx, :, None],
                 batch_size=args.batch_size, shuffle=True, seed=seed)
    va = _loader(val.lookback[va_idx, :, None], val.horizon[va_idx, :, None],
                 batch_size=args.batch_size, shuffle=False)
    return tr, va


@torch.no_grad()
def predict_horizon(model, lookback: np.ndarray, horizon: int, device, batch_size=512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(lookback), batch_size):
        xb = torch.from_numpy(lookback[i:i + batch_size, :, None]).to(device)
        out.append(model(xb, horizon=horizon).squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, horizon), np.float32)


@torch.no_grad()
def classify(model, X: np.ndarray, device, batch_size=1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


def forecast_errors(pred: np.ndarray, true: np.ndarray, y: np.ndarray) -> dict:
    err = pred - true
    res = {"MAE": float(np.abs(err).mean()), "MSE": float((err ** 2).mean())}
    for name, m in (("flare", y > 0), ("quiescent", y == 0)):
        if m.any():
            res[f"MAE_{name}"] = float(np.abs(err[m]).mean())
            res[f"MSE_{name}"] = float((err[m] ** 2).mean())
    return res


def run_one(ds: ForecastSamples, masks, H: int, seed: int, fold_id: int, args) -> tuple[dict, dict]:
    L = args.lookback
    train, val, test = (ds.subset(masks[k]) for k in ("train", "val", "test"))
    info = {k: {"n": len(s), "n_pos": int(s.y.sum()), "stars": sorted({ds.stars[i] for i in np.unique(s.star)})}
            for k, s in (("train", train), ("val", val), ("test", test))}
    logger.info("H=%d seed=%d fold=%d  train %d (%d pos)  val %d (%d pos)  test %d (%d pos)", H, seed, fold_id,
                len(train), train.y.sum(), len(val), val.y.sum(), len(test), test.y.sum())
    if train.y.sum() == 0 or val.y.sum() == 0 or test.y.sum() == 0:
        logger.warning("A split has no positives; skipping this fold.")
        return {}, {}

    scores_val, scores_test, fq = {}, {}, {}
    dev = args.device

    # -- forecasters ---------------------------------------------------------------
    forecasters = {}
    need_fc = [m for m in ("ETSformer", "LSTM", "Naive") if m in args.models]
    if any(m != "Naive" for m in need_fc):
        fc_tr, fc_va = forecast_loaders(train, val, args, seed)
    for name in need_fc:
        set_seed(seed)
        if name == "Naive":
            forecasters[name] = NaivePersistenceForecaster()
            continue
        model = ETSformerForecaster() if name == "ETSformer" else LSTMForecaster()
        forecasters[name] = train_forecaster(model, fc_tr, fc_va, epochs=args.forecaster_epochs,
                                             lr=args.forecaster_lr, device=dev, patience=args.patience)

    # -- classifier on (lookback + horizon): used by the hybrids and the Oracle -------------
    if need_fc or "Oracle" in args.models:
        set_seed(seed)
        win_cls = StellaCNN(window=L + H)
        win_cls = train_classifier(
            win_cls,
            _loader(train.window(), train.y, batch_size=args.batch_size, shuffle=True, seed=seed),
            _loader(val.window(), val.y, batch_size=args.batch_size, shuffle=False),
            epochs=args.classifier_epochs, lr=args.classifier_lr, device=dev)
        for split, s, store in (("val", val, scores_val), ("test", test, scores_test)):
            if "Oracle" in args.models:
                store["Oracle"] = classify(win_cls, s.window(), dev)
            for name, fc in forecasters.items():
                fut = predict_horizon(fc, s.lookback, H, dev)
                store[f"Hybrid-{name}"] = classify(win_cls, np.concatenate([s.lookback, fut], axis=1), dev)
                if split == "test":
                    fq[name] = forecast_errors(fut, s.horizon, s.y)

    # -- direct classifier on the lookback alone ----------------------------------------
    if "Direct-CNN" in args.models:
        set_seed(seed)
        direct = StellaCNN(window=L)
        direct = train_classifier(
            direct,
            _loader(train.lookback, train.y, batch_size=args.batch_size, shuffle=True, seed=seed),
            _loader(val.lookback, val.y, batch_size=args.batch_size, shuffle=False),
            epochs=args.classifier_epochs, lr=args.classifier_lr, device=dev)
        scores_val["Direct-CNN"] = classify(direct, val.lookback, dev)
        scores_test["Direct-CNN"] = classify(direct, test.lookback, dev)

    # -- non-learned references ------------------------------------------------------------
    if "Climatology" in args.models:
        scores_val["Climatology"] = climatology_scores(train, val)
        scores_test["Climatology"] = climatology_scores(train, test)
    if "Recency" in args.models:
        scores_val["Recency"] = recency_scores(val)
        scores_test["Recency"] = recency_scores(test)

    metrics = {}
    for name in scores_test:
        thr = select_threshold(val.y, scores_val[name], args.threshold_criterion)
        metrics[name] = evaluate_scores(test.y, scores_test[name], thr)

    record = {"horizon": H, "seed": seed, "fold": fold_id, "split": args.split, "data": info,
              "metrics": metrics, "forecast_quality": fq}
    preds = {"y": test.y, "star": test.star, "anchor": test.anchor,
             **{f"score/{k}": v for k, v in scores_test.items()}}
    return record, preds


def keep_awake():
    """Ask Windows not to sleep while this process runs (no-op elsewhere; reverts on exit)."""
    if platform.system() == "Windows":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    args = parse_args()
    keep_awake()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.makedirs(os.path.join(args.output_dir, "predictions"), exist_ok=True)
    records_path = os.path.join(args.output_dir, "runs.jsonl")
    done = set()
    if args.resume and os.path.exists(records_path):
        with open(records_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["horizon"], r["seed"], r["fold"]))
        logger.info("Resuming: %d runs already done, they will be skipped", len(done))
    elif os.path.exists(records_path) and os.path.getsize(records_path) > 0:
        raise RuntimeError(f"{records_path} already has results; use --resume to continue, "
                           "or choose another --output-dir.")
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "torch": torch.__version__, "python": platform.python_version(),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    curves = fetch_all(args.targets, sectors=args.sectors, cache_dir=args.cache_dir)
    logger.info("Loaded %d light curves", len(curves))
    missing = sorted(set(args.targets) - {lc.target for lc in curves})
    if missing and not args.allow_missing:
        raise RuntimeError(f"Failed to load {missing} (network?). Re-run, or pass --allow-missing.")
    if not done:
        open(records_path, "w").close()

    for H in args.horizons:
        ds = build_dataset(curves, args.lookback, H, args.stride, args.clip)
        logger.info("H=%d: %d samples, %d positive (%.2f%%)", H, len(ds), ds.y.sum(), 100 * ds.y.mean())
        if args.split == "star":
            splits = [star_split_masks(ds, f) for f in
                      star_folds(len(ds.stars), args.folds, args.val_stars, args.split_seed)]
        else:
            splits = [time_split_masks(ds, args.lookback, H)]
        for fold_id, masks in enumerate(splits):
            assert_no_leakage(ds, masks, args.split, args.lookback, H)
        for seed in args.seeds:
            for fold_id, masks in enumerate(splits):
                if (H, seed, fold_id) in done:
                    continue
                record, preds = run_one(ds, masks, H, seed, fold_id, args)
                if not record:
                    continue
                with open(records_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                np.savez_compressed(
                    os.path.join(args.output_dir, "predictions", f"H{H}_seed{seed}_fold{fold_id}.npz"), **preds)
                summary = "  ".join(f"{k}: AUC={v.get('roc_auc', float('nan')):.3f} TSS={v['tss']:.3f}"
                                    for k, v in record["metrics"].items())
                logger.info("H=%d seed=%d fold=%d  %s", H, seed, fold_id, summary)

    logger.info("Done. Per-run results in %s; aggregate with scripts/aggregate_v2.py", records_path)


if __name__ == "__main__":
    main()
