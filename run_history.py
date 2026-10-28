"""Flare-history models (src/history.py) on exactly the same samples, folds and thresholds
procedure as run_forecasting.py, so their predictions can be merged with the neural runs
in scripts/aggregate_v2.py (pass both result directories).

Usage:
    python run_history.py --split star --horizons 5 15 30 60 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from run_forecasting import HOLDOUT_TARGETS  # noqa: F401  (same default targets)
from run_forecasting import parse_args as parse_forecasting_args
from src.data import fetch_all
from src.history import fit_predict_all
from src.metrics import evaluate_scores, select_threshold
from src.task import assert_no_leakage, build_dataset, star_folds, star_split_masks, time_split_masks

logger = logging.getLogger("run_history")


def main():
    import sys

    if "--output-dir" not in sys.argv:
        sys.argv += ["--output-dir", "./results_v2/history"]
    args = parse_forecasting_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.makedirs(os.path.join(args.output_dir, "predictions"), exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    curves = fetch_all(args.targets, sectors=args.sectors, cache_dir=args.cache_dir)
    missing = sorted(set(args.targets) - {lc.target for lc in curves})
    if missing and not args.allow_missing:
        raise RuntimeError(f"Failed to load {missing}")
    records_path = os.path.join(args.output_dir, "runs.jsonl")
    open(records_path, "w").close()

    for H in args.horizons:
        ds = build_dataset(curves, args.lookback, H, args.stride, args.clip)
        if args.split == "star":
            splits = [star_split_masks(ds, f) for f in
                      star_folds(len(ds.stars), args.folds, args.val_stars, args.split_seed)]
        else:
            splits = [time_split_masks(ds, args.lookback, H)]
        for seed in args.seeds:
            for fold_id, masks in enumerate(splits):
                assert_no_leakage(ds, masks, args.split, args.lookback, H)
                train, val, test = (ds.subset(masks[k]) for k in ("train", "val", "test"))
                if min(train.y.sum(), val.y.sum(), test.y.sum()) == 0:
                    continue
                scores = fit_predict_all(train, {"val": val, "test": test}, curves, H, seed)
                metrics = {}
                for name, s_test in scores["test"].items():
                    thr = select_threshold(val.y, scores["val"][name], args.threshold_criterion)
                    metrics[name] = evaluate_scores(test.y, s_test, thr)
                with open(records_path, "a") as f:
                    f.write(json.dumps({"horizon": H, "seed": seed, "fold": fold_id, "split": args.split,
                                        "metrics": metrics}) + "\n")
                np.savez_compressed(
                    os.path.join(args.output_dir, "predictions", f"H{H}_seed{seed}_fold{fold_id}.npz"),
                    y=test.y, star=test.star, anchor=test.anchor,
                    **{f"score/{k}": v for k, v in scores["test"].items()})
                logger.info("H=%d seed=%d fold=%d  %s", H, seed, fold_id, "  ".join(
                    f"{k}: AUC={v.get('roc_auc', float('nan')):.3f} TSS={v['tss']:.3f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
