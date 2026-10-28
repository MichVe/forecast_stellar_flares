"""Evaluates the hybrid ETSformer-forecast + Stella-classify flare-prediction
methodology proposed in Vesela & Trenz, "Application of Artificial Intelligence
for Predicting Stellar Flares: Analysis and Implementation of Transformers"
(SPAICE 2024, https://zenodo.org/records/13885607), on real TESS light curves.

Pipeline:
  1. Fetch TESS 2-min light curves via lightkurve/MAST for the default targets (M dwarfs
     from Guenther et al. 2020's published TESS sector 1-2 flare catalog, VizieR
     J/AJ/159/60 -- the paper's own cited data source) or any --targets given.
  2. Label flare/non-flare events from that real published catalog (falls back to a
     robust-statistics auto-detector for targets not in the catalog).
  3. Build (a) 200-cadence classification windows (restricted to --sectors, i.e. the
     catalog's sector 1-2 coverage, so every label is real) and (b) lookback->horizon
     forecasting pairs (self-supervised, so free to use more sectors via --fc-sectors,
     with extra windows oversampled around known flares via --flare-oversample), split
     80/10/10.
  4. Train the Stella CNN classifier and the ETSformer forecaster (plus an LSTM
     and a naive-persistence baseline for comparison).
  5. Evaluate forecast quality (MAE/MSE/RMSE, incl. a flare-horizon vs. quiescent-horizon
     breakdown) and the full hybrid flare-prediction pipeline (accuracy/precision/recall/
     F1/ROC-AUC), against an oracle upper bound.

Usage:
    python main.py --targets "AU Mic" "TRAPPIST-1" --classifier-epochs 25 --forecaster-epochs 15

    # Forecasting only (skips the Stella classifier and hybrid flare-prediction step):
    python main.py --forecast-only --targets "AU Mic" "TRAPPIST-1" --forecaster-epochs 40
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data import DEFAULT_SECTORS, DEFAULT_TARGETS, fetch_all, make_classification_windows, make_forecast_windows
from src.evaluate import evaluate_forecast_quality, evaluate_hybrid
from src.forecaster import ETSformerForecaster, LSTMForecaster, NaivePersistenceForecaster
from src.plotting import plot_example_windows, plot_forecast_example
from src.stella_model import StellaCNN
from src.train import make_classifier_loaders, make_forecast_loaders, train_classifier, train_forecaster


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS, help="TESS target names (resolved via MAST).")
    p.add_argument("--sectors", nargs="+", type=int, default=DEFAULT_SECTORS,
                    help="Restrict the *classification* data to specific TESS sectors. Defaults to [1, 2] "
                         "because the Guenther et al. 2020 flare catalog (the paper's own cited data source) "
                         "only covers TESS's first data release (sectors 1-2); pass e.g. --sectors 1 2 3 ... or "
                         "omit filtering with an empty list if using --targets that aren't in that catalog "
                         "(falls back to the auto-detector).")
    p.add_argument("--fc-sectors", nargs="+", type=int, default=None,
                    help="Sectors to use for *forecaster* training data (default: all available). The forecaster "
                         "is self-supervised (no flare label needed), so unlike --sectors it isn't limited to the "
                         "Guenther catalog's sector 1-2 coverage -- using more sectors gives it more training data.")
    p.add_argument("--flare-oversample", type=int, default=4,
                    help="For the forecaster, add this many extra (jittered) training windows per known flare so "
                         "its horizon segment contains the flare, at various offsets. Flares are rare, so a plain "
                         "fixed stride leaves them a tiny minority and the aggregate MAE/MSE is dominated by easy "
                         "quiescent segments; 0 disables oversampling.")
    p.add_argument("--cache-dir", default="./tess_cache", help="lightkurve download cache directory.")
    p.add_argument("--min-flare-amplitude", type=float, default=0.01,
                    help="Minimum relative flux excursion above the local trend to count as a flare. "
                         "Approximates the paper's 'only large flares (in class M or C)' restriction, "
                         "since TESS photometry has no direct GOES X-ray-class equivalent.")
    p.add_argument("--window", type=int, default=200, help="Classification window length (cadences).")
    p.add_argument("--lookback", type=int, default=150, help="Forecaster input length (cadences).")
    p.add_argument("--horizon", type=int, default=50, help="Forecaster output length (cadences).")
    p.add_argument("--stride", type=int, default=100, help="Stride for classification windows.")
    p.add_argument("--classifier-epochs", type=int, default=25)
    p.add_argument("--forecaster-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--classifier-lr", type=float, default=1e-3)
    p.add_argument("--forecaster-lr", type=float, default=3e-4,
                    help="ETSformer/LSTM tend to need a lower LR than the CNN classifier; raise the "
                         "epoch count and/or lower this if forecast MAE stays worse than NaivePersistence.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--forecast-only", action="store_true",
                    help="Skip the Stella classifier and hybrid flare-prediction evaluation entirely; "
                         "only train/evaluate the forecasters (ETSformer, LSTM, NaivePersistence) on "
                         "MAE/MSE/RMSE. Faster to iterate on forecast quality alone.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.lookback + args.horizon != args.window:
        raise ValueError("--window must equal --lookback + --horizon (the classification window is "
                          "reused as the lookback+horizon pair for the hybrid pipeline).")

    cls_X_parts, cls_y_parts = [], []
    if not args.forecast_only:
        logging.info("Fetching TESS light curves for classification (sectors=%s): %s", args.sectors, args.targets)
        cls_curves = fetch_all(args.targets, sectors=args.sectors, cache_dir=args.cache_dir, min_flare_amplitude=args.min_flare_amplitude)
        for lc in cls_curves:
            cx, cy = make_classification_windows(lc, window=args.window, stride=args.stride)
            if len(cx):
                cls_X_parts.append(cx)
                cls_y_parts.append(cy)

    logging.info("Fetching TESS light curves for forecaster training (sectors=%s): %s",
                 args.fc_sectors or "all available", args.targets)
    fc_curves = fetch_all(args.targets, sectors=args.fc_sectors, cache_dir=args.cache_dir, min_flare_amplitude=args.min_flare_amplitude)

    fc_X_parts, fc_y_parts = [], []
    for lc in fc_curves:
        fx, fy = make_forecast_windows(
            lc, lookback=args.lookback, horizon=args.horizon, stride=max(args.stride // 4, 1),
            oversample_near_flares=args.flare_oversample,
        )
        if len(fx):
            fc_X_parts.append(fx)
            fc_y_parts.append(fy)

    if not fc_X_parts or (not args.forecast_only and not cls_X_parts):
        raise RuntimeError("No usable windows were built from the fetched light curves; try more targets "
                            "or a shorter --window.")

    fc_X, fc_y = np.concatenate(fc_X_parts), np.concatenate(fc_y_parts)
    logging.info("Forecast training pairs: %d", len(fc_y))
    fc_train, fc_val, fc_test = make_forecast_loaders(fc_X, fc_y, batch_size=args.batch_size, seed=args.seed)

    logging.info("Training ETSformer forecaster...")
    ets = ETSformerForecaster()
    ets = train_forecaster(ets, fc_train, fc_val, epochs=args.forecaster_epochs, lr=args.forecaster_lr, device=args.device)

    logging.info("Training LSTM baseline forecaster...")
    lstm = LSTMForecaster()
    lstm = train_forecaster(lstm, fc_train, fc_val, epochs=args.forecaster_epochs, lr=args.forecaster_lr, device=args.device)

    naive = NaivePersistenceForecaster()
    forecasters = [("ETSformer", ets), ("LSTM", lstm), ("NaivePersistence", naive)]

    report = {"forecast_quality": {}, "hybrid_flare_prediction": {}}
    for name, model in forecasters:
        report["forecast_quality"][name] = evaluate_forecast_quality(model, fc_test, device=args.device)

    classifier = None
    if not args.forecast_only:
        cls_X, cls_y = np.concatenate(cls_X_parts), np.concatenate(cls_y_parts)
        logging.info("Classification windows: %d (positives=%d)", len(cls_y), int(cls_y.sum()))
        cls_train, cls_val, _ = make_classifier_loaders(cls_X, cls_y, batch_size=args.batch_size, seed=args.seed)

        # Re-seed immediately before the classifier's own init/training. Without this,
        # the classifier's actual initial weights depend on however much of the global
        # RNG stream the forecasters happened to consume beforehand (which varies with
        # --fc-sectors / --flare-oversample / dataset size), making cross-run comparisons
        # of forecaster configs silently confounded by an incidentally different
        # classifier rather than isolating the forecaster's effect. See Discussion.
        torch.manual_seed(args.seed)
        logging.info("Training Stella CNN classifier...")
        classifier = StellaCNN(window=args.window)
        classifier = train_classifier(classifier, cls_train, cls_val, epochs=args.classifier_epochs, lr=args.classifier_lr, device=args.device)

        # Held-out classification split for the hybrid pipeline (same seed/ratios as make_classifier_loaders).
        _, cls_X_tmp, _, cls_y_tmp = train_test_split(cls_X, cls_y, test_size=0.2, stratify=cls_y, random_state=args.seed)
        _, cls_X_test, _, cls_y_test = train_test_split(cls_X_tmp, cls_y_tmp, test_size=0.5, stratify=cls_y_tmp, random_state=args.seed)

        oracle_recorded = False
        for name, model in forecasters:
            result = evaluate_hybrid(model, classifier, cls_X_test, cls_y_test, args.lookback, args.horizon, device=args.device)
            report["hybrid_flare_prediction"][name] = result["hybrid"]
            if not oracle_recorded:
                report["hybrid_flare_prediction"]["Oracle (true future)"] = result["oracle"]
                oracle_recorded = True

    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Wrote report to %s", report_path)
    print_report(report)

    torch.save(ets.state_dict(), os.path.join(args.output_dir, "etsformer.pt"))
    torch.save(lstm.state_dict(), os.path.join(args.output_dir, "lstm_baseline.pt"))
    if classifier is not None:
        torch.save(classifier.state_dict(), os.path.join(args.output_dir, "stella_classifier.pt"))
        plot_example_windows(cls_X_test, cls_y_test, os.path.join(args.output_dir, "example_windows.png"))

    xb, yb = next(iter(fc_test))
    xb, yb = xb.to(args.device), yb.to(args.device)
    with torch.no_grad():
        pred = ets(xb, horizon=yb.shape[1])
    plot_forecast_example(
        xb[0, :, 0].cpu().numpy(),
        yb[0, :, 0].cpu().numpy(),
        pred[0, :, 0].cpu().numpy(),
        os.path.join(args.output_dir, "forecast_example.png"),
        label="ETSformer forecast",
    )
    logging.info("Saved diagnostic plots and model checkpoints to %s", args.output_dir)


def print_report(report: dict):
    # RMSE is computed and kept in report.json for completeness, but per the paper's own
    # reasoning it's dropped from the headline table since it's derivable from MSE.
    print("\n=== Forecast quality (MAE / MSE on held-out horizon) ===")
    for name, metrics in report["forecast_quality"].items():
        line = f"{name:20s} MAE={metrics['MAE']:.5f}  MSE={metrics['MSE']:.5f}"
        if "MAE_flare_horizon" in metrics:
            line += (
                f"   | flare-horizon (n={metrics['n_flare_horizon']}): "
                f"MAE={metrics['MAE_flare_horizon']:.5f} MSE={metrics['MSE_flare_horizon']:.5f}"
                f"   | quiescent (n={metrics['n_quiescent_horizon']}): "
                f"MAE={metrics['MAE_quiescent_horizon']:.5f} MSE={metrics['MSE_quiescent_horizon']:.5f}"
            )
        print(line)

    if not report["hybrid_flare_prediction"]:
        return

    print("\n=== Hybrid flare prediction (forecast -> Stella classify) ===")
    for name, metrics in report["hybrid_flare_prediction"].items():
        line = (
            f"{name:20s} acc={metrics['accuracy']:.3f} prec={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}"
        )
        if "roc_auc" in metrics:
            line += f" auc={metrics['roc_auc']:.3f}"
        print(line)
        if "best_threshold" in metrics:
            print(
                f"{'':20s} @best_threshold={metrics['best_threshold']:.3f}: "
                f"prec={metrics['precision_at_best']:.3f} recall={metrics['recall_at_best']:.3f} "
                f"f1={metrics['f1_at_best']:.3f}"
            )


if __name__ == "__main__":
    main()
