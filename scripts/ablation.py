# -*- coding: utf-8 -*-
"""2x2 ablation: does ETSformer's advantage come from (a) more forecaster training
data (all sectors vs. sectors 1-2 only), (b) flare-aware oversampling, or both?

Two corners are already known from prior runs (see results/report.json history):
  (sectors=1-2, oversample=0): ETSformer MAE=0.01079 MSE=0.00024, hybrid AUC=0.941 F1@best=0.843
  (sectors=all, oversample=4): ETSformer MAE=0.00466 MSE=0.00010, hybrid AUC=0.969 F1@best=0.905
This script fills in the other two corners, reusing the already-trained (frozen) Stella
classifier and the exact same classification test split, so only the forecaster's
training-data configuration varies.
"""
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data import DEFAULT_TARGETS, DEFAULT_SECTORS, fetch_all, make_classification_windows, make_forecast_windows
from src.forecaster import ETSformerForecaster
from src.stella_model import StellaCNN
from src.train import make_forecast_loaders, train_forecaster
from src.evaluate import evaluate_forecast_quality, evaluate_hybrid

SEED = 42
LOOKBACK, HORIZON, WINDOW = 150, 50, 200
device = "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- fixed: classification data + frozen classifier + fixed test split (same as main report) ---
cls_curves = fetch_all(DEFAULT_TARGETS, sectors=DEFAULT_SECTORS, cache_dir="./tess_cache")
Xp, yp = [], []
for lc in cls_curves:
    cx, cy = make_classification_windows(lc, window=WINDOW, stride=100)
    if len(cx):
        Xp.append(cx)
        yp.append(cy)
cls_X, cls_y = np.concatenate(Xp), np.concatenate(yp)
_, X_tmp, _, y_tmp = train_test_split(cls_X, cls_y, test_size=0.2, stratify=cls_y, random_state=SEED)
_, X_test, _, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

classifier = StellaCNN(window=WINDOW)
classifier.load_state_dict(torch.load("./results/stella_classifier.pt", map_location=device))
classifier.eval()


def run_cell(sectors, oversample, label):
    logging.info("=== Ablation cell: sectors=%s oversample=%s ===", sectors, oversample)
    fc_curves = fetch_all(DEFAULT_TARGETS, sectors=sectors, cache_dir="./tess_cache")
    fc_Xp, fc_yp = [], []
    for lc in fc_curves:
        fx, fy = make_forecast_windows(lc, lookback=LOOKBACK, horizon=HORIZON, stride=25, oversample_near_flares=oversample)
        if len(fx):
            fc_Xp.append(fx)
            fc_yp.append(fy)
    fc_X, fc_y = np.concatenate(fc_Xp), np.concatenate(fc_yp)
    logging.info("%s: %d forecast training pairs", label, len(fc_y))

    fc_train, fc_val, fc_test = make_forecast_loaders(fc_X, fc_y, batch_size=64, seed=SEED)
    torch.manual_seed(SEED)
    ets = ETSformerForecaster()
    ets = train_forecaster(ets, fc_train, fc_val, epochs=15, lr=3e-4, device=device)

    fq = evaluate_forecast_quality(ets, fc_test, device=device)
    hybrid = evaluate_hybrid(ets, classifier, X_test, y_test, LOOKBACK, HORIZON, device=device)["hybrid"]

    print(f"\n### {label} ###")
    print(f"  n_forecast_pairs={len(fc_y)}")
    print(f"  MAE={fq['MAE']:.5f} MSE={fq['MSE']:.5f} "
          f"MAE_flare={fq.get('MAE_flare_horizon', float('nan')):.5f} MSE_flare={fq.get('MSE_flare_horizon', float('nan')):.5f}")
    print(f"  hybrid: acc={hybrid['accuracy']:.3f} auc={hybrid.get('roc_auc', float('nan')):.3f} "
          f"f1_at_best={hybrid.get('f1_at_best', float('nan')):.3f} (thr={hybrid.get('best_threshold', float('nan')):.3f})")
    return dict(n_pairs=len(fc_y), forecast_quality=fq, hybrid=hybrid)


import json
import os

results = {}
if os.path.exists("./results/ablation.json"):
    with open("./results/ablation.json") as f:
        results = json.load(f)
    logging.info("Loaded %d existing ablation cells from disk, reusing them: %s", len(results), list(results.keys()))

if "sectors12_no_oversample" not in results:
    results["sectors12_no_oversample"] = run_cell([1, 2], 0, "sectors=1-2, oversample=OFF")
if "all_sectors_no_oversample" not in results:
    results["all_sectors_no_oversample"] = run_cell(None, 0, "sectors=ALL, oversample=OFF")
if "sectors12_with_oversample" not in results:
    results["sectors12_with_oversample"] = run_cell([1, 2], 4, "sectors=1-2, oversample=ON")
if "all_sectors_with_oversample" not in results:
    results["all_sectors_with_oversample"] = run_cell(None, 4, "sectors=ALL, oversample=ON")

with open("./results/ablation.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved ./results/ablation.json")
