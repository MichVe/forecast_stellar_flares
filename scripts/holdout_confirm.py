# -*- coding: utf-8 -*-
"""Confirmatory test: evaluate the already-trained, frozen checkpoints (no retraining,
no hyperparameter changes) on 8 stars never seen during development or training, to
check the ETSformer-over-baselines result isn't an artifact of iterating on the same
8-star training set.
"""
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import torch

from src.data import make_classification_windows, fetch_light_curve
from src.forecaster import ETSformerForecaster, LSTMForecaster, NaivePersistenceForecaster
from src.stella_model import StellaCNN
from src.evaluate import evaluate_hybrid

HELDOUT_TICS = [441398770, 201883033, 364588501, 220536421, 425938795, 220556639, 220433364, 220433363]
LOOKBACK, HORIZON, WINDOW = 150, 50, 200
device = "cpu"

classifier = StellaCNN(window=WINDOW)
classifier.load_state_dict(torch.load("./results/stella_classifier.pt", map_location=device))
classifier.eval()

ets = ETSformerForecaster()
ets.load_state_dict(torch.load("./results/etsformer.pt", map_location=device))
ets.eval()

lstm = LSTMForecaster()
lstm.load_state_dict(torch.load("./results/lstm_baseline.pt", map_location=device))
lstm.eval()

naive = NaivePersistenceForecaster()

Xp, yp = [], []
for tic in HELDOUT_TICS:
    lc = fetch_light_curve(f"TIC {tic}", sectors=[1, 2], cache_dir="./tess_cache")
    if lc is None:
        print(f"TIC {tic}: no data, skipping")
        continue
    cx, cy = make_classification_windows(lc, window=WINDOW, stride=100)
    if len(cx):
        Xp.append(cx)
        yp.append(cy)

X_heldout, y_heldout = np.concatenate(Xp), np.concatenate(yp)
print(f"\nHeld-out set: {len(y_heldout)} windows, {int(y_heldout.sum())} positive\n")

forecasters = [("ETSformer", ets), ("LSTM", lstm), ("NaivePersistence", naive)]
oracle_done = False
for name, model in forecasters:
    result = evaluate_hybrid(model, classifier, X_heldout, y_heldout, LOOKBACK, HORIZON, device=device)
    h = result["hybrid"]
    line = f"{name:20s} acc={h['accuracy']:.3f} f1={h['f1']:.3f}"
    if "roc_auc" in h:
        line += f" auc={h['roc_auc']:.3f}"
    if "f1_at_best" in h:
        line += f"  f1@best={h['f1_at_best']:.3f} (thr={h['best_threshold']:.3f})"
    print(line)
    if not oracle_done:
        o = result["oracle"]
        oline = f"{'Oracle':20s} acc={o['accuracy']:.3f} f1={o['f1']:.3f}"
        if "roc_auc" in o:
            oline += f" auc={o['roc_auc']:.3f}"
        if "f1_at_best" in o:
            oline += f"  f1@best={o['f1_at_best']:.3f} (thr={o['best_threshold']:.3f})"
        print(oline)
        oracle_done = True
