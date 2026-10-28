# -*- coding: utf-8 -*-
"""Bootstrap confidence intervals on AUC/F1 for the hybrid flare-prediction pipeline,
reusing the already-trained checkpoints (no retraining) and the exact same
train/val/test split (seed=42) as the reported run.
"""
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

from src.data import DEFAULT_TARGETS, DEFAULT_SECTORS, fetch_all, make_classification_windows
from src.forecaster import ETSformerForecaster, LSTMForecaster, NaivePersistenceForecaster
from src.stella_model import StellaCNN

SEED = 42
LOOKBACK, HORIZON, WINDOW = 150, 50, 200
N_BOOT = 2000

torch.manual_seed(SEED)
np.random.seed(SEED)

curves = fetch_all(DEFAULT_TARGETS, sectors=DEFAULT_SECTORS, cache_dir="./tess_cache")
Xp, yp = [], []
for lc in curves:
    cx, cy = make_classification_windows(lc, window=WINDOW, stride=100)
    if len(cx):
        Xp.append(cx)
        yp.append(cy)
cls_X, cls_y = np.concatenate(Xp), np.concatenate(yp)

_, X_tmp, _, y_tmp = train_test_split(cls_X, cls_y, test_size=0.2, stratify=cls_y, random_state=SEED)
_, X_test, _, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
print("test set:", len(y_test), "positives:", int(y_test.sum()))

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

forecasters = [("ETSformer", ets), ("LSTM", lstm), ("NaivePersistence", naive)]


@torch.no_grad()
def get_probs(forecaster, X_windows):
    n = len(X_windows)
    y_prob_hybrid = np.zeros(n, dtype=np.float32)
    y_prob_oracle = np.zeros(n, dtype=np.float32)
    bs = 64
    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch = torch.from_numpy(X_windows[start:end]).to(device)
        lb = batch[:, :LOOKBACK].unsqueeze(-1)
        true_future = batch[:, LOOKBACK:LOOKBACK + HORIZON]
        forecast = forecaster(lb, horizon=HORIZON).squeeze(-1)
        recon_hybrid = torch.cat([batch[:, :LOOKBACK], forecast], dim=1)
        recon_oracle = torch.cat([batch[:, :LOOKBACK], true_future], dim=1)
        y_prob_hybrid[start:end] = torch.sigmoid(classifier(recon_hybrid)).cpu().numpy()
        y_prob_oracle[start:end] = torch.sigmoid(classifier(recon_oracle)).cpu().numpy()
    return y_prob_hybrid, y_prob_oracle


def bootstrap_ci(y_true, y_prob, threshold, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
        f1s.append(f1_score(yt, (yp >= threshold).astype(int), zero_division=0))
    return np.array(aucs), np.array(f1s)


def pct_ci(arr):
    return np.percentile(arr, 2.5), np.percentile(arr, 50), np.percentile(arr, 97.5)


results = {}
oracle_probs_ref = None
for name, model in forecasters:
    y_prob_hybrid, y_prob_oracle = get_probs(model, X_test)
    # best threshold from the point estimate (same approach as evaluate.py)
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_hybrid)
    denom = precisions + recalls
    f1s_curve = np.divide(2 * precisions * recalls, denom, out=np.zeros_like(denom), where=denom > 0)
    best_threshold = float(thresholds[int(np.argmax(f1s_curve[:-1]))])

    aucs, f1s = bootstrap_ci(y_test, y_prob_hybrid, best_threshold)
    auc_lo, auc_med, auc_hi = pct_ci(aucs)
    f1_lo, f1_med, f1_hi = pct_ci(f1s)
    point_auc = roc_auc_score(y_test, y_prob_hybrid)
    point_f1 = f1_score(y_test, (y_prob_hybrid >= best_threshold).astype(int))
    print(f"{name:20s} AUC={point_auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]   F1@best={point_f1:.3f} [{f1_lo:.3f}, {f1_hi:.3f}]  (threshold={best_threshold:.3f})")
    results[name] = dict(auc=point_auc, auc_ci=(auc_lo, auc_hi), f1=point_f1, f1_ci=(f1_lo, f1_hi))

    if oracle_probs_ref is None:
        aucs_o, f1s_o = bootstrap_ci(y_test, y_prob_oracle, best_threshold)
        auc_lo_o, auc_med_o, auc_hi_o = pct_ci(aucs_o)
        point_auc_o = roc_auc_score(y_test, y_prob_oracle)
        print(f"{'Oracle':20s} AUC={point_auc_o:.3f} [{auc_lo_o:.3f}, {auc_hi_o:.3f}]")
        oracle_probs_ref = y_prob_oracle

# Pairwise: is ETSformer's AUC advantage over LSTM/NaivePersistence outside bootstrap noise?
# (paired bootstrap on the *difference*, same resampled indices for both models)
print("\n--- Paired bootstrap: AUC(ETSformer) - AUC(other), on the same resamples ---")
y_prob_ets, _ = get_probs(ets, X_test)
y_prob_lstm, _ = get_probs(lstm, X_test)
y_prob_naive, _ = get_probs(naive, X_test)

rng = np.random.default_rng(SEED)
n = len(y_test)
for other_name, y_prob_other in [("LSTM", y_prob_lstm), ("NaivePersistence", y_prob_naive)]:
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yt = y_test[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(roc_auc_score(yt, y_prob_ets[idx]) - roc_auc_score(yt, y_prob_other[idx]))
    diffs = np.array(diffs)
    lo, med, hi = pct_ci(diffs)
    frac_positive = float((diffs > 0).mean())
    print(f"ETSformer - {other_name:16s}: diff median={med:+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]  P(diff>0)={frac_positive:.3f}")
