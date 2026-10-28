# -*- coding: utf-8 -*-
"""Trains and evaluates an Informer forecaster (Hugging Face `transformers`) on the
exact same forecaster training data as the reported ETSformer/LSTM run (all sectors,
flare-aware oversampling), and evaluates it the same way (forecast MAE/MSE with a
flare-horizon breakdown, plus the hybrid pipeline via the frozen Stella classifier).

Configuration choices are kept explicit and conservative given the risk that a naive
setup could under-train a genuinely more complex, probabilistic model:
  - context_length = LOOKBACK - max(lags_sequence), so past_values consumes exactly
    our 150-cadence lookback window (no extra data available or fabricated).
  - a single synthetic time feature (normalized cadence index), since our data has no
    real calendar time.
  - d_model=64, 2 encoder/2 decoder layers, 4 heads -- matched to ETSformer's own
    configuration (see Results) for comparable capacity.
  - trained with the model's own default negative-log-likelihood loss (Student-T
    output distribution); point forecasts for evaluation are the mean of 100 sampled
    trajectories from model.generate(), the standard approach for these HF models.
"""
import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import InformerConfig, InformerForPrediction

from src.data import DEFAULT_TARGETS, DEFAULT_SECTORS, fetch_all, make_classification_windows, make_forecast_windows
from src.stella_model import StellaCNN
from src.evaluate import mae, mse

SEED = 42
LOOKBACK, HORIZON, WINDOW = 150, 50, 200
LAGS = [1, 2, 3, 4, 5]
CONTEXT_LENGTH = LOOKBACK - max(LAGS)
EPOCHS = 10
BATCH_SIZE = 64
LR = 3e-4
device = "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- same forecaster training data as the reported ETSformer/LSTM run ----
logging.info("Fetching forecaster data (all sectors, flare-aware oversampling)...")
fc_curves = fetch_all(DEFAULT_TARGETS, sectors=None, cache_dir="./tess_cache")
fc_Xp, fc_yp = [], []
for lc in fc_curves:
    fx, fy = make_forecast_windows(lc, lookback=LOOKBACK, horizon=HORIZON, stride=25, oversample_near_flares=4)
    if len(fx):
        fc_Xp.append(fx)
        fc_yp.append(fy)
fc_X, fc_y = np.concatenate(fc_Xp), np.concatenate(fc_yp)
logging.info("Forecast training pairs: %d", len(fc_y))

X_train, X_tmp, y_train, y_tmp = train_test_split(fc_X, fc_y, test_size=0.2, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED)


def to_batches(X, y, batch_size, shuffle):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(SEED).shuffle(idx)
    for start in range(0, n, batch_size):
        sel = idx[start:start + batch_size]
        yield X[sel], y[sel]


def make_time_features(batch_size, length, offset=0):
    t = (torch.arange(offset, offset + length).float() / LOOKBACK).view(1, -1, 1)
    return t.repeat(batch_size, 1, 1)


config = InformerConfig(
    prediction_length=HORIZON,
    context_length=CONTEXT_LENGTH,
    lags_sequence=LAGS,
    num_time_features=1,
    num_static_categorical_features=0,
    num_static_real_features=0,
    d_model=64,
    encoder_layers=2,
    decoder_layers=2,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    dropout=0.1,
    scaling="mean",
)
model = InformerForPrediction(config).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

best_val = float("inf")
best_state = None

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    train_loss, n_batches = 0.0, 0
    for xb, yb in to_batches(X_train, y_train, BATCH_SIZE, shuffle=True):
        xb_t = torch.from_numpy(xb).float().to(device)
        yb_t = torch.from_numpy(yb).float().to(device)
        b = xb_t.shape[0]
        out = model(
            past_values=xb_t,
            past_time_features=make_time_features(b, LOOKBACK),
            past_observed_mask=torch.ones_like(xb_t),
            future_values=yb_t,
            future_time_features=make_time_features(b, HORIZON, offset=LOOKBACK),
        )
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        train_loss += out.loss.item()
        n_batches += 1
    train_loss /= max(n_batches, 1)

    model.eval()
    val_loss, n_val = 0.0, 0
    with torch.no_grad():
        for xb, yb in to_batches(X_val, y_val, BATCH_SIZE, shuffle=False):
            xb_t = torch.from_numpy(xb).float().to(device)
            yb_t = torch.from_numpy(yb).float().to(device)
            b = xb_t.shape[0]
            out = model(
                past_values=xb_t,
                past_time_features=make_time_features(b, LOOKBACK),
                past_observed_mask=torch.ones_like(xb_t),
                future_values=yb_t,
                future_time_features=make_time_features(b, HORIZON, offset=LOOKBACK),
            )
            val_loss += out.loss.item()
            n_val += 1
    val_loss /= max(n_val, 1)
    logging.info("informer epoch %d: train_loss=%.4f val_loss=%.4f (%.1fs)", epoch, train_loss, val_loss, time.time() - t0)
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)
torch.save(model.state_dict(), "./results/informer.pt")

# ---- point-forecast evaluation (forecast quality, same style as evaluate.py) ----
model.eval()


@torch.no_grad()
def point_forecast(xb_np, num_samples=100):
    xb_t = torch.from_numpy(xb_np).float().to(device)
    b = xb_t.shape[0]
    gen = model.generate(
        past_values=xb_t,
        past_time_features=make_time_features(b, LOOKBACK),
        past_observed_mask=torch.ones_like(xb_t),
        future_time_features=make_time_features(b, HORIZON, offset=LOOKBACK),
    )
    return gen.sequences.mean(dim=1).cpu().numpy()  # (b, HORIZON)


preds, trues = [], []
for xb, yb in to_batches(X_test, y_test, BATCH_SIZE, shuffle=False):
    preds.append(point_forecast(xb))
    trues.append(yb)
preds = np.concatenate(preds)[..., None]
trues = np.concatenate(trues)[..., None]

m = mse(preds, trues)
fq = {"MAE": mae(preds, trues), "MSE": m, "RMSE": float(np.sqrt(m))}
excursion = trues.max(axis=1) - 1.0
is_flare = excursion.reshape(-1) > 0.01
if is_flare.any():
    fq["MAE_flare_horizon"] = mae(preds[is_flare], trues[is_flare])
    fq["MSE_flare_horizon"] = mse(preds[is_flare], trues[is_flare])
    fq["n_flare_horizon"] = int(is_flare.sum())
if (~is_flare).any():
    fq["MAE_quiescent_horizon"] = mae(preds[~is_flare], trues[~is_flare])
    fq["MSE_quiescent_horizon"] = mse(preds[~is_flare], trues[~is_flare])
    fq["n_quiescent_horizon"] = int((~is_flare).sum())

print("\n=== Informer forecast quality ===")
print(fq)

# ---- hybrid pipeline: reuse the same frozen classifier + fixed classification test split ----
cls_curves = fetch_all(DEFAULT_TARGETS, sectors=DEFAULT_SECTORS, cache_dir="./tess_cache")
Xp, yp = [], []
for lc in cls_curves:
    cx, cy = make_classification_windows(lc, window=WINDOW, stride=100)
    if len(cx):
        Xp.append(cx)
        yp.append(cy)
cls_X, cls_y = np.concatenate(Xp), np.concatenate(yp)
_, Xt, _, yt = train_test_split(cls_X, cls_y, test_size=0.2, stratify=cls_y, random_state=SEED)
_, X_cls_test, _, y_cls_test = train_test_split(Xt, yt, test_size=0.5, stratify=yt, random_state=SEED)

classifier = StellaCNN(window=WINDOW)
classifier.load_state_dict(torch.load("./results/stella_classifier.pt", map_location=device))
classifier.eval()

from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score

n = len(X_cls_test)
y_prob = np.zeros(n, dtype=np.float32)
with torch.no_grad():
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = X_cls_test[start:end]
        lb = batch[:, :LOOKBACK]
        forecast = point_forecast(lb)
        recon = np.concatenate([lb, forecast], axis=1)
        y_prob[start:end] = torch.sigmoid(classifier(torch.from_numpy(recon).float())).cpu().numpy()

y_pred = (y_prob >= 0.5).astype(int)
precisions, recalls, thresholds = precision_recall_curve(y_cls_test, y_prob)
denom = precisions + recalls
f1s = np.divide(2 * precisions * recalls, denom, out=np.zeros_like(denom), where=denom > 0)
best_threshold = float(thresholds[int(np.argmax(f1s[:-1]))]) if len(thresholds) else 0.5
y_pred_best = (y_prob >= best_threshold).astype(int)

hybrid = {
    "accuracy": accuracy_score(y_cls_test, y_pred),
    "precision": precision_score(y_cls_test, y_pred, zero_division=0),
    "recall": recall_score(y_cls_test, y_pred, zero_division=0),
    "f1": f1_score(y_cls_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_cls_test, y_prob),
    "best_threshold": best_threshold,
    "f1_at_best": f1_score(y_cls_test, y_pred_best, zero_division=0),
}
print("\n=== Informer hybrid flare prediction ===")
print(hybrid)

import json
with open("./results/informer_report.json", "w") as f:
    json.dump({"forecast_quality": fq, "hybrid": hybrid}, f, indent=2)
print("\nSaved ./results/informer_report.json")
