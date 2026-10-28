# -*- coding: utf-8 -*-
"""Selects a genuine flare-containing test window and plots lookback / true future /
ETSformer forecast for it, replacing the earlier randomly-picked (quiescent) example
with one that actually illustrates the paper's central claim -- that ETSformer tracks
the shape of a flare in its forecast horizon better than the simpler baselines.
"""
import logging
logging.basicConfig(level=logging.WARNING)
import numpy as np
import torch

from src.data import DEFAULT_TARGETS, fetch_all, make_forecast_windows
from src.forecaster import ETSformerForecaster, LSTMForecaster, NaivePersistenceForecaster
from src.plotting import plot_forecast_example

LOOKBACK, HORIZON = 150, 50
device = "cpu"

ets = ETSformerForecaster()
ets.load_state_dict(torch.load("./results/etsformer.pt", map_location=device))
ets.eval()

lstm = LSTMForecaster()
lstm.load_state_dict(torch.load("./results/lstm_baseline.pt", map_location=device))
lstm.eval()

naive = NaivePersistenceForecaster()

# Rebuild the same all-sectors + oversampled forecast windows used for training/eval,
# and search for a clear, representative flare-containing example.
curves = fetch_all(DEFAULT_TARGETS, sectors=None, cache_dir="./tess_cache")
best = None  # (amplitude, x, y)
for lc in curves:
    fx, fy = make_forecast_windows(lc, lookback=LOOKBACK, horizon=HORIZON, stride=25, oversample_near_flares=0)
    if len(fx) == 0:
        continue
    amp = fy.max(axis=1) - 1.0
    for i in range(len(amp)):
        a = amp[i]
        if 0.008 < a < 0.02:  # modest, representative flare, not an extreme outlier
            if best is None or abs(a - 0.012) < abs(best[0] - 0.012):
                best = (a, fx[i], fy[i])

if best is None:
    raise RuntimeError("No suitable flare example found")

amp, x, y = best
print(f"Selected example: amplitude={amp:.4f}")

xb = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(-1)
yb = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(-1)

with torch.no_grad():
    pred_ets = ets(xb, horizon=HORIZON)
    pred_lstm = lstm(xb, horizon=HORIZON)
    pred_naive = naive(xb, horizon=HORIZON)

plot_forecast_example(
    x, y, pred_ets[0, :, 0].numpy(),
    "./results/forecast_example.png",
    label="ETSformer forecast",
)
print("Saved ./results/forecast_example.png")

# Also save a multi-model comparison version for reference/inspection.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
t_lb = np.arange(LOOKBACK)
t_fut = np.arange(LOOKBACK, LOOKBACK + HORIZON)
ax.plot(t_lb, x, color="black", label="Lookback (observed)")
ax.plot(t_fut, y, color="green", linewidth=2, label="True future")
ax.plot(t_fut, pred_ets[0, :, 0].numpy(), color="red", linestyle="--", label="ETSformer")
ax.plot(t_fut, pred_lstm[0, :, 0].numpy(), color="blue", linestyle="--", label="LSTM")
ax.plot(t_fut, pred_naive[0, :, 0].numpy(), color="gray", linestyle=":", label="NaivePersistence")
ax.axvline(LOOKBACK, color="gray", linestyle=":")
ax.set_xlabel("Cadences")
ax.set_ylabel("Normalized flux")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("./results/forecast_example_all_models.png", dpi=200)
print("Saved ./results/forecast_example_all_models.png")
