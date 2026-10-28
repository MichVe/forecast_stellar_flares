from __future__ import annotations

import numpy as np


def plot_example_windows(X: np.ndarray, y: np.ndarray, path: str, n: int = 8):
    """Reproduces the paper's Figure 1 style: a grid of flare vs non-flare windows."""
    import matplotlib.pyplot as plt

    half = n // 2
    idx_pos = np.where(y == 1)[0][:half]
    idx_neg = np.where(y == 0)[0][:half]
    idx = np.concatenate([idx_pos, idx_neg])
    if len(idx) == 0:
        return

    fig, axes = plt.subplots(2, half, figsize=(3 * half, 5))
    axes = np.atleast_2d(axes)
    for ax, i in zip(axes.flat, idx):
        ax.plot(X[i], color="red" if y[i] == 1 else "black", linewidth=0.8)
        ax.set_title("Flare" if y[i] == 1 else "Non-flare", fontsize=9)
        ax.set_xlabel("Cadences")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forecast_example(lookback: np.ndarray, true_future: np.ndarray, forecast: np.ndarray, path: str, label: str = "Forecast"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    t_lb = np.arange(len(lookback))
    t_fut = np.arange(len(lookback), len(lookback) + len(true_future))
    ax.plot(t_lb, lookback, color="black", label="Lookback (observed)")
    ax.plot(t_fut, true_future, color="green", label="True future")
    ax.plot(t_fut, forecast, color="red", linestyle="--", label=label)
    ax.axvline(len(lookback), color="gray", linestyle=":")
    ax.set_xlabel("Cadences")
    ax.set_ylabel("Normalized flux")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
