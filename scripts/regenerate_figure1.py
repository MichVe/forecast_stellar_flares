# -*- coding: utf-8 -*-
"""Regenerates a high-resolution 'light curve dataset sample' figure (replacement for
the low-res, borrowed Figure 1) from our own real, catalog-labeled TESS windows --
in the original Feinstein-style relative-flux view (median-normalized, ~1.0 baseline),
not the z-scored view used internally for classifier training.
"""
import logging
logging.basicConfig(level=logging.WARNING)
import numpy as np
import matplotlib.pyplot as plt

from src.data import fetch_light_curve, MEDIAN_CADENCE_GAP_FACTOR, _median_cadence

WINDOW = 200

def raw_windows(lc, window=200, stride=20, positive_margin=50):
    """Same slicing/labeling as make_classification_windows, but median-normalized
    (relative flux ~1.0 baseline) instead of z-scored -- for illustration only."""
    flux, time = lc.flux, lc.time
    n = len(flux)
    dt = _median_cadence(time)
    windows, labels, centers = [], [], []

    def add(start, label):
        end = start + window
        if start < 0 or end > n:
            return
        if np.max(np.diff(time[start:end])) > MEDIAN_CADENCE_GAP_FACTOR * dt:
            return
        w = flux[start:end].copy()
        w = w / np.median(w)
        windows.append(w)
        labels.append(label)
        centers.append(start + window // 2)

    for start in range(0, n - window, stride):
        center = start + window // 2
        label = 0
        for ev in lc.flares:
            if abs(ev.peak - center) <= positive_margin:
                label = 1
                break
        add(start, label)

    for ev in lc.flares:
        add(ev.peak - window // 2, 1)

    return np.array(windows), np.array(labels)


targets = ["TIC 441420236", "TIC 150359500", "TIC 231914259", "TIC 234506911"]
flare_examples, nonflare_examples = [], []

for t in targets:
    lc = fetch_light_curve(t, sectors=[1, 2], cache_dir="./tess_cache")
    X, y = raw_windows(lc, window=WINDOW, stride=20)
    pos = X[y == 1]
    neg = X[y == 0]
    # prefer clean, single-peak-looking flare windows: peak amplitude clearly above
    # baseline but not absurdly large, and low pre-flare scatter
    if len(pos):
        amp = pos.max(axis=1) - np.median(pos[:, :WINDOW // 3], axis=1)
        order = np.argsort(-amp)
        for i in order:
            if 0.005 < amp[i] < 0.08:
                flare_examples.append(pos[i])
                break
    if len(neg):
        scatter = neg.std(axis=1)
        order = np.argsort(scatter)
        # take a calm one, skip a couple of the very flattest to get some variety
        idx = order[min(3, len(order) - 1)]
        nonflare_examples.append(neg[idx])

flare_examples = flare_examples[:4]
nonflare_examples = nonflare_examples[:4]

fig, axes = plt.subplots(2, 4, figsize=(12, 5), dpi=300)
for ax, w in zip(axes[0], flare_examples):
    ax.plot(w, color="red", linewidth=0.9)
    ax.set_title("Flare", fontsize=11)
    ax.set_xlabel("Cadences")
for ax, w in zip(axes[1], nonflare_examples):
    ax.plot(w, color="black", linewidth=0.9)
    ax.set_title("Non-flare", fontsize=11)
    ax.set_xlabel("Cadences")
axes[0, 0].set_ylabel("Relative flux")
axes[1, 0].set_ylabel("Relative flux")

fig.tight_layout()
out = "./results/light_curve_dataset_sample.png"
fig.savefig(out, dpi=300)
print("Saved:", out, "-", flare_examples[0].shape if flare_examples else "no flare examples found")
print("n flare examples:", len(flare_examples), "n non-flare examples:", len(nonflare_examples))
