"""Paper figures for the v2 (leakage-free) experiments.

    fig_auc_horizon   ROC-AUC vs forecast horizon, star split and chronological split
    fig_pipeline      the compared model families and their inputs/outputs (replaces Fig. 2)
    fig_task_leakage  v1 vs v2 window definition on a real flare, and the resulting AUC drop

Reads results_v2/summary_{star,time}_vs_poisson.json, results_v1/report.json and the
processed light-curve cache; writes PDF + PNG to paper/figures/.

Usage:
    python scripts/make_figures_v2.py
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join("paper", "figures")
L, CADENCE_MIN = 150, 2

# Reference categorical palette, slots 1-3 (validated all-pairs incl. CVD); light-curve
# models share one neutral ink and are told apart by marker, since they overlap anyway.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
SERIES = [
    # (model, label, color, marker, linestyle)
    ("Recency", "Recency", ORANGE, "o", "-"),
    ("History-LR", "History-LR", AQUA, "D", "-"),
    ("Poisson", "Poisson", BLUE, "s", "-"),
    ("Climatology", "Climatology", INK, None, ":"),
    ("Hybrid-ETSformer", "Hybrid-ETSformer", INK2, "^", "--"),
    ("Hybrid-Naive", "Hybrid-Naive", MUTED, "v", "--"),
    ("Direct-CNN", "Direct-CNN", INK2, "x", "-."),
]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": True, "axes.spines.right": True, "savefig.dpi": 400,
    "xtick.top": True, "ytick.right": True, "xtick.direction": "in", "ytick.direction": "in",
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf/.png")


def load_rows(path):
    with open(path) as f:
        return {(r["horizon"], r["model"]): r for r in json.load(f)["runs"]}


# ---------------------------------------------------------------- AUC vs horizon
def fig_auc_horizon():
    panels = [
        ("(a) Star split: unseen stars", "results_v2/summary_star_vs_poisson.json"),
        ("(b) Chronological split: future of known stars", "results_v2/summary_time_vs_poisson.json"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, (title, path) in zip(axes, panels):
        rows = load_rows(path)
        horizons = sorted({h for h, _ in rows})
        x = np.arange(len(horizons))
        for i, (model, label, color, marker, ls) in enumerate(SERIES):
            m = np.array([rows[(h, model)]["roc_auc"] for h in horizons])
            s = np.array([rows[(h, model)]["roc_auc_std"] for h in horizons])
            dx = (i - (len(SERIES) - 1) / 2) * 0.035  # dodge so error bars don't overlap
            ax.errorbar(x + dx, m, yerr=s if s.max() > 0 else None, color=color, marker=marker,
                        ls=ls, lw=1.2, ms=4, mfc=color if marker != "x" else None,
                        elinewidth=0.6, capsize=1.5, label=label, zorder=3)
        oracle = [rows[(h, "Oracle")]["roc_auc"] for h in horizons]
        ax.text(0.98, 0.97, f"Oracle (true future): {min(oracle):.2f}–{max(oracle):.2f}, off scale",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=INK2)
        ax.axhline(0.5, color=GRID, lw=0.8, zorder=1)
        ax.set_xticks(x, [f"{h}\n({h * CADENCE_MIN} min)" for h in horizons])
        ax.set_xlabel("Horizon $H$ [cadences]")
        ax.set_title(title, loc="left")
        ax.set_ylim(0.47, 0.68)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    axes[0].set_ylabel("ROC-AUC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.07), handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_auc_horizon")


# ---------------------------------------------------------------- pipeline diagram
def _box(ax, xy, w, h, text, fc="#ffffff", ec=INK2, ls="-", bold=False, fs=7.2):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=0.8, ls=ls))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax, a, b, ls="-", color=INK2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.8,
                                 color=color, ls=ls, shrinkA=0, shrinkB=0))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(-0.05, 11.1)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    h = 0.66
    lb_tint, hist_tint, out_tint = "#eaf2fc", "#e8f7f1", "#f4f3f0"
    c1, c2, c3, c4, c5 = 0.0, 2.35, 5.05, 7.55, 9.15  # column x positions
    w1, w2, w3, w4, w5 = 1.85, 2.2, 2.0, 1.2, 1.85

    # inputs
    _box(ax, (c1, 2.55), w1, h, "Lookback\n$x_{t-L+1..t}$  ($L\\times1$)", fc=lb_tint)
    _box(ax, (c1, 0.3), w1, h, "Flare history\ncatalogue peaks $\\leq t$", fc=hist_tint)

    # (i) hybrid: forecast -> classify
    _box(ax, (c2, 3.4), w2, h, "Forecaster\nETSformer / LSTM / Naive")
    _box(ax, (c3, 3.4), w3, h, "Forecast\n$\\hat x_{t+1..t+H}$  ($H\\times1$)")
    ax.text(c3 + w3 / 2, 3.28, "Oracle: true $x_{t+1..t+H}$ instead", ha="center", va="top",
            fontsize=6.3, color=MUTED, style="italic")
    _box(ax, (c4, 2.55), w4, h, "Concatenate\n$(L+H)\\times1$")
    _box(ax, (c5, 2.55), w5, h, "Stella CNN classifier\n(trained on true\n$(L+H)$ windows)", fs=6.6)
    _arrow(ax, (c1 + w1, 3.05), (c2, 3.6))
    _arrow(ax, (c2 + w2, 3.73), (c3, 3.73))
    _arrow(ax, (c3 + w3, 3.6), (c4, 3.05))
    _arrow(ax, (c1 + w1, 2.75), (c4, 2.75))
    _arrow(ax, (c4 + w4, 2.88), (c5, 2.88))

    # (ii) direct classifier
    _box(ax, (c3, 1.4), w3, h, "Direct-CNN\n(Stella on lookback)")
    _arrow(ax, (0.92, 2.55), (0.92, 1.73))
    _arrow(ax, (0.92, 1.73), (c3, 1.73))

    # (iii) history models
    _box(ax, (c3 - 0.2, 0.3), w3 + 0.4, h, "Poisson · Renewal · Recency\nHistory-LR · History+LC-GBM",
         fs=6.6)
    _arrow(ax, (c1 + w1, 0.63), (c3 - 0.2, 0.63))

    # output
    _box(ax, (c5, 0.75), w5, 1.05,
         "$p\\,$(flare onset\nin $(t,\\,t+H]$)\nthreshold: max TSS\non validation", fc=out_tint, fs=6.8)
    _arrow(ax, (c5 + w5 / 2, 2.55), (c5 + w5 / 2, 1.8))
    _arrow(ax, (c3 + w3, 1.73), (c5, 1.4))
    _arrow(ax, (c3 + w3 + 0.2, 0.63), (c5, 1.0))

    for y, label in [(4.22, "(i) Hybrid: forecast, then classify the window"),
                     (2.2, "(ii) Direct classification of the lookback"),
                     (1.08, "(iii) Flare-history models (no light curve)")]:
        ax.text(c2, y, label, fontsize=7, color=INK2, style="italic")
    ax.text(c1, 4.22, "Inputs", fontsize=7, color=INK2, style="italic")
    save(fig, "fig_pipeline")


# ---------------------------------------------------------------- leakage illustration
def _pick_flare():
    """An isolated, moderate flare with clean data around it (deterministic choice)."""
    for path in sorted(glob.glob(os.path.join("tess_cache", "processed", "*.npz"))):
        d = np.load(path)
        time, flux, flares = d["time"], d["flux"], d["flares"]
        dt = np.median(np.diff(time))
        for start, peak, end in flares:
            lo, hi = peak - 230, peak + 120
            if lo < 0 or hi > len(flux) or np.max(np.diff(time[lo:hi])) > 1.5 * dt:
                continue
            others = flares[(np.abs(flares[:, 1] - peak) < 260) & (flares[:, 1] != peak)]
            amp = flux[peak] / np.nanmedian(flux[lo:hi]) - 1
            if len(others) == 0 and 0.03 < amp < 0.3 and peak - start >= 2:
                name = os.path.basename(path).split("_s")[0].replace("_", " ")
                return name, flux / np.nanmedian(flux[lo:hi]), int(start), int(peak)
    raise RuntimeError("no suitable flare found")


def fig_task_leakage():
    name, flux, onset, peak = _pick_flare()
    v1 = json.load(open("results_v1/report.json"))["hybrid_flare_prediction"]
    star = load_rows("results_v2/summary_star_vs_poisson.json")
    timesplit = load_rows("results_v2/summary_time_vs_poisson.json")

    fig = plt.figure(figsize=(7.2, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.32)

    def lc_panel(ax, w0, n_in, n_out, title, note):
        idx = np.arange(w0 - 20, w0 + n_in + n_out + 20)
        rel = idx - w0
        ax.axvspan(0, n_in, color="#eaf2fc", zorder=0)
        ax.axvspan(n_in, n_in + n_out, color="#fdeee6", zorder=0)
        ax.plot(rel, flux[idx], color=INK, lw=0.7, zorder=2)
        ax.axvline(peak - w0, color=ORANGE, lw=0.8, ls=":", zorder=1)
        top = np.nanmax(flux[idx])
        tr = ax.get_xaxis_transform()
        ax.text(n_in / 2, 0.96, "model input", ha="center", va="top", fontsize=6.8, color=BLUE,
                transform=tr)
        ax.text(n_in + n_out / 2, 0.96, "horizon", ha="center", va="top", fontsize=6.8,
                color=ORANGE, transform=tr)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Cadence relative to window start")
        ax.text(0.03, 0.85, note, transform=ax.transAxes, fontsize=6.4, color=INK2, va="top")
        ax.set_ylim(None, top + 0.035)

    ax_a = fig.add_subplot(gs[0])
    lc_panel(ax_a, peak - 100, 150, 50, "(a) v1: flare peak inside the input",
             "label 1 if peak near cadence 100\n→ task is detection")
    ax_a.set_ylabel("Relative flux")
    ax_b = fig.add_subplot(gs[1], sharey=ax_a)
    anchor = onset - 10  # last observed cadence, 10 cadences before the estimated onset
    lc_panel(ax_b, anchor - L + 1, L, 30, "(b) v2: onset after the anchor $t$",
             "label 1 if onset $> t$ and peak $\\leq t+H$\n→ task is forecasting")

    ax_c = fig.add_subplot(gs[2])
    models = [("ETSformer", "Hybrid-ETSformer"), ("NaivePersistence", "Hybrid-Naive"),
              ("Oracle (true future)", "Oracle")]
    labels = ["Hybrid-\nETSformer", "Hybrid-\nNaive", "Oracle"]
    x = np.arange(len(models))
    vals = [
        ("v1 (reported)", [v1[a]["roc_auc"] for a, _ in models], INK2),
        ("v2 star split", [star[(30, b)]["roc_auc"] for _, b in models], BLUE),
        ("v2 chronological", [timesplit[(30, b)]["roc_auc"] for _, b in models], AQUA),
    ]
    w = 0.26
    for i, (lab, v, c) in enumerate(vals):
        bars = ax_c.bar(x + (i - 1) * w, v, w * 0.9, color=c, label=lab, zorder=2)
        for b_, val in zip(bars, v):
            ax_c.text(b_.get_x() + b_.get_width() / 2, val + 0.01, f"{val:.2f}"[1:], ha="center",
                      fontsize=5.4, color=INK2)
    ax_c.axhline(0.5, color=MUTED, lw=0.7, ls=":", zorder=3)
    ax_c.set_xticks(x, labels)
    ax_c.set_ylim(0.4, 1.32)
    ax_c.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_c.set_ylabel("ROC-AUC")
    ax_c.set_title("(c) Effect on ROC-AUC ($H=30$)", loc="left")
    ax_c.legend(frameon=False, fontsize=6.2, loc="upper left", ncol=1, handlelength=1.0,
                bbox_to_anchor=(0.0, 1.02))
    ax_c.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    print(f"leakage example: {name}, onset {onset}, peak {peak}")
    save(fig, "fig_task_leakage")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_auc_horizon()
    fig_pipeline()
    fig_task_leakage()
