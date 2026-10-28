"""LaTeX tables for the v2 paper (RASTI style: horizontal rules only at top, under the
headings and at the bottom), generated from saved results (no retraining).

    tab_dataset.tex      per-star flares, samples and positives (H = 30)
    tab_star.tex         star split: ROC-AUC per horizon, PR-AUC and TSS at H = 30
    tab_star_boot.tex    star split: paired star-cluster bootstrap, AUC difference vs Poisson
    tab_time.tex         chronological split: ROC-AUC per horizon
    tab_forecast.tex     forecast error of the three forecasters (normalised flux units)

Usage:
    python scripts/make_tables_v2.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_forecasting import DEFAULT_TARGETS, HOLDOUT_TARGETS
from src.data import fetch_all
from src.task import build_dataset

OUT = os.path.join("paper", "tables")
HORIZONS = [5, 15, 30, 60]
ORDER = [  # (model key, label in the paper, group)
    ("Oracle", "Oracle (true future)$^\\dagger$", "ref"),
    ("History-LR", "History-LR", "hist"),
    ("Poisson", "Poisson", "hist"),
    ("History+LC-GBM", "History+LC-GBM", "hist"),
    ("Recency", "Recency", "hist"),
    ("Renewal", "Renewal", "hist"),
    ("Hybrid-ETSformer", "Hybrid-ETSformer", "lc"),
    ("Hybrid-LSTM", "Hybrid-LSTM", "lc"),
    ("Hybrid-Naive", "Hybrid-Naive", "lc"),
    ("Direct-CNN", "Direct-CNN", "lc"),
    ("Climatology", "Climatology", "ref"),
]
GROUP_HEAD = {"hist": "\\emph{Flare history}", "lc": "\\emph{Light curve}"}


def rows(path):
    with open(path) as f:
        return {(r["horizon"], r["model"]): r for r in json.load(f)["runs"]}


def boot(path):
    with open(path) as f:
        return {(b["horizon"], b["model"]): b["per_seed"] for b in json.load(f)["bootstrap"]}


def fmt(v, s=None, d=3):
    out = f"{v:.{d}f}"
    return out if s is None or s == 0 else f"{out}\\,$\\pm$\\,{s:.{d}f}"


def write(name, body):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(body)
    print(f"wrote {OUT}/{name}")


def model_rows(r, cells, best_by_h=None):
    lines, group = [], None
    for key, label, g in ORDER:
        if g in GROUP_HEAD and g != group:
            lines.append(f"\\multicolumn{{7}}{{l}}{{{GROUP_HEAD[g]}}} \\\\")
        group = g
        lines.append(f"{label} & " + " & ".join(cells(r, key)) + " \\\\")
    return "\n".join(lines)


def auc_table(path, caption, label, extra=True):
    r = rows(path)
    best = {h: max((r[(h, k)]["roc_auc"], k) for k, _, g in ORDER if g != "ref")[1] for h in HORIZONS}

    def cells(r, key):
        c = []
        for h in HORIZONS:
            x = fmt(r[(h, key)]["roc_auc"], r[(h, key)].get("roc_auc_std"))
            c.append(f"\\textbf{{{x}}}" if best[h] == key else x)
        if extra:
            c += [fmt(r[(30, key)]["pr_auc"]), fmt(r[(30, key)]["tss"])]
        return c

    base = " & ".join(f"{100 * r[(h, 'Oracle')]['base_rate']:.1f}\\%" for h in HORIZONS)
    head_extra = " & PR-AUC & TSS" if extra else ""
    sub_extra = " & \\multicolumn{2}{c}{$H=30$}" if extra else ""
    cols = "l" + "c" * (len(HORIZONS) + (2 if extra else 0))
    return f"""\\begin{{table*}}
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{cols}}}
\\hline
 & \\multicolumn{{4}}{{c}}{{ROC-AUC at horizon $H$ [cadences]}}{sub_extra} \\\\
Model & $H=5$ (10 min) & $H=15$ (30 min) & $H=30$ (1 h) & $H=60$ (2 h){head_extra} \\\\
\\hline
{model_rows(r, cells)}
Positive rate & {base}{' & & ' if extra else ''} \\\\
\\hline
\\end{{tabular}}
\\end{{table*}}
"""


def boot_table(path):
    b = boot(path)
    models = ["History-LR", "History+LC-GBM", "Recency", "Renewal", "Hybrid-ETSformer", "Hybrid-LSTM",
              "Hybrid-Naive", "Direct-CNN"]
    lines = []
    for m in models:
        cells = []
        for h in HORIZONS:
            res = b[(h, m)]
            d = np.mean([x["auc_diff"] for x in res])
            lo = np.mean([x["ci95"][0] for x in res])
            hi = np.mean([x["ci95"][1] for x in res])
            c = f"${d:+.3f}$ [${lo:+.3f}$, ${hi:+.3f}$]"
            cells.append(f"\\textbf{{{c}}}" if lo > 0 or hi < 0 else c)
        lines.append(f"{m} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(lines)
    return f"""\\begin{{table*}}
\\centering
\\caption{{Star split: paired star-cluster bootstrap of the ROC-AUC difference to the Poisson model
(test predictions of the four folds pooled per seed, 1000 resamples of stars, mean over five seeds).
Bold: 95\\% interval excludes zero.}}
\\label{{tab:star_boot}}
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{lcccc}}
\\hline
Model $-$ Poisson & $H=5$ & $H=15$ & $H=30$ & $H=60$ \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table*}}
"""


def forecast_table():
    acc = defaultdict(list)
    for split, d in (("star", "results_v2/full_star"), ("time", "results_v2/full_time")):
        with open(os.path.join(d, "runs.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                for m, v in r["forecast_quality"].items():
                    for k in ("MAE", "MAE_flare"):
                        acc[(split, r["horizon"], m, k)].append(v[k])
    lines = []
    for m in ("ETSformer", "LSTM", "Naive"):
        cells = [f"{np.mean(acc[('star', h, m, k)]):.3f}" for h in (5, 30, 60) for k in ("MAE", "MAE_flare")]
        lines.append(f"{m} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(lines)
    return f"""\\begin{{table}}
\\centering
\\caption{{Forecast error on the star-split test sets (mean over 20 runs), in units of the lookback's
robust standard deviation; MAE$_\\mathrm{{fl}}$: samples whose horizon contains a flare onset.}}
\\label{{tab:forecast}}
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\begin{{tabular}}{{lcccccc}}
\\hline
 & \\multicolumn{{2}}{{c}}{{$H=5$}} & \\multicolumn{{2}}{{c}}{{$H=30$}} & \\multicolumn{{2}}{{c}}{{$H=60$}} \\\\
Forecaster & MAE & MAE$_\\mathrm{{fl}}$ & MAE & MAE$_\\mathrm{{fl}}$ & MAE & MAE$_\\mathrm{{fl}}$ \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}
"""


def dataset_table():
    with open("results_v2/full_star/config.json") as f:
        cfg = json.load(f)
    curves = fetch_all(cfg["targets"], sectors=cfg["sectors"], cache_dir=cfg["cache_dir"])
    ds = build_dataset(curves, cfg["lookback"], 30, cfg["stride"], cfg["clip"])
    lines, tot = [], np.zeros(4, dtype=int)
    for i, lc in enumerate(curves):
        m = ds.star == i
        vals = [len(lc.flares), len(lc.flux), int(m.sum()), int(ds.y[m].sum())]
        tot += vals
        role = "" if lc.target in DEFAULT_TARGETS else "$^\\ast$"
        lines.append(f"{lc.target}{role} & " + " & ".join(f"{v:,}".replace(",", "{,}") for v in vals) + " \\\\")
    body = "\n".join(lines)
    total = " & ".join(f"{v:,}".replace(",", "{,}") for v in tot)
    assert set(cfg["targets"]) == set(DEFAULT_TARGETS + HOLDOUT_TARGETS)
    return f"""\\begin{{table}}
\\centering
\\caption{{Dataset: 16 M dwarfs observed in TESS sectors 1--2 at 2-min cadence, flares from
\\citet{{gunther2020stellar}}. Samples: anchors with stride 5 cadences, $L=150$, $H=30$.
$^\\ast$The eight stars used only for the held-out check of the v1 manuscript.}}
\\label{{tab:dataset}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{lrrrr}}
\\hline
Star & Flares & Cadences & Samples & Positive \\\\
\\hline
{body}
Total & {total} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    write("tab_dataset.tex", dataset_table())
    write("tab_star.tex", auc_table(
        "results_v2/summary_star_vs_poisson.json",
        "Star split (4-fold, unseen test stars): ROC-AUC, mean\\,$\\pm$\\,s.d.\\ over 20 runs (4 folds $\\times$ "
        "5 seeds); PR-AUC and TSS (threshold chosen on validation) at $H=30$. Bold: best non-oracle model. "
        "$^\\dagger$Not a forecast: classifies the true future.", "tab:star"))
    write("tab_star_boot.tex", boot_table("results_v2/summary_star_vs_poisson.json"))
    write("tab_time.tex", auc_table(
        "results_v2/summary_time_vs_poisson.json",
        "Chronological split (last 15\\% of each star's light curve as test): ROC-AUC, mean\\,$\\pm$\\,s.d.\\ "
        "over 5 seeds (flare-history models are deterministic); PR-AUC and TSS at $H=30$. "
        "Bold: best non-oracle model. $^\\dagger$Not a forecast.", "tab:time"))
    write("tab_forecast.tex", forecast_table())
