"""Aggregate v2 results (run_forecasting.py) into paper-ready tables.

1. Per (horizon, model): mean +- std over all seed x fold runs for ROC-AUC, PR-AUC, TSS,
   HSS, F1 and the base rate.
2. Paired star-cluster bootstrap of the ROC-AUC difference between each model and a
   reference model: resampling *stars* (not windows), since samples from one star are
   strongly correlated. Test predictions of all folds of one seed are pooled (every star
   is a test star exactly once under the star split), then results are summarised across seeds.

Usage:
    python scripts/aggregate_v2.py --results results_v2 --reference Direct-CNN
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

METRICS = ["roc_auc", "pr_auc", "tss", "hss", "f1", "base_rate"]


def summarise_runs(records):
    table = defaultdict(lambda: defaultdict(list))
    for r in records:
        for model, m in r["metrics"].items():
            for k in METRICS:
                if k in m:
                    table[(r["horizon"], model)][k].append(m[k])
    rows = []
    for (H, model), vals in sorted(table.items()):
        row = {"horizon": H, "model": model, "n_runs": len(vals["tss"])}
        for k in METRICS:
            if vals[k]:
                row[k] = float(np.mean(vals[k]))
                row[k + "_std"] = float(np.std(vals[k]))
        rows.append(row)
    return rows


def load_pooled(results_dirs):
    """{(H, seed): dict(y, star, score/<model>...)} with all folds of one seed concatenated.

    Predictions of the same (H, seed, fold) from several result directories are merged
    column-wise; they must describe identical test samples (checked via star/anchor/y)."""
    runs = defaultdict(dict)
    for results_dir in results_dirs:
        for path in glob.glob(os.path.join(results_dir, "predictions", "H*_seed*_fold*.npz")):
            key = tuple(map(int, re.findall(r"H(\d+)_seed(\d+)_fold(\d+)", path)[0]))
            d = dict(np.load(path))
            if runs[key]:
                for k in ("y", "star", "anchor"):
                    assert np.array_equal(runs[key][k], d[k]), f"{path}: test samples differ ({k})"
            runs[key].update(d)
    groups = defaultdict(list)
    for (H, seed, _), d in sorted(runs.items()):
        groups[(H, seed)].append(d)
    pooled = {}
    for key, parts in groups.items():
        keys = set.intersection(*(set(p) for p in parts))
        pooled[key] = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    return pooled


def fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic (ties averaged) -- identical to
    sklearn's roc_auc_score. Used in the bootstrap loop: roc_auc_score calls np.unique,
    which leaks memory in numpy 2.4.1 (~0.3 MB per call), so 10^5 bootstrap calls ran
    the machine out of memory."""
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    ranks = rankdata(score)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cluster_bootstrap_auc_diff(y, star, s_a, s_b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    stars = np.unique(star)
    idx_by_star = {s: np.flatnonzero(star == s) for s in stars}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(stars, size=len(stars), replace=True)
        idx = np.concatenate([idx_by_star[s] for s in pick])
        n_pos = int(y[idx].sum())  # not np.unique: it leaks ~0.3 MB per call in numpy 2.4.1
        if n_pos == 0 or n_pos == len(idx):
            continue
        diffs.append(fast_auc(y[idx], s_a[idx]) - fast_auc(y[idx], s_b[idx]))
    diffs = np.asarray(diffs)
    return {
        "auc_diff": float(roc_auc_score(y, s_a) - roc_auc_score(y, s_b)),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_greater": float((diffs > 0).mean()),
        "n_boot": int(len(diffs)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=["results_v2"],
                   help="One or more result directories (e.g. neural + history runs) to merge.")
    p.add_argument("--reference", default="Direct-CNN")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default=None, help="Summary JSON path (default: <first results dir>/summary.json).")
    args = p.parse_args()

    records = []
    for d in args.results:
        with open(os.path.join(d, "runs.jsonl")) as f:
            records += [json.loads(line) for line in f if line.strip()]
    rows = summarise_runs(records)

    print(f"{'H':>4} {'model':<18} {'runs':>4} " + " ".join(f"{k:>15}" for k in METRICS))
    for r in rows:
        cells = [f"{r[k]:.3f}+-{r[k + '_std']:.3f}" if k in r else "-" for k in METRICS]
        print(f"{r['horizon']:>4} {r['model']:<18} {r['n_runs']:>4} " + " ".join(f"{c:>15}" for c in cells))

    boot = defaultdict(lambda: defaultdict(list))
    ref_key = f"score/{args.reference}"
    pooled = load_pooled(args.results) if args.n_boot > 0 else {}  # --n-boot 0: quick table only
    for (H, seed), d in sorted(pooled.items()):
        if ref_key not in d:
            continue
        for k in d:
            if k.startswith("score/") and k != ref_key:
                res = cluster_bootstrap_auc_diff(d["y"], d["star"], d[k], d[ref_key], args.n_boot, seed)
                boot[H][k[6:]].append(res)

    print(f"\nPaired star-cluster bootstrap: AUC(model) - AUC({args.reference}), pooled folds, per seed")
    boot_rows = []
    for H in sorted(boot):
        for model, res in sorted(boot[H].items()):
            d = [r["auc_diff"] for r in res]
            lo = [r["ci95"][0] for r in res]
            hi = [r["ci95"][1] for r in res]
            pg = [r["p_greater"] for r in res]
            boot_rows.append({"horizon": H, "model": model, "reference": args.reference, "per_seed": res})
            print(f"H={H:>3} {model:<18} diff={np.mean(d):+.3f}  95%CI~[{np.mean(lo):+.3f}, {np.mean(hi):+.3f}]  "
                  f"P(diff>0)={np.mean(pg):.2f}  (seeds={len(res)})")

    out = args.out or os.path.join(args.results[0], "summary.json")
    with open(out, "w") as f:
        json.dump({"runs": rows, "bootstrap": boot_rows}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
