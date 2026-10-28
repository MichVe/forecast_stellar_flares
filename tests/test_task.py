"""Leakage checks for the v2 forecasting task on a synthetic light curve (no network).

Run:  python -m pytest tests/  (or: python tests/test_task.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import LightCurve  # noqa: E402
from src.flare_detection import FlareEvent  # noqa: E402
from src.metrics import evaluate_scores, select_threshold, skill_scores  # noqa: E402
from src.task import (  # noqa: E402
    assert_no_leakage,
    build_dataset,
    build_samples,
    star_folds,
    star_split_masks,
    time_split_masks,
)

L, H = 50, 20


def synthetic_lc(name="S", n=5000, seed=0, n_flares=25, gap_at=None):
    rng = np.random.default_rng(seed)
    time = np.arange(n) * (2 / 1440)
    if gap_at is not None:
        time[gap_at:] += 1.0  # one data gap
    flux = 1 + 0.001 * rng.standard_normal(n)
    peaks = np.sort(rng.choice(np.arange(200, n - 200), size=n_flares, replace=False))
    flares = []
    for p in peaks:
        flux[p:p + 15] += 0.05 * np.exp(-np.arange(15) / 3)
        flux[p - 2:p] += [0.01, 0.03]
        flares.append(FlareEvent(start=p - 2, peak=p, end=p + 10))
    return LightCurve(target=name, time=time, flux=flux, flares=flares)


def test_positive_flare_not_in_lookback():
    lc = synthetic_lc()
    s = build_samples(lc, 0, L, H, stride=1)
    onsets = np.array([f.start for f in lc.flares])
    peaks = np.array([f.peak for f in lc.flares])
    for t, y in zip(s["anchor"], s["y"]):
        in_horizon = (onsets > t) & (peaks <= t + H)
        assert bool(y) == bool(in_horizon.any())
        # no sample is anchored during a flare's rise
        assert not ((onsets <= t) & (peaks > t)).any()
    assert s["y"].sum() > 0


def test_normalisation_uses_lookback_only():
    lc = synthetic_lc()
    s1 = build_samples(lc, 0, L, H, stride=7)
    lc.flux = lc.flux.copy()
    lc.flux[-H * 10:] += 5.0  # change only far-future flux: no effect on earlier lookbacks
    s2 = build_samples(lc, 0, L, H, stride=7)
    t_limit = len(lc.flux) - H * 10 - H
    m = s1["anchor"] < t_limit
    assert np.allclose(s1["lookback"][m], s2["lookback"][m])
    # and changing a sample's own horizon leaves its lookback untouched
    lc2 = synthetic_lc()
    t = int(s1["anchor"][10])
    a = build_samples(lc2, 0, L, H, stride=7)
    lc2.flux = lc2.flux.copy()
    lc2.flux[t + 1:t + H + 1] *= 1.5
    b = build_samples(lc2, 0, L, H, stride=7)
    i = int(np.flatnonzero(a["anchor"] == t)[0])
    j = int(np.flatnonzero(b["anchor"] == t)[0])
    assert np.allclose(a["lookback"][i], b["lookback"][j])


def test_gap_samples_dropped():
    lc = synthetic_lc(gap_at=2500)
    s = build_samples(lc, 0, L, H, stride=1)
    lo, hi = s["anchor"] - L + 1, s["anchor"] + H
    assert not ((lo <= 2499) & (hi >= 2500)).any()


def test_splits_leakage_free():
    curves = [synthetic_lc(f"S{i}", seed=i) for i in range(6)]
    ds = build_dataset(curves, L, H, stride=3)
    folds = star_folds(len(curves), n_folds=3, n_val_stars=1, seed=0)
    tested = sorted(s for f in folds for s in f["test"])
    assert tested == list(range(6))
    for f in folds:
        assert_no_leakage(ds, star_split_masks(ds, f), "star", L, H)
    assert_no_leakage(ds, time_split_masks(ds, L, H), "time", L, H)


def test_metrics():
    y = np.array([0, 0, 1, 1])
    s = skill_scores(y, np.array([0, 1, 1, 1]))
    assert np.isclose(s["tss"], 1 - 0.5) and np.isclose(s["recall"], 1.0)
    score = np.array([0.1, 0.2, 0.8, 0.9])
    thr = select_threshold(y, score)
    r = evaluate_scores(y, score, thr)
    assert r["roc_auc"] == 1.0 and r["tss"] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
