"""Flare-history forecasts for the v2 task.

All features at anchor t use only catalogued flares whose peak is at or before t, i.e.
events already observed at forecast time (the task already drops anchors during a flare's
rise, see src/task.py). Models:

  Poisson         memoryless: P = 1 - exp(-rate * H), rate = the star's flare rate observed
                  so far (shrunk towards the training-set rate early in the light curve).
  Renewal         waiting-time model: P(next onset within H | s cadences since last flare),
                  from the empirical waiting-time distribution of the training stars.
                  Beating Poisson means flares cluster in time.
  History-LR      logistic regression on history features.
  History+LC-GBM  gradient boosting on history features + simple lookback statistics; tests
                  whether the light-curve shape adds anything beyond flare history.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import LightCurve
from src.task import ForecastSamples, flare_onsets

WINDOWS = (30, 180, 720)  # 1 h, 6 h, 24 h at 2-min cadence
PRIOR_CADENCES = 2000.0  # strength of the rate shrinkage (~2.8 days of pseudo-observation)

HISTORY_FEATURES = ["log_since", "has_past", *[f"n_{w}" for w in WINDOWS], "log_rate"]
LC_FEATURES = ["lc_max", "lc_std", "lc_last", "lc_slope", "lc_max_last30"]


def _per_star_peaks(curves: list[LightCurve]) -> list[np.ndarray]:
    return [flare_onsets(lc)[1] for lc in curves]


def history_matrix(ds: ForecastSamples, curves: list[LightCurve], prior_rate: float) -> np.ndarray:
    """(N, len(HISTORY_FEATURES)) features; `prior_rate` = flares per cadence (training set)."""
    peaks = _per_star_peaks(curves)
    X = np.zeros((len(ds), len(HISTORY_FEATURES)), dtype=np.float64)
    for s in np.unique(ds.star):
        idx = np.flatnonzero(ds.star == s)
        t = ds.anchor[idx]
        p = peaks[s]
        n_past = np.searchsorted(p, t, side="right")  # peaks <= t
        since = ds.cadences_since_flare[idx].astype(np.float64)
        has = np.isfinite(since)
        X[idx, 0] = np.where(has, np.log1p(np.where(has, since, 0)), np.log1p(1e5))
        X[idx, 1] = has
        for j, w in enumerate(WINDOWS):
            X[idx, 2 + j] = n_past - np.searchsorted(p, t - w, side="right")
        rate = (n_past + prior_rate * PRIOR_CADENCES) / (t + 1 + PRIOR_CADENCES)
        X[idx, 2 + len(WINDOWS)] = np.log(rate)
    return X


def lc_matrix(ds: ForecastSamples) -> np.ndarray:
    lb = ds.lookback.astype(np.float64)
    x = np.arange(lb.shape[1]) - (lb.shape[1] - 1) / 2
    slope = (lb * x).sum(1) / (x ** 2).sum()
    return np.column_stack([lb.max(1), lb.std(1), lb[:, -1], slope, lb[:, -30:].max(1)])


def training_rate(train: ForecastSamples, curves: list[LightCurve]) -> float:
    """Flares per cadence over the training stars' anchored spans."""
    peaks = _per_star_peaks(curves)
    n_fl, n_cad = 0, 0
    for s in np.unique(train.star):
        a = train.anchor[train.star == s]
        n_fl += int(((peaks[s] >= a.min()) & (peaks[s] <= a.max())).sum())
        n_cad += int(a.max() - a.min() + 1)
    return n_fl / max(n_cad, 1)


def poisson_scores(X: np.ndarray, horizon: int) -> np.ndarray:
    rate = np.exp(X[:, HISTORY_FEATURES.index("log_rate")])
    return 1.0 - np.exp(-rate * horizon)


class RenewalModel:
    """Empirical hazard of the next flare onset given the time since the last flare peak,
    estimated on training samples by binning log(since) (no parametric assumption)."""

    def __init__(self, n_bins: int = 30):
        self.n_bins = n_bins

    def fit(self, X: np.ndarray, y: np.ndarray):
        ls = X[:, HISTORY_FEATURES.index("log_since")]
        self.edges = np.unique(np.quantile(ls, np.linspace(0, 1, self.n_bins + 1)))
        b = np.clip(np.searchsorted(self.edges, ls, side="right") - 1, 0, len(self.edges) - 2)
        prior = y.mean()
        self.p = np.array([(y[b == k].sum() + prior * 20) / ((b == k).sum() + 20) for k in range(len(self.edges) - 1)])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        ls = X[:, HISTORY_FEATURES.index("log_since")]
        b = np.clip(np.searchsorted(self.edges, ls, side="right") - 1, 0, len(self.edges) - 2)
        return self.p[b]


def fit_predict_all(train: ForecastSamples, others: dict[str, ForecastSamples], curves: list[LightCurve],
                    horizon: int, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """{split: {model: scores}} for every split in `others` (e.g. val, test)."""
    r0 = training_rate(train, curves)
    Xh_tr = history_matrix(train, curves, r0)
    Xl_tr = np.column_stack([Xh_tr, lc_matrix(train)])
    y = train.y.astype(int)

    renewal = RenewalModel().fit(Xh_tr, y)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(Xh_tr, y)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         class_weight="balanced", random_state=seed)
    gbm.fit(Xl_tr, y)

    out = {}
    for split, s in others.items():
        Xh = history_matrix(s, curves, r0)
        Xl = np.column_stack([Xh, lc_matrix(s)])
        out[split] = {
            "Poisson": poisson_scores(Xh, horizon),
            "Renewal": renewal.predict(Xh),
            "History-LR": lr.predict_proba(Xh)[:, 1],
            "History+LC-GBM": gbm.predict_proba(Xl)[:, 1],
        }
    return out
