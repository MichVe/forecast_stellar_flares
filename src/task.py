"""Leakage-free stellar flare *forecasting* task (v2).

v1 (src/data.py:make_classification_windows) labelled a 200-cadence window positive when a
flare peak sat near its centre (~cadence 100), but the forecaster/classifier observed
cadences 0-149 -- so the "predicted" flare was already inside the observed lookback, and
the hybrid pipeline was effectively detecting, not forecasting. Windows were also z-scored
with statistics that included the future horizon, and overlapping windows were split at
random. This module replaces that with:

* Anchor time t (index of the last observed cadence). Input: flux[t-L+1 .. t] (L cadences).
* Label y = 1 iff a catalogued flare *begins* in the horizon, i.e. its estimated onset
  (peak - FWHM/2) is > t and its peak is <= t+H. Samples where a flare is rising at t
  (onset <= t < peak) are ambiguous -- part of the event is already visible -- and dropped.
* Normalisation uses lookback statistics only (median, MAD); the horizon is scaled with
  the same constants, so no future information leaks into the input.
* Splits are by star (unseen stars at test time) or chronological per star, with a gap of
  at least L+H cadences between blocks so no sample straddles two splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data import MEDIAN_CADENCE_GAP_FACTOR, LightCurve, _median_cadence

MAD_TO_SIGMA = 1.4826


@dataclass
class ForecastSamples:
    """Anchored (lookback, horizon, label) samples. Arrays share the first axis."""

    lookback: np.ndarray  # (N, L) float32, lookback-normalised
    horizon: np.ndarray  # (N, H) float32, same scaling as lookback
    y: np.ndarray  # (N,) float32, 1 = a flare begins within the horizon
    star: np.ndarray  # (N,) int, index into `stars`
    anchor: np.ndarray  # (N,) int, cadence index of the last observed point
    cadences_since_flare: np.ndarray  # (N,) float32, cadences since last catalogued peak <= t (inf if none)
    stars: list[str] = field(default_factory=list)

    def __len__(self):
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "ForecastSamples":
        return ForecastSamples(
            lookback=self.lookback[mask], horizon=self.horizon[mask], y=self.y[mask],
            star=self.star[mask], anchor=self.anchor[mask],
            cadences_since_flare=self.cadences_since_flare[mask], stars=self.stars,
        )

    def window(self) -> np.ndarray:
        """Lookback + true horizon, (N, L+H): the classifier's training input / Oracle input."""
        return np.concatenate([self.lookback, self.horizon], axis=1)


def flare_onsets(lc: LightCurve) -> tuple[np.ndarray, np.ndarray]:
    """(onset, peak) cadence indices of catalogued flares, sorted by peak.

    The Guenther et al. 2020 catalogue gives peak time and FWHM but no start time, so the
    onset is approximated by FlareEvent.start (= peak - FWHM/2, see src/data.py)."""
    if not lc.flares:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    onset = np.array([ev.start for ev in lc.flares], dtype=int)
    peak = np.array([ev.peak for ev in lc.flares], dtype=int)
    order = np.argsort(peak)
    return onset[order], peak[order]


def build_samples(
    lc: LightCurve,
    star_index: int,
    lookback: int,
    horizon: int,
    stride: int,
    clip: float = 30.0,
) -> dict[str, np.ndarray]:
    """All valid anchored samples for one light curve (see module docstring)."""
    flux, time = lc.flux, lc.time
    n = len(flux)
    dt = _median_cadence(time)
    onset, peak = flare_onsets(lc)

    # gap[i] is True when cadences i and i+1 are separated by a data gap
    gap = np.diff(time) > MEDIAN_CADENCE_GAP_FACTOR * dt
    gap_cum = np.concatenate([[0], np.cumsum(gap)])

    out = {k: [] for k in ("lookback", "horizon", "y", "anchor", "since")}
    for t in range(lookback - 1, n - horizon, stride):
        lo, hi = t - lookback + 1, t + horizon
        if gap_cum[hi] - gap_cum[lo] > 0:
            continue  # sample straddles a downlink/data gap

        rising = (onset <= t) & (peak > t)
        if rising.any():
            continue  # flare already under way at t: ambiguous, part of it is observed

        y = bool(((onset > t) & (peak <= t + horizon)).any())

        past = peak[peak <= t]
        since = float(t - past[-1]) if len(past) else np.inf

        seg = flux[lo:hi + 1]
        lb = seg[:lookback]
        med = np.median(lb)
        scale = MAD_TO_SIGMA * np.median(np.abs(lb - med))
        if not np.isfinite(scale) or scale <= 0:
            scale = np.std(lb) if np.std(lb) > 0 else 1.0
        z = np.clip((seg - med) / scale, -clip, clip)

        out["lookback"].append(z[:lookback])
        out["horizon"].append(z[lookback:])
        out["y"].append(y)
        out["anchor"].append(t)
        out["since"].append(since)

    m = len(out["y"])
    return {
        "lookback": np.asarray(out["lookback"], dtype=np.float32).reshape(m, lookback),
        "horizon": np.asarray(out["horizon"], dtype=np.float32).reshape(m, horizon),
        "y": np.asarray(out["y"], dtype=np.float32),
        "anchor": np.asarray(out["anchor"], dtype=int),
        "since": np.asarray(out["since"], dtype=np.float32),
        "star": np.full(m, star_index, dtype=int),
    }


def build_dataset(curves: list[LightCurve], lookback: int, horizon: int, stride: int,
                  clip: float = 30.0) -> ForecastSamples:
    parts = [build_samples(lc, i, lookback, horizon, stride, clip) for i, lc in enumerate(curves)]
    cat = lambda k: np.concatenate([p[k] for p in parts])
    return ForecastSamples(
        lookback=cat("lookback"), horizon=cat("horizon"), y=cat("y"), star=cat("star"),
        anchor=cat("anchor"), cadences_since_flare=cat("since"), stars=[lc.target for lc in curves],
    )


# ---------------------------------------------------------------------------------------
# Splits. Each returns {"train": mask, "val": mask, "test": mask} over a ForecastSamples.
# ---------------------------------------------------------------------------------------

def star_folds(n_stars: int, n_folds: int, n_val_stars: int, seed: int) -> list[dict[str, list[int]]]:
    """Group k-fold over stars: every star is a test star in exactly one fold; of the
    remaining stars, `n_val_stars` (chosen per fold) are held out for validation."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_stars)
    test_groups = np.array_split(order, n_folds)
    folds = []
    for test in test_groups:
        rest = [s for s in order if s not in set(test)]
        val = list(rng.choice(rest, size=n_val_stars, replace=False))
        train = [s for s in rest if s not in set(val)]
        folds.append({"train": sorted(int(s) for s in train), "val": sorted(int(s) for s in val),
                      "test": sorted(int(s) for s in test)})
    return folds


def star_split_masks(ds: ForecastSamples, fold: dict[str, list[int]]) -> dict[str, np.ndarray]:
    return {k: np.isin(ds.star, v) for k, v in fold.items()}


def time_split_masks(ds: ForecastSamples, lookback: int, horizon: int,
                     fractions=(0.7, 0.15, 0.15)) -> dict[str, np.ndarray]:
    """Chronological per-star split. Anchors within L+H cadences of a block boundary are
    discarded, so no sample's [t-L+1, t+H] span overlaps the next block's samples."""
    masks = {k: np.zeros(len(ds), dtype=bool) for k in ("train", "val", "test")}
    margin = lookback + horizon
    for s in np.unique(ds.star):
        idx = np.flatnonzero(ds.star == s)
        a = ds.anchor[idx]
        a0, a1 = a.min(), a.max()
        b1 = a0 + fractions[0] * (a1 - a0)
        b2 = a0 + (fractions[0] + fractions[1]) * (a1 - a0)
        masks["train"][idx[a <= b1 - margin]] = True
        masks["val"][idx[(a > b1) & (a <= b2 - margin)]] = True
        masks["test"][idx[a > b2]] = True
    return masks


def assert_no_leakage(ds: ForecastSamples, masks: dict[str, np.ndarray], mode: str,
                      lookback: int, horizon: int) -> None:
    """Hard checks that the split is leakage-free. Raises AssertionError otherwise."""
    for a in masks:
        for b in masks:
            if a < b:
                assert not (masks[a] & masks[b]).any(), f"samples shared between {a} and {b}"
    if mode == "star":
        stars = {k: set(np.unique(ds.star[m])) for k, m in masks.items()}
        assert not (stars["test"] & (stars["train"] | stars["val"])), "test star also used for training/validation"
        assert not (stars["val"] & stars["train"]), "validation star also used for training"
    else:
        for s in np.unique(ds.star):
            spans = {}
            for k, m in masks.items():
                sel = m & (ds.star == s)
                if sel.any():
                    spans[k] = (ds.anchor[sel].min() - lookback + 1, ds.anchor[sel].max() + horizon)
            keys = [k for k in ("train", "val", "test") if k in spans]
            for k1, k2 in zip(keys, keys[1:]):
                assert spans[k1][1] < spans[k2][0], f"star {s}: {k1} span overlaps {k2} span"
