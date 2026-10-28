"""Non-learned reference forecasts for the v2 task. Any learned model has to beat these
to claim forecasting skill."""

from __future__ import annotations

import numpy as np

from src.task import ForecastSamples


def climatology_scores(train: ForecastSamples, target: ForecastSamples) -> np.ndarray:
    """Positive rate of each star in the training data (global rate for stars not seen in
    training, e.g. every test star under the star split -> a constant, no-skill forecast)."""
    global_rate = float(train.y.mean()) if len(train) else 0.0
    rates = {int(s): float(train.y[train.star == s].mean()) for s in np.unique(train.star)}
    return np.array([rates.get(int(s), global_rate) for s in target.star], dtype=np.float32)


def recency_scores(target: ForecastSamples) -> np.ndarray:
    """Higher when the star flared recently (flares cluster in time). Uses only catalogued
    flare peaks at or before the anchor, i.e. information available at forecast time."""
    since = target.cadences_since_flare.astype(np.float64)
    return (1.0 / (1.0 + since)).astype(np.float32)  # inf -> 0
