from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation


class FlareEvent:
    __slots__ = ("start", "peak", "end")

    def __init__(self, start: int, peak: int, end: int):
        self.start = start
        self.peak = peak
        self.end = end

    def __repr__(self):
        return f"FlareEvent(start={self.start}, peak={self.peak}, end={self.end})"


def detect_flares(
    flux: np.ndarray,
    trend_window: int = 101,
    threshold_sigma: float = 3.0,
    min_consecutive: int = 2,
    min_amplitude: float = 0.01,
) -> list[FlareEvent]:
    """Detrend-and-threshold flare finder (simplified Davenport-style detector).

    The paper's training data comes from manually/CNN-labeled catalogs (Feinstein
    et al. 2020) that aren't reproducible offline, so real TESS light curves fetched
    here are auto-labeled with a robust-statistics flare finder: a rolling median
    removes slow stellar variability, and cadences whose residual exceeds
    `threshold_sigma` MAD-based sigmas for at least `min_consecutive` points in a
    row are grouped into flare events.

    The paper restricts training to "large flares (in class M or C)" only -- TESS
    photometry has no direct analog to that GOES X-ray scale, so `min_amplitude`
    approximates it: an event's peak must exceed the local trend by at least this
    fraction of the (~1-normalized) flux to count as a "large" flare. Smaller
    excursions still get flagged as significant by the sigma threshold above but
    are discarded here, matching the paper's intent of training only on large events.
    """
    if trend_window % 2 == 0:
        trend_window += 1
    trend = median_filter(flux, size=trend_window, mode="nearest")
    residual = flux - trend
    sigma = median_abs_deviation(residual, scale="normal")
    if sigma == 0:
        return []
    flagged = residual > threshold_sigma * sigma

    events: list[FlareEvent] = []
    i = 0
    n = len(flagged)
    while i < n:
        if flagged[i]:
            j = i
            while j < n and flagged[j]:
                j += 1
            if j - i >= min_consecutive:
                peak = i + int(np.argmax(flux[i:j]))
                if residual[peak] >= min_amplitude:
                    events.append(FlareEvent(start=i, peak=peak, end=j - 1))
            i = j
        else:
            i += 1
    return events
