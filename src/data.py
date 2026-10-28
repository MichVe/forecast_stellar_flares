from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from src.flare_detection import FlareEvent, detect_flares

logger = logging.getLogger(__name__)

# Targets drawn from Guenther et al. 2020 (VizieR J/AJ/159/60) -- the paper's own cited
# data source (ref [28]: "the original light curve data from Guenther et al. (2020)").
# Picked as the catalog's highest flare-count M dwarfs, by TIC id, so real published
# flare peak times/amplitudes/durations are used as training labels instead of an
# auto-detector. Restricted to sectors 1-2 (see DEFAULT_SECTORS below) because that
# catalog only covers TESS's "first data release" (sectors 1-2); AU Mic is kept as
# the one overlap with the earlier named-star target list, as a sanity check.
DEFAULT_TARGETS = [
    "TIC 441420236",  # AU Mic, 46 catalogued flares
    "TIC 150359500",  # 109 catalogued flares
    "TIC 140045538",  # 92 catalogued flares
    "TIC 231914259",  # 83 catalogued flares
    "TIC 234506911",  # 56 catalogued flares
    "TIC 231017428",  # 47 catalogued flares
    "TIC 231910539",  # 47 catalogued flares
    "TIC 237885807",  # 45 catalogued flares
]
DEFAULT_SECTORS = [1, 2]

GUENTHER_CATALOG_ID = "J/AJ/159/60"
GUENTHER_CATALOG_CACHE = os.path.join("tess_cache", "catalogs", "guenther2020_table1.ecsv")
GUENTHER_BJD_OFFSET = 2457000.0  # BJD_TDB -> TESS BTJD, matching lightkurve's lc.time.value convention

MEDIAN_CADENCE_GAP_FACTOR = 3.0

# Normal TESS 2-min photometric scatter for these targets is well under this after
# per-sector normalize() (typically std <~ 0.05). Some MAST holdings contain a second,
# corrupted/duplicate flux product for the same target+sector (e.g. an already
# background-subtracted or otherwise mis-scaled column with a near-zero raw median),
# which normalize() blows up into huge relative excursions once divided by that
# near-zero median. Sectors whose post-normalize scatter exceeds this are dropped
# before stitching so a single bad file can't dominate the light curve / training data.
MAX_SECTOR_NORM_STD = 0.5


@dataclass
class LightCurve:
    target: str
    time: np.ndarray
    flux: np.ndarray
    flares: list[FlareEvent]


def _parse_tic(target: str) -> int | None:
    target = target.strip()
    if target.upper().startswith("TIC"):
        try:
            return int(target[3:].strip())
        except ValueError:
            return None
    return None


_guenther_table1 = None


def _load_guenther_catalog():
    """Downloads (once per process) the individual-flare table of Guenther et al. 2020's
    published TESS sector 1-2 flare catalog -- ref [28] in the paper, the dataset the
    Methods/Results sections describe training on."""
    global _guenther_table1
    if _guenther_table1 is None:
        if os.path.exists(GUENTHER_CATALOG_CACHE):
            from astropy.table import Table

            _guenther_table1 = Table.read(GUENTHER_CATALOG_CACHE)
        else:
            from astroquery.vizier import Vizier

            Vizier.ROW_LIMIT = -1
            Vizier.TIMEOUT = 180
            cats = Vizier.get_catalogs(GUENTHER_CATALOG_ID)
            _guenther_table1 = cats[f"{GUENTHER_CATALOG_ID}/table1"]
            os.makedirs(os.path.dirname(GUENTHER_CATALOG_CACHE), exist_ok=True)
            _guenther_table1.write(GUENTHER_CATALOG_CACHE, overwrite=True)
        logger.info("Loaded Guenther et al. 2020 flare catalog: %d flares across %d TIC/sector entries",
                    len(_guenther_table1), len(set(_guenther_table1["TIC"])))
    return _guenther_table1


def _flares_from_catalog(tic: int, time: np.ndarray, dt: float) -> list[FlareEvent] | None:
    """Real, published flare events for `tic` from the Guenther et al. 2020 catalog,
    mapped onto this light curve's cadence indices, instead of the statistical
    auto-detector in flare_detection.py. Returns None if `tic` isn't in the catalog."""
    table = _load_guenther_catalog()
    rows = table[table["TIC"] == tic]
    if len(rows) == 0:
        return None

    events = []
    t_start, t_end = time[0], time[-1]
    for row in rows:
        tpeak_btjd = float(row["tpeak"]) - GUENTHER_BJD_OFFSET
        if not (t_start <= tpeak_btjd <= t_end):
            continue
        idx = int(np.argmin(np.abs(time - tpeak_btjd)))
        if abs(time[idx] - tpeak_btjd) > 2 * dt:
            continue  # catalogued peak falls in a data gap we don't actually have cadences for
        fwhm_d = row["FWHMdt"]
        fwhm_d = float(fwhm_d) if fwhm_d is not None and not np.ma.is_masked(fwhm_d) else 10 * dt
        half_width = max(int(round((fwhm_d / 2) / dt)), 1)
        events.append(FlareEvent(start=max(idx - half_width, 0), peak=idx, end=min(idx + half_width, len(time) - 1)))
    return events


def _processed_cache_path(target: str, sectors, cache_dir: str | None) -> str | None:
    if cache_dir is None:
        return None
    tag = "all" if not sectors else "-".join(str(s) for s in sorted(sectors))
    name = target.replace(" ", "_")
    return os.path.join(cache_dir, "processed", f"{name}_s{tag}.npz")


def fetch_light_curve(target: str, sectors=None, cache_dir: str | None = None, min_flare_amplitude: float = 0.01) -> LightCurve | None:
    """Stitched, normalised 2-min SPOC light curve with catalogued (or auto-detected) flares.

    The processed result is cached in <cache_dir>/processed/ so later runs need no network."""
    cache = _processed_cache_path(target, sectors, cache_dir)
    if cache and os.path.exists(cache):
        d = np.load(cache)
        flares = [FlareEvent(int(s), int(p), int(e)) for s, p, e in d["flares"]]
        return LightCurve(target=target, time=d["time"], flux=d["flux"], flares=flares)
    lc = _fetch_light_curve(target, sectors, cache_dir, min_flare_amplitude)
    if lc is not None and cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        fl = np.array([(f.start, f.peak, f.end) for f in lc.flares], dtype=int).reshape(-1, 3)
        np.savez_compressed(cache, time=lc.time, flux=lc.flux, flares=fl)
    return lc


def _fetch_light_curve(target: str, sectors=None, cache_dir: str | None = None, min_flare_amplitude: float = 0.01) -> LightCurve | None:
    import lightkurve as lk

    search = lk.search_lightcurve(target, mission="TESS", author="SPOC", exptime=120, sector=sectors)
    if len(search) == 0:
        logger.warning("No SPOC 2-min light curves found for %s", target)
        return None

    collection = search.download_all(download_dir=cache_dir, quality_bitmask="hard")
    if collection is None or len(collection) == 0:
        logger.warning("Download failed for %s", target)
        return None

    good = []
    for x in collection:
        xn = x.normalize()
        f = np.asarray(xn.flux.value, dtype=np.float64)
        f = f[np.isfinite(f)]
        if len(f) == 0:
            continue
        s = float(np.std(f))
        if s > MAX_SECTOR_NORM_STD:
            logger.warning(
                "%s: dropping sector %s file (post-normalize std=%.3f > %.2f, "
                "looks like a corrupted/duplicate flux product)",
                target, x.meta.get("SECTOR", "?"), s, MAX_SECTOR_NORM_STD,
            )
            continue
        good.append(x)

    if not good:
        logger.warning("All sectors filtered out as anomalous for %s", target)
        return None

    collection = lk.LightCurveCollection(good)
    lc = collection.stitch(corrector_func=lambda x: x.normalize()).remove_nans()
    time = np.asarray(lc.time.value, dtype=np.float64)
    flux = np.asarray(lc.flux.value, dtype=np.float64)

    order = np.argsort(time)
    time, flux = time[order], flux[order]
    dt = _median_cadence(time)

    tic = _parse_tic(target)
    flares = _flares_from_catalog(tic, time, dt) if tic is not None else None
    if flares is not None:
        logger.info("%s: %d cadences, %d real published flares (Guenther et al. 2020 catalog)",
                    target, len(flux), len(flares))
    else:
        flares = detect_flares(flux, min_amplitude=min_flare_amplitude)
        logger.info("%s: %d cadences, %d large flares (auto-detector fallback; not in Guenther catalog)",
                    target, len(flux), len(flares))
    return LightCurve(target=target, time=time, flux=flux, flares=flares)


def fetch_all(targets=None, sectors=None, cache_dir=None, min_flare_amplitude: float = 0.01) -> list[LightCurve]:
    targets = targets or DEFAULT_TARGETS
    curves = []
    for t in targets:
        try:
            lc = fetch_light_curve(t, sectors=sectors, cache_dir=cache_dir, min_flare_amplitude=min_flare_amplitude)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", t, exc)
            lc = None
        if lc is not None and len(lc.flux) > 0:
            curves.append(lc)
    if not curves:
        raise RuntimeError("No light curves could be fetched. Check network access and target names.")
    return curves


def _median_cadence(time: np.ndarray) -> float:
    return float(np.median(np.diff(time)))


def make_classification_windows(
    lc: LightCurve,
    window: int = 200,
    stride: int = 100,
    positive_margin: int = 50,
):
    """Slice a light curve into fixed-length windows labeled flare / non-flare,
    following the 200-cadence centered-windowing approach of Feinstein et al. 2020
    described in the paper's Data/Results sections.
    """
    flux, time = lc.flux, lc.time
    n = len(flux)
    if n <= window:
        return np.empty((0, window), dtype=np.float32), np.empty((0,), dtype=np.float32)
    dt = _median_cadence(time)
    windows, labels = [], []

    def add(start: int, label: int):
        end = start + window
        if start < 0 or end > n:
            return
        if np.max(np.diff(time[start:end])) > MEDIAN_CADENCE_GAP_FACTOR * dt:
            return  # window straddles a data-downlink gap
        w = flux[start:end].copy()
        # Standardize to zero mean / unit variance rather than just dividing by the
        # median: relative flux stays offset around ~1.0 with tiny scatter (std ~0.003),
        # which is a poor input scale for a freshly-initialized CNN (near-zero effective
        # gradients through the first conv/ReLU layers) and empirically left the Stella
        # classifier unable to learn at all (flat loss regardless of LR/epochs). Z-scoring
        # each window fixes this and matches standard practice for CNN light-curve
        # classifiers (e.g. Feinstein et al. 2020's own Stella implementation).
        std = w.std()
        w = (w - w.mean()) / std if std > 0 else w - w.mean()
        windows.append(w)
        labels.append(label)

    for start in range(0, n - window, stride):
        center = start + window // 2
        label = 0
        for ev in lc.flares:
            if abs(ev.peak - center) <= positive_margin:
                label = 1
                break
        add(start, label)

    # Explicitly center a window on every detected flare peak so the positive
    # class isn't starved by the fixed stride (mirrors the paper's centering step).
    for ev in lc.flares:
        add(ev.peak - window // 2, 1)

    return np.asarray(windows, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def make_forecast_windows(
    lc: LightCurve,
    lookback: int = 150,
    horizon: int = 50,
    stride: int = 25,
    oversample_near_flares: int = 0,
):
    """Sliding (lookback -> horizon) pairs for self-supervised forecaster training,
    skipping windows that straddle a data-downlink gap.

    Flares are rare, so a plain fixed stride makes the vast majority of training pairs
    "quiescent" (flat, near-persistence) segments -- the forecaster then mostly learns
    to reproduce the easy majority case and the (small) minority of flare-containing
    horizons, which is what actually matters for the downstream hybrid task, barely
    moves the aggregate loss. When `oversample_near_flares` > 0, additional windows are
    added (at `oversample_near_flares` jittered offsets each) so every known flare peak
    also appears inside the *horizon* segment of several training pairs, the same way
    make_classification_windows() explicitly centers windows on flare peaks.
    """
    flux, time = lc.flux, lc.time
    n = len(flux)
    total = lookback + horizon
    if n <= total:
        return np.empty((0, lookback), dtype=np.float32), np.empty((0, horizon), dtype=np.float32)
    dt = _median_cadence(time)
    xs, ys = [], []

    def add(start: int):
        end = start + total
        if start < 0 or end > n:
            return
        if np.max(np.diff(time[start:end])) > MEDIAN_CADENCE_GAP_FACTOR * dt:
            return
        seg = flux[start:end].copy()
        seg = seg / np.median(seg[:lookback])
        xs.append(seg[:lookback])
        ys.append(seg[lookback:])

    for start in range(0, n - total, stride):
        add(start)

    if oversample_near_flares > 0:
        for ev in lc.flares:
            for k in range(oversample_near_flares):
                # place the flare peak at a different cadence within the horizon each time,
                # so the model sees it both as an imminent and a more distant target
                offset_in_horizon = 1 + (k * (horizon - 2)) // max(oversample_near_flares - 1, 1) if oversample_near_flares > 1 else horizon // 2
                add(ev.peak - lookback - offset_in_horizon)

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
