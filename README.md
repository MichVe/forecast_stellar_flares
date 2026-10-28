# Forecasting stellar flares from TESS light curves: a leakage-free benchmark

Code and results for the manuscript

> M. Veselá & O. Trenz, *Can TESS light curves forecast stellar flares? A leakage-free benchmark
> of Transformer forecasting pipelines and flare-history baselines* (in preparation for
> RAS Techniques and Instruments).

**Task.** From the last *L* = 150 cadences (5 h) of a 2-min TESS light curve, predict whether a
catalogued flare ([Günther et al. 2020](https://doi.org/10.3847/1538-3881/ab5d3a)) *begins* within
the next *H* ∈ {5, 15, 30, 60} cadences (10 min – 2 h). The data are 16 M dwarfs in TESS sectors 1–2, with 930 flares.

**Main result.** With leakage removed, every light-curve model (Hybrid-ETSformer, Hybrid-LSTM,
Hybrid-Naive, Direct-CNN) is close to chance (ROC-AUC 0.51–0.55). A classifier that sees the
true future reaches 0.96–1.00, so the task is well posed. The forecastable signal comes from
flare history: the per-star flare rate for unseen stars (ROC-AUC ≈ 0.59), and the time since the
last flare for known stars (0.60–0.64).

## Contents

| Path | Content |
|---|---|
| `src/task.py` | Sample construction (anchored lookback and horizon, onset-based labels, lookback-only normalisation), star and chronological splits, and `assert_no_leakage` |
| `src/forecaster.py`, `src/stella_model.py`, `src/train.py` | ETSformer, LSTM and naive forecasters, the Stella CNN, and training loops |
| `src/history.py`, `src/baselines.py` | Flare-history models (Poisson, Renewal, History-LR, History+LC-GBM), climatology and recency |
| `src/metrics.py` | ROC-AUC, PR-AUC, Brier, TSS, HSS, F1, and threshold selection on validation |
| `src/data.py` | TESS light-curve download (lightkurve, SPOC 2-min) and catalogue labels (VizieR J/AJ/159/60) |
| `run_forecasting.py` | Main experiment: neural models and non-learned baselines |
| `run_history.py` | Flare-history models on identical samples and splits |
| `scripts/aggregate_v2.py` | Summary tables and the paired star-cluster bootstrap |
| `results_v2/` | Per-run metrics (`runs.jsonl`), per-sample test predictions (`predictions/*.npz`) and bootstrap summaries |
| `tests/test_task.py` | Synthetic tests of the task definition and the leakage checks |
| `main.py`, `results_v1/`, `scripts/ablation.py`, … | The earlier (v1) pipeline, kept for transparency. **Its window construction leaks the flare into the input** (see Section 6 of the paper), so do not use it as a forecasting baseline. |

## Installation

Python 3.11 or later is recommended.

```bash
pip install -r requirements.txt
```

The experiments run on a CPU. If PyTorch detects a GPU, it is used automatically.

## Reproducing the results

On first use, the light curves and the catalogue are downloaded into `./tess_cache` (a few hundred MB).

```bash
# star split (4 star folds x 5 seeds x 4 horizons)
python run_forecasting.py --split star --output-dir results_v2/full_star
python run_history.py     --split star --output-dir results_v2/history_star
# chronological split (70/15/15 per star, with gaps)
python run_forecasting.py --split time --output-dir results_v2/full_time
python run_history.py     --split time --output-dir results_v2/history_time

# summary tables and bootstrap: no retraining needed, the saved predictions are used
python scripts/aggregate_v2.py --results results_v2/full_star results_v2/history_star \
       --reference Poisson --n-boot 1000 --out results_v2/summary_star_vs_poisson.json
```

A full neural run takes tens of hours on a laptop CPU.

- `python run_forecasting.py --smoke` checks the whole pipeline in a few minutes.
- `--resume` continues an interrupted run. Every run re-seeds itself, so a resumed run gives the same results as an uninterrupted one.
Tests:

```bash
python tests/test_task.py
```

## Data

* TESS 2-min SPOC light curves from the [Mikulski Archive for Space Telescopes (MAST)](https://archive.stsci.edu/missions-and-data/tess).
* Flare catalogue: Günther et al. (2020), AJ 159, 60, VizieR [J/AJ/159/60](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/AJ/159/60).

## Reporting problems

Please open an [issue](https://github.com/MichVe/Articlle_forecast_stellar_flares/issues).

## Licence

MIT, see [LICENSE](LICENSE).
