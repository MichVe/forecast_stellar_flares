"""Classification metrics for the v2 forecasting task.

The decision threshold is chosen on the *validation* set and then applied unchanged to
the test set (v1 picked the F1-optimal threshold on the test set itself). TSS and HSS
are reported because they are the standard skill scores in flare forecasting and, unlike
accuracy, are not inflated by the strong class imbalance.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    return tp, fp, fn, tn


def skill_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = confusion(y_true, y_pred)
    pod = tp / (tp + fn) if tp + fn else 0.0  # recall / probability of detection
    pofd = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * pod / (precision + pod) if precision + pod else 0.0
    n = tp + fp + fn + tn
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {
        "precision": precision,
        "recall": pod,
        "f1": f1,
        "accuracy": (tp + tn) / n if n else 0.0,
        "tss": pod - pofd,
        "hss": 2 * (tp * tn - fn * fp) / hss_den if hss_den else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def select_threshold(y_true: np.ndarray, score: np.ndarray, criterion: str = "tss") -> float:
    """Threshold on `score` maximising `criterion` (tss or f1) on the given (validation) set."""
    candidates = np.unique(score)
    if len(candidates) > 512:
        candidates = np.quantile(score, np.linspace(0, 1, 513))
    best_t, best_v = float(candidates[0]), -np.inf
    for t in candidates:
        v = skill_scores(y_true, score >= t)[criterion]
        if v > best_v:
            best_t, best_v = float(t), v
    return best_t


def evaluate_scores(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    """Ranking metrics (threshold-free) + skill scores at a threshold fixed beforehand.

    `score` need not be a calibrated probability; Brier score is only reported when it
    lies in [0, 1]."""
    out: dict[str, float] = {"n": int(len(y_true)), "n_pos": int(y_true.sum()),
                             "base_rate": float(y_true.mean()) if len(y_true) else 0.0,
                             "threshold": float(threshold)}
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
        out["pr_auc"] = float(average_precision_score(y_true, score))
    if np.all((score >= 0) & (score <= 1)):
        out["brier"] = float(brier_score_loss(y_true, score))
    out.update(skill_scores(y_true, score >= threshold))
    return out
