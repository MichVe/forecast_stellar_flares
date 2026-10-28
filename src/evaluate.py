from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


@torch.no_grad()
def evaluate_forecast_quality(forecaster, loader, device="cpu", flare_amplitude_threshold: float = 0.01):
    """MAE / MSE / RMSE of the forecaster on held-out horizon cadences (Section 4's metrics).

    Also splits the same metrics by whether the true horizon contains a flare-like
    excursion or not. Flares are rare, so the aggregate MAE/MSE above is dominated by
    the (easy, near-persistence) quiescent majority; the flare-horizon subset is the
    part that actually matters for the downstream hybrid flare-prediction task.
    """
    forecaster.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = forecaster(xb, horizon=yb.shape[1])
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    m = mse(preds, trues)
    result = {"MAE": mae(preds, trues), "MSE": m, "RMSE": float(np.sqrt(m))}

    excursion = trues.max(axis=1) - 1.0  # horizon is lookback-median-normalized, so ~1.0-centered
    is_flare = excursion.reshape(-1) > flare_amplitude_threshold
    if is_flare.any():
        result["MAE_flare_horizon"] = mae(preds[is_flare], trues[is_flare])
        result["MSE_flare_horizon"] = mse(preds[is_flare], trues[is_flare])
        result["n_flare_horizon"] = int(is_flare.sum())
    if (~is_flare).any():
        result["MAE_quiescent_horizon"] = mae(preds[~is_flare], trues[~is_flare])
        result["MSE_quiescent_horizon"] = mse(preds[~is_flare], trues[~is_flare])
        result["n_quiescent_horizon"] = int((~is_flare).sum())
    return result


def _classification_report(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Reports metrics at the default 0.5 cutoff plus at the threshold that maximizes
    F1 on this same set. With a class-imbalanced positive rate (flares are rare), 0.5
    is frequently not where the model's decision boundary should sit -- AUC can look
    fine while every 0.5-thresholded prediction is negative. Reporting both makes that
    visible instead of silently hiding a well-ranking-but-miscalibrated classifier."""
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        denom = precisions + recalls
        f1s = np.divide(2 * precisions * recalls, denom, out=np.zeros_like(denom), where=denom > 0)
        if len(thresholds) > 0:
            best_idx = int(np.argmax(f1s[:-1]))
            best_threshold = float(thresholds[best_idx])
            y_pred_best = (y_prob >= best_threshold).astype(int)
            out["best_threshold"] = best_threshold
            out["precision_at_best"] = precision_score(y_true, y_pred_best, zero_division=0)
            out["recall_at_best"] = recall_score(y_true, y_pred_best, zero_division=0)
            out["f1_at_best"] = f1_score(y_true, y_pred_best, zero_division=0)
    return out


@torch.no_grad()
def evaluate_hybrid(forecaster, classifier, X_windows, y_labels, lookback, horizon, device="cpu", batch_size=64):
    """The paper's proposed hybrid pipeline (Section 6): forecast the horizon segment,
    concatenate it with the observed lookback, and classify the reconstructed window
    with the Stella CNN. Also reports an "oracle" variant that classifies the *true*
    future segment instead of a forecast -- the upper bound no forecaster can beat.
    """
    forecaster.eval()
    classifier.eval()

    n = len(X_windows)
    y_prob_hybrid = np.zeros(n, dtype=np.float32)
    y_prob_oracle = np.zeros(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.from_numpy(X_windows[start:end]).to(device)
        lb = batch[:, :lookback].unsqueeze(-1)
        true_future = batch[:, lookback:lookback + horizon]

        forecast = forecaster(lb, horizon=horizon).squeeze(-1)
        recon_hybrid = torch.cat([batch[:, :lookback], forecast], dim=1)
        recon_oracle = torch.cat([batch[:, :lookback], true_future], dim=1)

        y_prob_hybrid[start:end] = torch.sigmoid(classifier(recon_hybrid)).cpu().numpy()
        y_prob_oracle[start:end] = torch.sigmoid(classifier(recon_oracle)).cpu().numpy()

    return {
        "hybrid": _classification_report(y_labels, y_prob_hybrid),
        "oracle": _classification_report(y_labels, y_prob_oracle),
    }
