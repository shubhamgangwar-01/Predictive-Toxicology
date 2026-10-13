"""Evaluation metrics for imbalanced multi-task classification and regression."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, matthews_corrcoef, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)


def best_mcc_threshold(y_true, y_prob) -> float:
    """Decision threshold maximising MCC. Chosen on the VALIDATION split only, then
    frozen for the test split (a fixed 0.5 cut-off is meaningless at a 3% positive rate)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    candidates = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
    scores = [matthews_corrcoef(y_true, (y_prob >= t).astype(int)) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def classification_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "mcc": np.nan, "pos_rate": float(y_true.mean())}
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, (y_prob >= threshold).astype(int)),
        "pos_rate": float(y_true.mean()),
    }


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def masked_mean_auc(Y, P) -> float:
    """Mean ROC-AUC over tasks, ignoring missing labels (NaN) — used for early stopping."""
    aucs = []
    for t in range(Y.shape[1]):
        m = ~np.isnan(Y[:, t])
        if m.sum() > 1 and len(np.unique(Y[m, t])) == 2:
            aucs.append(roc_auc_score(Y[m, t], P[m, t]))
    return float(np.mean(aucs)) if aucs else np.nan
