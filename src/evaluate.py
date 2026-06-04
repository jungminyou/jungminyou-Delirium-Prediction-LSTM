from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_auc_prc_with_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, random_state: int = 42) -> tuple[float, tuple[float, float], float, tuple[float, float]]:
    rng = np.random.default_rng(random_state)
    auc_scores = []
    prc_scores = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
        prc_scores.append(average_precision_score(y_true[idx], y_prob[idx]))

    auc_mean = float(np.mean(auc_scores)) if auc_scores else np.nan
    prc_mean = float(np.mean(prc_scores)) if prc_scores else np.nan

    auc_ci = (float(np.percentile(auc_scores, 2.5)), float(np.percentile(auc_scores, 97.5))) if auc_scores else (np.nan, np.nan)
    prc_ci = (float(np.percentile(prc_scores, 2.5)), float(np.percentile(prc_scores, 97.5))) if prc_scores else (np.nan, np.nan)
    return auc_mean, auc_ci, prc_mean, prc_ci


def calc_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float]:
    brier = brier_score_loss(y_true, y_prob)
    x = np.clip(y_prob, 1e-7, 1 - 1e-7)
    logit_x = np.log(x / (1 - x))
    slope, intercept = np.polyfit(logit_x, y_true, 1)
    return float(brier), float(slope), float(intercept)


def calculate_net_benefit(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    benefits = []
    n = len(y_true)
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        nb = (tp / n) - (fp / n) * (thr / (1 - thr))
        benefits.append(nb)
    return np.array(benefits)


def evaluate_multi_class(models_dict: dict[str, np.ndarray], y_true: np.ndarray, n_classes: int) -> pd.DataFrame:
    results = []
    y_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2:
        y_bin = np.hstack((1 - y_bin, y_bin))

    for model_name, y_prob_matrix in models_dict.items():
        y_pred_class = np.argmax(y_prob_matrix, axis=1)
        for c in range(n_classes):
            y_true_c = y_bin[:, c]
            y_prob_c = y_prob_matrix[:, c]
            y_pred_c = (y_pred_class == c).astype(int)

            auc_m, auc_ci, prc_m, prc_ci = compute_auc_prc_with_ci(y_true_c, y_prob_c)
            brier, slope, _ = calc_calibration_metrics(y_true_c, y_prob_c)
            nb_vals = calculate_net_benefit(y_true_c, y_prob_c, np.array([0.1, 0.2, 0.3]))

            results.append(
                {
                    "Model": model_name,
                    "Class": f"C{c}",
                    "Acc": accuracy_score(y_true_c, y_pred_c),
                    "Prec": precision_score(y_true_c, y_pred_c, zero_division=0),
                    "Rec": recall_score(y_true_c, y_pred_c, zero_division=0),
                    "F1": f1_score(y_true_c, y_pred_c, zero_division=0),
                    "AUROC (95% CI)": f"{auc_m:.3f} ({auc_ci[0]:.3f}-{auc_ci[1]:.3f})",
                    "AUPRC (95% CI)": f"{prc_m:.3f} ({prc_ci[0]:.3f}-{prc_ci[1]:.3f})",
                    "Brier": brier,
                    "Slope": slope,
                    "NB (0.1/0.2/0.3)": f"{nb_vals[0]:.3f}/{nb_vals[1]:.3f}/{nb_vals[2]:.3f}",
                }
            )

    return pd.DataFrame(results)


def prepare_occurrence_probabilities(preds_next_1d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    models_next_2d = {}
    for name, prob_1d in preds_next_1d.items():
        models_next_2d[name] = np.hstack((1 - prob_1d[:, None], prob_1d[:, None]))
    return models_next_2d
