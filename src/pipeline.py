"""Shared stage orchestration.

These helpers hold the single copy of the train/evaluate and fidelity logic so
that ``run_all.py`` and the per-stage entry points stay in lock-step instead of
duplicating the same blocks.
"""

from __future__ import annotations

import pandas as pd

from .config import (
    FIDELITY_CAT_FEATURES,
    FIDELITY_CONT_FEATURES,
    FIDELITY_EXCLUDED_COLS,
    ProjectConfig,
)
from .evaluate import evaluate_multi_class, prepare_occurrence_probabilities
from .fidelity import (
    compute_distance_metrics,
    compute_jsd_table,
    run_pairwise_statistics,
    run_tsne_pairwise,
    save_tsne_pairwise_figure,
)
from .train import train_all_models


def run_train_eval(
    tensors: dict,
    cfg: ProjectConfig,
    lstm_epochs: int = 300,
    batch_size: int = 32,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train every model and build the occurrence/subtype/severity metric tables."""
    preds_next, preds_typ, preds_kdr = train_all_models(
        tensors, lstm_epochs=lstm_epochs, batch_size=batch_size
    )

    y_true_next = tensors["Y_val"][..., 1].ravel().astype(int)
    y_true_typ = tensors["Y_val"][..., 2].ravel().astype(int)
    y_true_kdr = tensors["Y_val"][..., 3].ravel().astype(int)

    models_next_2d = prepare_occurrence_probabilities(preds_next)
    df_occ = evaluate_multi_class(models_next_2d, y_true_next, n_classes=2)
    df_sub = evaluate_multi_class(preds_typ, y_true_typ, n_classes=int(y_true_typ.max()) + 1)
    df_sev = evaluate_multi_class(preds_kdr, y_true_kdr, n_classes=int(y_true_kdr.max()) + 1)

    if save:
        df_occ.to_csv(cfg.base_dir / cfg.metrics_occurrence_csv, index=False)
        df_sub.to_csv(cfg.base_dir / cfg.metrics_subtype_csv, index=False)
        df_sev.to_csv(cfg.base_dir / cfg.metrics_severity_csv, index=False)

    return df_occ, df_sub, df_sev


def run_fidelity(
    cfg: ProjectConfig,
    perplexity: int = 30,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare genuine vs. synthetic cohorts and (optionally) persist the outputs."""
    train_aug_df = pd.read_csv(cfg.base_dir / cfg.train_augmented_csv)
    feature_cols = [c for c in train_aug_df.columns if c not in FIDELITY_EXCLUDED_COLS]

    stats_df = run_pairwise_statistics(train_aug_df, FIDELITY_CONT_FEATURES, FIDELITY_CAT_FEATURES)
    jsd_df = compute_jsd_table(train_aug_df, FIDELITY_CONT_FEATURES, FIDELITY_CAT_FEATURES)
    dist_df = compute_distance_metrics(train_aug_df, feature_cols)
    tsne_result = run_tsne_pairwise(train_aug_df, feature_cols, perplexity=perplexity)

    if save:
        save_tsne_pairwise_figure(tsne_result, str(cfg.base_dir / cfg.fidelity_figure_png))
        stats_df.to_csv(cfg.base_dir / cfg.pairwise_statistics_csv, index=False)
        jsd_df.to_csv(cfg.base_dir / cfg.js_divergence_csv, index=False)
        dist_df.to_csv(cfg.base_dir / cfg.distance_metrics_csv, index=False)

    return stats_df, jsd_df, dist_df
