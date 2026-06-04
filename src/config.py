from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Canonical feature/label vocabulary for the longitudinal cohort. Defined here
# (a dependency-light module) so both the modeling code in `src.augmentation`
# and the synthetic-data generator in `src.generate_mock_data` share one source
# of truth without the generator having to import the heavy modeling stack.
LABEL_COLS: list[str] = ["delirium", "delirium_next_day", "delirium_typ", "K-DRS-R-98"]
STATIC_FEATURES: list[str] = [
    "delirium_drug_count", "GDS", "FRAIL", "ADL", "IADL", "MNA",
    "sex", "edu", "marst", "living", "smoking", "drinking",
    "hear_impair", "sight_impair", "hos_exp", "surgical_exp", "del_hx",
    "med_count", "benz_med", "antipsy_med", "narc_med", "other_med",
    "caregiver", "CCI", "age", "ht", "wt", "BMI", "op_level2",
    "anesthesia_duration", "op_duration", "ASAscore", "V1MMSE",
]


@dataclass(frozen=True)
class ProjectConfig:
    base_dir: Path
    raw_longitudinal_csv: str = "raw_longitudinal_data.csv"
    train_augmented_csv: str = "train_augmented.csv"
    val_original_csv: str = "validation_original.csv"
    test_original_csv: str = "test_original.csv"
    metrics_occurrence_csv: str = "metrics_occurrence.csv"
    metrics_subtype_csv: str = "metrics_subtype.csv"
    metrics_severity_csv: str = "metrics_severity.csv"
    pairwise_statistics_csv: str = "pairwise_statistics.csv"
    js_divergence_csv: str = "js_divergence_results.csv"
    distance_metrics_csv: str = "distance_metrics.csv"
    fidelity_figure_png: str = "fidelity_tsne.png"


# Single source of truth for the fidelity-stage feature selection.
# Shared by run_all.py and run_stage_04_fidelity.py so the two never drift apart.
FIDELITY_CONT_FEATURES: list[str] = [
    "GDS", "FRAIL", "ADL", "IADL", "MNA", "BMI", "age",
    "op_level2", "anesthesia_duration", "op_duration", "ASAscore", "CCI", "V1MMSE",
]
FIDELITY_CAT_FEATURES: list[str] = ["sex", "edu"]
FIDELITY_EXCLUDED_COLS: set[str] = {
    "pid",
    "time",
    "Dataset_Type",
    "delirium",
    "delirium_next_day",
    "delirium_typ",
    "K-DRS-R-98",
}


def get_default_config() -> ProjectConfig:
    # This repository is self-contained: the project root is the directory that
    # holds `src/`. All inputs and outputs live directly under that root, so the
    # workflow runs end to end immediately after cloning.
    base_dir = Path(__file__).resolve().parents[1]
    return ProjectConfig(base_dir=base_dir)


def ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def ensure_input_data(cfg: ProjectConfig) -> Path:
    """Guarantee that the longitudinal input table exists before a stage runs.

    If the de-identified clinical input is already present it is used as is and
    never overwritten. Otherwise a pipeline-compatible synthetic cohort is
    generated so reviewers can reproduce the full workflow without access to the
    IRB-restricted source data. The synthetic file mirrors the 135-variable
    structure and dependent-label design described in the manuscript appendix.
    """
    raw_path = cfg.base_dir / cfg.raw_longitudinal_csv
    if raw_path.exists():
        return raw_path

    # Imported lazily to avoid a circular import (generate_mock_data imports the
    # feature/label definitions from src.augmentation, which imports this module).
    from .generate_mock_data import create_pipeline_mock_data

    print(
        f"[info] '{cfg.raw_longitudinal_csv}' not found; generating a synthetic "
        f"cohort that mirrors the manuscript schema for reproducible review."
    )
    create_pipeline_mock_data(output_path=raw_path)
    return raw_path
