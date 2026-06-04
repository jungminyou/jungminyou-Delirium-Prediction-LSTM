"""
Synthetic Longitudinal Cohort Generator

The original patient-level clinical data cannot be shared because of IRB and
institutional data-transfer restrictions. To keep the analysis fully
reproducible, this module emits a single longitudinal table,
``raw_longitudinal_data.csv``, whose schema is *byte-for-byte* compatible with
the production pipeline (``src/augmentation.py`` onward).

It mirrors the structure documented in the manuscript appendix:

* 135 clinical variables = 33 static (baseline) + 98 temporal (daily) + 4 labels
  (``pid`` and ``time`` are record identifiers and are not counted among the 135).
* A balanced panel: every subject has exactly ``TIME_STEPS`` daily records.
* A dependent (soft-gated) label structure: subtype and severity are non-zero
  only when next-day occurrence is positive.

The feature and label *names* are imported directly from ``src.config`` (the same
definitions the modeling code uses) so the generated columns can never silently
drift from what the pipeline consumes, and data generation stays free of the
heavy modeling dependencies (TensorFlow, tslearn).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABEL_COLS, STATIC_FEATURES

# ----------------------------------------------------------------------------
# Schema definition
# ----------------------------------------------------------------------------
TIME_STEPS = 5          # Pre-op baseline (t=0) through POD3 (t=4)
NUM_PATIENTS = 120      # train (~84) keeps majority > minority for augmentation
RANDOM_SEED = 42        # Fixed seed aligned with the manuscript

# 98 temporal (daily) predictors. Named, clinically meaningful daily measures
# come first; the remainder are padded so the temporal block totals 98 without
# colliding with any static-feature or label name.
NAMED_TEMPORAL_FEATURES = [
    "delirium_hrs",
    "NRS_pain",
    "NRS_movement",
    "body_temperature",
    "heart_rate",
    "resp_rate",
    "SBP",
    "DBP",
    "SpO2",
    "eGFR_MDRD",
    "BUN",
    "creatinine",
    "sodium",
    "potassium",
    "chloride",
    "calcium",
    "magnesium",
    "glucose",
    "hemoglobin",
    "hematocrit",
    "WBC",
    "platelet",
    "albumin",
    "CRP",
    "RDW",
    "lactate",
    "daily_opioid_dose",
    "daily_benzo_dose",
    "daily_antipsychotic_dose",
    "sleep_hours",
    "mobility_score",
    "restraint_use",
    "foley_catheter",
    "iv_line_count",
    "urine_output",
]
N_TEMPORAL = 98
TEMPORAL_FEATURES = NAMED_TEMPORAL_FEATURES + [
    f"daily_marker_{i:02d}" for i in range(N_TEMPORAL - len(NAMED_TEMPORAL_FEATURES))
]

# Temporal columns that must round-trip through CSV as integers, so that
# `enforce_distribution_strict` in the pipeline recognises and snaps them.
INT_TEMPORAL_FEATURES = {
    "delirium_hrs", "NRS_pain", "NRS_movement", "heart_rate", "resp_rate",
    "SBP", "DBP", "platelet", "WBC", "restraint_use", "foley_catheter",
    "iv_line_count", "mobility_score", "daily_opioid_dose", "daily_benzo_dose",
    "daily_antipsychotic_dose", "urine_output",
}

EXPECTED_VARIABLE_COUNT = 135  # 33 static + 98 temporal + 4 labels


def _build_static_row(rng: np.random.Generator, prone: bool) -> dict:
    """Baseline, time-invariant markers for one subject.

    Delirium-prone subjects carry the geriatric-vulnerability profile
    (older, frailer, lower MMSE/MNA, higher comorbidity) seen in the cohort.
    """
    return {
        "delirium_drug_count": int(rng.integers(0, 6)),
        "GDS": int(rng.integers(6, 13) if prone else rng.integers(0, 7)),
        "FRAIL": int(rng.integers(3, 6) if prone else rng.integers(0, 3)),
        "ADL": int(rng.integers(14, 21) if prone else rng.integers(0, 8)),
        "IADL": int(rng.integers(12, 25) if prone else rng.integers(8, 16)),
        "MNA": int(rng.integers(6, 12) if prone else rng.integers(12, 15)),
        "sex": int(rng.integers(0, 2)),
        "edu": int(rng.integers(0, 4)),
        "marst": int(rng.integers(0, 2)),
        "living": int(rng.integers(0, 3)),
        "smoking": int(rng.integers(0, 2)),
        "drinking": int(rng.integers(0, 2)),
        "hear_impair": int(rng.integers(0, 2)),
        "sight_impair": int(rng.integers(0, 2)),
        "hos_exp": int(rng.integers(0, 2)),
        "surgical_exp": int(rng.integers(0, 2)),
        "del_hx": int(rng.integers(0, 2) if prone else 0),
        "med_count": int(rng.integers(3, 16)),
        "benz_med": int(rng.integers(0, 2)),
        "antipsy_med": int(rng.integers(0, 2)),
        "narc_med": int(rng.integers(0, 2)),
        "other_med": int(rng.integers(0, 2)),
        "caregiver": int(rng.integers(0, 2)),
        "CCI": int(rng.integers(3, 8) if prone else rng.integers(0, 4)),
        "age": int(rng.integers(72, 89) if prone else rng.integers(65, 80)),
        "ht": float(round(rng.uniform(150, 180), 1)),
        "wt": float(round(rng.uniform(45, 90), 1)),
        "BMI": float(round(rng.uniform(18.5, 30.0), 1)),
        "op_level2": int(rng.integers(1, 4)),
        "anesthesia_duration": int(rng.integers(120, 300)),
        "op_duration": int(rng.integers(100, 260)),
        "ASAscore": int(rng.integers(1, 5)),
        "V1MMSE": int(rng.integers(18, 26) if prone else rng.integers(26, 31)),
    }


def _label_structure(prone: bool, t: int) -> tuple[int, int, int, int]:
    """Dependent (soft-gated) label block for one subject-day.

    * ``delirium``        : current-day POD status (drives the minority mask).
    * ``delirium_next_day``: occurrence at t+1; the gate for the two tasks below.
    * ``delirium_typ``    : 0=None, 1=Hyper, 2=Hypo, 3=Mixed (non-zero only if gated).
    * ``K-DRS-R-98``      : 0=None, 1=Moderate, 2=Severe (non-zero only if gated).

    Each prone subject is given delirium on POD t in {2,3,4}; the gated days
    t in {1,2,3} deterministically cycle through every subtype/severity class so
    all label classes appear in both the training and validation partitions.
    """
    if not prone:
        return 0, 0, 0, 0

    delirium = 1 if t in (2, 3, 4) else 0
    next_day = 1 if t in (1, 2, 3) else 0
    if next_day:
        subtype = {1: 1, 2: 2, 3: 3}[t]      # covers {1, 2, 3}
        severity = {1: 1, 2: 2, 3: 1}[t]     # covers {1, 2}
    else:
        subtype, severity = 0, 0
    return delirium, next_day, subtype, severity


def _build_temporal_row(rng: np.random.Generator, prone: bool, delirium: int) -> dict:
    """Daily labs, vitals, and care markers for one subject-day.

    Prone subjects (and delirious days in particular) carry the expected
    physiological derangements so the synthetic minority class remains
    distributionally distinct from the majority class.
    """
    sick = prone and bool(delirium)
    row = {
        "delirium_hrs": int(rng.integers(2, 25)) if delirium else 0,
        "NRS_pain": int(rng.integers(1, 7)),
        "NRS_movement": int(rng.integers(2, 9)),
        "body_temperature": round(rng.uniform(36.2, 38.4 if sick else 37.4), 1),
        "heart_rate": int(rng.integers(70, 110 if sick else 95)),
        "resp_rate": int(rng.integers(12, 24 if sick else 20)),
        "SBP": int(rng.integers(100, 160)),
        "DBP": int(rng.integers(55, 95)),
        "SpO2": round(rng.uniform(92.0 if sick else 95.0, 100.0), 1),
        "eGFR_MDRD": round(rng.uniform(45, 70) if prone else rng.uniform(70, 110), 1),
        "BUN": round(rng.uniform(15, 35) if prone else rng.uniform(8, 20), 1),
        "creatinine": round(rng.uniform(0.9, 1.8) if prone else rng.uniform(0.6, 1.1), 2),
        "sodium": round(rng.uniform(133, 145), 1),
        "potassium": round(rng.uniform(3.1, 3.8) if prone else rng.uniform(3.8, 4.8), 2),
        "chloride": round(rng.uniform(98, 108), 1),
        "calcium": round(rng.uniform(8.2, 10.2), 2),
        "magnesium": round(rng.uniform(1.6, 2.4), 2),
        "glucose": round(rng.uniform(90, 180 if sick else 140), 1),
        "hemoglobin": round(rng.uniform(9.0, 14.0), 1),
        "hematocrit": round(rng.uniform(28.0, 42.0), 1),
        "WBC": int(rng.integers(5000, 15000 if sick else 11000)),
        "platelet": int(rng.integers(120, 400)),
        "albumin": round(rng.uniform(2.8, 4.5), 2),
        "CRP": round(rng.uniform(2.0, 18.0) if sick else rng.uniform(0.1, 5.0), 2),
        "RDW": round(rng.uniform(14.5, 17.0) if prone else rng.uniform(11.5, 14.5), 1),
        "lactate": round(rng.uniform(0.8, 3.5 if sick else 2.0), 2),
        "daily_opioid_dose": int(rng.integers(0, 60)),
        "daily_benzo_dose": int(rng.integers(0, 10)),
        "daily_antipsychotic_dose": int(rng.integers(0, 10)) if sick else 0,
        "sleep_hours": round(rng.uniform(2.0 if sick else 4.0, 9.0), 1),
        "mobility_score": int(rng.integers(0, 4)),
        "restraint_use": int(rng.integers(0, 2)) if sick else 0,
        "foley_catheter": int(rng.integers(0, 2)),
        "iv_line_count": int(rng.integers(1, 4)),
        "urine_output": int(rng.integers(300, 2500)),
    }
    # Padded markers: standard-normal with a small prone shift so distributions
    # differ between classes without becoming degenerate (constant) columns.
    shift = 0.6 if prone else 0.0
    for name in TEMPORAL_FEATURES[len(NAMED_TEMPORAL_FEATURES):]:
        row[name] = round(float(rng.normal(shift, 1.0)), 4)
    return row


def create_pipeline_mock_data(output_path: "str | Path | None" = None, output_dir: "str | Path | None" = None) -> Path:
    """Generate the pipeline-compatible synthetic cohort and write it to CSV.

    Parameters
    ----------
    output_path:
        Full path of the CSV to write. Takes precedence over ``output_dir``.
    output_dir:
        Directory to write ``raw_longitudinal_data.csv`` into (kept for
        backward-compatible / standalone use).
    """
    if output_path is not None:
        out = Path(output_path)
    elif output_dir is not None:
        out = Path(output_dir) / "raw_longitudinal_data.csv"
    else:
        from .config import get_default_config

        cfg = get_default_config()
        out = cfg.base_dir / cfg.raw_longitudinal_csv

    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    records: list[dict] = []
    for pid in range(1, NUM_PATIENTS + 1):
        prone = (pid % 3 == 0)            # ~33% synthetic incidence
        static_row = _build_static_row(rng, prone)  # time-invariant per subject
        for t in range(TIME_STEPS):
            delirium, next_day, subtype, severity = _label_structure(prone, t)
            record = {"pid": pid, "time": t}
            record.update(static_row)
            record.update(_build_temporal_row(rng, prone, delirium))
            record["delirium"] = delirium
            record["delirium_next_day"] = next_day
            record["delirium_typ"] = subtype
            record["K-DRS-R-98"] = severity
            records.append(record)

    column_order = (
        ["pid", "time"] + STATIC_FEATURES + TEMPORAL_FEATURES + LABEL_COLS
    )
    df = pd.DataFrame.from_records(records)[column_order]

    # Enforce integer dtypes so the CSV round-trips with the expected types.
    int_cols = (
        [c for c in STATIC_FEATURES if c not in ("ht", "wt", "BMI")]
        + [c for c in TEMPORAL_FEATURES if c in INT_TEMPORAL_FEATURES]
        + LABEL_COLS
    )
    df[int_cols] = df[int_cols].astype(int)

    # Self-verification: the schema must match the manuscript appendix exactly.
    n_variables = len(STATIC_FEATURES) + len(TEMPORAL_FEATURES) + len(LABEL_COLS)
    assert len(TEMPORAL_FEATURES) == N_TEMPORAL, "temporal block must be 98 features"
    assert n_variables == EXPECTED_VARIABLE_COUNT, (
        f"expected {EXPECTED_VARIABLE_COUNT} variables, built {n_variables}"
    )
    assert (df.groupby("pid").size() == TIME_STEPS).all(), "panel must be balanced"

    df.to_csv(out, index=False)
    print(
        f"[Success] Synthetic cohort written to {out} "
        f"({NUM_PATIENTS} subjects x {TIME_STEPS} days, "
        f"{n_variables} clinical variables)."
    )
    return out


if __name__ == "__main__":
    create_pipeline_mock_data()
