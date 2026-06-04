# Input Data Schema and Synthetic-Data Design

This document describes the single longitudinal input table consumed by the
pipeline (`raw_longitudinal_data.csv`) and the synthetic generator that
reproduces it (`src/generate_mock_data.py`).

## Why a Synthetic Generator

The original patient-level dataset cannot be distributed because of IRB and
institutional data-transfer restrictions. To keep the analysis fully
reproducible, the generator emits a table that is
schema-identical to the production input. The synthetic cohort exercises the
complete executable workflow but is **not** suitable for clinical inference.

## Variable Budget (135 clinical variables)

`pid` and `time` are record identifiers and are not counted among the 135.

| Block                | Count | Examples |
|----------------------|-------|----------|
| Static (baseline)    | 33    | `age`, `sex`, `edu`, `BMI`, `CCI`, `GDS`, `FRAIL`, `ADL`, `IADL`, `MNA`, `V1MMSE`, `ASAscore`, `op_duration`, `anesthesia_duration` |
| Temporal (daily)     | 98    | `delirium_hrs`, `NRS_pain`, `body_temperature`, `eGFR_MDRD`, `potassium`, `hemoglobin`, `albumin`, `RDW`, `CRP`, `daily_opioid_dose`, plus padded `daily_marker_NN` |
| Labels               | 4     | `delirium`, `delirium_next_day`, `delirium_typ`, `K-DRS-R-98` |
| **Total**            | 135   | |

The static and label *names* are defined once in `src/config.py`
(`STATIC_FEATURES`, `LABEL_COLS`) and imported by both the modeling code and the
generator, so the generated columns cannot drift from what the pipeline
consumes. The generator asserts the 135-variable total at write time, and keeps
data generation free of the heavy modeling dependencies (TensorFlow, tslearn).

## Column Order

```
["pid", "time"] + STATIC_FEATURES (33) + TEMPORAL_FEATURES (98) + LABEL_COLS (4)
```

The pipeline derives its temporal feature list as every column that is not an
identifier, a static feature, or a label, so this contiguous layout is required.

## Panel Structure

- Balanced panel: every subject has exactly **5** daily records
  (`time` = 0 for the pre-operative baseline through 4 for POD3). This is a hard
  requirement of the tensor reshaping in `src/augmentation.py` and
  `src/tensor_prep.py`.
- Static features are **time-invariant** within a subject, matching the
  `drop_duplicates("pid")` handling in the augmentation stage.
- Continuous measures (labs, vitals) are floats; counts, ordinal scores,
  binary flags, and all labels are integers, so that the distribution-enforcement
  step recognises and snaps integer columns correctly.

## Dependent (Soft-Gated) Label Structure

The label block mirrors the manuscript's hierarchical prediction target, where
subtype and severity are conditional on next-day occurrence:

- `delirium` — current-day POD status; drives the minority (delirium-prone) mask.
- `delirium_next_day` — occurrence at *t + 1*; the gate for the two tasks below.
- `delirium_typ` — `0`=None, `1`=Hyperactive, `2`=Hypoactive, `3`=Mixed; non-zero
  only when `delirium_next_day = 1`.
- `K-DRS-R-98` — `0`=None, `1`=Moderate, `2`=Severe; non-zero only when
  `delirium_next_day = 1`.

In the synthetic cohort, roughly one third of subjects are delirium-prone. Each
prone subject is delirious on POD `t ∈ {2, 3, 4}`, and the gated days
`t ∈ {1, 2, 3}` deterministically cycle through every subtype and severity class
so that all label classes appear in both the training and validation partitions.

## Reproducibility

Generation is fully deterministic under a fixed seed (`RANDOM_SEED = 42`).
Re-running `python -m src.generate_mock_data` reproduces the same table.
