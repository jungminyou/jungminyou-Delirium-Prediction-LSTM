# Delirium Prediction Pipeline

This repository contains the research pipeline used to study postoperative
delirium prediction from longitudinal clinical trajectories. The workflow is
organized into clear functional stages so the full analysis can be reproduced
without exposing private source data or internal notebook history.

## What This Repository Does

- builds leakage-free patient splits,
- augments minority trajectories with sequence-aware synthetic generation,
- trains baseline and deep multi-task prediction models,
- evaluates discrimination, calibration, and clinical utility,
- measures fidelity between original and synthetic cohorts.

## Functional Guide

- Data preparation and augmentation: cohort splitting, leakage control, TimeGAN generation, and DTW-based interpolation
- Tensor preparation: sequence conversion and train-only feature scaling
- Modeling: logistic regression baseline and soft-gated multi-task LSTM
- Evaluation: AUROC, AUPRC, calibration, and decision-curve metrics
- Fidelity analysis: statistical comparison, divergence analysis, distance metrics, and visualization

Additional documentation:

- `docs/PIPELINE.md` — stage-by-stage description of the workflow
- `docs/DATA_SCHEMA.md` — the 135-variable input schema and synthetic-data design
- `docs/MANUSCRIPT_CODE_MAPPING.md` — manuscript-to-repository map

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the full workflow with `python run_all.py`.

On first run, if `raw_longitudinal_data.csv` is not present, a synthetic cohort
that mirrors the manuscript schema is generated automatically so the pipeline
executes end to end. To create the synthetic input explicitly beforehand:

```
python -m src.generate_mock_data
```

When approved de-identified data are available, place the table in the project
root as `raw_longitudinal_data.csv` and it will be used instead of the synthetic
cohort (the synthetic generator never overwrites an existing input).

## Stage Execution

Each stage can also be run on its own:

1. `python run_stage_01_augmentation.py`
2. `python run_stage_02_tensor_prep.py`
3. `python run_stage_03_train_eval.py`
4. `python run_stage_04_fidelity.py`

## Public Input and Output Names

- Input: `raw_longitudinal_data.csv`
- Generated training cohort: `train_augmented.csv`
- Generated validation cohort: `validation_original.csv`
- Generated test cohort: `test_original.csv`
- Performance tables: occurrence, subtype, and severity metrics
- Fidelity outputs: pairwise statistics, divergence tables, distance metrics, and a t-SNE figure

## Reproducibility

- All cohort partitions are patient-level and controlled with fixed random seeds.
- Leakage-prone variables are removed before model training.
- Feature scaling is fit on the original training cohort only.
- The repository is self-contained: inputs and outputs live under the project
  root and the workflow runs immediately after cloning.

## Code Availability

All code required to reproduce the preprocessing, augmentation, modeling,
evaluation, and synthetic-data validation reported in the manuscript is provided
in this repository under the MIT License. The complete workflow can be executed
end to end (`python run_all.py`) or stage by stage.

## Data Availability

The original patient-level clinical dataset cannot be shared publicly because of
IRB and institutional data-transfer restrictions. To support full
reproducibility, this repository includes a synthetic
data generator (`src/generate_mock_data.py`) that emits a longitudinal table
matching the manuscript's 135-variable structure and dependent-label design (see
`docs/DATA_SCHEMA.md`). The synthetic cohort reproduces the executable workflow
but is **not** suitable for clinical inference. Access to the de-identified
source data may be requested from the data custodian subject to institutional
approval.

## Citation and License

Citation metadata is provided in `CITATION.cff`. This
project is released under the MIT License.
