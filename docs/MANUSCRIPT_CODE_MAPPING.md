# Study-to-Repository Map

## Purpose

This document translates the manuscript into feature-focused repository sections for public release. It avoids internal notebook history and keeps the description aligned with the scientific workflow rather than implementation detail.

## Privacy Statement

- The clinical source dataset is private and not included in this repository.
- Public release is limited to de-identified, policy-approved artifacts.
- The repository is intended to reproduce the computational workflow once approved input data are available.
- A schema-matched synthetic cohort (`src/generate_mock_data.py`, documented in `docs/DATA_SCHEMA.md`) lets reviewers run the complete workflow without the IRB-restricted source data.

## Functional Mapping

### Cohort Design and Leakage Control

- Manuscript focus:
  cohort definition, leakage prevention, and patient-level partitioning
- Repository function:
  data preparation logic that removes leakage-prone variables and preserves patient-level independence
- Public interpretation:
  the repository recreates the study cohort design without exposing raw clinical records

### Synthetic Minority Augmentation

- Manuscript focus:
  generative and interpolation-based augmentation for underrepresented outcome trajectories
- Repository function:
  sequence generation, alignment-aware interpolation, and realistic reconstruction of synthetic cohorts
- Public interpretation:
  the repository supports two complementary augmentation strategies for imbalance reduction

### Tensor Construction and Safe Scaling

- Manuscript focus:
  conversion from longitudinal tables to model-ready temporal tensors
- Repository function:
  sequence reshaping and train-only normalization
- Public interpretation:
  preprocessing is structured to prevent information leakage across cohorts

### Multi-Task Prediction

- Manuscript focus:
  joint prediction of occurrence, subtype, and severity
- Repository function:
  baseline classification and gated deep learning for correlated outcomes
- Public interpretation:
  the modeling stage reflects the manuscript's hierarchical clinical prediction objective

### Evaluation and Clinical Utility

- Manuscript focus:
  discrimination, calibration, class-wise performance, and decision analysis
- Repository function:
  reusable evaluation routines that generate tables and figures for each prediction target
- Public interpretation:
  the repository is organized to support reviewer-friendly performance reporting

### Synthetic Data Fidelity

- Manuscript focus:
  validation that synthetic sequences remain clinically plausible and distributionally consistent
- Repository function:
  statistical testing, divergence analysis, distance metrics, and low-dimensional visualization
- Public interpretation:
  the fidelity stage documents whether augmentation preserves meaningful cohort structure

## Reproducibility Summary

- Random seed is fixed for cohort splitting, augmentation, and evaluation resampling.
- All cohort partitions are patient-level.
- Feature scaling is learned from the original training cohort only.
- The full workflow can be executed end to end or by individual stage.

## Reviewer Note

"Due to privacy regulations and institutional policy, source-level patient data cannot be publicly shared. This repository provides the complete executable workflow for preprocessing, augmentation, modeling, evaluation, and synthetic-data validation. A schema-matched synthetic cohort is generated automatically so the pipeline can be reproduced end to end with approved de-identified inputs or with the bundled synthetic data."
