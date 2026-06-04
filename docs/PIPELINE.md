# Pipeline Documentation

## 1. Data Preparation

- Load the de-identified longitudinal cohort table.
- Remove variables that could leak outcome information into the model.
- Separate temporal predictors, static predictors, and longitudinal labels.
- Preserve patient identity and visit order for every downstream step.

## 2. Leakage-Free Cohort Construction

- Split the cohort by patient rather than by row.
- Create independent training, validation, and test cohorts.
- Keep temporal trajectories intact within each patient.

## 3. Synthetic Minority Sequence Generation

- Identify minority trajectories from the training cohort.
- Generate synthetic sequences with a TimeGAN-based model.
- Create additional sequence-aware synthetic samples with DTW-guided interpolation.
- Constrain synthetic values to remain within observed clinical ranges.
- Reattach realistic label trajectories sampled from genuine minority patients.

## 4. Tensor Preparation

- Convert tabular patient trajectories into three-dimensional sequence tensors.
- Fit feature scaling on the original training cohort only.
- Apply the same transform to validation, test, and synthetic cohorts.
- Build parallel training sets for the original, interpolation-augmented, and generative-augmented scenarios.

## 5. Multi-Task Modeling

- Train a class-balanced logistic regression baseline for next-day occurrence.
- Use gated probability transfer so downstream subtype and severity predictions depend on occurrence risk.
- Train a soft-gated multi-task LSTM under three settings:
  original data only, original plus interpolation-based augmentation, and original plus generative augmentation.

## 6. Performance Evaluation

- Measure AUROC and AUPRC with bootstrap confidence intervals.
- Report accuracy, precision, recall, and F1 scores for each outcome class.
- Estimate calibration with Brier score and calibration slope.
- Quantify clinical utility through decision-curve net benefit.

## 7. Fidelity Assessment

- Compare genuine and synthetic cohorts with nonparametric statistical testing.
- Measure distribution similarity with Jensen-Shannon divergence.
- Measure sequence similarity with DTW and Wasserstein distance.
- Visualize cohort overlap with pairwise t-SNE embeddings.

## 8. Public Outputs

- Cohort tables for training, validation, and test partitions
- Outcome-specific performance tables
- Fidelity summary tables
- Publication-ready figures for performance and synthetic-data validation
