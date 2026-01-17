# Multi-Output LSTM-Based Prediction of Postoperative Delirium in Older Spine Surgery Patients

This repository contains the official implementation of the research framework for predicting **Postoperative Delirium (POD)**. Our approach integrates preoperative baseline clinical assessments with high-frequency intraoperative time-series data using a **Multi-Output LSTM** architecture with a **Soft Gating mechanism**.

## Project Overview

Postoperative delirium is a significant complication in older patients undergoing spine surgery. This project proposes a deep learning model that not only predicts the occurrence of delirium but also accounts for its clinical manifestation (phenotypes and severity) by modeling the inter-dependencies between outcomes.

### Key Methodological Highlights

* **Balanced Cohort via PSM**: Utilized 1:1 Propensity Score Matching to resolve class imbalance (N=190; 95 Delirium vs. 95 Matched Controls).
* **Soft Gating Architecture**: Implemented a gating mechanism where the primary delirium prediction modulates the information flow to secondary tasks (phenotypes, duration, etc.).
* **Robust Optimization**: Extensive grid search (200 log-spaced L2 values, dropout 0.2–0.5) and stratified 80/20 validation.
* **Explainable AI (XAI)**: Clinical interpretability provided via SHAP (SHapley Additive exPlanations) using a custom LSTM wrapper.

---

## Model Architecture

The model utilizes a hybrid input structure to capture the full perioperative context:

1. **Static Branch**: Processes preoperative geriatric assessments, laboratory findings, and patient demographics.
2. **Sequential Branch**: Processes aggregated intraoperative hemodynamic and surgical stress data via an LSTM backbone.
3. **Soft Gating Layer**: The delirium prediction output acts as a soft gate, reflecting the clinical reality that the occurrence of delirium dictates its subsequent phenotypes and severity.

---

## Repository Structure

```text
├── data/
│   └── data_schema.md       # Definition of features (Ref: Supplement Table 1)
├── scripts/
│   ├── 01_psm_matching.py   # Cohort balancing using 1:1 PSM
│   ├── 02_model_train.py    # Soft Gated LSTM construction and optimization
│   └── 03_interpretation.py # SHAP-based feature attribution analysis
├── requirements.txt         # Environment dependencies
└── README.md                # Project documentation

```

---

## Data Specifications

The model incorporates 50+ variables as detailed in **Supplement Table 1** of the manuscript, including:

* **Preoperative**: CCI, Frailty (TO_FRAIL), Cognition (K-MMSE), and comprehensive laboratory findings (Albumin, Hemoglobin, etc.).
* **Intraoperative**: Phase-wise hemodynamic stability (MAP, HR), surgical stress (EBL, Op_Time), and fluid management.

*Note: Raw clinical data is not publicly available due to institutional privacy policies. Please refer to `data/data_schema.md` for variable definitions.*

---

## Getting Started

### Prerequisites

* Python 3.8+
* TensorFlow 2.x
* SHAP

### Installation

```bash
git clone https://github.com/jungminyou/Delirium-Prediction-LSTM.git
cd Delirium-Prediction-LSTM
pip install -r requirements.txt

```

### Usage

1. **PSM Matching**: `python scripts/01_psm_matching.py`
2. **Training & Optimization**: `python scripts/02_model_train.py`
3. **Interpretation**: `python scripts/03_interpretation.py`

---

## Authors and Affiliation

**Jungmin You, Research Professor** Mo-Im Kim Nursing Research Institute, College of Nursing, Yonsei University, Seoul, Republic of Korea.

Email: jmyou@yuhs.ac

## Citation

If you find this code or research helpful, please cite our paper:

> You, J., et al. (2026). Multi-Output LSTM-Based Prediction of Postoperative Delirium: Integrating Baseline and Perioperative Data for Enhanced Risk Stratification in Older Spine Surgery Patients.
