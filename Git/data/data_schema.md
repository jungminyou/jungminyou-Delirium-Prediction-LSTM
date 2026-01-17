# Data Schema and Feature Specifications

This document defines the comprehensive feature set used for the Multi-Output LSTM model development, as detailed in the **Manuscript Supplement (Table 1)**.

## 1. Primary and Secondary Outcomes
| Category | Variable Name | Type | Description |
| :--- | :--- | :--- | :--- |
| **Primary Outcome** | `delirium_occurrence` | Binary | Postoperative Delirium (POD) diagnosed by CAM or DSM-V |
| **Secondary Outcomes**| `delirium_phenotype` | Cat | Subtypes (Hypoactive, Hyperactive, Mixed) |
| | `delirium_severity` | Numeric | Intensity of delirium symptoms (e.g., CAM-S score) |
| | `onset_timing` | Numeric | Postoperative day of first delirium occurrence |
| | `duration` | Numeric | Total number of days in a delirious state |

## 2. Model Input Features (Predictors)

### A. Baseline Characteristics & Clinical Indicators (Static Inputs)
Represented as 'Static Inputs' in the fusion layer of the LSTM model.

| Domain | Representative Variables | Scale/Unit |
| :--- | :--- | :--- |
| **General Characteristics** | Age, Sex, BMI, Education level, Living arrangement | years, kg/m², categorical |
| **Clinical History** | CCI (Charlson Comorbidity Index), Previous delirium, Medication count | count, yes/no |
| **Physical & Functional** | TO_FRAIL (Frailty), TO_ADL/IADL (Functional status) | scores (ordinal/numeric) |
| **Cognitive & Mental** | K-MMSE (Cognition), TO_GDS (Depression) | scores (numeric) |
| **Laboratory Findings** | WBC, Albumin, Creatinine, Hemoglobin, Electrolytes (Na, K, Cl) | Various (mg/dL, g/dL, etc.) |

### B. Intraoperative Phase-wise Data (Sequential Inputs)
High-frequency data aggregated into surgical phases for the LSTM backbone.

| Domain | Variables | Description |
| :--- | :--- | :--- |
| **Hemodynamics** | `MAP`, `Heart Rate`, `SpO2`, `EtCO2` | Vital signs monitored during surgery |
| **Surgical Stress** | `Op_Time`, `EBL (Estimated Blood Loss)` | Procedure duration and fluid loss |
| **Anesthesia/Fluids** | `Input_Crystal`, `Input_Colloid`, `Urine_Output` | Intraoperative fluid management |

## 3. Data Processing Details
- **Normalization:** All continuous clinical and laboratory features were standardized using Z-score normalization.
- **Handling Missing Values:** Missing laboratory data were handled via median imputation or excluded based on the threshold (specified in the manuscript).
- **Cohort Balance:** 1:1 Propensity Score Matching (PSM) was applied to the primary outcome (Delirium vs. Non-Delirium).