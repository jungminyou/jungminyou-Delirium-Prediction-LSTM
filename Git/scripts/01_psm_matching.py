import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def perform_psm_pipeline(df, treatment_col, covariates):
    """
    Constructs a balanced cohort using 1:1 Propensity Score Matching (PSM).
    This strategy resolves class imbalance for the primary outcome.
    """
    # Preprocessing: Standardization of covariates
    scaler = StandardScaler()
    df[covariates] = scaler.fit_transform(df[covariates])

    # Propensity Score Estimation via Logistic Regression
    ps_model = LogisticRegression(solver='liblinear')
    ps_model.fit(df[covariates], df[treatment_col])
    df['propensity_score'] = ps_model.predict_proba(df[covariates])[:, 1]

    # 1:1 Nearest Neighbor Matching without replacement
    treated = df[df[treatment_col] == 1].reset_index(drop=True)
    control = df[df[treatment_col] == 0].reset_index(drop=True)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])
    distances, indices = nn.kneighbors(treated[['propensity_score']])

    matched_control = control.iloc[indices.flatten()]
    balanced_df = pd.concat([treated, matched_control])

    return balanced_df