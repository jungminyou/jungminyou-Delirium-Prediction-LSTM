from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def df_to_3d_tensor(df_subset: pd.DataFrame, feature_cols: list[str], label_cols: list[str], time_steps: int) -> tuple[np.ndarray, np.ndarray]:
    df_sorted = df_subset.sort_values(["pid", "time"])
    n_patients = df_sorted["pid"].nunique()

    x_flat = df_sorted[feature_cols].values.astype(np.float32)
    x_3d = x_flat.reshape(n_patients, time_steps, len(feature_cols))

    y_flat = df_sorted[label_cols].values.astype(np.float32)
    y_3d = y_flat.reshape(n_patients, time_steps, len(label_cols))

    return x_3d, y_3d


def build_scaled_tensors(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    df_tg: pd.DataFrame,
    df_sm: pd.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    time_steps: int,
) -> dict[str, np.ndarray]:
    x_tr_raw, y_tr = df_to_3d_tensor(df_train, feature_cols, label_cols, time_steps)
    x_val_raw, y_val = df_to_3d_tensor(df_val, feature_cols, label_cols, time_steps)
    x_test_raw, y_test = df_to_3d_tensor(df_test, feature_cols, label_cols, time_steps)

    x_tg_raw, y_tg = df_to_3d_tensor(df_tg, feature_cols, label_cols, time_steps)
    x_sm_raw, y_sm = df_to_3d_tensor(df_sm, feature_cols, label_cols, time_steps)

    n_tr, t, f = x_tr_raw.shape
    scaler = StandardScaler()

    x_tr = scaler.fit_transform(x_tr_raw.reshape(-1, f)).reshape(n_tr, t, f)
    x_val = scaler.transform(x_val_raw.reshape(-1, f)).reshape(x_val_raw.shape)
    x_test = scaler.transform(x_test_raw.reshape(-1, f)).reshape(x_test_raw.shape)

    x_tg_scaled = scaler.transform(x_tg_raw.reshape(-1, f)).reshape(x_tg_raw.shape)
    x_sm_scaled = scaler.transform(x_sm_raw.reshape(-1, f)).reshape(x_sm_raw.shape)

    x_tr_smote = np.concatenate([x_tr, x_sm_scaled], axis=0)
    y_tr_smote = np.concatenate([y_tr, y_sm], axis=0)

    x_tr_timegan = np.concatenate([x_tr, x_tg_scaled], axis=0)
    y_tr_timegan = np.concatenate([y_tr, y_tg], axis=0)

    return {
        "X_tr": x_tr,
        "Y_tr": y_tr,
        "X_val": x_val,
        "Y_val": y_val,
        "X_test": x_test,
        "Y_test": y_test,
        "X_tr_smote": x_tr_smote,
        "Y_tr_smote": y_tr_smote,
        "X_tr_timegan": x_tr_timegan,
        "Y_tr_timegan": y_tr_timegan,
    }
