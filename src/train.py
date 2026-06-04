from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping

from .models import build_multi_output_lstm_gated


def train_all_models(tensors: dict[str, np.ndarray], lstm_epochs: int = 300, batch_size: int = 32) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    x_tr = tensors["X_tr"]
    y_tr = tensors["Y_tr"]
    x_val = tensors["X_val"]
    y_val = tensors["Y_val"]

    x_tr_smote = tensors["X_tr_smote"]
    y_tr_smote = tensors["Y_tr_smote"]
    x_tr_timegan = tensors["X_tr_timegan"]
    y_tr_timegan = tensors["Y_tr_timegan"]

    time_steps = x_tr.shape[1]
    n_features = x_tr.shape[2]

    preds_dict_next: dict[str, np.ndarray] = {}
    preds_dict_typ: dict[str, np.ndarray] = {}
    preds_dict_kdr: dict[str, np.ndarray] = {}

    # Model 1: logistic regression baseline with gating
    x_tr_lr = x_tr.reshape(-1, x_tr.shape[2])
    x_val_lr = x_val.reshape(-1, x_val.shape[2])

    y_tr_lr_next = y_tr[..., 1].ravel()
    lr_next = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=300)
    lr_next.fit(x_tr_lr, y_tr_lr_next)
    p_occ = lr_next.predict_proba(x_val_lr)[:, 1]
    preds_dict_next["LR"] = p_occ

    y_tr_lr_typ = y_tr[..., 2].ravel()
    lr_typ = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=300)
    lr_typ.fit(x_tr_lr, y_tr_lr_typ)
    p_typ_raw = lr_typ.predict_proba(x_val_lr)
    p_typ_gated = p_typ_raw.copy()
    p_typ_gated[:, 1:] = p_typ_raw[:, 1:] * p_occ[:, np.newaxis]
    p_typ_gated[:, 0] = 1.0 - np.sum(p_typ_gated[:, 1:], axis=1)
    preds_dict_typ["LR"] = p_typ_gated

    y_tr_lr_kdr = y_tr[..., 3].ravel()
    lr_kdr = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=300)
    lr_kdr.fit(x_tr_lr, y_tr_lr_kdr)
    p_kdr_raw = lr_kdr.predict_proba(x_val_lr)
    p_kdr_gated = p_kdr_raw.copy()
    p_kdr_gated[:, 1:] = p_kdr_raw[:, 1:] * p_occ[:, np.newaxis]
    p_kdr_gated[:, 0] = 1.0 - np.sum(p_kdr_gated[:, 1:], axis=1)
    preds_dict_kdr["LR"] = p_kdr_gated

    # LSTM models
    c_typ = int(np.max(y_tr[..., 2])) + 1
    c_kdr = int(np.max(y_tr[..., 3])) + 1

    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    def fit_predict_lstm(x_train: np.ndarray, y_train: np.ndarray, tag: str) -> None:
        model = build_multi_output_lstm_gated(time_steps, n_features, c_typ, c_kdr)
        y_train_next = np.expand_dims(y_train[..., 1], axis=-1)
        y_val_next = np.expand_dims(y_val[..., 1], axis=-1)

        model.fit(
            x_train,
            [y_train_next, y_train[..., 2], y_train[..., 3]],
            validation_data=(x_val, [y_val_next, y_val[..., 2], y_val[..., 3]]),
            epochs=lstm_epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0,
        )

        pred = model.predict(x_val, verbose=0)
        preds_dict_next[tag] = pred[0].ravel()
        preds_dict_typ[tag] = pred[1].reshape(-1, pred[1].shape[-1])
        preds_dict_kdr[tag] = pred[2].reshape(-1, pred[2].shape[-1])

    # Weighted training for base model (occurrence imbalance)
    model_base = build_multi_output_lstm_gated(time_steps, n_features, c_typ, c_kdr)
    y_tr_next = np.expand_dims(y_tr[..., 1], axis=-1)
    y_val_next = np.expand_dims(y_val[..., 1], axis=-1)
    weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_tr[..., 1].ravel()), y=y_tr[..., 1].ravel())
    sw_next = np.where(y_tr_next == 1, weights[1], weights[0])
    sw_typ = np.ones_like(y_tr[..., 2])
    sw_kdr = np.ones_like(y_tr[..., 3])

    model_base.fit(
        x_tr,
        [y_tr_next, y_tr[..., 2], y_tr[..., 3]],
        validation_data=(x_val, [y_val_next, y_val[..., 2], y_val[..., 3]]),
        epochs=lstm_epochs,
        batch_size=batch_size,
        callbacks=[es],
        sample_weight=[sw_next, sw_typ, sw_kdr],
        verbose=0,
    )
    pred_base = model_base.predict(x_val, verbose=0)
    preds_dict_next["LSTM (Base)"] = pred_base[0].ravel()
    preds_dict_typ["LSTM (Base)"] = pred_base[1].reshape(-1, pred_base[1].shape[-1])
    preds_dict_kdr["LSTM (Base)"] = pred_base[2].reshape(-1, pred_base[2].shape[-1])

    fit_predict_lstm(x_tr_smote, y_tr_smote, "LSTM (DTW-SMOTE)")
    fit_predict_lstm(x_tr_timegan, y_tr_timegan, "LSTM (TimeGAN)")

    return preds_dict_next, preds_dict_typ, preds_dict_kdr
