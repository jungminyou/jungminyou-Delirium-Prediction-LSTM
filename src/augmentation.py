from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, LSTM, Masking
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tslearn.metrics import dtw_path

from .config import LABEL_COLS, STATIC_FEATURES, get_default_config

warnings.filterwarnings("ignore")


@dataclass
class AugmentationResult:
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    df_tg: pd.DataFrame
    df_sm: pd.DataFrame
    all_feature_cols: list[str]
    label_cols: list[str]
    time_steps: int


class TimeGAN:
    def __init__(self, seq_len: int, feature_dim: int, hidden_dim: int = 24, num_layers: int = 3, batch_size: int = 64, lr: float = 1e-3) -> None:
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lr = lr
        self.mse_loss = MeanSquaredError()
        self._build_models()
        self._init_optimizers()

    def _build_models(self) -> None:
        x_in = Input((self.seq_len, self.feature_dim))
        h = Masking()(x_in)
        for _ in range(self.num_layers):
            h = LSTM(self.hidden_dim, return_sequences=True)(h)
        h_emb = Dense(self.hidden_dim, activation="sigmoid")(h)
        self.embedder = Model(x_in, h_emb)

        h_in = Input((self.seq_len, self.hidden_dim))
        r = h_in
        for _ in range(self.num_layers):
            r = LSTM(self.hidden_dim, return_sequences=True)(r)
        x_tilde = Dense(self.feature_dim, activation="sigmoid")(r)
        self.recovery = Model(h_in, x_tilde)

        z = Input((self.seq_len, self.hidden_dim))
        g = z
        for _ in range(self.num_layers):
            g = LSTM(self.hidden_dim, return_sequences=True)(g)
        e_hat = Dense(self.hidden_dim, activation="sigmoid")(g)
        self.generator = Model(z, e_hat)

        s = Input((self.seq_len, self.hidden_dim))
        sup = s
        for _ in range(self.num_layers - 1):
            sup = LSTM(self.hidden_dim, return_sequences=True)(sup)
        h_hat = Dense(self.hidden_dim, activation="sigmoid")(sup)
        self.supervisor = Model(s, h_hat)

    def _init_optimizers(self) -> None:
        self.opt_e = Adam(self.lr)

    def train(self, x: np.ndarray, epochs: int = 500) -> None:
        x_input = Input(shape=(self.seq_len, self.feature_dim))
        h = self.embedder(x_input)
        x_tilde = self.recovery(h)
        autoencoder = Model(x_input, x_tilde)
        autoencoder.compile(optimizer=self.opt_e, loss="mse")
        autoencoder.fit(x, x, epochs=max(1, epochs // 10), batch_size=self.batch_size, verbose=0)

    def sample(self, n: int) -> np.ndarray:
        z = tf.random.normal((n, self.seq_len, self.hidden_dim))
        h_fake = self.supervisor(self.generator(z))
        x_hat = self.recovery(h_fake).numpy().astype(np.float32)
        return x_hat


def dtw_smote_pair(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    path, _ = dtw_path(x, y)
    z = np.array([(1 - alpha) * x[i] + alpha * y[j] for i, j in path])
    if z.shape[0] != x.shape[0]:
        idx_map = np.round(np.linspace(0, z.shape[0] - 1, x.shape[0])).astype(int)
        z = z[idx_map]
    return z


def dtw_smote(x: np.ndarray, n_samples: int, alpha_range: tuple[float, float] = (0.1, 0.9)) -> np.ndarray:
    synthetic = []
    for _ in range(n_samples):
        i, j = np.random.choice(len(x), 2, replace=False)
        alpha = np.random.uniform(*alpha_range)
        synthetic.append(dtw_smote_pair(x[i], x[j], alpha))
    return np.stack(synthetic).astype(np.float32)


def enforce_distribution_strict(x_aug: np.ndarray, df_train: pd.DataFrame, all_cols: list[str]) -> np.ndarray:
    n, t, d = x_aug.shape
    x_flat = x_aug.reshape(n * t, d)
    for j, col_name in enumerate(all_cols):
        orig_series = df_train[col_name]
        min_val, max_val = orig_series.min(), orig_series.max()
        x_flat[:, j] = np.clip(x_flat[:, j], min_val, max_val)

        if np.issubdtype(orig_series.dtype, np.integer):
            unique_vals = np.sort(orig_series.unique())
            idx = np.abs(x_flat[:, j][:, None] - unique_vals[None, :]).argmin(axis=1)
            x_flat[:, j] = unique_vals[idx]
        else:
            x_flat[:, j] = x_flat[:, j].astype(np.float32)
    return x_flat.reshape(n, t, d)


def rebuild_dataframe_refined(
    x_synth: np.ndarray,
    df_train: pd.DataFrame,
    original_minority_pids: np.ndarray,
    features: list[str],
    labels: list[str],
    start_pid: int,
    source_name: str,
) -> tuple[pd.DataFrame, int]:
    n_synth, t, d = x_synth.shape
    new_pids = np.arange(start_pid, start_pid + n_synth)

    label_patterns = []
    for pid in original_minority_pids:
        p_labels = df_train[df_train["pid"] == pid].sort_values("time")[labels].values
        label_patterns.append(p_labels)
    label_patterns = np.array(label_patterns)

    sampled_idx = np.random.choice(len(label_patterns), n_synth, replace=True)
    y_synth = label_patterns[sampled_idx]

    pid_list = np.repeat(new_pids, t)
    time_list = np.tile(np.arange(t), n_synth)
    x_flat = x_synth.reshape(n_synth * t, d)
    y_flat = y_synth.reshape(n_synth * t, len(labels))

    full_arr = np.column_stack([pid_list, time_list, x_flat, y_flat])
    df_cols = ["pid", "time"] + features + labels
    df_synth = pd.DataFrame(full_arr, columns=df_cols)

    for col in df_cols:
        df_synth[col] = df_synth[col].astype(df_train[col].dtype)

    df_synth["Dataset_Type"] = source_name
    return df_synth, int(new_pids.max()) + 1


def run_augmentation_pipeline(base_dir: Path, seed: int = 42, timegan_epochs: int = 500) -> AugmentationResult:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    cfg = get_default_config()

    src_path = base_dir / cfg.raw_longitudinal_csv
    df = pd.read_csv(src_path)

    if "CAM_diagnosis" in df.columns:
        df = df.drop(columns=["CAM_diagnosis"])

    feature_cols = [c for c in df.columns if c not in ["pid", "time"] + STATIC_FEATURES + LABEL_COLS]
    all_feature_cols = feature_cols + STATIC_FEATURES

    pids = df["pid"].unique()
    train_pids, temp_pids = train_test_split(pids, test_size=0.3, random_state=seed)
    val_pids, test_pids = train_test_split(temp_pids, test_size=0.5, random_state=seed)

    df_train = df[df["pid"].isin(train_pids)].copy()
    df_val = df[df["pid"].isin(val_pids)].copy()
    df_test = df[df["pid"].isin(test_pids)].copy()

    train_pid_order = df_train["pid"].drop_duplicates().values
    n_train_patients = len(train_pid_order)
    time_steps = int(df["time"].nunique())

    x_dyn_train = df_train.sort_values(["pid", "time"])[feature_cols].values
    seqs_dyn_train = x_dyn_train.reshape(n_train_patients, time_steps, len(feature_cols)).astype(np.float32)

    static_df_train = df_train.drop_duplicates("pid").set_index("pid").loc[train_pid_order, STATIC_FEATURES]
    static_arr_train = static_df_train.values.astype(np.float32)

    mask_train = df_train.groupby("pid")["delirium"].max().loc[train_pid_order].values == 1
    x_min_dyn = seqs_dyn_train[mask_train]
    static_min_seq = np.repeat(static_arr_train[mask_train][:, None, :], time_steps, axis=1)

    x_min = np.concatenate([x_min_dyn, static_min_seq], axis=-1)
    n_to_generate = int((~mask_train).sum() - x_min.shape[0])

    tg = TimeGAN(seq_len=time_steps, feature_dim=x_min.shape[2])
    tg.train(x_min, epochs=timegan_epochs)
    x_tg = tg.sample(max(1, n_to_generate // 2))

    x_sm = dtw_smote(np.vstack([x_min, x_tg]), max(1, n_to_generate // 2))

    x_tg = enforce_distribution_strict(x_tg, df_train, all_feature_cols)
    x_sm = enforce_distribution_strict(x_sm, df_train, all_feature_cols)

    max_pid = int(df["pid"].max())
    minority_pids = train_pid_order[mask_train]

    df_tg, next_pid = rebuild_dataframe_refined(x_tg, df_train, minority_pids, all_feature_cols, LABEL_COLS, max_pid + 1, "TimeGAN")
    df_sm, _ = rebuild_dataframe_refined(x_sm, df_train, minority_pids, all_feature_cols, LABEL_COLS, next_pid, "SMOTE")

    df_train["Dataset_Type"] = "Original_Train"
    df_train_augmented = pd.concat([df_train, df_tg, df_sm], ignore_index=True)

    df_train_augmented.to_csv(base_dir / cfg.train_augmented_csv, index=False)
    df_val.assign(Dataset_Type="Original_Val").to_csv(base_dir / cfg.val_original_csv, index=False)
    df_test.assign(Dataset_Type="Original_Test").to_csv(base_dir / cfg.test_original_csv, index=False)

    return AugmentationResult(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        df_tg=df_tg,
        df_sm=df_sm,
        all_feature_cols=all_feature_cols,
        label_cols=LABEL_COLS,
        time_steps=time_steps,
    )
