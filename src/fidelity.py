from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, mannwhitneyu, wasserstein_distance
from sklearn.manifold import TSNE


def run_pairwise_statistics(train_aug: pd.DataFrame, cont_features: list[str], cat_features: list[str]) -> pd.DataFrame:
    orig_label = "Original_Train"
    tg_label = "TimeGAN"
    sm_label = "SMOTE"

    if "Dataset_Type" not in train_aug.columns or "delirium" not in train_aug.columns:
        raise ValueError("train_aug must contain `Dataset_Type` and `delirium` columns.")

    cont_features = [c for c in cont_features if c in train_aug.columns]
    cat_features = [c for c in cat_features if c in train_aug.columns]

    df_cont_agg = train_aug.groupby(["pid", "Dataset_Type"])[["delirium"] + cont_features].mean().reset_index()
    df_cat_agg = train_aug.sort_values(["pid", "time"]).drop_duplicates(["pid", "Dataset_Type"])[
        ["pid", "Dataset_Type", "delirium"] + cat_features
    ]

    rows = []

    def _summary_cont(s: pd.Series) -> str:
        if len(s) == 0:
            return "NA"
        return f"{s.median():.2f} ({s.quantile(0.25):.2f}, {s.quantile(0.75):.2f})"

    def _summary_cat(s: pd.Series, levels: list) -> str:
        if len(s) == 0:
            return "NA"
        counts = [(s == lv).sum() for lv in levels]
        total = len(s)
        return " / ".join([f"{int(c)} ({100 * c / total:.1f}%)" for c in counts])

    def _safe_mwu(a: pd.Series, b: pd.Series) -> float:
        if len(a) == 0 or len(b) == 0:
            return np.nan
        if np.all(a.values == a.values[0]) and np.all(b.values == b.values[0]) and a.values[0] == b.values[0]:
            return 1.0
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)

    def _safe_chi2(orig_vals: pd.Series, synth_vals: pd.Series, levels: list) -> float:
        if len(orig_vals) == 0 or len(synth_vals) == 0:
            return np.nan
        contingency = pd.DataFrame(
            {
                "Original": orig_vals.value_counts().reindex(levels, fill_value=0),
                "Synthetic": synth_vals.value_counts().reindex(levels, fill_value=0),
            }
        )
        if (contingency.sum(axis=1) == 0).all():
            return np.nan
        return float(chi2_contingency(contingency)[1])

    for col in cont_features:
        orig_vals = df_cont_agg[
            (df_cont_agg["Dataset_Type"] == orig_label) & (df_cont_agg["delirium"] == 1)
        ][col].dropna()
        tg_vals = df_cont_agg[df_cont_agg["Dataset_Type"] == tg_label][col].dropna()
        sm_vals = df_cont_agg[df_cont_agg["Dataset_Type"] == sm_label][col].dropna()

        p_tg = _safe_mwu(orig_vals, tg_vals)
        p_sm = _safe_mwu(orig_vals, sm_vals)

        rows.append(
            {
                "Variable": col,
                "Original (Genuine)": _summary_cont(orig_vals),
                "TimeGAN": _summary_cont(tg_vals),
                "p-value (vs TG)": f"{p_tg:.4f}" if pd.notna(p_tg) else "NA",
                "SMOTE": _summary_cont(sm_vals),
                "p-value (vs SM)": f"{p_sm:.4f}" if pd.notna(p_sm) else "NA",
            }
        )

    for col in cat_features:
        levels = sorted(train_aug[col].dropna().unique())
        orig_cat = df_cat_agg[(df_cat_agg["Dataset_Type"] == orig_label) & (df_cat_agg["delirium"] == 1)][col].dropna()
        tg_cat = df_cat_agg[df_cat_agg["Dataset_Type"] == tg_label][col].dropna()
        sm_cat = df_cat_agg[df_cat_agg["Dataset_Type"] == sm_label][col].dropna()

        p_tg = _safe_chi2(orig_cat, tg_cat, levels)
        p_sm = _safe_chi2(orig_cat, sm_cat, levels)

        rows.append(
            {
                "Variable": f"{col} (Levels: {levels})",
                "Original (Genuine)": _summary_cat(orig_cat, levels),
                "TimeGAN": _summary_cat(tg_cat, levels),
                "p-value (vs TG)": f"{p_tg:.4f}" if pd.notna(p_tg) else "NA",
                "SMOTE": _summary_cat(sm_cat, levels),
                "p-value (vs SM)": f"{p_sm:.4f}" if pd.notna(p_sm) else "NA",
            }
        )

    return pd.DataFrame(rows)


def get_js_divergence(p_data: np.ndarray, q_data: np.ndarray, is_categorical: bool = False) -> float:
    p_data = np.asarray(p_data)
    q_data = np.asarray(q_data)
    if len(p_data) == 0 or len(q_data) == 0:
        return np.nan

    if is_categorical:
        all_vals = np.unique(np.concatenate([p_data, q_data]))
        p_prob = pd.Series(p_data).value_counts().reindex(all_vals, fill_value=0).values
        q_prob = pd.Series(q_data).value_counts().reindex(all_vals, fill_value=0).values
    else:
        combined = np.concatenate([p_data, q_data])
        min_v, max_v = np.min(combined), np.max(combined)
        if min_v == max_v:
            return 0.0
        bins = np.linspace(min_v, max_v, 51)
        p_prob, _ = np.histogram(p_data, bins=bins)
        q_prob, _ = np.histogram(q_data, bins=bins)

    js_distance = jensenshannon(p_prob, q_prob)
    return float(js_distance**2)


def compute_jsd_table(train_aug: pd.DataFrame, cont_features: list[str], cat_features: list[str]) -> pd.DataFrame:
    orig_label = "Original_Train"
    tg_label = "TimeGAN"
    sm_label = "SMOTE"

    cont_features = [c for c in cont_features if c in train_aug.columns]
    cat_features = [c for c in cat_features if c in train_aug.columns]

    rows = []
    for col in cont_features + cat_features:
        if col not in train_aug.columns:
            continue
        is_cat = col in cat_features
        orig_vals = train_aug[
            (train_aug["Dataset_Type"] == orig_label) & (train_aug["delirium"] == 1)
        ][col].dropna().values
        tg_vals = train_aug[train_aug["Dataset_Type"] == tg_label][col].dropna().values
        sm_vals = train_aug[train_aug["Dataset_Type"] == sm_label][col].dropna().values

        rows.append(
            {
                "Variable": col,
                "Type": "Categorical" if is_cat else "Continuous",
                "JSD (Orig vs TimeGAN)": get_js_divergence(orig_vals, tg_vals, is_categorical=is_cat),
                "JSD (Orig vs SMOTE)": get_js_divergence(orig_vals, sm_vals, is_categorical=is_cat),
            }
        )
    return pd.DataFrame(rows)


def run_tsne_pairwise(train_aug: pd.DataFrame, feature_cols: list[str], perplexity: int = 30) -> dict[str, np.ndarray]:
    time_steps = int(train_aug["time"].nunique())

    def get_flattened(df_type: str, only_delirium: bool = False) -> np.ndarray:
        subset = train_aug[train_aug["Dataset_Type"] == df_type]
        if only_delirium:
            target_pids = subset.groupby("pid")["delirium"].max()
            target_pids = target_pids[target_pids == 1].index
            subset = subset[subset["pid"].isin(target_pids)]

        seqs = []
        for _, group in subset.groupby("pid"):
            group = group.sort_values("time")
            if len(group) == time_steps:
                seqs.append(group[feature_cols].values.flatten())
        return np.array(seqs)

    orig_flat = get_flattened("Original_Train", only_delirium=True)
    tg_flat = get_flattened("TimeGAN")
    sm_flat = get_flattened("SMOTE")

    if len(orig_flat) == 0 or len(tg_flat) == 0 or len(sm_flat) == 0:
        raise ValueError("t-SNE input arrays are empty. Check Dataset_Type labels and time steps.")

    perpl_tg = max(5, min(perplexity, len(orig_flat) + len(tg_flat) - 1))
    perpl_sm = max(5, min(perplexity, len(orig_flat) + len(sm_flat) - 1))

    tsne_tg = TSNE(n_components=2, perplexity=perpl_tg, random_state=42)
    tsne_sm = TSNE(n_components=2, perplexity=perpl_sm, random_state=42)

    emb_orig_tg = tsne_tg.fit_transform(np.vstack([orig_flat, tg_flat]))
    emb_orig_sm = tsne_sm.fit_transform(np.vstack([orig_flat, sm_flat]))

    return {
        "orig_tg": emb_orig_tg,
        "orig_sm": emb_orig_sm,
        "n_orig": len(orig_flat),
        "n_tg": len(tg_flat),
        "n_sm": len(sm_flat),
    }


def save_tsne_pairwise_figure(tsne_result: dict[str, np.ndarray], out_png_path: str) -> None:
    emb_orig_tg = tsne_result["orig_tg"]
    emb_orig_sm = tsne_result["orig_sm"]
    n_orig = int(tsne_result["n_orig"])
    n_tg = int(tsne_result["n_tg"])
    n_sm = int(tsne_result["n_sm"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    y_tg = ["Original"] * n_orig + ["TimeGAN"] * n_tg
    y_sm = ["Original"] * n_orig + ["SMOTE"] * n_sm

    sns.scatterplot(
        x=emb_orig_tg[:, 0],
        y=emb_orig_tg[:, 1],
        hue=y_tg,
        palette={"Original": "#003366", "TimeGAN": "#D11141"},
        alpha=0.8,
        s=60,
        edgecolor="white",
        linewidth=0.5,
        ax=ax1,
    )
    ax1.set_title("(a) Original vs. TimeGAN", fontsize=16, fontweight="bold", pad=15)
    ax1.grid(True, linestyle="--", alpha=0.3)

    sns.scatterplot(
        x=emb_orig_sm[:, 0],
        y=emb_orig_sm[:, 1],
        hue=y_sm,
        palette={"Original": "#003366", "SMOTE": "#00A1AB"},
        alpha=0.8,
        s=60,
        edgecolor="white",
        linewidth=0.5,
        ax=ax2,
    )
    ax2.set_title("(b) Original vs. SMOTE", fontsize=16, fontweight="bold", pad=15)
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Pairwise t-SNE Comparison for Data Fidelity Analysis", fontsize=22, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def dtw_dist(s1: np.ndarray, s2: np.ndarray) -> float:
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def compute_distance_metrics(train_aug: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    time_steps = int(train_aug["time"].nunique())

    def extract_3d(df_type: str, only_delirium: bool = False) -> np.ndarray:
        subset = train_aug[train_aug["Dataset_Type"] == df_type]
        if only_delirium:
            target_pids = subset.groupby("pid")["delirium"].max()
            target_pids = target_pids[target_pids == 1].index
            subset = subset[subset["pid"].isin(target_pids)]

        seqs = []
        for _, group in subset.groupby("pid"):
            group = group.sort_values("time")
            if len(group) == time_steps:
                seqs.append(group[feature_cols].values)
        return np.array(seqs)

    orig_3d = extract_3d("Original_Train", only_delirium=True)
    tg_3d = extract_3d("TimeGAN")
    sm_3d = extract_3d("SMOTE")

    if len(orig_3d) == 0 or len(tg_3d) == 0 or len(sm_3d) == 0:
        raise ValueError("Distance metric input arrays are empty. Check Dataset_Type labels and time steps.")

    rows = []
    for name, target in [("TimeGAN", tg_3d), ("SMOTE", sm_3d)]:
        dtw_mins = [min([dtw_dist(ts, osq) for osq in orig_3d]) for ts in target]
        t_flat = target.reshape(len(target), -1)
        o_flat = orig_3d.reshape(len(orig_3d), -1)
        wass_mins = [min([wasserstein_distance(tf, of) for of in o_flat]) for tf in t_flat]

        rows.append(
            {
                "Method": name,
                "Mean Min DTW Distance": float(np.mean(dtw_mins)),
                "Mean Min Wasserstein Distance": float(np.mean(wass_mins)),
            }
        )

    return pd.DataFrame(rows)
