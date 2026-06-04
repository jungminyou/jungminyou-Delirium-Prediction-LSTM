from __future__ import annotations

from src.augmentation import run_augmentation_pipeline
from src.config import ensure_input_data, get_default_config
from src.pipeline import run_fidelity, run_train_eval
from src.tensor_prep import build_scaled_tensors


def main() -> None:
    cfg = get_default_config()

    # 0) Make sure a longitudinal input table exists. The real de-identified
    #    cohort is used if present; otherwise a schema-matched synthetic cohort
    #    is generated so the full workflow is reproducible without it.
    ensure_input_data(cfg)

    # 1) Augmentation and patient-level split
    aug = run_augmentation_pipeline(base_dir=cfg.base_dir, seed=42, timegan_epochs=500)

    # 2) Tensor prep and leakage-safe scaling
    tensors = build_scaled_tensors(
        df_train=aug.df_train,
        df_val=aug.df_val,
        df_test=aug.df_test,
        df_tg=aug.df_tg,
        df_sm=aug.df_sm,
        feature_cols=aug.all_feature_cols,
        label_cols=aug.label_cols,
        time_steps=aug.time_steps,
    )

    # 3) Train, collect predictions, and write evaluation tables
    run_train_eval(tensors, cfg, lstm_epochs=300, batch_size=32)

    # 4) Synthetic-data fidelity analyses
    run_fidelity(cfg, perplexity=30)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
