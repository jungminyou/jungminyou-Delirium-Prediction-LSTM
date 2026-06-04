from src.augmentation import run_augmentation_pipeline
from src.config import ensure_input_data, get_default_config
from src.pipeline import run_train_eval
from src.tensor_prep import build_scaled_tensors


if __name__ == "__main__":
    cfg = get_default_config()
    ensure_input_data(cfg)
    aug = run_augmentation_pipeline(base_dir=cfg.base_dir, seed=42, timegan_epochs=500)

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

    run_train_eval(tensors, cfg, lstm_epochs=300, batch_size=32)
    print("Stage 03 completed: train + evaluation metrics saved.")
