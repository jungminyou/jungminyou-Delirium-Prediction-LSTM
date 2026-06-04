from src.augmentation import run_augmentation_pipeline
from src.config import ensure_input_data, get_default_config


if __name__ == "__main__":
    cfg = get_default_config()
    ensure_input_data(cfg)
    run_augmentation_pipeline(base_dir=cfg.base_dir, seed=42, timegan_epochs=500)
    print("Stage 01 completed: augmentation + split outputs saved.")
