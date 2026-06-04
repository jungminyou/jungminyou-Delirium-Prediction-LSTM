from src.config import get_default_config
from src.pipeline import run_fidelity


if __name__ == "__main__":
    cfg = get_default_config()
    run_fidelity(cfg, perplexity=30)
    print("Stage 04 completed: fidelity tables and t-SNE figure saved.")
