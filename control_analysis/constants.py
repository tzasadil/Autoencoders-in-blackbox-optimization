from __future__ import annotations

import json
from pathlib import Path

TABLE_DEFAULT_COLUMN_FORMAT = "|lc|"
BASELINE_COLOR = "#E04836"
DEFAULT_COLOR = "forestgreen"

DEFAULT_BEST_DOE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "doe_sweep" / "best_doe_config.json"


def load_best_doe_config(path: Path | str = DEFAULT_BEST_DOE_CONFIG_PATH) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {"model": "doe_2_8", "n_samples": 2, "latent_dim": 8}
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_doe_models(path: Path | str = DEFAULT_BEST_DOE_CONFIG_PATH) -> tuple[str, str]:
    config = load_best_doe_config(path)
    n_samples = int(config.get("n_samples", 2))
    latent_dim = int(config.get("latent_dim", 8))
    primary = str(config.get("model") or f"doe_{n_samples}_{latent_dim}")
    plain = f"doe_plain_{n_samples}_{latent_dim}"
    return primary, plain


PRIMARY_DOE_MODEL, PLAIN_DOE_MODEL = resolve_doe_models()

MODEL_DISPLAY_LABELS = {
    PRIMARY_DOE_MODEL: "DOE",
    PLAIN_DOE_MODEL: "DOE plain",
    "elm100": "ELM",
    "fitloss": "Fitloss",
    "none": "No surrogate",
    "oracle": "Oracle",
    "gp": "GP",
    "nn3": "NN",
}


def display_model_label(model_name: object) -> str:
    text = str(model_name)
    return MODEL_DISPLAY_LABELS.get(text, text.replace("_", " "))

FUNC_GROUP_LABELS = [
    ("f1-f5", 1, 5, "Separable Functions"),
    ("f6-f9", 6, 9, "Functions with low or moderate conditioning"),
    ("f10-f14", 10, 14, "Ill conditioned functions"),
    ("f15-f19", 15, 19, "Adequately structured multimodal functions"),
    ("f20-f24", 20, 24, "Weakly structured multimodal functions"),
]

EVAL_WINDOW_FUNC_GROUPS = [
    (1, 5, "Separable Functions"),
    (6, 9, "Functions with low or moderate conditioning"),
    (10, 14, "Ill conditioned functions"),
    (15, 19, "Adequately structured multimodal functions"),
    (20, 24, "Weakly structured multimodal functions"),
    (1, 24, "All functions"),
]