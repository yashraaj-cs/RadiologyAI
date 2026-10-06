"""Central configuration for RadiologyAI Copilot.

Defines thresholds, filesystem paths, model weight identifiers,
and loads environment variables from .env using python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directories
BASE_DIR: Path = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Path to the data directory (sample images, eval sets)
DATA_DIR: Path = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

# TorchXRayVision pretrained model weights
# Note: Input resolution is derived dynamically from the chosen weights' metadata
# in torchxrayvision.models.model_urls to prevent configuration drift.
MODEL_WEIGHTS: str = os.getenv("MODEL_WEIGHTS", "densenet121-res224-all")

# Pathology classification thresholds per model weights configuration.
# NOTE: These values are UNTUNED initial placeholders and must be tuned
# on a dedicated validation/dev split to balance clinical sensitivity and specificity.
# ResNet scores run much lower than DenseNet scores, so lower thresholds are used.
THRESHOLDS_BY_WEIGHTS: dict[str, dict[str, float]] = {
    # UNTUNED placeholder thresholds for DenseNet-121
    "densenet121-res224-all": {
        "present": 0.6,
        "absent": 0.4,
    },
    # UNTUNED placeholder thresholds for ResNet-50 (scores run much lower)
    "resnet50-res512-all": {
        "present": 0.5,
        "absent": 0.25,
    },
}


def get_thresholds(weights: str) -> tuple[float, float]:
    """Return (present_threshold, absent_threshold) for the specified model weights.

    Args:
        weights: Model weights identifier.

    Returns:
        Tuple of (present_threshold, absent_threshold).

    Raises:
        ValueError: If weights is not configured in THRESHOLDS_BY_WEIGHTS.
    """
    if weights in THRESHOLDS_BY_WEIGHTS:
        cfg = THRESHOLDS_BY_WEIGHTS[weights]
        return cfg["present"], cfg["absent"]

    valid_names = sorted(THRESHOLDS_BY_WEIGHTS.keys())
    raise ValueError(
        f"Unknown weights '{weights}'. Valid options are: {valid_names}"
    )

# LLM settings
LLM_API_KEY: str | None = os.getenv("LLM_API_KEY")
LLM_MODEL: str = os.getenv("LLM_MODEL", "default")
