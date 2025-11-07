"""
exp100: Qwen3-VL Fine-tuning Pipeline

This experiment implements a Vision-Language model (Qwen3-VL) fine-tuning pipeline
for grassland biomass prediction using StratifiedGroupKFold cross-validation.
"""

from .models import Qwen3VLForRegression
from .dataset import Qwen3VLDataset, DataCollatorForQwen3VL
from .utils import (
    get_device,
    get_lora_config,
    set_seed,
    compute_weighted_r2,
    save_predictions,
    print_metrics,
)

__all__ = [
    "Qwen3VLForRegression",
    "Qwen3VLDataset",
    "DataCollatorForQwen3VL",
    "get_device",
    "get_lora_config",
    "set_seed",
    "compute_weighted_r2",
    "save_predictions",
    "print_metrics",
]
