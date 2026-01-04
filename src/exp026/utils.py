import os
import random

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from loss import (
    AuxLoss,
    WeightedMSELoss,
    WeightedSmoothL1Loss,
)
from metrics import (
    WeightedR2Score,
)
from omegaconf import DictConfig


def seed_everything(seed: int):
    L.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_loss_fn(params: DictConfig, loss_name: str) -> nn.Module:
    if loss_name == "l2_loss":
        return WeightedMSELoss()
    elif loss_name == "smooth_l1_loss":
        return WeightedSmoothL1Loss()
    elif loss_name == "aux_loss":
        return AuxLoss()
    else:
        raise ValueError(f"Loss function {loss_name} not found")


def get_metrics(params: DictConfig, metrics_name: str) -> nn.Module:
    if metrics_name == "r2_score":
        return WeightedR2Score()
    else:
        raise ValueError(f"Metrics {metrics_name} not found")


def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    aux_targets: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup data augmentation for regression tasks.

    Args:
        images: (B, C, H, W)
        targets: (B, 5) raw space targets
        aux_targets: (B, 2) raw space auxiliary targets
        alpha: Beta distribution parameter

    Returns:
        mixed_images: (B, C, H, W)
        mixed_targets: (B, 5) mixed targets
        mixed_aux_targets: (B, 2) mixed auxiliary targets
        lam: mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    # Mix images
    mixed_images = lam * images + (1 - lam) * images[index]

    # Mix targets (linear interpolation for regression)
    mixed_targets = lam * targets + (1 - lam) * targets[index]
    mixed_aux_targets = lam * aux_targets + (1 - lam) * aux_targets[index]

    return mixed_images, mixed_targets, mixed_aux_targets, lam
