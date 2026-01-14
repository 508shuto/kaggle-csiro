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
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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
