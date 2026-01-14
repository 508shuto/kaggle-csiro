import os
import random

import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from loss import (
    AuxLoss,
    InverseWeightedSmoothL1Loss,
    PinballLoss,
    QuantileWeightedSmoothL1Loss,
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
    elif loss_name == "quantile_weighted_smooth_l1":
        return QuantileWeightedSmoothL1Loss()
    elif loss_name == "inverse_weighted_smooth_l1":
        return InverseWeightedSmoothL1Loss()
    elif loss_name == "pinball":
        quantile = params.get("quantile", 0.5) if params else 0.5
        return PinballLoss(quantile=quantile)
    elif loss_name == "aux_loss":
        return AuxLoss()
    else:
        raise ValueError(f"Loss function {loss_name} not found")


def get_metrics(params: DictConfig, metrics_name: str) -> nn.Module:
    if metrics_name == "r2_score":
        return WeightedR2Score()
    else:
        raise ValueError(f"Metrics {metrics_name} not found")


def get_transforms(mode: str, config: DictConfig) -> A.Compose:
    assert mode in ["train", "valid", "test"]

    if mode == "train":
        augmentations = [
            A.Resize(
                height=config.augmentation.train.image_size,
                width=config.augmentation.train.image_size,
                p=1.0,
            )
        ]
        if config.augmentation.train.horizontal_flip > 0:
            augmentations.append(A.HorizontalFlip(p=config.augmentation.train.horizontal_flip))
        if config.augmentation.train.vertical_flip > 0:
            augmentations.append(A.VerticalFlip(p=config.augmentation.train.vertical_flip))
        if config.augmentation.train.rotation_limit > 0:
            augmentations.append(
                A.Rotate(
                    limit=config.augmentation.train.rotation_limit,
                    p=0.2,
                )
            )
        if config.augmentation.train.elastic_transform > 0:
            augmentations.append(
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    p=0.2,
                )
            )
        if config.augmentation.train.brightness_contrast:
            augmentations.append(
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.2,
                )
            )
        augmentations.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            )
        )
        augmentations.append(ToTensorV2(p=1.0))
        print(augmentations)
        return A.Compose(augmentations, p=1.0, seed=config.experiment.seed)
    else:
        return A.Compose(
            [
                A.Resize(
                    height=config.augmentation.valid.image_size,
                    width=config.augmentation.valid.image_size,
                    p=1.0,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
            seed=config.experiment.seed,
        )


def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    aux_targets: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup data augmentation for regression tasks.

    Args:
        images: (B, C, H, W)
        targets: (B, 5) ターゲット（raw空間）
        aux_targets: (B, 2) 補助ターゲット（raw空間）
        alpha: Beta分布のパラメータ

    Returns:
        mixed_images: (B, C, H, W)
        mixed_targets: (B, 5) 混合ターゲット（raw空間）
        mixed_aux_targets: (B, 2) 混合補助ターゲット（raw空間）
        lam: mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    # 画像をMix
    mixed_images = lam * images + (1 - lam) * images[index]

    # ターゲットをMix（回帰なので線形補間が適切）
    mixed_targets = lam * targets + (1 - lam) * targets[index]
    mixed_aux_targets = lam * aux_targets + (1 - lam) * aux_targets[index]

    return mixed_images, mixed_targets, mixed_aux_targets, lam
