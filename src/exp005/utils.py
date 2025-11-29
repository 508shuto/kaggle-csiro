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
    WeightedHuberLoss,
    WeightedMSELoss,
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
    elif loss_name == "huber_loss":
        return WeightedHuberLoss()
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
    targets_log: torch.Tensor,
    aux_targets_log: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup data augmentation for regression tasks.

    Args:
        images: (B, C, H, W)
        targets_log: (B, 5) log空間のターゲット
        aux_targets_log: (B, 2) log空間の補助ターゲット
        alpha: Beta分布のパラメータ

    Returns:
        mixed_images: (B, C, H, W)
        mixed_targets_log: (B, 5) log空間の混合ターゲット
        mixed_aux_targets_log: (B, 2) log空間の混合補助ターゲット
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

    # ターゲットを元の空間に戻してMix（回帰なので線形補間が適切）
    targets = torch.expm1(targets_log)  # 元の空間
    aux_targets = torch.expm1(aux_targets_log)  # 元の空間

    mixed_targets = lam * targets + (1 - lam) * targets[index]
    mixed_aux_targets = lam * aux_targets + (1 - lam) * aux_targets[index]

    # log空間に戻す
    mixed_targets_log = torch.log1p(mixed_targets)
    mixed_aux_targets_log = torch.log1p(mixed_aux_targets)

    return mixed_images, mixed_targets_log, mixed_aux_targets_log, lam
