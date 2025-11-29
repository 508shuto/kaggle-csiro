import os
import random
from typing import Any

import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from loss import (
    AuxLoss,
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
    elif loss_name == "aux_loss":
        return AuxLoss()
    else:
        raise ValueError(f"Loss function {loss_name} not found")


def get_metrics(params: DictConfig, metrics_name: str) -> nn.Module:
    if metrics_name == "r2_score":
        return WeightedR2Score()
    else:
        raise ValueError(f"Metrics {metrics_name} not found")


class AddNDICIVEChannels(ImageOnlyTransform):
    """RGB画像からNDIとCIVEチャンネルを計算して5chに拡張する変換.

    Args:
        always_apply: 常に適用するかどうか
        p: 適用確率
    """

    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        """RGB画像からNDIとCIVEチャンネルを計算して5chに拡張.

        Args:
            image: (H, W, 3) RGB画像、値域は[0, 255]（整数型）または[0, 1]（浮動小数点型）
            **params: その他のパラメータ

        Returns:
            (H, W, 5) array [R, G, B, NDI, CIVE]（値域は入力と同じ）
        """
        # 入力が整数型（0-255）か浮動小数点型（0-1）かを判定
        if image.dtype == np.uint8 or np.max(image) > 1.0:
            # 0-255空間で計算
            r, g, b = (
                image[:, :, 0].astype(np.float32),
                image[:, :, 1].astype(np.float32),
                image[:, :, 2].astype(np.float32),
            )
        else:
            # 0-1空間で計算（255倍して0-255空間に変換）
            r, g, b = (
                image[:, :, 0] * 255.0,
                image[:, :, 1] * 255.0,
                image[:, :, 2] * 255.0,
            )

        # NDI = (G - R) / (G + R + eps)
        def calculate_ndi(r: np.ndarray, g: np.ndarray) -> np.ndarray:
            eps = 1e-6
            return (g - r) / (g + r + eps)

        def calculate_cive(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
            """CIVE = 0.441R - 0.881G + 0.385B + 18.78745

            Args:
                r: (H, W) array（0-255空間）
                g: (H, W) array（0-255空間）
                b: (H, W) array（0-255空間）

            Returns:
                (H, W) array（0-255空間）
            """
            return 0.441 * r - 0.881 * g + 0.385 * b + 18.78745

        ndi = calculate_ndi(r, g)
        cive = calculate_cive(r, g, b)

        # RGBも0-255空間に統一
        rgb_5ch = np.stack([r, g, b, ndi, cive], axis=2).astype(np.float32)

        return rgb_5ch


def get_transforms(mode: str, config: DictConfig) -> A.Compose:
    assert mode in ["train", "valid", "test"]

    # 5ch対応かどうかを判定（in_channels=5なら5chモード）
    use_5ch = config.model.in_channels == 5

    if mode == "train":
        augmentations = [
            A.Resize(
                height=config.augmentation.train.image_size,
                width=config.augmentation.train.image_size,
                p=1.0,
            ),
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
        # NDI/CIVEチャンネル追加（5chモードの場合）
        if use_5ch:
            augmentations.append(AddNDICIVEChannels(always_apply=True, p=1.0))
        # 正規化: 5chモードでは[RGB用ImageNet統計, NDI/CIVE用mean=0,std=1]
        if use_5ch:
            augmentations.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.0, 0.0],
                    std=[0.229, 0.224, 0.225, 1.0, 1.0],
                    max_pixel_value=1.0,
                    p=1.0,
                )
            )
        else:
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
        augmentations_valid = [
            A.Resize(
                height=config.augmentation.valid.image_size,
                width=config.augmentation.valid.image_size,
                p=1.0,
            ),
        ]
        # NDI/CIVEチャンネル追加（5chモードの場合、ToFloatの前で0-255空間で計算）
        if use_5ch:
            augmentations_valid.append(AddNDICIVEChannels(always_apply=True, p=1.0))
        # 正規化
        if use_5ch:
            augmentations_valid.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.0, 0.0],
                    std=[0.229, 0.224, 0.225, 1.0, 1.0],
                    max_pixel_value=255.0,
                    p=1.0,
                )
            )
        else:
            augmentations_valid.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                )
            )
        augmentations_valid.append(ToTensorV2(p=1.0))
        return A.Compose(
            augmentations_valid,
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
