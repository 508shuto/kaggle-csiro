import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from loss import (
    WeightedMSELoss,
)
from metrics import (
    WeightedR2Score,
)
from omegaconf import DictConfig


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
            augmentations.append(
                A.HorizontalFlip(p=config.augmentation.train.horizontal_flip)
            )
        if config.augmentation.train.vertical_flip > 0:
            augmentations.append(
                A.VerticalFlip(p=config.augmentation.train.vertical_flip)
            )
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
        return A.Compose(augmentations, p=1.0)
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
        )
