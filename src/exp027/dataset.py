import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import AutoImageProcessor


class CSIRODataset(Dataset):
    """Dataset for Qwen3-VL based regression model.

    This dataset loads images and applies VLM-compatible preprocessing.
    """

    def __init__(self, config: DictConfig, df: pd.DataFrame, mode: str):
        assert mode in ["train", "valid", "test"], mode
        self.config = config
        self.df = df
        self.mode = mode
        self.transform = self._get_transforms(mode)
        self.processor = AutoImageProcessor.from_pretrained(config.model.name)

    def _get_transforms(self, mode: str) -> T.Compose:
        """Get torchvision transforms for VLM input.

        Note: Returns PIL images without ToTensor/Normalize because
        Qwen3VLImageProcessor handles the full preprocessing.
        """
        image_size = (
            self.config.augmentation.train.image_size if mode == "train" else self.config.augmentation.valid.image_size
        )

        if mode == "train":
            transform_list = [
                T.Resize((image_size, image_size)),
            ]
            if self.config.augmentation.train.horizontal_flip > 0:
                transform_list.append(T.RandomHorizontalFlip(p=self.config.augmentation.train.horizontal_flip))
            if self.config.augmentation.train.vertical_flip > 0:
                transform_list.append(T.RandomVerticalFlip(p=self.config.augmentation.train.vertical_flip))
            if self.config.augmentation.train.rotation_limit > 0:
                transform_list.append(T.RandomRotation(degrees=self.config.augmentation.train.rotation_limit))
            if self.config.augmentation.train.brightness_contrast:
                transform_list.append(T.ColorJitter(brightness=0.2, contrast=0.2))
            # Note: No ToTensor/Normalize - Qwen3VLImageProcessor handles this
        else:
            transform_list = [
                T.Resize((image_size, image_size)),
            ]

        return T.Compose(transform_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        # Load image with PIL
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            raise RuntimeError(f"Failed to load image {row['image_path']}: {e}") from e

        # Apply transforms (returns PIL image, not tensor)
        image = self.transform(image)

        # Process with Qwen3VL processor (resize + normalize + convert to tensor)
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed.pixel_values.squeeze(0)  # Remove batch dim
        grid_thw = processed.image_grid_thw.squeeze(0)  # Remove batch dim

        # Get raw target values
        target = torch.tensor(
            [
                float(row["clover_target"]),
                float(row["dead_target"]),
                float(row["green_target"]),
                float(row["gdm_target"]),
                float(row["total_target"]),
            ],
            dtype=torch.float32,
        )

        # Auxiliary targets
        aux_target = torch.tensor(
            [
                float(row["pre_gshh_ndvi"]),
                float(row["height_ave_cm"]),
            ],
            dtype=torch.float32,
        )

        return pixel_values, grid_thw, target, aux_target
