import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


class CSIRODataset(Dataset):
    """Dataset for Qwen3-VL based model.

    Returns PIL images that will be processed by Qwen3-VL processor in collate_fn.
    """

    def __init__(self, config: DictConfig, df: pd.DataFrame, mode: str):
        assert mode in ["train", "valid", "test"], mode
        self.config = config
        self.df = df
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load image as PIL (Qwen3-VL processor expects PIL images)
        image = Image.open(row["image_path"]).convert("RGB")

        # Resize image for consistent processing
        image_size = self.config.augmentation.train.image_size
        image = image.resize((image_size, image_size), Image.BILINEAR)

        # Targets
        target = torch.tensor(
            [
                row["clover_target"].item(),
                row["dead_target"].item(),
                row["green_target"].item(),
                row["gdm_target"].item(),
                row["total_target"].item(),
            ],
            dtype=torch.float32,
        )

        # Auxiliary targets
        aux_target = torch.tensor(
            [
                row["pre_gshh_ndvi"].item(),
                row["height_ave_cm"].item(),
            ],
            dtype=torch.float32,
        )

        return image, target, aux_target


class Qwen3VLCollator:
    """Collate function for Qwen3-VL model.

    Processes PIL images using Qwen3-VL processor.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, batch: list) -> dict:
        images, targets, aux_targets = zip(*batch, strict=True)

        # Stack targets
        targets = torch.stack(targets, dim=0)
        aux_targets = torch.stack(aux_targets, dim=0)

        # Process images with Qwen3-VL processor
        # Create a simple message format for image processing
        messages_batch = []
        for img in images:
            messages_batch.append(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": img}],
                    }
                ]
            )

        # Process all images
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]

        inputs = self.processor(
            text=texts,
            images=list(images),
            return_tensors="pt",
            padding=True,
        )

        return {
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "targets": targets,
            "aux_targets": aux_targets,
        }
