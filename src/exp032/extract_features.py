"""Feature extraction script for Vision Encoder caching.

Extracts pooled features from Qwen3-VL vision encoder and saves them as .npy files.
This enables fast head-only training by skipping the expensive vision encoder forward pass.

Usage:
    uv run ./src/exp027/extract_features.py --output-dir ./output/exp027/features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro
from dataset import CSIRODataset
from models import Qwen3VLRegressionModel
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Collate function for preprocessed tensors."""
    pixel_values = torch.stack([item[0] for item in batch])
    grid_thw = torch.stack([item[1] for item in batch])
    targets = torch.stack([item[2] for item in batch])
    aux_targets = torch.stack([item[3] for item in batch])
    return pixel_values, grid_thw, targets, aux_targets


def extract_features(
    config_path: Path | None = None,
    output_dir: Path | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
    device: str = "cuda",
) -> None:
    """Extract vision encoder features and save as .npy files.

    Args:
        config_path: Path to config file. Defaults to ./config/exp027.yaml
        output_dir: Directory to save features. Defaults to ./output/exp027/features
        batch_size: Batch size for feature extraction
        num_workers: Number of dataloader workers
        device: Device to use (cuda, mps, cpu)
    """
    exp_name = Path(__file__).parent.name
    if config_path is None:
        config_path = Path("./config") / f"{exp_name}.yaml"
    config = OmegaConf.load(config_path)

    if output_dir is None:
        output_dir = Path(config.dataset.output_dir) / exp_name / "features"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {config.model.name}")
    model = Qwen3VLRegressionModel(
        model_name=config.model.name,
        pretrained=config.model.pretrained,
        freeze_backbone=True,
        hidden_dim=config.model.hidden_dim,
        head_hidden_dim=config.model.head_hidden_dim,
        dropout=config.model.dropout,
    )
    model = model.to(device)
    model.eval()

    # Load train data
    data_dir = Path(config.dataset.output_dir) / exp_name
    train_df = pd.read_csv(data_dir / "preprocessed_train.csv")
    print(f"Processing {len(train_df)} training samples")

    # Create dataset and dataloader
    dataset = CSIRODataset(config, train_df, "valid")  # Use valid mode (no augmentation)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

    # Extract features
    all_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            pixel_values, grid_thw, _, _ = batch
            pixel_values = pixel_values.to(device)
            grid_thw = grid_thw.to(device)

            # Extract vision features (same as model forward, but stop at pooling)
            image_embeds, _ = model.vlm.model.get_image_features(
                pixel_values.type(model.vlm.model.visual.dtype),
                grid_thw,
            )

            # Pool over patches for each image
            pooled = torch.stack([emb.mean(dim=0) for emb in image_embeds], dim=0)
            pooled = pooled.float().cpu().numpy()

            all_features.append(pooled)

    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    print(f"Extracted features shape: {all_features.shape}")

    # Save features per image
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Saving features"):
        image_id = row["image_id"]
        feature = all_features[idx]
        np.save(output_dir / f"{image_id}.npy", feature)

    print(f"Features saved to {output_dir}")
    print(f"Total size: {all_features.nbytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    tyro.cli(extract_features)
