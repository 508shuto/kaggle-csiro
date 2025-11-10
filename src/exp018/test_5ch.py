"""Quick test script to verify 5-channel input shape for EXP017."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from dataset import CSIRODataset
from models import CSIROModel
from utils import get_transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load config
config_path = Path("./config/exp017.yaml")
config: DictConfig = OmegaConf.load(str(config_path))  # type: ignore

# Load preprocessed data
output_dir = Path("./output/exp017")
if not (output_dir / "preprocessed_train.csv").exists():
    print(f"ERROR: {output_dir / 'preprocessed_train.csv'} not found")
    print("Please run create_dataset.py first")
    sys.exit(1)

df = pd.read_csv(output_dir / "preprocessed_train.csv")

print(f"Loaded {len(df)} samples")
print(f"Config model.in_channels: {config.model.in_channels}")

# Test transforms
transform = get_transforms("train", config)
print(f"\nTransform created successfully")

# Test dataset
train_df = df[df["fold"] != 0].head(2).reset_index(drop=True)
dataset = CSIRODataset(config, train_df, "train")

print(f"\nDataset created with {len(dataset)} samples")

# Test data loading
sample_image, sample_target, sample_aux = dataset[0]

print(f"\nImage shape: {sample_image.shape}")
print(f"Expected: torch.Size([5, 256, 256])")
print(f"Target shape: {sample_target.shape}")
print(f"Aux target shape: {sample_aux.shape}")

# Check for NaN
if torch.isnan(sample_image).any():
    print("\nERROR: NaN values found in image!")
    sys.exit(1)
else:
    print("\nOK: No NaN values in image")

# Check value ranges
print(f"\nImage value range: [{sample_image.min():.4f}, {sample_image.max():.4f}]")
print(f"Expected: RGB normalized, NDI/ExG centered around 0")

# Test model
model = CSIROModel(
    model_name=config.model.name,
    pretrained=False,  # Faster for testing
    in_channels=config.model.in_channels,
)

print(f"\nModel created with in_channels={config.model.in_channels}")

# Test forward pass
model.eval()
with torch.no_grad():
    dummy_input = sample_image.unsqueeze(0)  # Add batch dimension
    print(f"\nInput shape: {dummy_input.shape}")
    pred_log, aux_pred_log = model(dummy_input)

print(f"Prediction shape: {pred_log.shape}")
print(f"Aux prediction shape: {aux_pred_log.shape}")
print(f"Expected: torch.Size([1, 5]) and torch.Size([1, 2])")

# Check for NaN in predictions
if torch.isnan(pred_log).any() or torch.isnan(aux_pred_log).any():
    print("\nERROR: NaN values found in predictions!")
    sys.exit(1)
else:
    print("\nOK: No NaN values in predictions")

print("\nAll tests passed!")
