from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tyro
from lightning_module import CSIROModule
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm
from transformers import AutoImageProcessor


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for preprocessed tensors."""
    pixel_values = torch.stack([item[0] for item in batch])
    grid_thw = torch.stack([item[1] for item in batch])
    return pixel_values, grid_thw


def detect_device(device: str) -> str:
    """Detect available device automatically.

    Args:
        device: Device string ("auto", "cuda", "mps", "cpu")

    Returns:
        Detected device string
    """
    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def build_test_index(test_csv_path: Path, input_dir: Path) -> pd.DataFrame:
    """Build test dataset index from test CSV.

    Args:
        test_csv_path: Path to test.csv file
        input_dir: Input directory containing JPEG files

    Returns:
        DataFrame with columns: sample_id, image_path
    """
    df = pd.read_csv(test_csv_path)

    # Group by unique image_path
    image_paths = df["image_path"].unique()
    data_items = []

    for image_path in image_paths:
        # Extract sample_id root (e.g., "ID1001187975" from "ID1001187975__Dry_Clover_g")
        sample_id_row = df[df["image_path"] == image_path].iloc[0]
        sample_id = sample_id_row["sample_id"].split("__")[0]

        # Generate full JPEG path (input/test/xxx.jpg)
        jpeg_path = input_dir / image_path

        data_items.append(
            {
                "sample_id": sample_id,
                "image_path": jpeg_path,
            }
        )

    return pd.DataFrame(data_items)


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str, config: DictConfig):
        self.df = df
        self.mode = mode
        self.config = config
        self.transform = self._get_transforms()
        self.processor = AutoImageProcessor.from_pretrained(config.model.name)

    def _get_transforms(self) -> T.Compose:
        """Get transforms for test data.

        Note: Only PIL-level augmentations. Processor handles resize/normalize.
        """
        image_size = self.config.augmentation.valid.image_size
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        # Load image with PIL
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            raise RuntimeError(f"Failed to load image {row['image_path']}: {e}") from e
        image = self.transform(image)

        # Process with Qwen3VL processor
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed.pixel_values.squeeze(0)
        grid_thw = processed.image_grid_thw.squeeze(0)

        return pixel_values, grid_thw


def predict_fold(
    fold: int,
    config: DictConfig,
    df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    model_dir: Path,
    device: str,
    use_amp: bool = False,
    use_tta: bool = False,
) -> np.ndarray:
    dataset = TestDataset(df, "test", config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),  # Can now pin tensors
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

    checkpoints = list(model_dir.glob(f"fold{fold}*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found for fold {fold} in {model_dir}")
    model_path = checkpoints[0]

    # Use torch.device for map_location to avoid MPS issues
    map_location = torch.device(device)
    model = (
        CSIROModule.load_from_checkpoint(
            checkpoint_path=model_path,
            config=config,
            map_location=map_location,
        )
        .model_ema.module.eval()
        .to(device)
    )

    fold_predictions = predict(
        fold=fold,
        model=model,
        dataloader=dataloader,
        device=device,
        use_amp=use_amp,
        use_tta=use_tta,
    )
    return fold_predictions


def predict(
    fold: int,
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    use_amp: bool = False,
    use_tta: bool = False,
) -> np.ndarray:
    """Predict with optional AMP support.

    Args:
        fold: Fold number for logging
        model: Model to use for prediction
        dataloader: DataLoader for test data (returns preprocessed tensors)
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        use_tta: Whether to use test time augmentation

    Returns:
        Array of predictions shape (N, 5)
    """
    if use_tta:
        print("Warning: TTA is not yet implemented. Ignoring use_tta=True.")

    fold_predictions = []
    device_type = "cuda" if device.startswith("cuda") else ("mps" if device.startswith("mps") else "cpu")

    for pixel_values, grid_thw in tqdm(dataloader, desc=f"Fold {fold}"):
        pixel_values = pixel_values.to(device)
        grid_thw = grid_thw.to(device)

        # Main prediction
        with torch.no_grad():
            if use_amp and device_type == "cuda":
                with torch.amp.autocast(device_type=device_type):
                    pred, _ = model(pixel_values, grid_thw)
            else:
                pred, _ = model(pixel_values, grid_thw)

        # Clamp to non-negative
        pred = torch.clamp(pred, min=0.0)
        fold_predictions.append(pred.cpu().numpy())

    fold_predictions = np.concatenate(fold_predictions, axis=0)
    return fold_predictions


def main(
    test_csv_path: Path = Path("./input/test.csv"),
    config_path: Path = Path("./config/exp027.yaml"),
    model_dir: Path = Path("./output/exp027"),
    output_dir: Path = Path("./output/exp027"),
    folds: list[int] | None = None,
    device: str = "auto",
    batch_size: int = 16,
    num_workers: int = 4,
    use_amp: bool = False,
    use_tta: bool = False,
):
    """Run inference and generate submission file.

    Args:
        test_csv_path: Path to test.csv file
        config_path: Path to config YAML file
        model_dir: Directory containing model checkpoints
        output_dir: Output directory for submission file
        folds: List of fold numbers to use for ensemble
        device: Device to run inference on (cuda/cpu/mps)
        batch_size: Batch size for inference
        num_workers: Number of DataLoader workers
        use_amp: Whether to use automatic mixed precision
        use_tta: Whether to use test time augmentation
    """
    if folds is None:
        folds = [0, 1, 2, 3, 4]
    config: DictConfig = OmegaConf.load(str(config_path))  # type: ignore
    config.model.pretrained = False

    # Detect device if auto
    device = detect_device(device)
    print(f"Using device: {device}")

    # Build test index (sample_id, image_path)
    input_dir = Path(config.dataset.input_dir)
    df = build_test_index(test_csv_path, input_dir)

    # Collect predictions from all folds
    fold_predictions: list[np.ndarray] = []
    for fold in folds:
        preds = predict_fold(
            fold=fold,
            config=config,
            df=df,
            batch_size=batch_size,
            num_workers=num_workers,
            model_dir=model_dir,
            device=device,
            use_amp=use_amp,
            use_tta=use_tta,
        )
        fold_predictions.append(preds)

    # Average predictions across folds (shape: (N, 5))
    ensemble_predictions = np.mean(fold_predictions, axis=0)

    # Convert to submission format
    target_names = [
        "Dry_Clover_g",
        "Dry_Dead_g",
        "Dry_Green_g",
        "GDM_g",
        "Dry_Total_g",
    ]

    num_samples = len(df)
    num_targets = len(target_names)

    # Create sample_id column: repeat each sample_id num_targets times
    sample_ids = np.repeat(df["sample_id"].to_numpy(), num_targets)

    # Create target_name column: tile target names num_samples times
    target_names_array = np.tile(target_names, num_samples)

    # Create target column: reshape predictions to (N * 5,)
    targets = ensemble_predictions.flatten()

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {
            "sample_id": [f"{sid}__{tname}" for sid, tname in zip(sample_ids, target_names_array, strict=True)],
            "target": targets,
        }
    )

    submission_df.to_csv(output_dir / "submission.csv", index=False)
    print(f"Submission file saved to {output_dir / 'submission.csv'}")


if __name__ == "__main__":
    tyro.cli(main)
