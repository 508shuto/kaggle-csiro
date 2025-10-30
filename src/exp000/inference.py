import tyro
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
from lightning_module import CSIROModule
from utils import get_transforms


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


def build_test_index(test_csv_path: Path, output_dir: Path) -> pd.DataFrame:
    """Build test dataset index from test CSV.

    Args:
        test_csv_path: Path to test.csv file
        output_dir: Output directory containing converted numpy files

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

        # Convert image_path to numpy file path
        image_stem = Path(image_path).stem
        npy_path = output_dir / "test" / (image_stem + ".npy")

        data_items.append(
            {
                "sample_id": sample_id,
                "image_path": npy_path,
            }
        )

    return pd.DataFrame(data_items)


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str, config: DictConfig):
        self.df = df
        self.transform = get_transforms(mode, config)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = np.load(row["image_path"])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).float()
        return image

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
        pin_memory=device.startswith("cuda"),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    model_path = list(model_dir.glob(f"fold{fold}*.ckpt"))[0]
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
    """Predict with optional AMP and TTA support.

    Args:
        fold: Fold number for logging
        model: Model to use for prediction
        dataloader: DataLoader for test data
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        use_tta: Whether to use test time augmentation (not yet implemented)

    Returns:
        Array of predictions shape (N, 5)
    """
    fold_predictions = []
    device_type = "cuda" if device.startswith("cuda") else ("mps" if device.startswith("mps") else "cpu")

    for batch in tqdm(dataloader, desc=f"Fold {fold}"):
        images = batch.to(device)
        batch_predictions = []

        if use_amp and device_type == "cuda":
            with torch.no_grad():
                with torch.amp.autocast(device_type=device_type):
                    pred = model(images)
            batch_predictions.append(pred.cpu().detach().numpy())
        else:
            with torch.no_grad():
                pred = model(images)
            batch_predictions.append(pred.cpu().detach().numpy())

        # TTA (Test Time Augmentation)
        if use_tta and use_amp and device_type == "cuda":
            with torch.no_grad():
                with torch.amp.autocast(device_type=device_type):
                    # Horizontal flip
                    hflip_pred = model(torch.flip(images, dims=[3]))
                    batch_predictions.append(hflip_pred.cpu().detach().numpy())
                    # Vertical flip
                    vflip_pred = model(torch.flip(images, dims=[2]))
                    batch_predictions.append(vflip_pred.cpu().detach().numpy())
                    # Rot90
                    rot90_pred = model(torch.rot90(images, k=1, dims=[2, 3]))
                    batch_predictions.append(rot90_pred.cpu().detach().numpy())
                    # Rot270
                    rot270_pred = model(torch.rot90(images, k=3, dims=[2, 3]))
                    batch_predictions.append(rot270_pred.cpu().detach().numpy())
        elif use_tta:
            with torch.no_grad():
                # Horizontal flip
                hflip_pred = model(torch.flip(images, dims=[3]))
                batch_predictions.append(hflip_pred.cpu().detach().numpy())
                # Vertical flip
                vflip_pred = model(torch.flip(images, dims=[2]))
                batch_predictions.append(vflip_pred.cpu().detach().numpy())
                # Rot90
                rot90_pred = model(torch.rot90(images, k=1, dims=[2, 3]))
                batch_predictions.append(rot90_pred.cpu().detach().numpy())
                # Rot270
                rot270_pred = model(torch.rot90(images, k=3, dims=[2, 3]))
                batch_predictions.append(rot270_pred.cpu().detach().numpy())

        # Average predictions if TTA is used
        batch_pred = np.mean(batch_predictions, axis=0)
        fold_predictions.append(batch_pred)

    fold_predictions = np.concatenate(fold_predictions, axis=0)
    return fold_predictions

def main(
    test_csv_path: Path = Path("./input/test.csv"),
    config_path: Path = Path("./config/exp000.yaml"),
    model_dir: Path = Path("./output/exp000"),
    output_dir: Path = Path("./output/exp000"),
    folds: list[int] = [0, 1, 2, 3, 4],
    device: str = "auto",
    batch_size: int = 32,
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
    config: DictConfig = OmegaConf.load(str(config_path))  # type: ignore
    config.model.pretrained = False

    # Detect device if auto
    device = detect_device(device)
    print(f"Using device: {device}")

    # Build test index (sample_id, image_path)
    df = build_test_index(test_csv_path, output_dir)

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

    # Convert to submission format using vectorized operations
    # Target order: ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g", "Dry_Total_g"]
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
            "sample_id": [f"{sid}__{tname}" for sid, tname in zip(sample_ids, target_names_array)],
            "target": targets,
        }
    )

    submission_df.to_csv(output_dir / "submission.csv", index=False)
    print(f"Submission file saved to {output_dir / 'submission.csv'}")



if __name__ == "__main__":
    tyro.cli(main)
