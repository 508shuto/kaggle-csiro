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
from tqdm import tqdm


def detect_device(device: str) -> str:
    """Detect available device automatically."""
    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def build_test_index(test_csv_path: Path, input_dir: Path) -> pd.DataFrame:
    """Build test dataset index from test CSV."""
    df = pd.read_csv(test_csv_path)

    image_paths = df["image_path"].unique()
    data_items = []

    for image_path in image_paths:
        sample_id_row = df[df["image_path"] == image_path].iloc[0]
        sample_id = sample_id_row["sample_id"].split("__")[0]
        jpeg_path = input_dir / image_path

        data_items.append(
            {
                "sample_id": sample_id,
                "image_path": jpeg_path,
            }
        )

    return pd.DataFrame(data_items)


class TestDataset(Dataset):
    """Test dataset for Qwen3-VL model."""

    def __init__(self, df: pd.DataFrame, config: DictConfig):
        self.df = df
        self.image_size = config.augmentation.valid.image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        return image


class TestCollator:
    """Collate function for test dataset (no targets)."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, batch: list) -> dict:
        images = batch

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
        }


def predict_fold(
    fold: int,
    config: DictConfig,
    df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    model_dir: Path,
    device: str,
    collator: TestCollator,
) -> np.ndarray:
    dataset = TestDataset(df, config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=collator,
    )

    # Load checkpoint with validation
    ckpt_paths = list(model_dir.glob(f"fold{fold}*.ckpt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint found for fold {fold} in {model_dir}")
    if len(ckpt_paths) > 1:
        print(f"Warning: Multiple checkpoints found for fold {fold}, using {ckpt_paths[0]}")
    model_path = ckpt_paths[0]
    map_location = torch.device(device)
    model = (
        CSIROModule.load_from_checkpoint(
            checkpoint_path=model_path,
            config=config,
            map_location=map_location,
            weights_only=False,
        )
        .model_ema.module.eval()
        .to(device)
    )

    fold_predictions = predict(
        fold=fold,
        model=model,
        dataloader=dataloader,
        device=device,
    )
    return fold_predictions


def predict(
    fold: int,
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> np.ndarray:
    """Predict with Qwen3-VL model."""
    fold_predictions = []

    for batch in tqdm(dataloader, desc=f"Fold {fold}"):
        pixel_values = batch["pixel_values"].to(device)
        image_grid_thw = batch["image_grid_thw"].to(device)

        with torch.no_grad():
            pred, _ = model(pixel_values, image_grid_thw)

        pred = torch.clamp(pred, min=0.0)
        fold_predictions.append(pred.cpu().detach().numpy())

    fold_predictions = np.concatenate(fold_predictions, axis=0)
    return fold_predictions


def main(
    test_csv_path: Path = Path("./input/test.csv"),
    config_path: Path = Path("./config/exp048.yaml"),
    model_dir: Path = Path("./output/exp048"),
    output_dir: Path = Path("./output/exp048"),
    folds: list[int] | None = None,
    device: str = "auto",
    batch_size: int = 16,
    num_workers: int = 4,
):
    """Run inference and generate submission file."""
    if folds is None:
        folds = [0, 1, 2]  # 3-fold CV

    config: DictConfig = OmegaConf.load(str(config_path))
    config.model.pretrained = False

    device = detect_device(device)
    print(f"Using device: {device}")

    # Build test index
    input_dir = Path(config.dataset.input_dir)
    df = build_test_index(test_csv_path, input_dir)

    # Initialize collator
    print(f"Loading Qwen3-VL processor: {config.model.name}")
    collator = TestCollator(model_name=config.model.name)

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
            collator=collator,
        )
        fold_predictions.append(preds)

    # Average predictions
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

    sample_ids = np.repeat(df["sample_id"].to_numpy(), num_targets)
    target_names_array = np.tile(target_names, num_samples)
    targets = ensemble_predictions.flatten()

    submission_df = pd.DataFrame(
        {
            "sample_id": [f"{sid}__{tname}" for sid, tname in zip(sample_ids, target_names_array, strict=True)],
            "target": targets,
        }
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_dir / "submission.csv", index=False)
    print(f"Submission file saved to {output_dir / 'submission.csv'}")


if __name__ == "__main__":
    tyro.cli(main)
