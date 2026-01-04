from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytorch_lightning as L
import torch
import tyro
import wandb
from dataset import CachedFeatureDataset, CSIRODataset
from lightning_module import CachedCSIROModule, CSIROModule
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from utils import seed_everything


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Collate function for preprocessed tensors."""
    pixel_values = torch.stack([item[0] for item in batch])
    grid_thw = torch.stack([item[1] for item in batch])
    targets = torch.stack([item[2] for item in batch])
    aux_targets = torch.stack([item[3] for item in batch])
    return pixel_values, grid_thw, targets, aux_targets


def cached_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Collate function for cached feature tensors."""
    features = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    aux_targets = torch.stack([item[2] for item in batch])
    return features, targets, aux_targets


def train_fold(
    config: DictConfig,
    df: pd.DataFrame,
    fold: int,
    use_cache: bool = False,
    feature_dir: Path | None = None,
) -> None:
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    if use_cache:
        assert feature_dir is not None, "feature_dir must be provided when use_cache=True"
        print(f"[Cache Mode] Loading features from {feature_dir}")
        train_dataset = CachedFeatureDataset(config, train_df, feature_dir, "train")
        valid_dataset = CachedFeatureDataset(config, valid_df, feature_dir, "valid")
        current_collate_fn = cached_collate_fn
        # Can use larger batch size with cached features (no vision encoder)
        train_batch_size = config.trainer.train.get("cached_batch_size", config.trainer.train.batch_size * 4)
        valid_batch_size = config.trainer.valid.get("cached_batch_size", config.trainer.valid.batch_size * 4)
    else:
        train_dataset = CSIRODataset(config, train_df, "train")
        valid_dataset = CSIRODataset(config, valid_df, "valid")
        current_collate_fn = collate_fn
        train_batch_size = config.trainer.train.batch_size
        valid_batch_size = config.trainer.valid.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=config.trainer.train.num_workers,
        pin_memory=config.trainer.train.pin_memory,
        drop_last=config.trainer.train.drop_last,
        prefetch_factor=config.trainer.train.prefetch_factor,
        persistent_workers=config.trainer.train.num_workers > 0,
        collate_fn=current_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=config.trainer.valid.num_workers,
        pin_memory=True,  # Always use pin_memory for validation
        drop_last=config.trainer.valid.drop_last,
        prefetch_factor=config.trainer.valid.prefetch_factor,
        persistent_workers=config.trainer.valid.num_workers > 0,
        collate_fn=current_collate_fn,
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val_score",
            mode="max",
            save_top_k=1,
            save_last=False,
            dirpath=config.dataset.output_dir,
            filename=f"fold{fold}_{{epoch:02d}}-{{val_score:.4f}}",
        ),
    ]
    if config.trainer.train.patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_score",
                mode="max",
                patience=config.trainer.train.patience,
            ),
        )

    logger = None
    if config.trainer.train.use_wandb:
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.experiment.name}_qwen3vl_fold{fold}"
        logger = WandbLogger(
            project="kaggle-csiro",
            name=f"{config.experiment.name}_fold{fold}",
            group=config.experiment.name,
            id=run_id,
            config=OmegaConf.to_container(config, resolve=True),
        )

    if use_cache:
        module = CachedCSIROModule(config=config)
    else:
        module = CSIROModule(config=config)

    trainer = L.Trainer(
        max_epochs=config.trainer.train.epochs,
        accelerator=config.trainer.train.device_type,
        devices=config.trainer.train.devices,
        precision=config.trainer.train.precision,
        strategy=config.trainer.train.strategy,
        accumulate_grad_batches=config.trainer.train.accumulate_grad_batches,
        val_check_interval=config.trainer.train.val_check_interval,
        deterministic=config.trainer.train.deterministic,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    if config.trainer.train.use_wandb:
        wandb.finish()


def main(
    config_path: Path | None = None,
    folds: list[int] | None = None,
    debug: bool = False,
    use_cache: bool = False,
    feature_dir: Path | None = None,
) -> None:
    """Train Qwen3-VL based regression model with R2 score monitoring.

    Args:
        config_path: Path to config file. Defaults to ./config/exp027.yaml
        folds: List of folds to train. Defaults to [0, 1, 2, 3, 4]
        debug: If True, only train fold 0
        use_cache: If True, use cached features for fast head-only training
        feature_dir: Directory containing cached features (required if use_cache=True)
    """
    if folds is None:
        folds = [0, 1, 2, 3, 4]

    if use_cache and feature_dir is None:
        raise ValueError("feature_dir must be provided when use_cache=True")

    # Derive default config path from experiment directory name
    exp_name: str = Path(__file__).parent.name  # exp027
    if config_path is None:
        config_path = Path("./config") / f"{exp_name}.yaml"
    config = OmegaConf.load(config_path)

    seed_everything(config.experiment.seed)

    # Output dir
    config.dataset.output_dir = str(Path(config.dataset.output_dir) / exp_name)
    out_dir = Path(config.dataset.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(out_dir / "preprocessed_train.csv")

    print(f"Loaded dataset with {len(df)} samples")

    if max(folds) + 1 > config.dataset.n_folds or len(folds) > config.dataset.n_folds:
        raise ValueError(f"folds must be less than {config.dataset.n_folds} and must be unique")

    # Train folds
    for fold in folds:
        train_fold(config, df, fold, use_cache=use_cache, feature_dir=feature_dir)
        if debug:
            break


if __name__ == "__main__":
    tyro.cli(main)
