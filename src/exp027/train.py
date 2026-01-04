from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytorch_lightning as L
import tyro
import wandb
from dataset import CSIRODataset
from lightning_module import CSIROModule
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from utils import seed_everything


def train_fold(config: DictConfig, df: pd.DataFrame, fold: int) -> None:
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    train_dataset = CSIRODataset(config, train_df, "train")
    valid_dataset = CSIRODataset(config, valid_df, "valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.trainer.train.batch_size,
        shuffle=True,
        num_workers=config.trainer.train.num_workers,
        pin_memory=config.trainer.train.pin_memory,
        drop_last=config.trainer.train.drop_last,
        prefetch_factor=config.trainer.train.prefetch_factor,
        persistent_workers=config.trainer.train.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.trainer.valid.batch_size,
        shuffle=False,
        num_workers=config.trainer.valid.num_workers,
        pin_memory=config.trainer.valid.pin_memory,
        drop_last=config.trainer.valid.drop_last,
        prefetch_factor=config.trainer.valid.prefetch_factor,
        persistent_workers=config.trainer.valid.num_workers > 0,
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

    module = CSIROModule(
        config=config,
    )

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
) -> None:
    """Train Qwen3-VL based regression model with R2 score monitoring.
    - Trains model per fold with R2 score monitoring
    """
    if folds is None:
        folds = [0, 1, 2, 3, 4]
    # Derive default config path from experiment directory name
    exp_name: str = Path(__file__).parent.name  # exp026
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
        train_fold(config, df, fold)
        if debug:
            break


if __name__ == "__main__":
    tyro.cli(main)
