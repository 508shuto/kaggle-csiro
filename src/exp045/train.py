from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytorch_lightning as L
import torch
import tyro
import wandb
from dataset import CSIRODataset, Qwen3VLCollator
from lightning_module import CSIROModule
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from utils import seed_everything


def train_fold(config: DictConfig, df: pd.DataFrame, fold: int, collator: Qwen3VLCollator) -> None:
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
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.trainer.valid.batch_size,
        shuffle=False,
        num_workers=config.trainer.valid.num_workers,
        pin_memory=config.trainer.valid.pin_memory,
        drop_last=config.trainer.valid.drop_last,
        collate_fn=collator,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.dataset.output_dir,
        save_top_k=0,
        save_last=True,
    )
    callbacks = [checkpoint_callback]
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
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.experiment.name}_fold{fold}"
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

    # Strategy
    strategy = config.trainer.train.strategy
    if strategy == "ddp_find_unused_parameters_true":
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = L.Trainer(
        max_epochs=config.trainer.train.epochs,
        accelerator=config.trainer.train.device_type,
        devices=config.trainer.train.devices,
        precision=config.trainer.train.precision,
        strategy=strategy,
        accumulate_grad_batches=config.trainer.train.accumulate_grad_batches,
        val_check_interval=config.trainer.train.val_check_interval,
        deterministic=config.trainer.train.deterministic,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # Synchronize all ranks
    if hasattr(trainer.strategy, "barrier"):
        trainer.strategy.barrier()

    # Rename checkpoint
    if trainer.is_global_zero:
        last_ckpt = Path(config.dataset.output_dir) / "last.ckpt"
        if last_ckpt.exists():
            ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            epoch = ckpt.get("epoch", trainer.current_epoch)
            if epoch is None:
                epoch = trainer.current_epoch

            if "val_score" not in trainer.callback_metrics:
                raise ValueError(
                    f"val_score not found in trainer.callback_metrics. "
                    f"Available keys: {list(trainer.callback_metrics.keys())}"
                )
            val_score_tensor = trainer.callback_metrics["val_score"]
            if isinstance(val_score_tensor, torch.Tensor):
                val_score = val_score_tensor.item()
            else:
                val_score = float(val_score_tensor)

            new_name = Path(config.dataset.output_dir) / f"fold{fold}_epoch={epoch}-val_score={val_score:.4f}.ckpt"
            last_ckpt.rename(new_name)
            print(f"Saved checkpoint: {new_name}")

        if config.trainer.train.use_wandb:
            wandb.finish()

    if hasattr(trainer.strategy, "barrier"):
        trainer.strategy.barrier()


def main(
    config_path: Path | None = None,
    folds: list[int] | None = None,
    debug: bool = False,
) -> None:
    """Train Qwen3-VL based regression model."""
    if folds is None:
        folds = [0, 1, 2]  # 3-fold CV

    exp_name: str = Path(__file__).parent.name
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

    # Initialize collator (processor is loaded once)
    print(f"Loading Qwen3-VL processor: {config.model.name}")
    collator = Qwen3VLCollator(model_name=config.model.name)

    # Train folds
    for fold in folds:
        print(f"\n{'='*50}")
        print(f"Training fold {fold}")
        print(f"{'='*50}")
        train_fold(config, df, fold, collator)
        if debug:
            break


if __name__ == "__main__":
    tyro.cli(main)
