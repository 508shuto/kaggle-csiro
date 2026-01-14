import pytorch_lightning as L
import torch
from metrics import CLASS_NAMES, WEIGHTS, WeightedR2Score
from models import CSIROModel
from omegaconf import DictConfig
from timm.utils.model_ema import ModelEmaV3
from torch.optim import AdamW
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score
from transformers import get_cosine_schedule_with_warmup

from utils import get_loss_fn


class CSIROModule(L.LightningModule):
    """Lightning module for Qwen3-VL based regression model.

    The encoder is frozen; only the regression head is trained.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        self.config = config

        # Initialize model (encoder frozen, head trainable)
        self.model = CSIROModel(
            model_name=config.model.name,
            pretrained=config.model.pretrained,
            in_channels=config.model.in_channels,
            hidden_size=config.model.get("hidden_size", 1536),
            freeze_encoder=config.model.get("freeze_encoder", True),
        )

        # EMA for head only
        self.model_ema = ModelEmaV3(
            self.model,
            **self.config.trainer.train.ema,
        )

        self.loss_fn = get_loss_fn(self.config.loss.params, self.config.loss.name)
        self.aux_loss_fn = get_loss_fn(self.config.aux_loss.params, self.config.aux_loss.name)

        self.metrics = MetricCollection(
            {
                "r2_score": R2Score(multioutput="raw_values"),
                "weighted_r2_score": WeightedR2Score(),
                "rmse": MeanSquaredError(squared=False),
                "mae": MeanAbsoluteError(),
            }
        )
        self.aux_metrics = MetricCollection(
            {
                "r2_score": R2Score(multioutput="raw_values"),
            }
        )
        self.aux_weight = config.aux_loss.weight

    def forward(self, pixel_values, image_grid_thw):
        return self.model(pixel_values, image_grid_thw)

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model, self.global_step)

        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]
        targets = batch["targets"]
        aux_targets = batch["aux_targets"]

        # Forward pass
        preds, aux_preds = self.model(pixel_values, image_grid_thw)

        # Compute loss
        loss = self.loss_fn(preds, targets) + self.aux_weight * self.aux_loss_fn(aux_preds, aux_targets)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train_aux_loss",
            self.aux_loss_fn(aux_preds, aux_targets),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]
        targets = batch["targets"]
        aux_targets = batch["aux_targets"]

        # Forward pass with EMA model
        preds, aux_preds = self.model_ema.module(pixel_values, image_grid_thw)

        # Compute loss
        loss = self.loss_fn(preds, targets) + self.aux_weight * self.aux_loss_fn(aux_preds, aux_targets)

        # Clamp predictions
        preds = torch.clamp(preds, min=0.0)
        aux_preds = torch.clamp(aux_preds, min=0.0)

        # Update metrics
        self.metrics.update(preds, targets)
        self.aux_metrics.update(aux_preds, aux_targets)

        # Store for per-class metrics
        if not hasattr(self, "val_preds"):
            self.val_preds = []
            self.val_targets = []
        self.val_preds.append(preds)
        self.val_targets.append(targets)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_aux_loss",
            self.aux_loss_fn(aux_preds, aux_targets),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute()
        aux_metrics = self.aux_metrics.compute()

        # Per-class RMSE/MAE
        if hasattr(self, "val_preds") and len(self.val_preds) > 0:
            all_preds = torch.cat(self.val_preds, dim=0)
            all_targets = torch.cat(self.val_targets, dim=0)

            for i, class_name in enumerate(CLASS_NAMES):
                class_mse = torch.mean((all_preds[:, i] - all_targets[:, i]) ** 2)
                class_rmse = torch.sqrt(class_mse)
                self.log(f"val_rmse_{class_name}", class_rmse)

                class_mae = torch.mean(torch.abs(all_preds[:, i] - all_targets[:, i]))
                self.log(f"val_mae_{class_name}", class_mae)

            self.val_preds = []
            self.val_targets = []

        # R2 scores
        if "r2_score" in metrics:
            r2_scores = metrics["r2_score"]
            weights = torch.tensor(WEIGHTS, device=r2_scores.device)

            for i, class_name in enumerate(CLASS_NAMES):
                self.log(f"val_r2_{class_name}", r2_scores[i])
                weighted_r2 = weights[i] * r2_scores[i]
                self.log(f"val_weighted_r2_{class_name}", weighted_r2)

            self.log("val_r2_mean", r2_scores.mean())

        if "r2_score" in aux_metrics:
            r2_scores = aux_metrics["r2_score"]
            self.log("val_aux_r2_mean", r2_scores.mean())

        if "weighted_r2_score" in metrics:
            self.log("val_score", metrics["weighted_r2_score"])

        if "rmse" in metrics:
            self.log("val_rmse", metrics["rmse"])

        if "mae" in metrics:
            self.log("val_mae", metrics["mae"])

        self.metrics.reset()
        self.aux_metrics.reset()

    def configure_optimizers(self):
        # Only head parameters are trainable
        trainable_params = self.model.get_trainable_parameters()

        optimizer = AdamW(
            trainable_params,
            lr=self.config.trainer.train.optimizer.lr,
            weight_decay=self.config.trainer.train.optimizer.weight_decay,
        )

        # Cosine schedule with warmup
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.config.trainer.train.warmup_epochs * (total_steps // self.config.trainer.train.epochs)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        lr_dict = dict(
            scheduler=scheduler,
            interval="step",
            frequency=1,
        )
        return dict(optimizer=optimizer, lr_scheduler=lr_dict)
