import pytorch_lightning as L
import torch
from models import CSIROModel
from omegaconf import DictConfig
from timm.optim._optim_factory import create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2
from timm.utils.model_ema import ModelEmaV3
from torchmetrics import R2Score, MetricCollection
from metrics import WeightedR2Score, WEIGHTS, CLASS_NAMES

from utils import get_loss_fn


class CSIROModule(L.LightningModule):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        self.config = config
        self.model = CSIROModel(
            model_name=config.model.name,
            pretrained=config.model.pretrained,
            in_channels=config.model.in_channels,
        )
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
            }
        )
        self.aux_metrics = MetricCollection(
            {
                "r2_score": R2Score(multioutput="raw_values"),
            }
        )
        self.aux_weight = config.aux_loss.weight

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model, self.global_step)
        # NOTE: Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g
        #       so how is it possible to reduce the num_outputs 5 to 4(drop to learn Dry_Total_g)?
        # ?: Does the model learn the correlation between the targets, especially Dry_Green_g, Dry_Dead_g, Dry_Clover_g and Dry_Total_g?
        # NOTE: how about using the Pre_GSHH_NDVI and Height_Ave_cm as a auxiliary_target?
        # ?: How to use Sampling_Date and State?

        # TODO: implement mixup
        image, targets, aux_targets = batch
        pred, aux_pred = self.model(image)
        loss = self.loss_fn(pred, targets) + self.aux_weight * self.aux_loss_fn(aux_pred, aux_targets)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_aux_loss", self.aux_loss_fn(aux_pred, aux_targets), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, targets ,aux_targets = batch
        preds, aux_preds = self.model_ema.module(image)  # (B, 5)

        loss = self.loss_fn(preds, targets) + self.aux_weight * self.aux_loss_fn(aux_preds, aux_targets)

        # TODO: inverse log1p transform
        self.metrics.update(preds, targets)
        self.aux_metrics.update(aux_preds, aux_targets)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_aux_loss", self.aux_loss_fn(aux_preds, aux_targets), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute()
        aux_metrics = self.aux_metrics.compute()
        # R2スコアの処理
        if "r2_score" in metrics:
            r2_scores = metrics["r2_score"]  # (5,)
            weights = torch.tensor(WEIGHTS, device=r2_scores.device)

            # 各クラスごとにログ
            for i, class_name in enumerate(CLASS_NAMES):
                # 各クラスのR2スコア
                self.log(f"val_r2_{class_name}", r2_scores[i])
                # 各クラスの重み付きR2スコア (weight × R2)
                weighted_r2 = weights[i] * r2_scores[i]
                self.log(f"val_weighted_r2_{class_name}", weighted_r2)

            # 平均R2スコア（参考用）
            self.log("val_r2_mean", r2_scores.mean())

        # Auxiliary Metrics（全クラスの重み付きR2の合計）
        if "r2_score" in aux_metrics:
            r2_scores = aux_metrics["r2_score"]  # (2,)
            self.log("val_aux_r2_mean", r2_scores.mean())

        # CompetitionMetrics（全クラスの重み付きR2の合計）
        if "weighted_r2_score" in metrics:
            self.log("val_score", metrics["weighted_r2_score"])

        self.metrics.reset()
        self.aux_metrics.reset()

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            model_or_params=self.model,
            **self.config.trainer.train.optimizer,
        )
        updates_per_epoch = self._get_steps_per_epoch()

        scheduler_config = dict(self.config.trainer.train.scheduler)

        scheduler, _ = create_scheduler_v2(
            optimizer=optimizer,
            num_epochs=self.config.trainer.train.epochs,
            warmup_lr=0,
            **scheduler_config,
            step_on_epochs=False,
            updates_per_epoch=updates_per_epoch,
        )
        lr_dict = dict(
            scheduler=scheduler,
            interval="step",
            frequency=1,  # same as default
        )
        return dict(optimizer=optimizer, lr_scheduler=lr_dict)

    def _get_steps_per_epoch(self) -> int:
        total_steps: int = self.trainer.estimated_stepping_batches
        steps_per_epoch: int = total_steps // self.config.trainer.train.epochs
        return steps_per_epoch

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step_update(num_updates=self.global_step)
