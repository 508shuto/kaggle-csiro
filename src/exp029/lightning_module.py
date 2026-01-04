import pytorch_lightning as L
import torch
from metrics import CLASS_NAMES, WEIGHTS, WeightedR2Score
from models import CSIROModel
from omegaconf import DictConfig
from timm.optim._optim_factory import create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2
from timm.utils.model_ema import ModelEmaV3
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score

from utils import get_loss_fn, mixup_batch


class CSIROModule(L.LightningModule):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        self.config = config
        self.freeze_backbone = config.model.get("freeze_backbone", True)
        self.model = CSIROModel(
            model_name=config.model.name,
            pretrained=config.model.pretrained,
            in_channels=config.model.in_channels,
            freeze_backbone=self.freeze_backbone,
            use_tile=config.model.get("use_tile", False),
            tile_grid=config.model.get("tile_grid", 2),
            gem_p=config.model.get("gem_p", 3.0),
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

        # Mixup設定
        self.mixup_enabled = config.augmentation.train.get("mixup", {}).get("enabled", False)
        self.mixup_alpha = config.augmentation.train.get("mixup", {}).get("alpha", 0.2)
        self.mixup_prob = config.augmentation.train.get("mixup", {}).get("prob", 0.5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model, self.global_step)

        image, targets, aux_targets = batch

        # Mixup適用
        if self.mixup_enabled and torch.rand(1).item() < self.mixup_prob:
            image, targets, aux_targets, lam = mixup_batch(image, targets, aux_targets, alpha=self.mixup_alpha)
            self.log("train_mixup_lambda", lam, on_step=False, on_epoch=True)

        # raw空間で予測・損失計算
        preds, aux_preds = self.model(image)
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
        image, targets, aux_targets = batch

        # raw空間で予測
        preds, aux_preds = self.model_ema.module(image)  # (B, 5), (B, 2)

        # raw空間でLoss計算
        loss = self.loss_fn(preds, targets) + self.aux_weight * self.aux_loss_fn(aux_preds, aux_targets)

        # 負値を除去（念のため）
        preds = torch.clamp(preds, min=0.0)
        aux_preds = torch.clamp(aux_preds, min=0.0)

        # メトリクスはraw空間で計算
        self.metrics.update(preds, targets)
        self.aux_metrics.update(aux_preds, aux_targets)

        # Per-class RMSE/MAE計算のためにバッチごとのpreds/targetsを保存
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

        # Per-class RMSE/MAE計算
        if hasattr(self, "val_preds") and len(self.val_preds) > 0:
            all_preds = torch.cat(self.val_preds, dim=0)  # (N, 5)
            all_targets = torch.cat(self.val_targets, dim=0)  # (N, 5)

            # 各クラスごとにRMSEとMAEを計算
            for i, class_name in enumerate(CLASS_NAMES):
                # Per-class RMSE
                class_mse = torch.mean((all_preds[:, i] - all_targets[:, i]) ** 2)
                class_rmse = torch.sqrt(class_mse)
                self.log(f"val_rmse_{class_name}", class_rmse)

                # Per-class MAE
                class_mae = torch.mean(torch.abs(all_preds[:, i] - all_targets[:, i]))
                self.log(f"val_mae_{class_name}", class_mae)

            # リストをクリア
            self.val_preds = []
            self.val_targets = []

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

        # RMSE（全体）
        if "rmse" in metrics:
            self.log("val_rmse", metrics["rmse"])

        # MAE（全体）
        if "mae" in metrics:
            self.log("val_mae", metrics["mae"])

        self.metrics.reset()
        self.aux_metrics.reset()

    def configure_optimizers(self):
        # Freeze時は学習可能なパラメータのみをoptimizerに渡す
        trainable_params = self.model.get_trainable_parameters()
        optimizer = create_optimizer_v2(
            model_or_params=trainable_params,
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
