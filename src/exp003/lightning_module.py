import pytorch_lightning as L
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from models import CSIROModel
from omegaconf import DictConfig
from timm.optim._optim_factory import create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2
from timm.utils.model_ema import ModelEmaV3
from torchmetrics import R2Score, MetricCollection
from metrics import WeightedR2Score, WEIGHTS, CLASS_NAMES

from utils import get_loss_fn, mixup_batch, denormalize_image


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
        self.aux_loss_fn = get_loss_fn(
            self.config.aux_loss.params, self.config.aux_loss.name
        )
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

        # Mixup設定
        self.mixup_enabled = config.augmentation.train.get("mixup", {}).get(
            "enabled", False
        )
        self.mixup_alpha = config.augmentation.train.get("mixup", {}).get("alpha", 0.2)
        self.mixup_prob = config.augmentation.train.get("mixup", {}).get("prob", 0.5)

        # 予測結果のロギング設定
        self.log_predictions = config.trainer.train.get("log_predictions", True)
        self.max_log_images = config.trainer.train.get("max_log_images", 16)

        # 検証時の予測結果収集用
        self._val_images = []
        self._val_preds = []
        self._val_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model, self.global_step)

        image, targets, aux_targets = batch

        # Mixup適用
        if self.mixup_enabled and torch.rand(1).item() < self.mixup_prob:
            image, targets, aux_targets, lam = mixup_batch(
                image, targets, aux_targets, alpha=self.mixup_alpha
            )
            self.log("train_mixup_lambda", lam, on_step=False, on_epoch=True)

        # 対数空間で予測・損失計算
        pred_log, aux_pred_log = self.model(image)
        loss = self.loss_fn(pred_log, targets) + self.aux_weight * self.aux_loss_fn(
            aux_pred_log, aux_targets
        )

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train_aux_loss",
            self.aux_loss_fn(aux_pred_log, aux_targets),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image, targets_log, aux_targets_log = batch

        # 対数空間で予測
        preds_log, aux_preds_log = self.model_ema.module(image)  # (B, 5), (B, 2)

        # 対数空間でLoss計算
        loss = self.loss_fn(
            preds_log, targets_log
        ) + self.aux_weight * self.aux_loss_fn(aux_preds_log, aux_targets_log)

        # 元の空間に戻してメトリクス計算
        preds = torch.expm1(preds_log)  # exp(x) - 1
        aux_preds = torch.expm1(aux_preds_log)
        targets_original = torch.expm1(targets_log)
        aux_targets_original = torch.expm1(aux_targets_log)

        # 負値を除去（念のため）
        preds = torch.clamp(preds, min=0.0)
        aux_preds = torch.clamp(aux_preds, min=0.0)

        # メトリクスは元の空間で計算
        self.metrics.update(preds, targets_original)
        self.aux_metrics.update(aux_preds, aux_targets_original)

        # 予測結果をロギング用に収集（最初のmax_log_images枚のみ）
        if self.log_predictions:
            current_count = sum(len(imgs) for imgs in self._val_images)
            if current_count < self.max_log_images:
                remaining = self.max_log_images - current_count
                n_samples = min(remaining, image.size(0))
                self._val_images.append(image[:n_samples].detach().cpu())
                self._val_preds.append(preds[:n_samples].detach().cpu())
                self._val_targets.append(targets_original[:n_samples].detach().cpu())

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_aux_loss",
            self.aux_loss_fn(aux_preds_log, aux_targets_log),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
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

        # wandbに予測結果をロギング
        if self.log_predictions and self._val_images and wandb.run is not None:
            self._log_predictions_to_wandb()

        self.metrics.reset()
        self.aux_metrics.reset()

        # 予測結果収集用リストをリセット
        self._val_images = []
        self._val_preds = []
        self._val_targets = []

    def _log_predictions_to_wandb(self):
        """wandbに予測画像、予測値テーブル、散布図をログ."""
        # 収集した予測結果を結合
        all_images = torch.cat(self._val_images, dim=0)  # (N, C, H, W)
        all_preds = torch.cat(self._val_preds, dim=0)  # (N, 5)
        all_targets = torch.cat(self._val_targets, dim=0)  # (N, 5)

        n_samples = all_images.size(0)

        # 1. 予測画像と予測値のテーブルを作成
        columns = ["image"] + [f"pred_{c}" for c in CLASS_NAMES] + [f"true_{c}" for c in CLASS_NAMES] + [f"error_{c}" for c in CLASS_NAMES]
        table = wandb.Table(columns=columns)

        for i in range(n_samples):
            # 画像を逆正規化
            img_array = denormalize_image(all_images[i])

            # 各ターゲットの予測値、実測値、誤差
            preds_i = all_preds[i].numpy()
            targets_i = all_targets[i].numpy()
            errors_i = preds_i - targets_i

            row = [wandb.Image(img_array)]
            row.extend(preds_i.tolist())
            row.extend(targets_i.tolist())
            row.extend(errors_i.tolist())
            table.add_data(*row)

        wandb.log({"val_predictions": table}, step=self.current_epoch)

        # 2. 各ターゲットごとの散布図（予測 vs 実測）を作成
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, class_name in enumerate(CLASS_NAMES):
            ax = axes[i]
            preds_i = all_preds[:, i].numpy()
            targets_i = all_targets[:, i].numpy()

            ax.scatter(targets_i, preds_i, alpha=0.6, s=20)

            # 対角線（完璧な予測）
            max_val = max(targets_i.max(), preds_i.max())
            min_val = min(targets_i.min(), preds_i.min())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

            ax.set_xlabel(f"True {class_name}")
            ax.set_ylabel(f"Pred {class_name}")
            ax.set_title(f"{class_name} (weight={WEIGHTS[i]})")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 6番目のsubplotは使わないので非表示
        axes[5].axis('off')

        plt.tight_layout()
        wandb.log({"val_scatter_plots": wandb.Image(fig)}, step=self.current_epoch)
        plt.close(fig)

        # 3. 誤差分布のヒストグラム
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, class_name in enumerate(CLASS_NAMES):
            ax = axes[i]
            errors_i = (all_preds[:, i] - all_targets[:, i]).numpy()

            ax.hist(errors_i, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', lw=2)
            ax.axvline(x=errors_i.mean(), color='g', linestyle='-', lw=2, label=f'Mean: {errors_i.mean():.2f}')

            ax.set_xlabel(f"Error (Pred - True)")
            ax.set_ylabel("Count")
            ax.set_title(f"{class_name} Error Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[5].axis('off')

        plt.tight_layout()
        wandb.log({"val_error_histograms": wandb.Image(fig)}, step=self.current_epoch)
        plt.close(fig)

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
