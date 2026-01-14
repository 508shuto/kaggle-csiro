import numpy as np
import torch
from sklearn.metrics import r2_score
from torchmetrics import Metric

WEIGHTS = [
    0.1,  # Dry Clover
    0.1,  # Dry Dead
    0.1,  # Dry Green
    0.2,  # GDM
    0.5,  # Dry Total
]

CLASS_NAMES = [
    "Dry_Clover",
    "Dry_Dead",
    "Dry_Green",
    "GDM",
    "Dry_Total",
]


class WeightedR2Score(Metric):
    """Weighted R2 Score metric compatible with torchmetrics.

    Computes a single globally weighted R² score according to Kaggle specification.
    All (image, target) pairs are combined into one array, and each row is weighted
    according to its target type. This matches the Kaggle leaderboard evaluation.

    R² = 1 - (SS_res / SS_tot)
    where SS_res = Σ w_i * (y_true_i - y_pred_i)²
          SS_tot = Σ w_i * (y_true_i - weighted_mean)²
          weighted_mean = Σ w_i * y_true_i / Σ w_i
    """

    def __init__(self):
        super().__init__()
        self.weights = torch.tensor(WEIGHTS, dtype=torch.float32)
        # 予測値とターゲットを蓄積（分散学習対応）
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state with batch predictions and targets.

        Args:
            preds: Predictions tensor of shape (batch_size, num_classes)
            targets: Targets tensor of shape (batch_size, num_classes)
        """
        if preds.shape != targets.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape} != targets {targets.shape}")

        num_classes = preds.shape[1]

        if num_classes != len(WEIGHTS):
            raise ValueError(f"Number of classes {num_classes} != number of weights {len(WEIGHTS)}")

        # バッチをそのまま追加（後で結合してflatten）
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self) -> torch.Tensor:
        """Compute globally weighted R2 score from accumulated state.

        Returns:
            torch.Tensor: Globally weighted R2 score (scalar)
        """
        # 空の state チェック（list と Tensor の両方に対応）
        if isinstance(self.preds, list):
            if len(self.preds) == 0:
                return torch.tensor(0.0, device=self.weights.device)
        else:
            if self.preds.numel() == 0:
                return torch.tensor(0.0, device=self.weights.device)

        # DDP環境では dist_reduce_fx="cat" により state が Tensor に変換される可能性がある
        # list の場合は結合、Tensor の場合はそのまま使用
        if isinstance(self.preds, list):
            preds = torch.cat(self.preds, dim=0)  # (N, 5)
        else:
            preds = self.preds  # 既に Tensor

        if isinstance(self.targets, list):
            targets = torch.cat(self.targets, dim=0)  # (N, 5)
        else:
            targets = self.targets  # 既に Tensor

        # Flatten: row-major order (サンプルごとに5つのターゲットを順番に並べる)
        y_pred = preds.flatten()  # shape: (N*5,) [s0_t0, s0_t1, s0_t2, s0_t3, s0_t4, s1_t0, ...]
        y_true = targets.flatten()

        # sample_weight: 各要素にターゲットタイプの重みを割り当て
        weights = self.weights.to(preds.device)
        sample_weights = weights.repeat(preds.shape[0])  # [w0,w1,w2,w3,w4, w0,w1,w2,w3,w4, ...]

        # グローバル重み付き平均を計算
        weighted_mean = (sample_weights * y_true).sum() / sample_weights.sum()

        # SS_res: 重み付き残差平方和
        ss_res = (sample_weights * (y_true - y_pred) ** 2).sum()

        # SS_tot: 重み付き総平方和
        ss_tot = (sample_weights * (y_true - weighted_mean) ** 2).sum()

        # ゼロ除算を避ける
        eps = 1e-8
        ss_tot = torch.clamp(ss_tot, min=eps)

        # グローバル重み付きR²を計算
        r2_score = 1.0 - ss_res / ss_tot

        return r2_score


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute globally weighted R2 score according to Kaggle specification.

    All (image, target) pairs are combined into one array, and each row is weighted
    according to its target type. This matches the Kaggle leaderboard evaluation.

    Args:
        preds: Predictions array of shape (n_samples, 5)
        targets: Targets array of shape (n_samples, 5)

    Returns:
        float: Globally weighted R2 score
    """
    n_samples = preds.shape[0]

    # Flatten: row-major order (サンプルごとに5つのターゲットを順番に並べる)
    y_pred = preds.flatten()  # shape: (n_samples * 5,) [s0_t0, s0_t1, s0_t2, s0_t3, s0_t4, s1_t0, ...]
    y_true = targets.flatten()

    # sample_weight: 各要素にターゲットタイプの重みを割り当て
    sample_weights = np.tile(WEIGHTS, n_samples)  # [w0,w1,w2,w3,w4, w0,w1,w2,w3,w4, ...]

    # グローバル重み付きR²を計算
    return r2_score(y_true, y_pred, sample_weight=sample_weights)
