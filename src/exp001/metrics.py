import numpy as np
import torch
from sklearn.metrics import r2_score
from torchmetrics import Metric

WEIGHTS = [
    0.1,  # Dry Green
    0.1,  # Dry Dead
    0.1,  # Dry Clover
    0.2,  # GDM
    0.5,  # Dry Total
]

CLASS_NAMES = [
    "Dry_Green",
    "Dry_Dead",
    "Dry_Clover",
    "GDM",
    "Dry_Total",
]


class WeightedR2Score(Metric):
    """Weighted R2 Score metric compatible with torchmetrics.

    Computes R2 score for each class and returns weighted average.
    R² = 1 - (SS_res / SS_tot)
    where SS_res = Σ(y_true - y_pred)², SS_tot = Σ(y_true - y_mean)²
    """

    def __init__(self):
        super().__init__()
        self.weights = torch.tensor(WEIGHTS, dtype=torch.float32)
        self.add_state(
            "sum_squared_residuals",
            default=torch.zeros(len(WEIGHTS)),
            dist_reduce_fx="sum",
        )
        self.add_state("sum_squared_total", default=torch.zeros(len(WEIGHTS)), dist_reduce_fx="sum")
        self.add_state("sum_targets", default=torch.zeros(len(WEIGHTS)), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state with batch predictions and targets.

        Args:
            preds: Predictions tensor of shape (batch_size, num_classes)
            targets: Targets tensor of shape (batch_size, num_classes)
        """
        if preds.shape != targets.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape} != targets {targets.shape}")

        batch_size = preds.shape[0]
        num_classes = preds.shape[1]

        if num_classes != len(WEIGHTS):
            raise ValueError(f"Number of classes {num_classes} != number of weights {len(WEIGHTS)}")

        # Calculate residuals: (y_true - y_pred)² for each class
        residuals = (targets - preds) ** 2  # (batch_size, num_classes)
        sum_squared_residuals = residuals.sum(dim=0)  # (num_classes,)

        # Calculate sum of targets for mean calculation
        sum_targets = targets.sum(dim=0)  # (num_classes,)

        # Calculate total sum of squares: (y_true - y_mean)²
        # We'll need the mean for SS_tot calculation in compute()
        # For now, accumulate sum_targets and total_samples
        sum_squared_total = targets**2  # (batch_size, num_classes)
        sum_squared_total = sum_squared_total.sum(dim=0)  # (num_classes,)

        # Update state
        self.sum_squared_residuals += sum_squared_residuals
        self.sum_squared_total += sum_squared_total
        self.sum_targets += sum_targets
        self.total_samples += batch_size

    def compute(self) -> torch.Tensor:
        """Compute weighted R2 score from accumulated state.

        Returns:
            torch.Tensor: Weighted R2 score (scalar)
        """
        if self.total_samples == 0:
            return torch.tensor(0.0, device=self.weights.device)

        # Calculate mean for each class
        mean_targets = self.sum_targets / self.total_samples  # type: ignore  # (num_classes,)

        # Calculate SS_res (sum of squared residuals)
        ss_res = self.sum_squared_residuals  # (num_classes,)

        # Calculate SS_tot (total sum of squares)
        # SS_tot = Σ(y_true - y_mean)² = Σ(y_true²) - n * y_mean²
        ss_tot = self.sum_squared_total - self.total_samples * mean_targets**2  # (num_classes,)

        # Avoid division by zero
        eps = 1e-8
        ss_tot = torch.clamp(ss_tot, min=eps)

        # Calculate R2 score for each class: R² = 1 - (SS_res / SS_tot)
        r2_per_class = 1.0 - (ss_res / ss_tot)  # (num_classes,)

        # Calculate weighted average
        weights = self.weights.to(r2_per_class.device)
        weighted_r2 = torch.sum(weights * r2_per_class)

        return weighted_r2


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute metrics for weighted R2 score.

    Args:
        preds: Predictions array of shape (B,)
        targets: Targets array of shape (B,)

    Returns:
        float: Weighted R2 score
    """
    weighted_r2 = 0.0
    for i in range(preds.shape[1]):
        weighted_r2 += WEIGHTS[i] * r2_score(targets[:, i], preds[:, i])
    return weighted_r2
