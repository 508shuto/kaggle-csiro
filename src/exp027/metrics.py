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

    Computes a single globally weighted R2 score according to Kaggle specification.
    All (image, target) pairs are combined into one array, and each row is weighted
    according to its target type. This matches the Kaggle leaderboard evaluation.

    R2 = 1 - (SS_res / SS_tot)
    where SS_res = sum w_i * (y_true_i - y_pred_i)^2
          SS_tot = sum w_i * (y_true_i - weighted_mean)^2
          weighted_mean = sum w_i * y_true_i / sum w_i
    """

    def __init__(self):
        super().__init__()
        self.weights = torch.tensor(WEIGHTS, dtype=torch.float32)
        # Accumulate predictions and targets (distributed training compatible)
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

        # Append batch directly (concatenate later)
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self) -> torch.Tensor:
        """Compute globally weighted R2 score from accumulated state.

        Returns:
            torch.Tensor: Globally weighted R2 score (scalar)
        """
        if len(self.preds) == 0:
            return torch.tensor(0.0, device=self.weights.device)

        # Concatenate all batches
        preds = torch.cat(self.preds, dim=0)  # (N, 5)
        targets = torch.cat(self.targets, dim=0)  # (N, 5)

        # Flatten: row-major order
        y_pred = preds.flatten()  # (N*5,)
        y_true = targets.flatten()

        # sample_weight: assign target type weight to each element
        weights = self.weights.to(preds.device)
        sample_weights = weights.repeat(preds.shape[0])  # [w0,w1,w2,w3,w4, ...]

        # Compute global weighted mean
        weighted_mean = (sample_weights * y_true).sum() / sample_weights.sum()

        # SS_res: weighted residual sum of squares
        ss_res = (sample_weights * (y_true - y_pred) ** 2).sum()

        # SS_tot: weighted total sum of squares
        ss_tot = (sample_weights * (y_true - weighted_mean) ** 2).sum()

        # Avoid division by zero
        eps = 1e-8
        ss_tot = torch.clamp(ss_tot, min=eps)

        # Compute global weighted R2
        r2_score_val = 1.0 - ss_res / ss_tot

        return r2_score_val


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute globally weighted R2 score according to Kaggle specification.

    Args:
        preds: Predictions array of shape (n_samples, 5)
        targets: Targets array of shape (n_samples, 5)

    Returns:
        float: Globally weighted R2 score
    """
    n_samples = preds.shape[0]

    # Flatten: row-major order
    y_pred = preds.flatten()  # (n_samples * 5,)
    y_true = targets.flatten()

    # sample_weight: assign target type weight to each element
    sample_weights = np.tile(WEIGHTS, n_samples)  # [w0,w1,w2,w3,w4, ...]

    # Compute global weighted R2
    return r2_score(y_true, y_pred, sample_weight=sample_weights)
