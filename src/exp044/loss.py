import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.weight.to(preds.device) * (preds - targets) ** 2)


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # SmoothL1Lossを要素ごとに計算し、重み付けして平均
        loss_per_element = self.smooth_l1_loss(preds, targets)  # (B, 5)
        weighted_loss = self.weight.to(preds.device) * loss_per_element  # (B, 5)
        return torch.mean(weighted_loss)


class QuantileWeightedSmoothL1Loss(nn.Module):
    """Quantile-based sample weighting for imbalanced regression.

    Precomputed from train.csv using src/exp044/precompute_weights.py
    """

    # Bin edges: [min, Q20, Q40, Q60, Q80, max] for each target
    BIN_EDGES = {
        0: [0.0, 0.0, 0.2809, 2.8448, 10.2173, 71.7865],  # clover_target
        1: [0.0, 2.6240, 5.8, 10.5219, 19.7535, 83.8407],  # dead_target
        2: [0.0, 6.3839, 16.1592, 25.0993, 40.1330, 157.9836],  # green_target
        3: [1.04, 13.7226, 22.588, 31.9491, 49.2461, 157.9836],  # gdm_target
        4: [1.04, 22.1598, 34.0095, 46.52, 61.4763, 185.7],  # total_target
    }

    # Inverse frequency weights (normalized to mean 1)
    BIN_WEIGHTS = {
        0: [1.8172, 0.4543, 0.9137, 0.9137, 0.9010],  # clover_target (imbalanced)
        1: [0.9915, 1.0199, 0.9915, 1.0055, 0.9915],  # dead_target
        2: [0.9916, 1.0056, 1.0056, 1.0056, 0.9916],  # green_target
        3: [0.9916, 1.0056, 1.0056, 1.0056, 0.9916],  # gdm_target
        4: [0.9916, 1.0056, 1.0056, 1.0056, 0.9916],  # total_target
    }

    def __init__(self):
        super().__init__()
        self.class_weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none")

        # Register as buffers for proper device handling
        for i in range(5):
            self.register_buffer(
                f"bin_edges_{i}",
                torch.tensor(self.BIN_EDGES[i], dtype=torch.float32),
            )
            self.register_buffer(
                f"bin_weights_{i}",
                torch.tensor(self.BIN_WEIGHTS[i], dtype=torch.float32),
            )

    def _get_sample_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """Get quantile-based weights for each sample and target."""
        B, T = targets.shape  # (batch_size, 5)
        sample_weights = torch.ones_like(targets)

        for t in range(T):
            edges = getattr(self, f"bin_edges_{t}")
            weights = getattr(self, f"bin_weights_{t}")

            # Find bin index for each sample (0 to n_bins-1)
            # torch.bucketize returns index where value would be inserted
            # Using edges[1:-1] gives us internal boundaries
            bin_idx = torch.bucketize(targets[:, t].contiguous(), edges[1:-1])
            bin_idx = torch.clamp(bin_idx, 0, len(weights) - 1)

            sample_weights[:, t] = weights[bin_idx]

        return sample_weights

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_per_element = self.smooth_l1_loss(preds, targets)  # (B, 5)

        # Apply class weights (per-target importance)
        weighted_loss = self.class_weights.to(preds.device) * loss_per_element

        # Apply quantile-based sample weights
        sample_weights = self._get_sample_weights(targets)  # (B, 5)
        weighted_loss = sample_weights * weighted_loss

        return torch.mean(weighted_loss)


class InverseWeightedSmoothL1Loss(nn.Module):
    """GT値の逆数で重み付けするSmoothL1Loss.

    高GT値サンプルの勾配支配を抑制し、低〜中GT値サンプルの学習を強化。
    w = 1 / (1 + gt / median)
    """

    # 各ターゲットのmedian（train.csvから算出）
    MEDIANS = [1.42, 7.98, 20.80, 27.11, 40.30]

    def __init__(self, w_min: float = 0.3, w_max: float = 2.0):
        super().__init__()
        self.class_weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none")
        self.register_buffer("medians", torch.tensor(self.MEDIANS, dtype=torch.float32))
        self.w_min = w_min
        self.w_max = w_max

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_per_element = self.smooth_l1_loss(preds, targets)  # (B, 5)

        # Apply class weights (per-target importance)
        weighted_loss = self.class_weights.to(preds.device) * loss_per_element

        # Compute inverse weights based on GT values
        sample_w = 1.0 / (1.0 + targets / self.medians)  # (B, 5)
        sample_w = torch.clamp(sample_w, self.w_min, self.w_max)
        sample_w = sample_w / sample_w.mean()  # normalize to mean 1

        weighted_loss = sample_w * weighted_loss

        return torch.mean(weighted_loss)


class AuxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, aux_preds: torch.Tensor, aux_targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(aux_preds, aux_targets)
