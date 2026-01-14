import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Weight rationale: Prioritize Dry_Total (0.5) as the primary target,
        # GDM (0.2) as secondary, and individual components (0.1 each) equally.
        # Register as buffer to avoid repeated device transfers during forward pass
        self.register_buffer("weight", torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5]))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Validate inputs to prevent numerical instability
        assert not torch.isnan(preds).any(), "NaN detected in predictions"
        assert not torch.isinf(preds).any(), "Inf detected in predictions"
        assert not torch.isnan(targets).any(), "NaN detected in targets"
        assert not torch.isinf(targets).any(), "Inf detected in targets"

        return torch.mean(self.weight * (preds - targets) ** 2)


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # Weight rationale: Prioritize Dry_Total (0.5) as the primary target,
        # GDM (0.2) as secondary, and individual components (0.1 each) equally.
        # Register as buffer to avoid repeated device transfers during forward pass
        self.register_buffer("weight", torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5]))
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Validate inputs to prevent numerical instability
        assert not torch.isnan(preds).any(), "NaN detected in predictions"
        assert not torch.isinf(preds).any(), "Inf detected in predictions"
        assert not torch.isnan(targets).any(), "NaN detected in targets"
        assert not torch.isinf(targets).any(), "Inf detected in targets"

        loss_per_element = self.smooth_l1_loss(preds, targets)  # (B, 5)
        weighted_loss = self.weight * loss_per_element  # (B, 5)
        return torch.mean(weighted_loss)


class AuxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, aux_preds: torch.Tensor, aux_targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(aux_preds, aux_targets)
