import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self, weights: list[float] | None = None):
        super().__init__()
        if weights is None:
            weights = [0.1, 0.1, 0.1, 0.2, 0.5]
        self.weight = torch.tensor(weights)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.weight.to(preds.device) * (preds - targets) ** 2)


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, weights: list[float] | None = None):
        super().__init__()
        if weights is None:
            weights = [0.1, 0.1, 0.1, 0.2, 0.5]
        self.weight = torch.tensor(weights)
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute SmoothL1Loss per element, then weight and average
        loss_per_element = self.smooth_l1_loss(preds, targets)  # (B, 5)
        weighted_loss = self.weight.to(preds.device) * loss_per_element  # (B, 5)
        return torch.mean(weighted_loss)


class AuxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, aux_preds: torch.Tensor, aux_targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(aux_preds, aux_targets)
