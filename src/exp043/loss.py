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


class AuxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, aux_preds: torch.Tensor, aux_targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(aux_preds, aux_targets)
