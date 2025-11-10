import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.weight.to(preds.device) * (preds - targets) ** 2)


class WeightedHuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])
        self.delta = torch.tensor(1.0)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            self.weight.to(preds.device)
            * torch.where(
                torch.abs(preds - targets) < self.delta,
                (preds - targets) ** 2,
                self.delta * (torch.abs(preds - targets) - 0.5 * self.delta),
            )
        )


class AuxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(
        self, aux_preds: torch.Tensor, aux_targets: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(aux_preds, aux_targets)
