import timm
import torch
import torch.nn as nn


class CSIROModel(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3",
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 3,  # [clover, dead, green]
        aux_out_channels: int = 2,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone

        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            global_pool="",
            num_classes=0,
        )

        # Backbone をフリーズ
        if freeze_backbone:
            self.model.requires_grad_(False)
            self.model.eval()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(self.model.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_channels),
        )

        self.aux_head = nn.Sequential(
            nn.Linear(self.model.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, aux_out_channels),
        )

    def train(self, mode: bool = True):
        """Override train to keep backbone in eval mode when frozen."""
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self

    def forward(self, x):
        # Backbone は no_grad でフリーズ時の計算を高速化
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.model(x)
        else:
            features = self.model(x)

        # ViT の場合は (B, N, C) で出力されるので pooling
        if features.dim() == 3:
            # (B, N, C) -> (B, C)
            features = features.mean(dim=1)
        else:
            # CNN の場合は (B, C, H, W)
            features = self.pool(features).flatten(1)

        # 3つ予測: [Clover, Dead, Green] (raw空間)
        pred_3 = self.head(features)  # (B, 3)

        # 負値を除去
        pred_3 = torch.clamp(pred_3, min=0.0)  # (B, 3)

        # 物理制約を厳密に守る
        # GDM = Clover + Green
        # Total = Clover + Dead + Green
        clover = pred_3[:, 0]  # (B,)
        dead = pred_3[:, 1]  # (B,)
        green = pred_3[:, 2]  # (B,)
        gdm = clover + green  # (B,)
        total_g = clover + dead + green  # (B,)

        # 5つに拡張: [Clover, Dead, Green, GDM, Total]
        pred = torch.stack([clover, dead, green, gdm, total_g], dim=1)  # (B, 5)

        # 補助ターゲットもraw空間で予測
        aux_pred = self.aux_head(features)  # (B, 2)

        return pred, aux_pred

    def get_trainable_parameters(self):
        """Return only trainable parameters (heads only when backbone is frozen)."""
        if self.freeze_backbone:
            return list(self.head.parameters()) + list(self.aux_head.parameters())
        else:
            return list(self.parameters())


if __name__ == "__main__":
    from argparse import ArgumentParser

    import torch

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3")
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    model = CSIROModel(
        model_name=args.model_name,
        pretrained=args.pretrained,
        in_channels=args.in_channels,
        out_channels=3,
        aux_out_channels=2,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    model.eval()

    # パラメータ数の確認
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    batch_size = 4
    channels = 3
    height = 256
    width = 256

    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    with torch.no_grad():
        pred, aux_pred = model(dummy_input)
    print(f"pred shape: {pred.shape}")
    print(f"aux_pred shape: {aux_pred.shape}")
