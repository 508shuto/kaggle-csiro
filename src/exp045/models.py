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
        unfreeze_blocks: list[int] | None = None,  # Partial Unfreeze用
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.unfreeze_blocks = unfreeze_blocks or []

        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            global_pool="",
            num_classes=0,
        )

        # Backbone の設定
        if freeze_backbone and not unfreeze_blocks:
            # 完全フリーズ
            self.model.requires_grad_(False)
            self.model.eval()
        elif unfreeze_blocks:
            # Partial Unfreeze: 指定ブロックのみ学習可能
            self._setup_partial_unfreeze()

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

    def _setup_partial_unfreeze(self):
        """指定されたブロックのみ学習可能にする"""
        # まず全てフリーズ
        self.model.requires_grad_(False)

        # 指定ブロックのみunfreeze
        for name, param in self.model.named_parameters():
            for block_idx in self.unfreeze_blocks:
                if f"blocks.{block_idx}." in name:
                    param.requires_grad = True
                    break

        # 学習可能パラメータ数をログ
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Partial Unfreeze: {trainable:,} / {total:,} backbone params trainable")
        print(f"Unfreeze blocks: {self.unfreeze_blocks}")

    def train(self, mode: bool = True):
        """Override train to handle partial unfreeze correctly."""
        super().train(mode)
        if self.freeze_backbone and not self.unfreeze_blocks:
            # 完全フリーズの場合はevalモード維持
            self.model.eval()
        elif self.unfreeze_blocks:
            # Partial Unfreezeの場合、フリーズ部分のみevalモード
            for name, module in self.model.named_modules():
                is_unfrozen = any(f"blocks.{idx}" in name for idx in self.unfreeze_blocks)
                if not is_unfrozen and hasattr(module, "training"):
                    module.eval()
        return self

    def forward(self, x):
        # Partial Unfreezeの場合はno_gradを使わない
        if self.freeze_backbone and not self.unfreeze_blocks:
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
        """Return trainable parameters grouped by type for different learning rates."""
        if self.unfreeze_blocks:
            # Partial Unfreeze: backbone と head を分けて返す
            backbone_params = []
            head_params = []

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "head" in name or "aux_head" in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

            return {
                "backbone": backbone_params,
                "head": head_params,
            }
        elif self.freeze_backbone:
            # 完全フリーズ: head のみ
            return list(self.head.parameters()) + list(self.aux_head.parameters())
        else:
            # 全て学習
            return list(self.parameters())


if __name__ == "__main__":
    from argparse import ArgumentParser

    import torch

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vit_large_patch16_dinov3.lvd1689m")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--unfreeze_blocks", nargs="+", type=int, default=[22, 23])
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
        unfreeze_blocks=args.unfreeze_blocks,
    ).to(device)
    model.eval()

    # パラメータ数の確認
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 学習可能パラメータのグループ確認
    trainable = model.get_trainable_parameters()
    if isinstance(trainable, dict):
        print(f"Backbone params: {sum(p.numel() for p in trainable['backbone']):,}")
        print(f"Head params: {sum(p.numel() for p in trainable['head']):,}")

    batch_size = 4
    channels = 3
    height = 512
    width = 512

    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    with torch.no_grad():
        pred, aux_pred = model(dummy_input)
    print(f"pred shape: {pred.shape}")
    print(f"aux_pred shape: {aux_pred.shape}")
