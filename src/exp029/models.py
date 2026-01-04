import timm
import torch
import torch.nn as nn


class GeM(nn.Module):
    """Generalized Mean Pooling.

    GeM pooling is a generalization of average and max pooling.
    - p=1: Average Pooling
    - p→∞: Max Pooling
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if learnable else p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_tiles, features)
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)


class CSIROModel(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3",
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 3,  # [clover, dead, green]
        aux_out_channels: int = 2,
        freeze_backbone: bool = True,
        use_tile: bool = True,
        tile_grid: int = 2,  # 2×2 grid
        gem_p: float = 3.0,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.use_tile = use_tile
        self.tile_grid = tile_grid
        self.num_tiles = tile_grid * tile_grid

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

        # Tile処理用のGeM Pooling
        if use_tile:
            self.gem_pool = GeM(p=gem_p, learnable=True)

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

    def _split_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """Split image into tiles.

        Args:
            x: (B, C, H, W)

        Returns:
            tiles: (B * num_tiles, C, tile_h, tile_w)
        """
        B, C, H, W = x.shape
        tile_h = H // self.tile_grid
        tile_w = W // self.tile_grid

        # (B, C, H, W) -> (B, C, grid, tile_h, grid, tile_w)
        x = x.view(B, C, self.tile_grid, tile_h, self.tile_grid, tile_w)
        # -> (B, grid, grid, C, tile_h, tile_w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # -> (B * num_tiles, C, tile_h, tile_w)
        x = x.view(B * self.num_tiles, C, tile_h, tile_w)

        return x

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone."""
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

        return features

    def forward(self, x: torch.Tensor):
        B = x.shape[0]

        if self.use_tile:
            # Tile処理: 画像を分割して各タイルから特徴抽出
            tiles = self._split_tiles(x)  # (B * num_tiles, C, tile_h, tile_w)
            tile_features = self._extract_features(tiles)  # (B * num_tiles, feat_dim)

            # reshape: (B, num_tiles, feat_dim)
            tile_features = tile_features.view(B, self.num_tiles, -1)

            # GeM Pooling: (B, num_tiles, feat_dim) -> (B, feat_dim)
            features = self.gem_pool(tile_features)
        else:
            # 通常処理（Tileなし）
            features = self._extract_features(x)

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
        params = list(self.head.parameters()) + list(self.aux_head.parameters())
        if self.use_tile:
            params += list(self.gem_pool.parameters())
        if not self.freeze_backbone:
            params = list(self.parameters())
        return params


if __name__ == "__main__":
    from argparse import ArgumentParser

    import torch

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3")
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--use_tile", action="store_true", default=True)
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
        use_tile=args.use_tile,
    ).to(device)
    model.eval()

    # パラメータ数の確認
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    batch_size = 4
    channels = 3
    height = 512  # Tile処理用に512に変更
    width = 512

    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    with torch.no_grad():
        pred, aux_pred = model(dummy_input)
    print(f"pred shape: {pred.shape}")
    print(f"aux_pred shape: {aux_pred.shape}")
