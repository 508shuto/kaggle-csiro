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


class FeatureFusion(nn.Module):
    """Multi Head Attention based feature fusion.

    Fuses main features with auxiliary prediction embeddings.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, main_feat: torch.Tensor, aux_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            main_feat: (B, dim) - main task features
            aux_feat: (B, dim) - auxiliary prediction embeddings

        Returns:
            fused: (B, dim) - fused features
        """
        # Stack as sequence: (B, 2, dim)
        x = torch.stack([main_feat, aux_feat], dim=1)

        # Self-attention with residual
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        x = self.norm2(x + self.dropout(self.ffn(x)))

        # Take main feature position (first token)
        return x[:, 0, :]


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
        # MHA fusion settings
        use_mha_fusion: bool = True,
        mha_dim: int = 256,
        mha_heads: int = 4,
        mha_dropout: float = 0.1,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.use_tile = use_tile
        self.tile_grid = tile_grid
        self.num_tiles = tile_grid * tile_grid
        self.use_mha_fusion = use_mha_fusion

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

        # Main projection: backbone features → mha_dim
        self.main_proj = nn.Sequential(
            nn.Linear(self.model.num_features, mha_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Auxiliary head: backbone features → aux predictions
        self.aux_head = nn.Sequential(
            nn.Linear(self.model.num_features, mha_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mha_dim, aux_out_channels),
        )

        if use_mha_fusion:
            # Auxiliary prediction embedding: aux_pred → mha_dim
            self.aux_embed = nn.Sequential(
                nn.Linear(aux_out_channels, mha_dim),
                nn.ReLU(),
            )

            # MHA fusion
            self.fusion = FeatureFusion(
                dim=mha_dim,
                num_heads=mha_heads,
                dropout=mha_dropout,
            )

            # Final head: fused features → predictions
            self.final_head = nn.Linear(mha_dim, out_channels)
        else:
            # Legacy head (no fusion)
            self.head = nn.Sequential(
                nn.Linear(self.model.num_features, mha_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(mha_dim, out_channels),
            )

    def train(self, mode: bool = True):
        """Override train to keep backbone in eval mode when frozen."""
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self

    def _split_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """Split image into tiles."""
        B, C, H, W = x.shape
        tile_h = H // self.tile_grid
        tile_w = W // self.tile_grid

        x = x.view(B, C, self.tile_grid, tile_h, self.tile_grid, tile_w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B * self.num_tiles, C, tile_h, tile_w)

        return x

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone."""
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.model(x)
        else:
            features = self.model(x)

        if features.dim() == 3:
            features = features.mean(dim=1)
        else:
            features = self.pool(features).flatten(1)

        return features

    def forward(self, x: torch.Tensor):
        B = x.shape[0]

        if self.use_tile:
            tiles = self._split_tiles(x)
            tile_features = self._extract_features(tiles)
            tile_features = tile_features.view(B, self.num_tiles, -1)
            features = self.gem_pool(tile_features)
        else:
            features = self._extract_features(x)

        # Auxiliary prediction (NDVI, height)
        aux_pred = self.aux_head(features)  # (B, 2)

        if self.use_mha_fusion:
            # Main projection
            main_feat = self.main_proj(features)  # (B, mha_dim)

            # Embed auxiliary predictions
            aux_feat = self.aux_embed(aux_pred)  # (B, mha_dim)

            # MHA fusion
            fused = self.fusion(main_feat, aux_feat)  # (B, mha_dim)

            # Final prediction
            pred_3 = self.final_head(fused)  # (B, 3)
        else:
            pred_3 = self.head(features)  # (B, 3)

        # Clamp to non-negative
        pred_3 = torch.clamp(pred_3, min=0.0)

        # Physical constraints
        clover = pred_3[:, 0]
        dead = pred_3[:, 1]
        green = pred_3[:, 2]
        gdm = clover + green
        total_g = clover + dead + green

        pred = torch.stack([clover, dead, green, gdm, total_g], dim=1)

        return pred, aux_pred

    def get_trainable_parameters(self):
        """Return only trainable parameters."""
        params = []

        # Main projection
        params += list(self.main_proj.parameters())

        # Auxiliary head
        params += list(self.aux_head.parameters())

        if self.use_mha_fusion:
            params += list(self.aux_embed.parameters())
            params += list(self.fusion.parameters())
            params += list(self.final_head.parameters())
        else:
            params += list(self.head.parameters())

        if self.use_tile:
            params += list(self.gem_pool.parameters())

        if not self.freeze_backbone:
            params = list(self.parameters())

        return params


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_dinov3")
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--use_tile", action="store_true", default=True)
    parser.add_argument("--use_mha_fusion", action="store_true", default=True)
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
        use_mha_fusion=args.use_mha_fusion,
    ).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    batch_size = 4
    channels = 3
    height = 512
    width = 512

    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    with torch.no_grad():
        pred, aux_pred = model(dummy_input)
    print(f"pred shape: {pred.shape}")
    print(f"aux_pred shape: {aux_pred.shape}")
