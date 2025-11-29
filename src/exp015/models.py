import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionMIL(nn.Module):
    """Gated Attention MIL (Ilse et al., 2018) for bag-level prediction.

    Args:
        in_features: Input feature dimension
        hidden_dim: Hidden dimension for attention network
    """

    def __init__(self, in_features: int, hidden_dim: int = 256):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, N, D) bag of instances

        Returns:
            bag_feature: (B, D) aggregated bag representation
            attention_weights: (B, N) attention weights for each instance
        """
        # (B, N, D) -> (B, N, hidden_dim)
        V = self.attention_V(x)
        U = self.attention_U(x)
        # Element-wise multiplication
        gated = V * U  # (B, N, hidden_dim)
        # Attention weights
        attn = self.attention_weights(gated)  # (B, N, 1)
        attn = torch.softmax(attn, dim=1)  # (B, N, 1)

        # Weighted aggregation
        bag_feature = torch.sum(attn * x, dim=1)  # (B, D)

        return bag_feature, attn.squeeze(-1)  # (B, D), (B, N)


class CSIROModel(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 3,  # [clover, dead, green]
        aux_out_channels: int = 2,
        patch_size: int | None = None,
        stride: int | None = None,
        attn_hidden_dim: int = 256,
    ):
        super().__init__()
        self.use_mil = patch_size is not None and stride is not None
        self.patch_size = patch_size
        self.stride = stride

        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            global_pool="",
            num_classes=0,
        )

        if self.use_mil:
            # MILモード: パッチレベルのエンコーダ + Gated Attention
            self.pool = None  # MILでは使わない
            self.attention_mil = GatedAttentionMIL(
                in_features=self.model.num_features,
                hidden_dim=attn_hidden_dim,
            )
        else:
            # 通常モード: グローバル平均プーリング
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.attention_mil = None

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

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patches from input image using unfold.

        Args:
            x: (B, C, H, W) input image

        Returns:
            patches: (B, N, C, patch_size, patch_size) where N is number of patches
        """
        B, C, H, W = x.shape
        assert self.patch_size is not None and self.stride is not None, "patch_size and stride must be set for MIL mode"

        # Unfold: (B, C, H, W) -> (B, C * patch_size * patch_size, num_patches)
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)  # (B, C*P*P, N)

        # Reshape to (B, N, C, patch_size, patch_size)
        num_patches = patches.shape[2]
        patches = patches.view(B, C, self.patch_size, self.patch_size, num_patches)
        patches = patches.permute(0, 4, 1, 2, 3).contiguous()  # (B, N, C, patch_size, patch_size)

        return patches

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, C, H, W) input image

        Returns:
            pred_log: (B, 5) log空間の予測 [Clover, Dead, Green, GDM, Total]
            aux_pred_log: (B, 2) log空間の補助ターゲット予測
        """
        if self.use_mil:
            # MILモード: パッチ抽出 -> エンコード -> Attention集約
            B, C, H, W = x.shape

            # パッチ抽出: (B, C, H, W) -> (B, N, C, patch_size, patch_size)
            patches = self._extract_patches(x)  # (B, N, C, P, P)
            N = patches.shape[1]

            # バッチ×パッチをまとめて処理: (B*N, C, P, P)
            patches_flat = patches.reshape(B * N, C, self.patch_size, self.patch_size)

            # 各パッチをエンコード: (B*N, C, P, P) -> (B*N, D, H', W')
            patch_features = self.model(patches_flat)

            # グローバル平均プーリングでパッチレベルの特徴を抽出
            patch_pooled = F.adaptive_avg_pool2d(patch_features, 1).flatten(1)  # (B*N, D)

            # バッチ×パッチに戻す: (B, N, D)
            patch_features_bag = patch_pooled.view(B, N, -1)

            # Gated Attention MILで集約: (B, N, D) -> (B, D)
            x = self.attention_mil(patch_features_bag)[0]  # (B, D)
        else:
            # 通常モード: グローバル平均プーリング
            x = self.model(x)
            x = self.pool(x).flatten(1)

        # 3つ予測: [Clover, Dead, Green] (log空間)
        pred_3_log = self.head(x)  # (B, 3)

        # 元の空間に戻して負値を除去
        pred_3 = torch.clamp(torch.expm1(pred_3_log), min=0.0)  # (B, 3)

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

        # log空間に戻す
        pred_log = torch.log1p(pred)  # (B, 5)

        # 補助ターゲットもlog空間で予測
        aux_pred_log = self.aux_head(x)  # (B, 2)

        return pred_log, aux_pred_log


if __name__ == "__main__":
    from argparse import ArgumentParser

    import torch

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--attn_hidden_dim", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    model = CSIROModel(
        model_name=args.model_name,
        pretrained=args.pretrained,
        in_channels=args.in_channels,
        out_channels=3,
        aux_out_channels=2,
        patch_size=args.patch_size,
        stride=args.stride,
        attn_hidden_dim=args.attn_hidden_dim,
    ).to(device)
    model.eval()

    batch_size = 4
    channels = args.in_channels
    height = 256
    width = 256

    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    with torch.no_grad():
        pred, aux_pred = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Pred shape: {pred.shape}")
    print(f"Aux pred shape: {aux_pred.shape}")
    if args.patch_size is not None:
        print(f"MIL mode: patch_size={args.patch_size}, stride={args.stride}")
    else:
        print("Standard mode (no MIL)")
