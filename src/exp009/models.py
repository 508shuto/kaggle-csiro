import timm
import torch
import torch.nn as nn


class CSIROModel(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 3,  # [clover, dead, green]
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            global_pool="",
            num_classes=0,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(self.model.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_channels),
        )

    def forward(self, x):
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

        return pred_log


if __name__ == "__main__":
    from argparse import ArgumentParser

    import torch

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    model = CSIROModel(
        model_name=args.model_name,
        pretrained=args.pretrained,
        in_channels=args.in_channels,
        out_channels=3,
    ).to(device)
    model.eval()

    batch_size = 4
    channels = 3
    height = 256
    width = 256

    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    with torch.no_grad():
        pred = model(dummy_input)
    print(pred.shape)
