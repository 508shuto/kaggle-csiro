import timm
import torch
import torch.nn as nn


class CSIROModel(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 5,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=out_channels,
        )

    def forward(self, x):
        # (B, C, H, W) -> (B, 5)
        x = self.model(x)
        return x


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
    ).to(device)
    model.eval()

    batch_size = 4
    channels = 3
    height = 256
    width = 256

    dummy_input = torch.randn(batch_size, channels, height, width).to(device)

    with torch.no_grad():
        output = model(dummy_input)
    print(output.shape)
