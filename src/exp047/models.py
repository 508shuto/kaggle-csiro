import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModel


class CSIROModel(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 3,  # [clover, dead, green]
        aux_out_channels: int = 2,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: list[str] | None = None,
        lora_layers_to_transform: list[int] | None = None,
    ):
        super().__init__()
        self.use_lora = use_lora

        # HF AutoModel/AutoConfig（DINOv2/v3両対応）
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)

        self.hidden_size = self.backbone.config.hidden_size  # 1024

        # LoRA適用（最後の6レイヤーのみ）
        if use_lora:
            if lora_target_modules is None:
                # DINOv3のモジュール構造に合わせた正規表現
                # layer.18-23のattention層のみにLoRAを適用
                lora_target_modules = r"layer\.(1[89]|2[0-3])\.attention\.(q|k|v|o)_proj"

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()

        # ヘッド
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_channels),
        )

        self.aux_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, aux_out_channels),
        )

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (B, 1024)

        # 3つ予測: [Clover, Dead, Green] (raw空間)
        pred_3 = self.head(cls_token)  # (B, 3)

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
        aux_pred = self.aux_head(cls_token)  # (B, 2)

        return pred, aux_pred

    def get_trainable_parameters(self):
        """Return trainable parameters grouped by type."""
        if self.use_lora:
            lora_params = []
            head_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "lora_" in name:
                    lora_params.append(param)
                else:
                    head_params.append(param)
            return {"lora": lora_params, "head": head_params}
        else:
            return list(self.parameters())


if __name__ == "__main__":
    from argparse import ArgumentParser

    import torch

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    model = CSIROModel(
        model_name=args.model_name,
        pretrained=args.pretrained,
        in_channels=args.in_channels,
        out_channels=3,
        aux_out_channels=2,
        use_lora=args.use_lora,
    ).to(device)
    model.eval()

    # パラメータ数の確認
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.get_trainable_parameters()
    if isinstance(trainable_params, dict):
        lora_count = sum(p.numel() for p in trainable_params["lora"])
        head_count = sum(p.numel() for p in trainable_params["head"])
        print(f"Total parameters: {total_params:,}")
        print(f"LoRA parameters: {lora_count:,}")
        print(f"Head parameters: {head_count:,}")
        print(f"Trainable parameters: {lora_count + head_count:,}")
    else:
        trainable_count = sum(p.numel() for p in trainable_params)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_count:,}")

    batch_size = 4
    channels = 3
    height = 512
    width = 512

    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    with torch.no_grad():
        pred, aux_pred = model(dummy_input)
    print(f"pred shape: {pred.shape}")
    print(f"aux_pred shape: {aux_pred.shape}")
