import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration


class CSIROModel(nn.Module):
    """Qwen3-VL-2B based regression model.

    Uses the vision encoder from Qwen3-VL to extract image features,
    then applies a regression head to predict biomass values.
    The vision encoder is frozen; only the regression head is trained.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 3,  # [clover, dead, green]
        aux_out_channels: int = 2,
        hidden_size: int = 1536,  # Qwen3-VL-2B visual hidden size
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.hidden_size = hidden_size

        # Load Qwen3-VL model
        if pretrained:
            self.encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            )
        else:
            raise ValueError("pretrained=False is not supported for Qwen3-VL")

        # Freeze encoder
        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Regression head (float32 for stability)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_channels),
        )

        # Auxiliary head
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, aux_out_channels),
        )

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        """Forward pass.

        Args:
            pixel_values: (N, C) flattened pixel values from processor
            image_grid_thw: (B, 3) tensor with [temporal, height_patches, width_patches]

        Returns:
            pred: (B, 5) predictions [clover, dead, green, gdm, total]
            aux_pred: (B, 2) auxiliary predictions
        """
        # Extract visual features from Qwen3-VL vision encoder
        with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
            # visual module expects (N, C) pixel_values and grid_thw
            visual_outputs = self.encoder.visual(
                pixel_values,
                grid_thw=image_grid_thw,
            )
            # visual_outputs: (total_patches, hidden_size)
            # We need to pool per image

            # Calculate number of patches per image from grid_thw
            batch_size = image_grid_thw.shape[0]
            patches_per_image = (image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()

            # Split and pool
            features_list = []
            start_idx = 0
            for n_patches in patches_per_image:
                img_features = visual_outputs[start_idx : start_idx + n_patches]  # (n_patches, hidden)
                pooled = img_features.mean(dim=0)  # (hidden,)
                features_list.append(pooled)
                start_idx += n_patches

            features = torch.stack(features_list, dim=0)  # (B, hidden_size)

        # Cast to float32 for head computation
        features = features.float()

        # Predict 3 base targets
        pred_3 = self.head(features)  # (B, 3)

        # Clamp to non-negative
        pred_3 = torch.clamp(pred_3, min=0.0)

        # Apply physical constraints
        clover = pred_3[:, 0]
        dead = pred_3[:, 1]
        green = pred_3[:, 2]
        gdm = clover + green
        total_g = clover + dead + green

        pred = torch.stack([clover, dead, green, gdm, total_g], dim=1)  # (B, 5)

        # Auxiliary predictions
        aux_pred = self.aux_head(features)  # (B, 2)

        return pred, aux_pred

    def get_trainable_parameters(self):
        """Return trainable parameters (head only)."""
        return [p for p in self.parameters() if p.requires_grad]


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading model...")
    model = CSIROModel(
        model_name=args.model_name,
        pretrained=True,
        freeze_encoder=True,
    ).to(device)
    model.eval()

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
