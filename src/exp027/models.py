import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration


class Qwen3VLRegressionModel(nn.Module):
    """Qwen3-VL based regression model.

    Uses Qwen3-VL vision encoder as feature extractor and adds
    regression heads for predicting biomass targets.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        hidden_dim: int = 1536,
        head_hidden_dim: int = 256,
        dropout: float = 0.2,
        out_channels: int = 3,  # [clover, dead, green]
        aux_out_channels: int = 2,  # [ndvi, height]
    ):
        super().__init__()

        # Load Qwen3-VL model
        if pretrained:
            self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=None,  # Manual device placement
            )
        else:
            from transformers import Qwen3VLConfig

            config = Qwen3VLConfig.from_pretrained(model_name)
            self.vlm = Qwen3VLForConditionalGeneration(config)

        # Extract vision encoder
        self.vision_encoder = self.vlm.model.vision_model

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # Get hidden dimension from vision encoder
        self.hidden_dim = hidden_dim

        # Regression head for main targets (predicts 3: clover, dead, green)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, out_channels),
        )

        # Auxiliary head for NDVI and Height
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, aux_out_channels),
        )

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor | None = None):
        """Forward pass.

        Args:
            pixel_values: Image tensor of shape (B, C, H, W)
            grid_thw: Grid dimensions tensor for Qwen3-VL (optional)

        Returns:
            pred: Predictions of shape (B, 5) [Clover, Dead, Green, GDM, Total]
            aux_pred: Auxiliary predictions of shape (B, 2) [NDVI, Height]
        """
        # Get vision features from Qwen3-VL vision encoder
        # The vision encoder expects pixel_values and grid_thw
        if grid_thw is None:
            # Create default grid_thw for single images
            batch_size = pixel_values.shape[0]
            # Qwen3-VL expects grid_thw as (num_images, 3) where each row is [t, h, w]
            # For static images: t=1, h and w depend on image patches
            h = pixel_values.shape[2] // 14  # patch size is 14
            w = pixel_values.shape[3] // 14
            grid_thw = torch.tensor([[1, h, w]] * batch_size, device=pixel_values.device)

        # Extract vision features
        with torch.set_grad_enabled(not self.training or any(p.requires_grad for p in self.vision_encoder.parameters())):
            vision_outputs = self.vision_encoder(
                pixel_values=pixel_values,
                grid_thw=grid_thw,
            )

        # Get hidden states: shape (total_patches, hidden_dim)
        hidden_states = vision_outputs.last_hidden_state

        # Global average pooling over sequence dimension
        # For batched processing, we need to handle variable sequence lengths
        # For simplicity, take mean over all patches
        if hidden_states.dim() == 2:
            # Shape: (total_patches, hidden_dim) - need to reshape per batch
            # Use grid_thw to determine batch boundaries
            features = []
            start_idx = 0
            for i in range(grid_thw.shape[0]):
                t, h, w = grid_thw[i].tolist()
                num_patches = int(t * h * w)
                batch_hidden = hidden_states[start_idx : start_idx + num_patches]
                features.append(batch_hidden.mean(dim=0))
                start_idx += num_patches
            pooled = torch.stack(features, dim=0)  # (B, hidden_dim)
        else:
            # Shape: (B, seq_len, hidden_dim)
            pooled = hidden_states.mean(dim=1)  # (B, hidden_dim)

        # Convert to float32 for regression head
        pooled = pooled.float()

        # Predict 3 base targets: [Clover, Dead, Green]
        pred_3 = self.head(pooled)  # (B, 3)

        # Clamp to non-negative values
        pred_3 = torch.clamp(pred_3, min=0.0)

        # Apply physical constraints
        clover = pred_3[:, 0]  # (B,)
        dead = pred_3[:, 1]  # (B,)
        green = pred_3[:, 2]  # (B,)
        gdm = clover + green  # (B,)
        total_g = clover + dead + green  # (B,)

        # Stack to 5 outputs: [Clover, Dead, Green, GDM, Total]
        pred = torch.stack([clover, dead, green, gdm, total_g], dim=1)  # (B, 5)

        # Auxiliary predictions
        aux_pred = self.aux_head(pooled)  # (B, 2)

        return pred, aux_pred


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    model = Qwen3VLRegressionModel(
        model_name=args.model_name,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)
    model.eval()

    batch_size = 2
    channels = 3
    height = 384
    width = 384

    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    with torch.no_grad():
        pred, aux_pred = model(dummy_input)
    print(f"pred shape: {pred.shape}")  # Expected: (2, 5)
    print(f"aux_pred shape: {aux_pred.shape}")  # Expected: (2, 2)
