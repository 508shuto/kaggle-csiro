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
        hidden_dim: int = 2048,  # Qwen3-VL-2B out_hidden_size
        head_hidden_dim: int = 256,
        dropout: float = 0.2,
        out_channels: int = 3,  # [clover, dead, green]
        aux_out_channels: int = 2,  # [ndvi, height]
    ):
        super().__init__()

        # Load Qwen3-VL model
        if pretrained:
            # Load in float32 and let PyTorch Lightning handle mixed precision
            self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None,  # Manual device placement
            )
        else:
            from transformers import Qwen3VLConfig

            config = Qwen3VLConfig.from_pretrained(model_name)
            self.vlm = Qwen3VLForConditionalGeneration(config)

        # Extract vision encoder
        self.vision_encoder = self.vlm.model.visual

        # Freeze backbone if specified
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            # Freeze entire VLM (vision encoder + language model)
            # Only regression heads will be trainable
            for param in self.vlm.parameters():
                param.requires_grad = False

        # Validate hidden dimension matches vision encoder output
        actual_hidden_dim = self.vision_encoder.config.out_hidden_size
        assert hidden_dim == actual_hidden_dim, (
            f"Config hidden_dim ({hidden_dim}) must match Qwen3-VL out_hidden_size ({actual_hidden_dim})"
        )

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

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        """Forward pass.

        Args:
            pixel_values: Preprocessed image tensor from Dataset
            grid_thw: Grid information tensor (B, 3)

        Returns:
            pred: Predictions of shape (B, 5) [Clover, Dead, Green, GDM, Total]
            aux_pred: Auxiliary predictions of shape (B, 2) [NDVI, Height]
        """
        # Extract vision features using get_image_features
        # Returns tuple: (image_embeds, deepstack_embeds)
        image_embeds, _ = self.vlm.model.get_image_features(
            pixel_values.type(self.vlm.model.visual.dtype),
            grid_thw,
        )

        # image_embeds is a tuple of tensors, one per image
        # Each tensor has shape (num_patches, hidden_dim)
        # Pool over patches for each image
        pooled = torch.stack([emb.mean(dim=0) for emb in image_embeds], dim=0)  # (B, hidden_dim)

        # Convert to float32 for regression head
        pooled = pooled.float()

        # Predict 3 base targets: [Clover, Dead, Green]
        pred_3 = self.head(pooled)  # (B, 3)

        # Note: Clamping removed to preserve gradients during training
        # Non-negative constraint is applied in validation_step instead

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

    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor

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

    # Create dummy PIL images and process with AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    batch_size = 2
    height = 384
    width = 384

    dummy_images = [
        Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)) for _ in range(batch_size)
    ]
    processed = processor(images=dummy_images, return_tensors="pt")
    pixel_values = processed.pixel_values.to(device)
    grid_thw = processed.image_grid_thw.to(device)

    with torch.no_grad():
        pred, aux_pred = model(pixel_values, grid_thw)
    print(f"pred shape: {pred.shape}")  # Expected: (2, 5)
    print(f"aux_pred shape: {aux_pred.shape}")  # Expected: (2, 2)
