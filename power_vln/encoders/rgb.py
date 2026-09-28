"""RGB feature-map to spatial-token encoder."""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .projector import TokenProjector
from .token_batch import TokenBatch


class RgbTokenEncoder(nn.Module):
    """Convert Open-Nav/Qwen visual feature maps into aligned model tokens.

    This module deliberately starts after the image backbone. It accepts either
    the Open-Nav ResNet feature map or an equivalent Qwen visual feature map,
    which keeps heavyweight backbone loading outside the trainable fusion code.
    """

    def __init__(
        self,
        source_feature_size: int = 2048,
        output_size: int = 2048,
        output_tokens: int = 196,
    ) -> None:
        super().__init__()
        if output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        self.output_tokens = output_tokens
        self.grid_height = int(math.floor(math.sqrt(output_tokens)))
        self.grid_width = int(math.ceil(output_tokens / self.grid_height))
        self.projector = TokenProjector(source_feature_size, output_size)

    def forward(
        self,
        feature_map: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> TokenBatch:
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [B, C, H, W]")
        batch_size, _, source_height, source_width = feature_map.shape
        pooled = F.adaptive_avg_pool2d(
            feature_map, (self.grid_height, self.grid_width)
        )
        source_tokens = pooled.flatten(2).transpose(1, 2)[:, : self.output_tokens]
        features = self.projector(source_tokens)

        y = torch.linspace(
            -1.0, 1.0, self.grid_height, device=feature_map.device,
            dtype=feature_map.dtype,
        )
        x = torch.linspace(
            -1.0, 1.0, self.grid_width, device=feature_map.device,
            dtype=feature_map.dtype,
        )
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        positions = torch.stack(
            (grid_x.flatten(), grid_y.flatten(), torch.zeros_like(grid_x).flatten()),
            dim=-1,
        )[: self.output_tokens]
        positions = positions.unsqueeze(0).expand(batch_size, -1, -1)

        if valid_mask is None:
            mask = torch.ones(
                batch_size, self.output_tokens, dtype=torch.bool,
                device=feature_map.device,
            )
        else:
            if valid_mask.shape != (batch_size, source_height, source_width):
                raise ValueError("valid_mask must have shape [B, H, W]")
            pooled_mask = F.adaptive_max_pool2d(
                valid_mask.to(dtype=feature_map.dtype).unsqueeze(1),
                (self.grid_height, self.grid_width),
            )
            mask = pooled_mask.flatten(1)[:, : self.output_tokens] > 0

        confidence = mask.to(dtype=features.dtype)
        features = features * mask.unsqueeze(-1).to(dtype=features.dtype)
        return TokenBatch(features, positions, mask, confidence, "rgb")
