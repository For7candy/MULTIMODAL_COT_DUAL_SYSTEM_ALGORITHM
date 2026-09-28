"""Projection layers shared by RGB, point-cloud, and pose encoders."""

from torch import nn


class TokenProjector(nn.Module):
    """Project one modality into the language-model hidden space."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        if input_size <= 0 or output_size <= 0:
            raise ValueError("projector dimensions must be positive")
        self.layers = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, output_size),
            nn.GELU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, features):
        return self.layers(features)
