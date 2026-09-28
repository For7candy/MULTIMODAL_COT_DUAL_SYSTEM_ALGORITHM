"""Common tensor contract emitted by every modality encoder."""

from dataclasses import dataclass

import torch


@dataclass
class TokenBatch:
    """A padded batch of spatial tokens in the foundation-model hidden space.

    Shapes are ``features=[B, N, D]``, ``positions=[B, N, 3]``, and
    ``mask/confidence=[B, N]``. ``mask=True`` marks a real token rather than
    padding. Positions are geometry anchors in the modality frame: RGB uses the
    normalized image plane, while point-cloud and pose tokens use their resolved
    metric reference frame. The fusion stage is responsible for frame alignment.
    """

    features: torch.Tensor
    positions: torch.Tensor
    mask: torch.Tensor
    confidence: torch.Tensor
    modality: str

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must have shape [B, N, D]")
        batch_size, token_count, _ = self.features.shape
        if self.positions.shape != (batch_size, token_count, 3):
            raise ValueError("positions must have shape [B, N, 3]")
        if self.mask.shape != (batch_size, token_count):
            raise ValueError("mask must have shape [B, N]")
        if self.confidence.shape != (batch_size, token_count):
            raise ValueError("confidence must have shape [B, N]")
        if self.mask.dtype != torch.bool:
            raise ValueError("mask must use torch.bool")
        if not isinstance(self.modality, str) or not self.modality.strip():
            raise ValueError("modality must not be empty")
        if self.features.device != self.positions.device:
            raise ValueError("features and positions must share a device")
        if self.features.device != self.mask.device:
            raise ValueError("features and mask must share a device")
        if self.features.device != self.confidence.device:
            raise ValueError("features and confidence must share a device")
        if not torch.isfinite(self.features).all():
            raise ValueError("features must be finite")
        if not torch.isfinite(self.positions).all():
            raise ValueError("positions must be finite")
        if not torch.isfinite(self.confidence).all():
            raise ValueError("confidence must be finite")
        if torch.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("confidence must be within [0, 1]")

    @property
    def hidden_size(self) -> int:
        return self.features.shape[-1]

    @property
    def token_count(self) -> int:
        return self.features.shape[1]
