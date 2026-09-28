"""Translation, orientation, time, and uncertainty pose encoder."""

import math
from typing import Optional

import torch
from torch import nn

from .token_batch import TokenBatch


def quaternion_xyzw_to_rot6d(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized-or-normalizable xyzw quaternions to continuous Rot6D."""

    if quaternion.ndim != 2 or quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have shape [B, 4] in xyzw order")
    if not torch.isfinite(quaternion).all():
        raise ValueError("quaternion must be finite")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if torch.any(norm < 1e-8):
        raise ValueError("quaternion norm must be non-zero")
    x, y, z, w = (quaternion / norm).unbind(dim=-1)
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    return rotation[:, :, :2].transpose(1, 2).reshape(-1, 6)


class PoseTokenEncoder(nn.Module):
    """Encode one world-frame robot pose as one foundation-model token."""

    def __init__(
        self,
        output_size: int = 2048,
        fourier_bands: int = 8,
        include_timestamp: bool = True,
        include_covariance: bool = True,
    ) -> None:
        super().__init__()
        if output_size <= 0 or fourier_bands <= 0:
            raise ValueError("output_size and fourier_bands must be positive")
        self.fourier_bands = fourier_bands
        self.include_timestamp = include_timestamp
        self.include_covariance = include_covariance
        input_size = 3 + 3 * 2 * fourier_bands + 6
        input_size += 1 if include_timestamp else 0
        input_size += 36 if include_covariance else 0
        hidden_size = min(output_size, 512)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

    def _fourier(self, translation: torch.Tensor) -> torch.Tensor:
        frequencies = (2.0 ** torch.arange(
            self.fourier_bands,
            device=translation.device,
            dtype=translation.dtype,
        )) * math.pi
        phase = translation.unsqueeze(-1) * frequencies
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(1)

    def forward(
        self,
        translation: torch.Tensor,
        quaternion_xyzw: torch.Tensor,
        timestamp_s: Optional[torch.Tensor] = None,
        covariance: Optional[torch.Tensor] = None,
    ) -> TokenBatch:
        if translation.ndim != 2 or translation.shape[-1] != 3:
            raise ValueError("translation must have shape [B, 3]")
        if not torch.isfinite(translation).all():
            raise ValueError("translation must be finite")
        if quaternion_xyzw.shape[0] != translation.shape[0]:
            raise ValueError("translation and quaternion batch sizes must match")

        parts = [translation, self._fourier(translation)]
        parts.append(quaternion_xyzw_to_rot6d(quaternion_xyzw))
        batch_size = translation.shape[0]

        if self.include_timestamp:
            if timestamp_s is None:
                raise ValueError("timestamp_s is required by this encoder")
            timestamp_s = timestamp_s.reshape(batch_size, 1).to(translation)
            if not torch.isfinite(timestamp_s).all():
                raise ValueError("timestamp_s must be finite")
            parts.append(timestamp_s)

        flat_covariance = None
        if self.include_covariance:
            if covariance is None:
                raise ValueError("covariance is required by this encoder")
            if covariance.shape == (batch_size, 6, 6):
                flat_covariance = covariance.reshape(batch_size, 36)
            elif covariance.shape == (batch_size, 36):
                flat_covariance = covariance
            else:
                raise ValueError("covariance must have shape [B, 6, 6] or [B, 36]")
            flat_covariance = flat_covariance.to(translation)
            if not torch.isfinite(flat_covariance).all():
                raise ValueError("covariance must be finite")
            parts.append(flat_covariance)

        features = self.encoder(torch.cat(parts, dim=-1)).unsqueeze(1)
        positions = translation.unsqueeze(1)
        mask = torch.ones(batch_size, 1, dtype=torch.bool, device=translation.device)
        if flat_covariance is None:
            confidence = torch.ones(
                batch_size, 1, dtype=features.dtype, device=translation.device
            )
        else:
            diagonal = flat_covariance.reshape(batch_size, 6, 6).diagonal(
                dim1=-2, dim2=-1
            )
            uncertainty = diagonal.clamp_min(0).mean(dim=-1, keepdim=True)
            confidence = torch.exp(-uncertainty).to(features.dtype)
        return TokenBatch(features, positions, mask, confidence, "pose")
