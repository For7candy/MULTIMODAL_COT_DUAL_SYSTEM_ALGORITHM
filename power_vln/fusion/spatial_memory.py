"""Deterministic confidence-weighted spatial token memory."""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch

from .cross_modal_fusion import FusedSceneTokens


VoxelKey = Tuple[int, int, int]


@dataclass
class _MemoryCell:
    features: torch.Tensor
    position: torch.Tensor
    confidence: float
    modality_weights: torch.Tensor
    uncertainty: float
    last_seen_ms: int
    observation_count: int


@dataclass(frozen=True)
class SpatialMemorySnapshot:
    version: int
    timestamp_ms: int
    frame_id: str
    voxel_keys: Tuple[VoxelKey, ...]
    features: torch.Tensor
    positions: torch.Tensor
    confidence: torch.Tensor
    modality_weights: torch.Tensor
    uncertainty: torch.Tensor
    last_seen_ms: torch.Tensor
    observation_count: torch.Tensor

    def __post_init__(self) -> None:
        token_count = len(self.voxel_keys)
        if self.version < 0 or self.timestamp_ms < 0:
            raise ValueError("version and timestamp_ms must be non-negative")
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if self.features.ndim != 2 or self.features.shape[0] != token_count:
            raise ValueError("features must have shape [N, D]")
        if self.positions.shape != (token_count, 3):
            raise ValueError("positions must have shape [N, 3]")
        if self.confidence.shape != (token_count,):
            raise ValueError("confidence must have shape [N]")
        if self.modality_weights.shape != (token_count, 2):
            raise ValueError("modality_weights must have shape [N, 2]")
        if self.uncertainty.shape != (token_count,):
            raise ValueError("uncertainty must have shape [N]")
        if self.last_seen_ms.shape != (token_count,):
            raise ValueError("last_seen_ms must have shape [N]")
        if self.observation_count.shape != (token_count,):
            raise ValueError("observation_count must have shape [N]")

    @property
    def token_count(self) -> int:
        return len(self.voxel_keys)

    @property
    def scene_id(self) -> str:
        return "memory-{:08d}".format(self.version)

    def to_fused_tokens(self) -> FusedSceneTokens:
        mask = torch.ones(1, self.token_count, dtype=torch.bool)
        return FusedSceneTokens(
            features=self.features.unsqueeze(0),
            positions=self.positions.unsqueeze(0),
            mask=mask,
            confidence=self.confidence.unsqueeze(0),
            modality_weights=self.modality_weights.unsqueeze(0),
            uncertainty=self.uncertainty.unsqueeze(0),
        )


class SpatialMemory:
    """World-frame voxel memory for one navigation stream.

    Runtime storage is detached on CPU by design: this is persistent inference
    state, not part of backpropagation through an unbounded history.
    """

    def __init__(
        self,
        feature_size: int,
        frame_id: str = "world",
        voxel_size_m: float = 0.25,
        confidence_half_life_ms: int = 5000,
        maximum_age_ms: int = 20000,
    ) -> None:
        if feature_size <= 0 or voxel_size_m <= 0:
            raise ValueError("feature_size and voxel_size_m must be positive")
        if confidence_half_life_ms <= 0 or maximum_age_ms <= 0:
            raise ValueError("memory time constants must be positive")
        if not frame_id:
            raise ValueError("frame_id must not be empty")
        self.feature_size = feature_size
        self.frame_id = frame_id
        self.voxel_size_m = voxel_size_m
        self.confidence_half_life_ms = confidence_half_life_ms
        self.maximum_age_ms = maximum_age_ms
        self._cells: Dict[VoxelKey, _MemoryCell] = {}
        self._version = 0
        self._timestamp_ms = 0

    def _decay(self, elapsed_ms: int) -> float:
        return math.exp(-math.log(2.0) * elapsed_ms / self.confidence_half_life_ms)

    def _expire(self, timestamp_ms: int) -> bool:
        expired = [
            key
            for key, cell in self._cells.items()
            if timestamp_ms - cell.last_seen_ms > self.maximum_age_ms
        ]
        for key in expired:
            del self._cells[key]
        return bool(expired)

    def _voxel_key(self, position: torch.Tensor) -> VoxelKey:
        coordinate = torch.floor(position / self.voxel_size_m).to(torch.int64)
        return tuple(int(value) for value in coordinate.tolist())

    def update(
        self,
        tokens: FusedSceneTokens,
        timestamp_ms: int,
        dynamic_mask: Optional[torch.Tensor] = None,
    ) -> SpatialMemorySnapshot:
        if timestamp_ms < self._timestamp_ms:
            raise ValueError("memory timestamps must be monotonic")
        if tokens.features.shape[0] != 1:
            raise ValueError("SpatialMemory supports one runtime stream per instance")
        if tokens.hidden_size != self.feature_size:
            raise ValueError("token hidden size does not match memory feature size")
        if dynamic_mask is None:
            dynamic_mask = torch.zeros_like(tokens.mask)
        if dynamic_mask.shape != tokens.mask.shape or dynamic_mask.dtype != torch.bool:
            raise ValueError("dynamic_mask must be bool with shape [1, N]")

        self._expire(timestamp_ms)
        features = tokens.features[0].detach().float().cpu()
        positions = tokens.positions[0].detach().float().cpu()
        confidence = tokens.confidence[0].detach().float().cpu()
        modality = tokens.modality_weights[0].detach().float().cpu()
        uncertainty = tokens.uncertainty[0].detach().float().cpu()
        valid = (tokens.mask & ~dynamic_mask)[0].detach().cpu()

        grouped = {}
        for index in torch.nonzero(valid, as_tuple=False).flatten().tolist():
            item_confidence = float(confidence[index])
            if item_confidence <= 0:
                continue
            key = self._voxel_key(positions[index])
            grouped.setdefault(key, []).append(index)

        for key in sorted(grouped):
            indices = grouped[key]
            weights = confidence[indices].clamp_min(1e-6)
            weight_sum = weights.sum()
            new_feature = (features[indices] * weights[:, None]).sum(0) / weight_sum
            new_position = (positions[indices] * weights[:, None]).sum(0) / weight_sum
            new_modality = (modality[indices] * weights[:, None]).sum(0) / weight_sum
            new_uncertainty = float((uncertainty[indices] * weights).sum() / weight_sum)
            new_confidence = float(1.0 - torch.prod(1.0 - confidence[indices]))
            observation_count = len(indices)
            previous = self._cells.get(key)
            if previous is None:
                self._cells[key] = _MemoryCell(
                    new_feature,
                    new_position,
                    new_confidence,
                    new_modality,
                    new_uncertainty,
                    timestamp_ms,
                    observation_count,
                )
                continue
            elapsed_ms = timestamp_ms - previous.last_seen_ms
            old_confidence = previous.confidence * self._decay(elapsed_ms)
            total_confidence = old_confidence + new_confidence
            new_weight = new_confidence / max(total_confidence, 1e-8)
            old_weight = 1.0 - new_weight
            merged_modality = old_weight * previous.modality_weights + new_weight * new_modality
            merged_modality = merged_modality / merged_modality.sum().clamp_min(1e-8)
            self._cells[key] = _MemoryCell(
                features=old_weight * previous.features + new_weight * new_feature,
                position=old_weight * previous.position + new_weight * new_position,
                confidence=min(1.0, 1.0 - (1.0 - old_confidence) * (1.0 - new_confidence)),
                modality_weights=merged_modality,
                uncertainty=old_weight * previous.uncertainty + new_weight * new_uncertainty,
                last_seen_ms=timestamp_ms,
                observation_count=previous.observation_count + observation_count,
            )

        self._timestamp_ms = timestamp_ms
        self._version += 1
        return self.snapshot(timestamp_ms)

    def advance(self, timestamp_ms: int) -> SpatialMemorySnapshot:
        """Advance time and expire stale cells without adding observations."""

        if timestamp_ms < self._timestamp_ms:
            raise ValueError("memory timestamps must be monotonic")
        changed = self._expire(timestamp_ms)
        self._timestamp_ms = timestamp_ms
        if changed:
            self._version += 1
        return self.snapshot(timestamp_ms)

    def snapshot(self, timestamp_ms: Optional[int] = None) -> SpatialMemorySnapshot:
        timestamp = self._timestamp_ms if timestamp_ms is None else timestamp_ms
        if timestamp < self._timestamp_ms:
            raise ValueError("snapshot timestamp cannot precede memory state")
        keys = tuple(sorted(self._cells))
        if not keys:
            return SpatialMemorySnapshot(
                self._version,
                timestamp,
                self.frame_id,
                (),
                torch.empty(0, self.feature_size),
                torch.empty(0, 3),
                torch.empty(0),
                torch.empty(0, 2),
                torch.empty(0),
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
            )
        cells = [self._cells[key] for key in keys]
        decayed_confidence = torch.tensor(
            [
                cell.confidence * self._decay(timestamp - cell.last_seen_ms)
                for cell in cells
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        return SpatialMemorySnapshot(
            version=self._version,
            timestamp_ms=timestamp,
            frame_id=self.frame_id,
            voxel_keys=keys,
            features=torch.stack([cell.features for cell in cells]),
            positions=torch.stack([cell.position for cell in cells]),
            confidence=decayed_confidence,
            modality_weights=torch.stack([cell.modality_weights for cell in cells]),
            uncertainty=torch.tensor([cell.uncertainty for cell in cells]).clamp(0.0, 1.0),
            last_seen_ms=torch.tensor([cell.last_seen_ms for cell in cells]),
            observation_count=torch.tensor([cell.observation_count for cell in cells]),
        )
