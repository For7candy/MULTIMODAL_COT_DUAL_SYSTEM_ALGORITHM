"""Composable losses for alignment, geometry, language, ranking, and signals."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


def _masked_mean(values: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return values.mean()
    weights = mask.to(device=values.device, dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def alignment_loss(rgb_features: Tensor, point_features: Tensor, mask=None) -> Tensor:
    if rgb_features.shape != point_features.shape:
        raise ValueError("aligned RGB and point features must have identical shapes")
    rgb = functional.normalize(rgb_features.float(), dim=-1)
    point = functional.normalize(point_features.float(), dim=-1)
    return _masked_mean(1.0 - (rgb * point).sum(dim=-1), mask)


def geometry_loss(
    predicted_centers: Tensor,
    target_centers: Tensor,
    predicted_sizes: Tensor,
    target_sizes: Tensor,
    mask=None,
) -> Tensor:
    if predicted_centers.shape != target_centers.shape:
        raise ValueError("predicted and target centers must match")
    if predicted_sizes.shape != target_sizes.shape:
        raise ValueError("predicted and target sizes must match")
    center = functional.smooth_l1_loss(
        predicted_centers.float(), target_centers.float(), reduction="none"
    )
    size = functional.smooth_l1_loss(
        predicted_sizes.float(), target_sizes.float(), reduction="none"
    )
    return _masked_mean(center, mask) + _masked_mean(size, mask)


def token_generation_loss(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    if logits.ndim != targets.ndim + 1 or logits.shape[:-1] != targets.shape:
        raise ValueError("token logits and targets have incompatible shapes")
    return functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def candidate_ranking_loss(scores: Tensor, target_indices: Tensor) -> Tensor:
    if scores.ndim != 2 or target_indices.shape != scores.shape[:1]:
        raise ValueError("candidate scores must be [batch, candidates]")
    return functional.cross_entropy(scores.float(), target_indices.long())


def navigation_signal_loss(
    status_logits: Tensor,
    status_targets: Tensor,
    confidence: Tensor,
    confidence_targets: Tensor,
) -> Tensor:
    status = functional.cross_entropy(status_logits.float(), status_targets.long())
    confidence_term = functional.smooth_l1_loss(
        confidence.float(), confidence_targets.float()
    )
    return status + confidence_term


@dataclass(frozen=True)
class LossWeights:
    alignment: float = 1.0
    geometry: float = 1.0
    scene_generation: float = 1.0
    cot_generation: float = 1.0
    candidate_ranking: float = 1.0
    navigation_signal: float = 1.0

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.__dict__.values()):
            raise ValueError("loss weights must be non-negative")


class MultiTaskNavigationLoss(nn.Module):
    """Aggregate already-computed task losses with explicit stable names."""

    _NAMES = (
        "alignment",
        "geometry",
        "scene_generation",
        "cot_generation",
        "candidate_ranking",
        "navigation_signal",
    )

    def __init__(self, weights: LossWeights = LossWeights()) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, losses: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        unknown = set(losses) - set(self._NAMES)
        if unknown:
            raise ValueError("unknown task losses: {}".format(sorted(unknown)))
        if not losses:
            raise ValueError("at least one task loss is required")
        weighted = {
            name: value * getattr(self.weights, name) for name, value in losses.items()
        }
        total = torch.stack(tuple(weighted.values())).sum()
        return {**weighted, "total": total}
