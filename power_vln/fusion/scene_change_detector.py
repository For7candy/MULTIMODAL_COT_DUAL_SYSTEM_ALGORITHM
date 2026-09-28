"""Version-aware scene change scoring for slow-system triggers."""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.nn import functional as F

from .spatial_memory import SpatialMemorySnapshot


@dataclass(frozen=True)
class SceneChangeEvent:
    changed: bool
    score: float
    previous_version: int
    current_version: int
    reasons: Tuple[str, ...]


class SceneChangeDetector:
    def __init__(
        self,
        threshold: float = 0.35,
        match_radius_m: float = 0.40,
        occupancy_weight: float = 0.6,
        appearance_weight: float = 0.4,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        if match_radius_m <= 0:
            raise ValueError("match_radius_m must be positive")
        if occupancy_weight < 0 or appearance_weight < 0:
            raise ValueError("change weights must be non-negative")
        total = occupancy_weight + appearance_weight
        if total <= 0:
            raise ValueError("at least one change weight must be positive")
        self.threshold = threshold
        self.match_radius_m = match_radius_m
        self.occupancy_weight = occupancy_weight / total
        self.appearance_weight = appearance_weight / total

    def compare(
        self,
        previous: SpatialMemorySnapshot,
        current: SpatialMemorySnapshot,
    ) -> SceneChangeEvent:
        if previous.frame_id != current.frame_id:
            raise ValueError("scene snapshots must use the same frame")
        if previous.version == current.version:
            return SceneChangeEvent(False, 0.0, previous.version, current.version, ())
        if previous.token_count == 0 and current.token_count == 0:
            return SceneChangeEvent(False, 0.0, previous.version, current.version, ())
        if previous.token_count == 0 or current.token_count == 0:
            return SceneChangeEvent(
                True,
                1.0,
                previous.version,
                current.version,
                ("occupancy_changed",),
            )

        distance = torch.cdist(previous.positions.float(), current.positions.float())
        candidate_pairs = []
        for previous_index, current_index in torch.nonzero(
            distance <= self.match_radius_m, as_tuple=False
        ).tolist():
            candidate_pairs.append(
                (
                    float(distance[previous_index, current_index]),
                    previous_index,
                    current_index,
                )
            )
        matched_pairs = []
        used_previous = set()
        used_current = set()
        for _, previous_index, current_index in sorted(candidate_pairs):
            if previous_index in used_previous or current_index in used_current:
                continue
            used_previous.add(previous_index)
            used_current.add(current_index)
            matched_pairs.append((previous_index, current_index))
        match_count = len(matched_pairs)
        occupancy_change = 1.0 - match_count / max(
            previous.token_count, current.token_count
        )
        if match_count:
            previous_features = torch.stack(
                [previous.features[left] for left, _ in matched_pairs]
            ).float()
            current_features = torch.stack(
                [current.features[right] for _, right in matched_pairs]
            ).float()
            appearance_change = float(
                (1.0 - F.cosine_similarity(previous_features, current_features)).mean()
            )
            appearance_change = min(1.0, max(0.0, appearance_change))
        else:
            appearance_change = 1.0
        score = min(
            1.0,
            self.occupancy_weight * occupancy_change
            + self.appearance_weight * appearance_change,
        )
        reasons = []
        if occupancy_change >= self.threshold:
            reasons.append("occupancy_changed")
        if appearance_change >= self.threshold:
            reasons.append("appearance_changed")
        changed = score >= self.threshold
        if changed and not reasons:
            reasons.append("combined_change")
        return SceneChangeEvent(
            changed,
            score,
            previous.version,
            current.version,
            tuple(reasons),
        )
