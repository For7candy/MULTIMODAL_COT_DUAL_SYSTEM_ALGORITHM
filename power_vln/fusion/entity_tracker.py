"""Deterministic static-entity association and dynamic trajectory tracking."""

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Tuple

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class EntityObservation:
    observation_id: str
    category: str
    center: torch.Tensor
    appearance: torch.Tensor
    confidence: float
    dynamic: bool = False

    def __post_init__(self) -> None:
        if not self.observation_id or not self.category:
            raise ValueError("observation_id and category must not be empty")
        if self.center.shape != (3,) or self.appearance.ndim != 1:
            raise ValueError("center must be [3] and appearance must be [D]")
        if not torch.isfinite(self.center).all() or not torch.isfinite(self.appearance).all():
            raise ValueError("entity tensors must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")


@dataclass(frozen=True)
class TrackedEntity:
    track_id: str
    category: str
    center: torch.Tensor
    appearance: torch.Tensor
    confidence: float
    dynamic: bool
    first_seen_ms: int
    last_seen_ms: int
    observation_ids: Tuple[str, ...]
    velocity_mps: torch.Tensor


@dataclass(frozen=True)
class EntityTrackerSnapshot:
    version: int
    timestamp_ms: int
    static_entities: Tuple[TrackedEntity, ...]
    dynamic_tracks: Tuple[TrackedEntity, ...]


class EntityTracker:
    """Associate observations using category, distance, and appearance."""

    def __init__(
        self,
        appearance_size: int,
        match_radius_m: float = 0.75,
        appearance_similarity: float = 0.70,
        static_maximum_age_ms: int = 20000,
        dynamic_maximum_age_ms: int = 3000,
    ) -> None:
        if appearance_size <= 0 or match_radius_m <= 0:
            raise ValueError("appearance_size and match_radius_m must be positive")
        if not -1.0 <= appearance_similarity <= 1.0:
            raise ValueError("appearance_similarity must be within [-1, 1]")
        if static_maximum_age_ms <= 0 or dynamic_maximum_age_ms <= 0:
            raise ValueError("track maximum ages must be positive")
        self.appearance_size = appearance_size
        self.match_radius_m = match_radius_m
        self.appearance_similarity = appearance_similarity
        self.static_maximum_age_ms = static_maximum_age_ms
        self.dynamic_maximum_age_ms = dynamic_maximum_age_ms
        self._tracks: Dict[str, TrackedEntity] = {}
        self._next_static_id = 1
        self._next_dynamic_id = 1
        self._version = 0
        self._timestamp_ms = 0

    def _new_id(self, dynamic: bool) -> str:
        if dynamic:
            track_id = "dynamic-{:06d}".format(self._next_dynamic_id)
            self._next_dynamic_id += 1
        else:
            track_id = "static-{:06d}".format(self._next_static_id)
            self._next_static_id += 1
        return track_id

    def _similarity(self, left: torch.Tensor, right: torch.Tensor) -> float:
        if torch.linalg.vector_norm(left) < 1e-8 or torch.linalg.vector_norm(right) < 1e-8:
            return 0.0
        return float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())

    def _find_match(
        self,
        observation: EntityObservation,
        used_tracks: set,
    ) -> str:
        candidates = []
        for track_id, track in self._tracks.items():
            if track_id in used_tracks:
                continue
            if track.dynamic != observation.dynamic or track.category != observation.category:
                continue
            distance = float(torch.linalg.vector_norm(track.center - observation.center))
            if distance > self.match_radius_m:
                continue
            similarity = self._similarity(track.appearance, observation.appearance)
            if similarity < self.appearance_similarity:
                continue
            score = distance / self.match_radius_m + (1.0 - similarity)
            candidates.append((score, track_id))
        return min(candidates)[1] if candidates else ""

    def _merge(
        self,
        track: TrackedEntity,
        observation: EntityObservation,
        timestamp_ms: int,
    ) -> TrackedEntity:
        total = max(track.confidence + observation.confidence, 1e-8)
        new_weight = observation.confidence / total
        center = (1.0 - new_weight) * track.center + new_weight * observation.center
        appearance = (
            (1.0 - new_weight) * track.appearance + new_weight * observation.appearance
        )
        elapsed_s = (timestamp_ms - track.last_seen_ms) / 1000.0
        if track.dynamic and elapsed_s > 0:
            velocity = (observation.center - track.center) / elapsed_s
        else:
            velocity = torch.zeros(3)
        confidence = min(
            1.0, 1.0 - (1.0 - track.confidence) * (1.0 - observation.confidence)
        )
        return replace(
            track,
            center=center,
            appearance=appearance,
            confidence=confidence,
            last_seen_ms=timestamp_ms,
            observation_ids=track.observation_ids + (observation.observation_id,),
            velocity_mps=velocity,
        )

    def update(
        self,
        observations: Iterable[EntityObservation],
        timestamp_ms: int,
    ) -> EntityTrackerSnapshot:
        if timestamp_ms < self._timestamp_ms:
            raise ValueError("tracker timestamps must be monotonic")
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if timestamp_ms - track.last_seen_ms
            > (
                self.dynamic_maximum_age_ms
                if track.dynamic
                else self.static_maximum_age_ms
            )
        ]
        for track_id in expired:
            del self._tracks[track_id]

        normalized = []
        for observation in observations:
            if observation.appearance.shape != (self.appearance_size,):
                raise ValueError("observation appearance size does not match tracker")
            normalized.append(
                replace(
                    observation,
                    center=observation.center.detach().float().cpu(),
                    appearance=observation.appearance.detach().float().cpu(),
                )
            )
        normalized.sort(
            key=lambda item: (item.dynamic, item.category, item.observation_id)
        )
        used_tracks = set()
        for observation in normalized:
            track_id = self._find_match(observation, used_tracks)
            if track_id:
                self._tracks[track_id] = self._merge(
                    self._tracks[track_id], observation, timestamp_ms
                )
            else:
                track_id = self._new_id(observation.dynamic)
                self._tracks[track_id] = TrackedEntity(
                    track_id=track_id,
                    category=observation.category,
                    center=observation.center,
                    appearance=observation.appearance,
                    confidence=observation.confidence,
                    dynamic=observation.dynamic,
                    first_seen_ms=timestamp_ms,
                    last_seen_ms=timestamp_ms,
                    observation_ids=(observation.observation_id,),
                    velocity_mps=torch.zeros(3),
                )
            used_tracks.add(track_id)
        self._timestamp_ms = timestamp_ms
        self._version += 1
        return self.snapshot()

    def snapshot(self) -> EntityTrackerSnapshot:
        tracks = tuple(self._tracks[key] for key in sorted(self._tracks))
        return EntityTrackerSnapshot(
            version=self._version,
            timestamp_ms=self._timestamp_ms,
            static_entities=tuple(track for track in tracks if not track.dynamic),
            dynamic_tracks=tuple(track for track in tracks if track.dynamic),
        )
