"""Structured three-dimensional scene description produced by stage one."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .common import (
    JsonSchemaMixin,
    Pose3D,
    Vector3,
    require_confidence,
    require_finite,
    require_text,
)


@dataclass(frozen=True)
class Entity3D(JsonSchemaMixin):
    entity_id: str
    category: str
    center: Vector3
    size: Vector3
    yaw: float
    attributes: Tuple[str, ...]
    confidence: float
    source_observations: Tuple[str, ...]

    def __post_init__(self) -> None:
        require_text(self.entity_id, "entity_id")
        require_text(self.category, "category")
        require_finite((self.yaw,), "yaw")
        require_confidence(self.confidence)
        if min(self.size.x, self.size.y, self.size.z) <= 0:
            raise ValueError("entity size must be positive")
        if not self.source_observations:
            raise ValueError("source_observations must not be empty")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Entity3D":
        return cls(
            entity_id=payload["entity_id"],
            category=payload["category"],
            center=Vector3.from_value(payload["center"]),
            size=Vector3.from_value(payload["size"]),
            yaw=payload["yaw"],
            attributes=tuple(payload.get("attributes", ())),
            confidence=payload["confidence"],
            source_observations=tuple(payload["source_observations"]),
        )


@dataclass(frozen=True)
class Relation3D(JsonSchemaMixin):
    subject_id: str
    predicate: str
    object_id: str
    confidence: float

    def __post_init__(self) -> None:
        require_text(self.subject_id, "subject_id")
        require_text(self.predicate, "predicate")
        require_text(self.object_id, "object_id")
        require_confidence(self.confidence)
        if self.subject_id == self.object_id:
            raise ValueError("a relation must not reference the same entity twice")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Relation3D":
        return cls(**payload)


@dataclass(frozen=True)
class FreeSpaceRegion(JsonSchemaMixin):
    region_id: str
    vertices: Tuple[Vector3, ...]
    clearance_height_m: float
    confidence: float

    def __post_init__(self) -> None:
        require_text(self.region_id, "region_id")
        if len(self.vertices) < 3:
            raise ValueError("free-space region requires at least three vertices")
        if self.clearance_height_m <= 0:
            raise ValueError("clearance_height_m must be positive")
        require_confidence(self.confidence)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FreeSpaceRegion":
        return cls(
            region_id=payload["region_id"],
            vertices=tuple(Vector3.from_value(item) for item in payload["vertices"]),
            clearance_height_m=payload["clearance_height_m"],
            confidence=payload["confidence"],
        )


@dataclass(frozen=True)
class Obstacle3D(JsonSchemaMixin):
    obstacle_id: str
    category: str
    center: Vector3
    size: Vector3
    dynamic: bool
    confidence: float
    observed_at_ms: int
    velocity: Optional[Vector3] = None

    def __post_init__(self) -> None:
        require_text(self.obstacle_id, "obstacle_id")
        require_text(self.category, "category")
        if min(self.size.x, self.size.y, self.size.z) <= 0:
            raise ValueError("obstacle size must be positive")
        if self.observed_at_ms < 0:
            raise ValueError("observed_at_ms must be non-negative")
        require_confidence(self.confidence)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Obstacle3D":
        velocity = payload.get("velocity")
        return cls(
            obstacle_id=payload["obstacle_id"],
            category=payload["category"],
            center=Vector3.from_value(payload["center"]),
            size=Vector3.from_value(payload["size"]),
            dynamic=payload["dynamic"],
            confidence=payload["confidence"],
            observed_at_ms=payload["observed_at_ms"],
            velocity=Vector3.from_value(velocity) if velocity is not None else None,
        )


@dataclass(frozen=True)
class CandidateApproach(JsonSchemaMixin):
    candidate_id: str
    target_entity_id: str
    position: Vector3
    yaw: float
    clearance_m: float
    visibility_score: float
    confidence: float

    def __post_init__(self) -> None:
        require_text(self.candidate_id, "candidate_id")
        require_text(self.target_entity_id, "target_entity_id")
        require_finite((self.yaw,), "yaw")
        if self.clearance_m < 0:
            raise ValueError("clearance_m must be non-negative")
        require_confidence(self.visibility_score, "visibility_score")
        require_confidence(self.confidence)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CandidateApproach":
        return cls(
            candidate_id=payload["candidate_id"],
            target_entity_id=payload["target_entity_id"],
            position=Vector3.from_value(payload["position"]),
            yaw=payload["yaw"],
            clearance_m=payload["clearance_m"],
            visibility_score=payload["visibility_score"],
            confidence=payload["confidence"],
        )


@dataclass(frozen=True)
class SceneUncertainty(JsonSchemaMixin):
    uncertainty_type: str
    description: str
    confidence: float
    entity_id: Optional[str] = None

    def __post_init__(self) -> None:
        require_text(self.uncertainty_type, "uncertainty_type")
        require_text(self.description, "description")
        require_confidence(self.confidence)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SceneUncertainty":
        return cls(**payload)


@dataclass(frozen=True)
class SceneDescription3D(JsonSchemaMixin):
    schema_version: int
    scene_id: str
    timestamp_ms: int
    frame_id: str
    observer_pose: Pose3D
    entities: Tuple[Entity3D, ...] = ()
    relations: Tuple[Relation3D, ...] = ()
    free_space: Tuple[FreeSpaceRegion, ...] = ()
    obstacles: Tuple[Obstacle3D, ...] = ()
    candidate_approaches: Tuple[CandidateApproach, ...] = ()
    uncertainties: Tuple[SceneUncertainty, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        require_text(self.scene_id, "scene_id")
        require_text(self.frame_id, "frame_id")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SceneDescription3D":
        return cls(
            schema_version=payload["schema_version"],
            scene_id=payload["scene_id"],
            timestamp_ms=payload["timestamp_ms"],
            frame_id=payload["frame_id"],
            observer_pose=Pose3D.from_dict(payload["observer_pose"]),
            entities=tuple(Entity3D.from_dict(item) for item in payload.get("entities", ())),
            relations=tuple(Relation3D.from_dict(item) for item in payload.get("relations", ())),
            free_space=tuple(FreeSpaceRegion.from_dict(item) for item in payload.get("free_space", ())),
            obstacles=tuple(Obstacle3D.from_dict(item) for item in payload.get("obstacles", ())),
            candidate_approaches=tuple(
                CandidateApproach.from_dict(item)
                for item in payload.get("candidate_approaches", ())
            ),
            uncertainties=tuple(
                SceneUncertainty.from_dict(item)
                for item in payload.get("uncertainties", ())
            ),
        )
