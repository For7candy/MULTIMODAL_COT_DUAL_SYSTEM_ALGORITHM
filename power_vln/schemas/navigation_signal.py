"""Structured reasoning and navigation signal contracts for stage two."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .common import (
    JsonSchemaMixin,
    Vector3,
    require_confidence,
    require_finite,
    require_text,
)


class NavigationIntent(str, Enum):
    APPROACH = "approach"
    OBSERVE = "observe"
    PASS_THROUGH = "pass_through"
    STOP = "stop"


class NavigationStatus(str, Enum):
    READY = "ready"
    NEED_OBSERVATION = "need_observation"
    AMBIGUOUS_TARGET = "ambiguous_target"
    UNREACHABLE = "unreachable"
    INVALID_INSTRUCTION = "invalid_instruction"
    INVALID = "invalid"


@dataclass(frozen=True)
class TaskParse(JsonSchemaMixin):
    action: str
    target: str
    constraints: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.action, "action")
        require_text(self.target, "target")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TaskParse":
        return cls(
            action=payload["action"],
            target=payload["target"],
            constraints=tuple(payload.get("constraints", ())),
        )


@dataclass(frozen=True)
class GroundingResult(JsonSchemaMixin):
    entity_id: str
    confidence: float

    def __post_init__(self) -> None:
        require_text(self.entity_id, "entity_id")
        require_confidence(self.confidence)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GroundingResult":
        return cls(**payload)


@dataclass(frozen=True)
class CandidateEvaluation(JsonSchemaMixin):
    candidate_id: str
    reachable: bool
    score: float
    reason_codes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.candidate_id, "candidate_id")
        require_confidence(self.score, "score")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CandidateEvaluation":
        return cls(
            candidate_id=payload["candidate_id"],
            reachable=payload["reachable"],
            score=payload["score"],
            reason_codes=tuple(payload.get("reason_codes", ())),
        )


@dataclass(frozen=True)
class NavigationDecision(JsonSchemaMixin):
    selected_candidate_id: Optional[str]
    reason_codes: Tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NavigationDecision":
        return cls(
            selected_candidate_id=payload.get("selected_candidate_id"),
            reason_codes=tuple(payload.get("reason_codes", ())),
        )


@dataclass(frozen=True)
class NavigationReasoning(JsonSchemaMixin):
    task_parse: TaskParse
    grounding: Optional[GroundingResult]
    candidate_ids: Tuple[str, ...]
    evaluations: Tuple[CandidateEvaluation, ...]
    decision: NavigationDecision

    def __post_init__(self) -> None:
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NavigationReasoning":
        grounding = payload.get("grounding")
        return cls(
            task_parse=TaskParse.from_dict(payload["task_parse"]),
            grounding=GroundingResult.from_dict(grounding) if grounding else None,
            candidate_ids=tuple(payload.get("candidate_ids", ())),
            evaluations=tuple(
                CandidateEvaluation.from_dict(item)
                for item in payload.get("evaluations", ())
            ),
            decision=NavigationDecision.from_dict(payload["decision"]),
        )


@dataclass(frozen=True)
class NavigationGoal3D(JsonSchemaMixin):
    frame_id: str
    position: Vector3
    yaw: float
    tolerance_m: float

    def __post_init__(self) -> None:
        require_text(self.frame_id, "frame_id")
        require_finite((self.yaw,), "yaw")
        if self.tolerance_m < 0:
            raise ValueError("tolerance_m must be non-negative")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NavigationGoal3D":
        return cls(
            frame_id=payload["frame_id"],
            position=Vector3.from_value(payload["position"]),
            yaw=payload["yaw"],
            tolerance_m=payload["tolerance_m"],
        )


@dataclass(frozen=True)
class NavigationConstraints(JsonSchemaMixin):
    minimum_clearance_m: float
    maximum_speed_mps: float
    forbidden_region_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.minimum_clearance_m < 0:
            raise ValueError("minimum_clearance_m must be non-negative")
        if self.maximum_speed_mps <= 0:
            raise ValueError("maximum_speed_mps must be positive")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NavigationConstraints":
        return cls(
            minimum_clearance_m=payload["minimum_clearance_m"],
            maximum_speed_mps=payload["maximum_speed_mps"],
            forbidden_region_ids=tuple(payload.get("forbidden_region_ids", ())),
        )


@dataclass(frozen=True)
class NavigationSignal(JsonSchemaMixin):
    schema_version: int
    mission_id: str
    scene_id: str
    subtask_id: str
    intent: NavigationIntent
    status: NavigationStatus
    constraints: NavigationConstraints
    confidence: float
    generated_at_ms: int
    valid_for_ms: int
    replan_triggers: Tuple[str, ...]
    target_entity_id: Optional[str] = None
    candidate_id: Optional[str] = None
    goal: Optional[NavigationGoal3D] = None

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        require_text(self.mission_id, "mission_id")
        require_text(self.scene_id, "scene_id")
        require_text(self.subtask_id, "subtask_id")
        require_confidence(self.confidence)
        if self.generated_at_ms < 0:
            raise ValueError("generated_at_ms must be non-negative")
        if self.valid_for_ms <= 0:
            raise ValueError("valid_for_ms must be positive")
        if self.status == NavigationStatus.READY:
            if not self.target_entity_id or not self.candidate_id or self.goal is None:
                raise ValueError(
                    "ready signal requires target_entity_id, candidate_id and goal"
                )

    @property
    def expires_at_ms(self) -> int:
        return self.generated_at_ms + self.valid_for_ms

    def is_expired(self, now_ms: int) -> bool:
        return now_ms >= self.expires_at_ms

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NavigationSignal":
        goal = payload.get("goal")
        return cls(
            schema_version=payload["schema_version"],
            mission_id=payload["mission_id"],
            scene_id=payload["scene_id"],
            subtask_id=payload["subtask_id"],
            intent=NavigationIntent(payload["intent"]),
            status=NavigationStatus(payload["status"]),
            constraints=NavigationConstraints.from_dict(payload["constraints"]),
            confidence=payload["confidence"],
            generated_at_ms=payload["generated_at_ms"],
            valid_for_ms=payload["valid_for_ms"],
            replan_triggers=tuple(payload.get("replan_triggers", ())),
            target_entity_id=payload.get("target_entity_id"),
            candidate_id=payload.get("candidate_id"),
            goal=NavigationGoal3D.from_dict(goal) if goal else None,
        )
