"""Dependency-free data contracts shared by slow and fast navigation systems."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any, Dict, List, Tuple


class RiskLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HARD_STOP = "hard_stop"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float
    frame_id: str = "map"


@dataclass(frozen=True)
class CandidateGoal:
    goal_id: str
    pose: Pose2D
    approach_yaw: float
    stop_distance_m: float
    source: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.goal_id:
            raise ValueError("goal_id must not be empty")
        if self.stop_distance_m < 0:
            raise ValueError("stop_distance_m must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("goal confidence must be within [0, 1]")


@dataclass(frozen=True)
class RiskFinding:
    rule_id: str
    level: RiskLevel
    passed: bool
    evidence: str


@dataclass(frozen=True)
class SemanticPlan:
    mission_id: str
    current_subtask: str
    candidates: Tuple[CandidateGoal, ...]
    selected_goal_id: str
    findings: Tuple[RiskFinding, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    valid_for_ms: int = 0
    replan_triggers: Tuple[str, ...] = field(default_factory=tuple)
    audit_summary: str = ""

    def __post_init__(self) -> None:
        if not self.mission_id:
            raise ValueError("mission_id must not be empty")
        if not self.current_subtask:
            raise ValueError("current_subtask must not be empty")
        candidate_ids = {candidate.goal_id for candidate in self.candidates}
        if self.selected_goal_id not in candidate_ids:
            raise ValueError("selected_goal_id must reference a candidate")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("plan confidence must be within [0, 1]")
        if self.valid_for_ms <= 0:
            raise ValueError("valid_for_ms must be positive")

    @property
    def selected_goal(self) -> CandidateGoal:
        for candidate in self.candidates:
            if candidate.goal_id == self.selected_goal_id:
                return candidate
        raise RuntimeError("validated plan lost its selected goal")

    @property
    def has_hard_stop(self) -> bool:
        return any(
            finding.level == RiskLevel.HARD_STOP and not finding.passed
            for finding in self.findings
        )

    def to_dict(self) -> Dict[str, Any]:
        return _to_primitive(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def _to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_primitive(item) for item in value]
    return value


def semantic_plan_from_dict(payload: Dict[str, Any]) -> SemanticPlan:
    candidates: List[CandidateGoal] = []
    for item in payload["candidates"]:
        candidates.append(
            CandidateGoal(
                goal_id=item["goal_id"],
                pose=Pose2D(**item["pose"]),
                approach_yaw=item["approach_yaw"],
                stop_distance_m=item["stop_distance_m"],
                source=item["source"],
                confidence=item["confidence"],
            )
        )

    findings: List[RiskFinding] = []
    for item in payload.get("findings", []):
        findings.append(
            RiskFinding(
                rule_id=item["rule_id"],
                level=RiskLevel(item["level"]),
                passed=item["passed"],
                evidence=item["evidence"],
            )
        )

    return SemanticPlan(
        mission_id=payload["mission_id"],
        current_subtask=payload["current_subtask"],
        candidates=tuple(candidates),
        selected_goal_id=payload["selected_goal_id"],
        findings=tuple(findings),
        confidence=payload["confidence"],
        valid_for_ms=payload["valid_for_ms"],
        replan_triggers=tuple(payload.get("replan_triggers", [])),
        audit_summary=payload.get("audit_summary", ""),
    )
