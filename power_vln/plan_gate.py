"""Fail-closed validation between semantic reasoning and motion execution."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from .models import SemanticPlan


class GateCode(str, Enum):
    APPROVED = "approved"
    RULES_UNAVAILABLE = "rules_unavailable"
    HARD_STOP = "hard_stop"
    LOW_CONFIDENCE = "low_confidence"
    EXPIRED = "expired"
    FRAME_UNAVAILABLE = "frame_unavailable"
    UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class GateContext:
    plan_age_ms: int
    rules_available: bool
    transform_available: bool
    selected_goal_reachable: bool

    def __post_init__(self) -> None:
        if self.plan_age_ms < 0:
            raise ValueError("plan_age_ms must be non-negative")


@dataclass(frozen=True)
class GateDecision:
    approved: bool
    codes: Tuple[GateCode, ...]


def evaluate_plan(
    plan: SemanticPlan,
    context: GateContext,
    minimum_confidence: float = 0.70,
) -> GateDecision:
    """Evaluate every safety-relevant condition without short-circuiting."""
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be within [0, 1]")

    failures = []
    if not context.rules_available:
        failures.append(GateCode.RULES_UNAVAILABLE)
    if plan.has_hard_stop:
        failures.append(GateCode.HARD_STOP)
    if plan.confidence < minimum_confidence:
        failures.append(GateCode.LOW_CONFIDENCE)
    if context.plan_age_ms >= plan.valid_for_ms:
        failures.append(GateCode.EXPIRED)
    if not context.transform_available:
        failures.append(GateCode.FRAME_UNAVAILABLE)
    if not context.selected_goal_reachable:
        failures.append(GateCode.UNREACHABLE)

    if failures:
        return GateDecision(approved=False, codes=tuple(failures))
    return GateDecision(approved=True, codes=(GateCode.APPROVED,))
