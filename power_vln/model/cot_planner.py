"""Stage-two structured reasoning and navigation-signal generation."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from power_vln.prompts import build_navigation_cot_prompts
from power_vln.schemas import (
    NavigationReasoning,
    NavigationSignal,
    NavigationStatus,
    SceneDescription3D,
)
from power_vln.validators import validate_navigation_signal, validate_reasoning

from .candidate_ranker import expected_candidate
from .json_schema import validate_json_schema_subset
from .multimodal_backbone import (
    MultimodalBackboneRequest,
    StructuredMultimodalBackbone,
)


def _confidence_schema() -> Dict[str, Any]:
    return {"type": "number", "minimum": 0, "maximum": 1}


def _vector_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["x", "y", "z"],
        "properties": {
            axis: {"type": "number"} for axis in ("x", "y", "z")
        },
    }


def navigation_plan_json_schema() -> Dict[str, Any]:
    task_parse = {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "target", "constraints"],
        "properties": {
            "action": {"type": "string", "minLength": 1},
            "target": {"type": "string", "minLength": 1},
            "constraints": {"type": "array", "items": {"type": "string"}},
        },
    }
    grounding = {
        "type": ["object", "null"],
        "additionalProperties": False,
        "required": ["entity_id", "confidence"],
        "properties": {
            "entity_id": {"type": "string", "minLength": 1},
            "confidence": _confidence_schema(),
        },
    }
    evaluation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidate_id", "reachable", "score", "reason_codes"],
        "properties": {
            "candidate_id": {"type": "string", "minLength": 1},
            "reachable": {"type": "boolean"},
            "score": _confidence_schema(),
            "reason_codes": {"type": "array", "items": {"type": "string"}},
        },
    }
    decision = {
        "type": "object",
        "additionalProperties": False,
        "required": ["selected_candidate_id", "reason_codes"],
        "properties": {
            "selected_candidate_id": {"type": ["string", "null"]},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
        },
    }
    reasoning = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "task_parse", "grounding", "candidate_ids", "evaluations", "decision"
        ],
        "properties": {
            "task_parse": task_parse,
            "grounding": grounding,
            "candidate_ids": {"type": "array", "items": {"type": "string"}},
            "evaluations": {"type": "array", "items": evaluation},
            "decision": decision,
        },
    }
    goal = {
        "type": ["object", "null"],
        "additionalProperties": False,
        "required": ["frame_id", "position", "yaw", "tolerance_m"],
        "properties": {
            "frame_id": {"type": "string", "minLength": 1},
            "position": _vector_schema(),
            "yaw": {"type": "number"},
            "tolerance_m": {"type": "number", "minimum": 0},
        },
    }
    constraints = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "minimum_clearance_m", "maximum_speed_mps", "forbidden_region_ids"
        ],
        "properties": {
            "minimum_clearance_m": {"type": "number", "minimum": 0},
            "maximum_speed_mps": {"type": "number", "exclusiveMinimum": 0},
            "forbidden_region_ids": {
                "type": "array", "items": {"type": "string"}
            },
        },
    }
    signal = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "mission_id", "scene_id", "subtask_id", "intent",
            "status", "constraints", "confidence", "generated_at_ms", "valid_for_ms",
            "replan_triggers", "target_entity_id", "candidate_id", "goal",
        ],
        "properties": {
            "schema_version": {"const": 1},
            "mission_id": {"type": "string", "minLength": 1},
            "scene_id": {"type": "string", "minLength": 1},
            "subtask_id": {"type": "string", "minLength": 1},
            "intent": {
                "type": "string",
                "enum": ["approach", "observe", "pass_through", "stop"],
            },
            "status": {
                "type": "string",
                "enum": [
                    "ready", "need_observation", "ambiguous_target", "unreachable",
                    "invalid_instruction",
                ],
            },
            "constraints": constraints,
            "confidence": _confidence_schema(),
            "generated_at_ms": {"type": "integer", "minimum": 0},
            "valid_for_ms": {"type": "integer", "exclusiveMinimum": 0},
            "replan_triggers": {"type": "array", "items": {"type": "string"}},
            "target_entity_id": {"type": ["string", "null"]},
            "candidate_id": {"type": ["string", "null"]},
            "goal": goal,
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "NavigationPlan-v1",
        "type": "object",
        "additionalProperties": False,
        "required": ["reasoning", "signal"],
        "properties": {"reasoning": reasoning, "signal": signal},
    }


@dataclass(frozen=True)
class NavigationPlanningContext:
    instruction: str
    mission_id: str
    subtask_id: str
    generated_at_ms: int
    valid_for_ms: int = 5000
    minimum_clearance_m: float = 0.5
    maximum_speed_mps: float = 0.3
    replan_triggers: Tuple[str, ...] = (
        "scene_changed",
        "path_blocked",
        "target_missing",
    )
    task_history: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.instruction.strip() or not self.mission_id.strip():
            raise ValueError("instruction and mission_id must not be empty")
        if not self.subtask_id.strip():
            raise ValueError("subtask_id must not be empty")
        if self.generated_at_ms < 0 or self.valid_for_ms <= 0:
            raise ValueError("planning timestamps must be positive")
        if self.minimum_clearance_m < 0 or self.maximum_speed_mps <= 0:
            raise ValueError("navigation policy limits are invalid")
        if not self.replan_triggers:
            raise ValueError("replan_triggers must not be empty")


class NavigationPlanningError(ValueError):
    def __init__(self, code: str, message: str, details: Tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


@dataclass(frozen=True)
class NavigationPlanResult:
    reasoning: NavigationReasoning
    signal: NavigationSignal


class StructuredCoTPlanner:
    """Generate and audit stage-two reasoning with the same foundation model."""

    def __init__(
        self,
        backbone: StructuredMultimodalBackbone,
        hidden_size: int = 2048,
        minimum_ready_confidence: float = 0.70,
    ) -> None:
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= minimum_ready_confidence <= 1.0:
            raise ValueError("minimum_ready_confidence must be within [0, 1]")
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.minimum_ready_confidence = minimum_ready_confidence
        self.json_schema = navigation_plan_json_schema()

    def _request(
        self,
        scene: SceneDescription3D,
        context: NavigationPlanningContext,
    ) -> MultimodalBackboneRequest:
        system_prompt, user_prompt = build_navigation_cot_prompts(
            scene=scene.to_dict(),
            instruction=context.instruction,
            mission_id=context.mission_id,
            subtask_id=context.subtask_id,
            generated_at_ms=context.generated_at_ms,
            valid_for_ms=context.valid_for_ms,
            minimum_clearance_m=context.minimum_clearance_m,
            maximum_speed_mps=context.maximum_speed_mps,
            replan_triggers=context.replan_triggers,
            task_history=context.task_history,
        )
        return MultimodalBackboneRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="NavigationPlan-v1",
            json_schema=self.json_schema,
            embeddings=torch.empty(1, 0, self.hidden_size),
            attention_mask=torch.empty(1, 0, dtype=torch.bool),
            token_type_ids=torch.empty(1, 0, dtype=torch.long),
            positions=torch.empty(1, 0, 3),
            confidence=torch.empty(1, 0),
        )

    def _decode(
        self,
        output: Any,
        scene: SceneDescription3D,
        context: NavigationPlanningContext,
    ) -> NavigationPlanResult:
        if not isinstance(output, dict):
            raise NavigationPlanningError(
                "invalid_root", "navigation output must be one JSON object"
            )
        schema_issues = validate_json_schema_subset(output, self.json_schema)
        if schema_issues:
            raise NavigationPlanningError(
                "json_schema_violation",
                "navigation output violates NavigationPlan-v1",
                schema_issues,
            )
        try:
            reasoning = NavigationReasoning.from_dict(output["reasoning"])
            signal = NavigationSignal.from_dict(output["signal"])
        except (KeyError, TypeError, ValueError) as error:
            raise NavigationPlanningError(
                "typed_schema_violation", "navigation fields are invalid", (str(error),)
            ) from error

        identity_issues = []
        if signal.mission_id != context.mission_id:
            identity_issues.append("mission_id")
        if signal.subtask_id != context.subtask_id:
            identity_issues.append("subtask_id")
        if signal.generated_at_ms != context.generated_at_ms:
            identity_issues.append("generated_at_ms")
        if signal.valid_for_ms != context.valid_for_ms:
            identity_issues.append("valid_for_ms")
        if signal.constraints.minimum_clearance_m != context.minimum_clearance_m:
            identity_issues.append("minimum_clearance_m")
        if signal.constraints.maximum_speed_mps != context.maximum_speed_mps:
            identity_issues.append("maximum_speed_mps")
        if signal.replan_triggers != context.replan_triggers:
            identity_issues.append("replan_triggers")
        if identity_issues:
            raise NavigationPlanningError(
                "request_identity_mismatch",
                "signal does not preserve request identity",
                tuple(identity_issues),
            )

        reasoning_validation = validate_reasoning(reasoning, scene)
        if not reasoning_validation.is_valid:
            raise NavigationPlanningError(
                "invalid_reasoning_references",
                "reasoning references are inconsistent with the scene",
                tuple(issue.code for issue in reasoning_validation.issues),
            )
        minimum_confidence = (
            self.minimum_ready_confidence
            if signal.status == NavigationStatus.READY
            else 0.0
        )
        signal_validation = validate_navigation_signal(
            signal,
            scene,
            now_ms=context.generated_at_ms,
            minimum_confidence=minimum_confidence,
            require_ready=False,
        )
        if not signal_validation.is_valid:
            raise NavigationPlanningError(
                "invalid_navigation_signal",
                "navigation signal is inconsistent with the scene",
                tuple(issue.code for issue in signal_validation.issues),
            )

        selected = reasoning.decision.selected_candidate_id
        ranked_candidate = expected_candidate(reasoning)
        if selected is not None and selected != ranked_candidate:
            raise NavigationPlanningError(
                "non_deterministic_candidate_selection",
                "decision did not select the highest-ranked reachable candidate",
            )
        if signal.status == NavigationStatus.READY:
            if reasoning.grounding is None:
                raise NavigationPlanningError(
                    "ready_without_grounding", "ready requires grounded target"
                )
            if selected is None or selected != signal.candidate_id:
                raise NavigationPlanningError(
                    "decision_signal_mismatch", "decision and signal candidates differ"
                )
            if reasoning.grounding.entity_id != signal.target_entity_id:
                raise NavigationPlanningError(
                    "grounding_signal_mismatch", "grounding and signal targets differ"
                )
        else:
            if selected is not None or any(
                item is not None
                for item in (signal.target_entity_id, signal.candidate_id, signal.goal)
            ):
                raise NavigationPlanningError(
                    "unsafe_non_ready_signal",
                    "non-ready result must not contain an executable target",
                )
            if (
                signal.status == NavigationStatus.AMBIGUOUS_TARGET
                and reasoning.grounding is not None
            ):
                raise NavigationPlanningError(
                    "ambiguous_but_grounded",
                    "ambiguous target must not claim one grounded entity",
                )
        return NavigationPlanResult(reasoning=reasoning, signal=signal)

    def plan(
        self,
        scene: SceneDescription3D,
        context: NavigationPlanningContext,
    ) -> NavigationPlanResult:
        request = self._request(scene, context)
        output = self.backbone.generate_structured(request)
        return self._decode(output, scene, context)
