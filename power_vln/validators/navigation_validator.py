"""Validation across scene, structured reasoning and navigation signal."""

import math
from typing import Optional

from power_vln.schemas.common import ValidationIssue, ValidationResult
from power_vln.schemas.navigation_signal import (
    NavigationReasoning,
    NavigationSignal,
    NavigationStatus,
)
from power_vln.schemas.scene_description import SceneDescription3D


def validate_reasoning(
    reasoning: NavigationReasoning,
    scene: SceneDescription3D,
) -> ValidationResult:
    issues = []
    scene_entities = {item.entity_id for item in scene.entities}
    scene_candidates = {
        item.candidate_id: item for item in scene.candidate_approaches
    }
    declared_candidates = set(reasoning.candidate_ids)

    if reasoning.grounding and reasoning.grounding.entity_id not in scene_entities:
        issues.append(
            ValidationIssue(
                code="unknown_grounding_entity",
                path="grounding.entity_id",
                message="grounding does not reference a scene entity",
            )
        )

    for candidate_id in sorted(declared_candidates):
        candidate = scene_candidates.get(candidate_id)
        if candidate is None:
            issues.append(
                ValidationIssue(
                    code="unknown_declared_candidate",
                    path="candidate_ids",
                    message="declared candidate is absent from the scene",
                )
            )
        elif reasoning.grounding and candidate.target_entity_id != reasoning.grounding.entity_id:
            issues.append(
                ValidationIssue(
                    code="candidate_grounding_mismatch",
                    path="candidate_ids",
                    message="candidate belongs to a different grounded entity",
                )
            )

    evaluated_candidates = set()
    for index, evaluation in enumerate(reasoning.evaluations):
        if evaluation.candidate_id in evaluated_candidates:
            issues.append(
                ValidationIssue(
                    code="duplicate_candidate_evaluation",
                    path="evaluations[{}].candidate_id".format(index),
                    message="candidate was evaluated more than once",
                )
            )
        evaluated_candidates.add(evaluation.candidate_id)
        if evaluation.candidate_id not in declared_candidates:
            issues.append(
                ValidationIssue(
                    code="undeclared_evaluation_candidate",
                    path="evaluations[{}].candidate_id".format(index),
                    message="evaluation candidate is absent from candidate_ids",
                )
            )
        if evaluation.candidate_id not in scene_candidates:
            issues.append(
                ValidationIssue(
                    code="unknown_scene_candidate",
                    path="evaluations[{}].candidate_id".format(index),
                    message="evaluation candidate is absent from the scene",
                )
            )

    for candidate_id in sorted(declared_candidates - evaluated_candidates):
        issues.append(
            ValidationIssue(
                code="missing_candidate_evaluation",
                path="evaluations",
                message="declared candidate was not evaluated: {}".format(candidate_id),
            )
        )

    selected = reasoning.decision.selected_candidate_id
    if selected is not None and selected not in declared_candidates:
        issues.append(
            ValidationIssue(
                code="undeclared_selected_candidate",
                path="decision.selected_candidate_id",
                message="selected candidate is absent from candidate_ids",
            )
        )
    if selected is not None:
        selected_evaluation = next(
            (
                item
                for item in reasoning.evaluations
                if item.candidate_id == selected
            ),
            None,
        )
        if selected_evaluation is not None and not selected_evaluation.reachable:
            issues.append(
                ValidationIssue(
                    code="selected_candidate_unreachable",
                    path="decision.selected_candidate_id",
                    message="selected candidate is marked unreachable",
                )
            )

    return ValidationResult(issues=tuple(issues))


def validate_navigation_signal(
    signal: NavigationSignal,
    scene: SceneDescription3D,
    now_ms: int,
    minimum_confidence: float = 0.70,
    require_ready: bool = True,
) -> ValidationResult:
    if now_ms < 0:
        raise ValueError("now_ms must be non-negative")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be within [0, 1]")

    issues = []
    if require_ready and signal.status != NavigationStatus.READY:
        issues.append(
            ValidationIssue(
                code="signal_not_ready",
                path="status",
                message="only a ready signal may enter the fast system",
            )
        )
    if signal.scene_id != scene.scene_id:
        issues.append(
            ValidationIssue(
                code="scene_version_mismatch",
                path="scene_id",
                message="signal and scene identifiers differ",
            )
        )
    if signal.status == NavigationStatus.READY and signal.confidence < minimum_confidence:
        issues.append(
            ValidationIssue(
                code="low_confidence",
                path="confidence",
                message="signal confidence is below the configured threshold",
            )
        )
    if signal.is_expired(now_ms):
        issues.append(
            ValidationIssue(
                code="expired",
                path="valid_for_ms",
                message="navigation signal has expired",
            )
        )

    entity_ids = {item.entity_id for item in scene.entities}
    candidates = {item.candidate_id: item for item in scene.candidate_approaches}
    if signal.target_entity_id and signal.target_entity_id not in entity_ids:
        issues.append(
            ValidationIssue(
                code="unknown_target_entity",
                path="target_entity_id",
                message="target entity is absent from the referenced scene",
            )
        )

    candidate = candidates.get(signal.candidate_id) if signal.candidate_id else None
    if signal.candidate_id and candidate is None:
        issues.append(
            ValidationIssue(
                code="unknown_candidate",
                path="candidate_id",
                message="candidate is absent from the referenced scene",
            )
        )
    if candidate and signal.target_entity_id != candidate.target_entity_id:
        issues.append(
            ValidationIssue(
                code="candidate_target_mismatch",
                path="candidate_id",
                message="candidate belongs to a different target entity",
            )
        )

    if signal.goal:
        if signal.goal.frame_id != scene.frame_id:
            issues.append(
                ValidationIssue(
                    code="frame_mismatch",
                    path="goal.frame_id",
                    message="goal and scene frame identifiers differ",
                )
            )
        if candidate:
            distance = math.sqrt(
                (signal.goal.position.x - candidate.position.x) ** 2
                + (signal.goal.position.y - candidate.position.y) ** 2
                + (signal.goal.position.z - candidate.position.z) ** 2
            )
            if distance > max(signal.goal.tolerance_m, 1e-6):
                issues.append(
                    ValidationIssue(
                        code="goal_candidate_mismatch",
                        path="goal.position",
                        message="goal position does not match the selected candidate",
                    )
                )

    return ValidationResult(issues=tuple(issues))
