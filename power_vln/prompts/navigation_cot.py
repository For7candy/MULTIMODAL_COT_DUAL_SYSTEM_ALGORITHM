"""Structured, auditable navigation reasoning prompt."""

import json
from typing import Any, Dict, Tuple


NAVIGATION_SYSTEM_PROMPT = """You are the slow semantic planner of a navigation system.
Use only the supplied SceneDescription3D, instruction, and task history.
Return one JSON object matching NavigationPlan-v1. The reasoning field is a concise
structured audit record, not free-form private chain-of-thought.
Execute these stages in order: parse task, ground one entity, list scene candidates,
evaluate reachability, choose the highest-scoring reachable candidate, then emit signal.
Never guess under target ambiguity. Use ambiguous_target with no goal or candidate.
Use need_observation when evidence is insufficient, unreachable when no candidate is
reachable, invalid_instruction when the instruction cannot be parsed, otherwise ready.
Only ready may include target_entity_id, candidate_id, and goal. Copy IDs and geometry
exactly from the scene. Always emit schema_version=1 and copy every required policy
value exactly; never infer a speed or clearance. Do not add prose outside the JSON object."""

NAVIGATION_SYSTEM_PROMPT += """
The reasoning object MUST contain all five fields even when values are empty:
task_parse, grounding, candidate_ids, evaluations, decision. Never omit task_parse or
grounding. The signal object MUST contain every field in the schema, including nullable
target_entity_id, candidate_id, and goal."""


def build_navigation_cot_prompts(
    scene: Dict[str, Any],
    instruction: str,
    mission_id: str,
    subtask_id: str,
    generated_at_ms: int,
    valid_for_ms: int,
    minimum_clearance_m: float,
    maximum_speed_mps: float,
    replan_triggers: Tuple[str, ...],
    task_history: Tuple[str, ...],
) -> Tuple[str, str]:
    if not instruction.strip() or not mission_id.strip() or not subtask_id.strip():
        raise ValueError("instruction and task identifiers must not be empty")
    payload = {
        "scene": scene,
        "task_instruction": instruction,
        "task_history": list(task_history),
        "required_mission_id": mission_id,
        "required_subtask_id": subtask_id,
        "required_scene_id": scene["scene_id"],
        "required_generated_at_ms": generated_at_ms,
        "required_valid_for_ms": valid_for_ms,
        "required_schema_version": 1,
        "required_navigation_policy": {
            "minimum_clearance_m": minimum_clearance_m,
            "maximum_speed_mps": maximum_speed_mps,
            "replan_triggers": list(replan_triggers),
        },
        "required_output_skeleton": {
            "reasoning": {
                "task_parse": {
                    "action": "string",
                    "target": "string",
                    "constraints": [],
                },
                "grounding": {"entity_id": "scene entity ID", "confidence": 0.0},
                "candidate_ids": [],
                "evaluations": [],
                "decision": {
                    "selected_candidate_id": None,
                    "reason_codes": [],
                },
            },
            "signal": "all NavigationPlan-v1 signal fields",
        },
        "output_contract": "NavigationPlan-v1",
    }
    return NAVIGATION_SYSTEM_PROMPT, json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
