"""Prompt construction for stage-one structured 3D scene generation."""

import json
from typing import Any, Dict, Tuple


SCENE_SYSTEM_PROMPT = """You are the geometric perception stage of a navigation system.
Infer only facts supported by the supplied world-frame scene tokens and metadata.
Return exactly one JSON object matching the supplied SceneDescription3D JSON Schema.
Use stable unique IDs. Every relation and candidate must reference an existing entity.
Do not invent coordinates, silently repair uncertainty, or add prose outside JSON.
If evidence is insufficient, omit the item and add a structured uncertainty entry."""


def build_scene_prompts(
    instruction: str,
    timestamp_ms: int,
    frame_id: str,
    observer_pose: Dict[str, Any],
    memory_version: int,
    observation_ids: Tuple[str, ...],
    token_summary: Dict[str, Any],
) -> Tuple[str, str]:
    if not instruction.strip() or not frame_id.strip():
        raise ValueError("instruction and frame_id must not be empty")
    payload = {
        "task_instruction": instruction,
        "required_timestamp_ms": timestamp_ms,
        "required_frame_id": frame_id,
        "required_observer_pose": observer_pose,
        "spatial_memory_version": memory_version,
        "source_observation_ids": list(observation_ids),
        "token_summary": token_summary,
        "output_contract": "SceneDescription3D-v1",
    }
    return SCENE_SYSTEM_PROMPT, json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
