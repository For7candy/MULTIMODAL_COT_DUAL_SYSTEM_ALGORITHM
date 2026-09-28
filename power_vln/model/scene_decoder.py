"""Strict SceneDescription3D schema, decoding, and generation orchestration."""

from dataclasses import dataclass
import json
from typing import Any, Dict, Tuple

from power_vln.encoders import TokenBatch
from power_vln.fusion import FusedSceneTokens, SpatialMemorySnapshot
from power_vln.schemas import SceneDescription3D
from power_vln.validators import validate_scene

from .multimodal_backbone import (
    MultimodalBackboneRequest,
    MultimodalInputAssembler,
    SceneGenerationContext,
    StructuredMultimodalBackbone,
)
from .json_schema import validate_json_schema_subset


def _vector_schema(positive: bool = False) -> Dict[str, Any]:
    coordinate = {"type": "number"}
    if positive:
        coordinate["exclusiveMinimum"] = 0
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["x", "y", "z"],
        "properties": {axis: dict(coordinate) for axis in ("x", "y", "z")},
    }


def _confidence_schema() -> Dict[str, Any]:
    return {"type": "number", "minimum": 0, "maximum": 1}


def scene_description_json_schema() -> Dict[str, Any]:
    """Return the strict JSON Schema supplied to grammar-capable backends."""

    vector = _vector_schema()
    positive_vector = _vector_schema(positive=True)
    pose = {
        "type": "object",
        "additionalProperties": False,
        "required": ["position", "orientation"],
        "properties": {
            "position": vector,
            "orientation": {
                "type": "object",
                "additionalProperties": False,
                "required": ["x", "y", "z", "w"],
                "properties": {
                    axis: {"type": "number"} for axis in ("x", "y", "z", "w")
                },
            },
        },
    }
    entity = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "entity_id", "category", "center", "size", "yaw", "attributes",
            "confidence", "source_observations",
        ],
        "properties": {
            "entity_id": {"type": "string", "minLength": 1},
            "category": {"type": "string", "minLength": 1},
            "center": vector,
            "size": positive_vector,
            "yaw": {"type": "number"},
            "attributes": {"type": "array", "items": {"type": "string"}},
            "confidence": _confidence_schema(),
            "source_observations": {
                "type": "array", "minItems": 1, "items": {"type": "string"}
            },
        },
    }
    relation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["subject_id", "predicate", "object_id", "confidence"],
        "properties": {
            "subject_id": {"type": "string"},
            "predicate": {"type": "string"},
            "object_id": {"type": "string"},
            "confidence": _confidence_schema(),
        },
    }
    free_space = {
        "type": "object",
        "additionalProperties": False,
        "required": ["region_id", "vertices", "clearance_height_m", "confidence"],
        "properties": {
            "region_id": {"type": "string"},
            "vertices": {"type": "array", "minItems": 3, "items": vector},
            "clearance_height_m": {"type": "number", "exclusiveMinimum": 0},
            "confidence": _confidence_schema(),
        },
    }
    obstacle = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "obstacle_id", "category", "center", "size", "dynamic",
            "confidence", "observed_at_ms",
        ],
        "properties": {
            "obstacle_id": {"type": "string"},
            "category": {"type": "string"},
            "center": vector,
            "size": positive_vector,
            "dynamic": {"type": "boolean"},
            "confidence": _confidence_schema(),
            "observed_at_ms": {"type": "integer", "minimum": 0},
            "velocity": vector,
        },
    }
    candidate = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "candidate_id", "target_entity_id", "position", "yaw", "clearance_m",
            "visibility_score", "confidence",
        ],
        "properties": {
            "candidate_id": {"type": "string"},
            "target_entity_id": {"type": "string"},
            "position": vector,
            "yaw": {"type": "number"},
            "clearance_m": {"type": "number", "minimum": 0},
            "visibility_score": _confidence_schema(),
            "confidence": _confidence_schema(),
        },
    }
    uncertainty = {
        "type": "object",
        "additionalProperties": False,
        "required": ["uncertainty_type", "description", "confidence"],
        "properties": {
            "uncertainty_type": {"type": "string"},
            "description": {"type": "string"},
            "confidence": _confidence_schema(),
            "entity_id": {"type": ["string", "null"]},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SceneDescription3D-v1",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "scene_id", "timestamp_ms", "frame_id", "observer_pose",
            "entities", "relations", "free_space", "obstacles",
            "candidate_approaches", "uncertainties",
        ],
        "properties": {
            "schema_version": {"const": 1},
            "scene_id": {"type": "string", "minLength": 1},
            "timestamp_ms": {"type": "integer", "minimum": 0},
            "frame_id": {"type": "string", "minLength": 1},
            "observer_pose": pose,
            "entities": {"type": "array", "items": entity},
            "relations": {"type": "array", "items": relation},
            "free_space": {"type": "array", "items": free_space},
            "obstacles": {"type": "array", "items": obstacle},
            "candidate_approaches": {"type": "array", "items": candidate},
            "uncertainties": {"type": "array", "items": uncertainty},
        },
    }


class SceneGenerationError(ValueError):
    def __init__(self, code: str, message: str, details: Tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


@dataclass(frozen=True)
class SceneGenerationResult:
    scene: SceneDescription3D
    memory_version: int
    source_token_count: int


class SceneGenerator:
    """Run stage-one generation and reject, rather than repair, invalid geometry."""

    _TOP_LEVEL_FIELDS = {
        "schema_version", "scene_id", "timestamp_ms", "frame_id", "observer_pose",
        "entities", "relations", "free_space", "obstacles", "candidate_approaches",
        "uncertainties",
    }

    def __init__(
        self,
        assembler: MultimodalInputAssembler,
        backbone: StructuredMultimodalBackbone,
    ) -> None:
        self.assembler = assembler
        self.backbone = backbone
        self.json_schema = scene_description_json_schema()

    def _parse(self, output: Any) -> Dict[str, Any]:
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError as error:
                raise SceneGenerationError(
                    "invalid_json", "backbone output is not valid JSON"
                ) from error
        if not isinstance(output, dict):
            raise SceneGenerationError(
                "invalid_root", "backbone output must be one JSON object"
            )
        unknown = sorted(set(output) - self._TOP_LEVEL_FIELDS)
        missing = sorted(self._TOP_LEVEL_FIELDS - set(output))
        if unknown:
            raise SceneGenerationError(
                "unknown_fields", "scene contains unknown fields", tuple(unknown)
            )
        if missing:
            raise SceneGenerationError(
                "missing_fields", "scene is missing required fields", tuple(missing)
            )
        return output

    def _decode(
        self,
        output: Any,
        context: SceneGenerationContext,
    ) -> SceneDescription3D:
        payload = self._parse(output)
        schema_issues = validate_json_schema_subset(payload, self.json_schema)
        if schema_issues:
            raise SceneGenerationError(
                "json_schema_violation",
                "scene JSON does not conform to SceneDescription3D-v1",
                schema_issues,
            )
        try:
            scene = SceneDescription3D.from_dict(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise SceneGenerationError(
                "schema_violation", "scene fields violate the typed schema", (str(error),)
            ) from error
        if scene.timestamp_ms != context.timestamp_ms:
            raise SceneGenerationError(
                "timestamp_mismatch", "scene timestamp does not match the input packet"
            )
        if scene.frame_id != context.frame_id:
            raise SceneGenerationError(
                "frame_mismatch", "scene frame does not match the aligned world frame"
            )
        if scene.observer_pose != context.observer_pose:
            raise SceneGenerationError(
                "observer_pose_mismatch", "scene observer pose does not match input pose"
            )
        allowed_observations = set(context.observation_ids)
        invalid_sources = sorted(
            {
                source
                for entity in scene.entities
                for source in entity.source_observations
                if source not in allowed_observations
            }
        )
        if invalid_sources:
            raise SceneGenerationError(
                "unknown_source_observation",
                "entity cites an observation not present in the request",
                tuple(invalid_sources),
            )
        validation = validate_scene(scene)
        if not validation.is_valid:
            details = tuple(
                "{}@{}".format(issue.code, issue.path) for issue in validation.issues
            )
            raise SceneGenerationError(
                "invalid_scene_geometry", "scene cross-reference validation failed", details
            )
        return scene

    def generate(
        self,
        current_scene: FusedSceneTokens,
        memory: SpatialMemorySnapshot,
        pose: TokenBatch,
        context: SceneGenerationContext,
    ) -> SceneGenerationResult:
        request: MultimodalBackboneRequest = self.assembler(
            current_scene, memory, pose, context, self.json_schema
        )
        output = self.backbone.generate_structured(request)
        scene = self._decode(output, context)
        return SceneGenerationResult(
            scene=scene,
            memory_version=memory.version,
            source_token_count=int(request.attention_mask.sum()),
        )
