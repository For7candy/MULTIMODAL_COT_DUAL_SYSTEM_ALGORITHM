"""Model-neutral request contract for inserting spatial tokens into an MLLM."""

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

import torch
from torch import nn

from power_vln.encoders import TokenBatch
from power_vln.fusion import FusedSceneTokens, SpatialMemorySnapshot
from power_vln.prompts import build_scene_prompts
from power_vln.schemas import Pose3D


@dataclass(frozen=True)
class SceneGenerationContext:
    instruction: str
    timestamp_ms: int
    frame_id: str
    observer_pose: Pose3D
    observation_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.instruction.strip() or not self.frame_id.strip():
            raise ValueError("instruction and frame_id must not be empty")
        if self.timestamp_ms < 0 or not self.observation_ids:
            raise ValueError("timestamp must be non-negative and observations non-empty")


@dataclass
class MultimodalBackboneRequest:
    system_prompt: str
    user_prompt: str
    schema_name: str
    json_schema: Dict[str, Any]
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    positions: torch.Tensor
    confidence: torch.Tensor

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [B, N, D]")
        batch_size, token_count, _ = self.embeddings.shape
        if self.attention_mask.shape != (batch_size, token_count):
            raise ValueError("attention_mask must have shape [B, N]")
        if self.attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must use torch.bool")
        if self.token_type_ids.shape != (batch_size, token_count):
            raise ValueError("token_type_ids must have shape [B, N]")
        if self.positions.shape != (batch_size, token_count, 3):
            raise ValueError("positions must have shape [B, N, 3]")
        if self.confidence.shape != (batch_size, token_count):
            raise ValueError("confidence must have shape [B, N]")
        if not self.schema_name or not self.json_schema:
            raise ValueError("schema metadata must not be empty")


class StructuredMultimodalBackbone(Protocol):
    def generate_structured(self, request: MultimodalBackboneRequest) -> Any:
        ...


class MultimodalInputAssembler(nn.Module):
    """Combine current scene, persistent memory, and pose tokens for the MLLM."""

    CURRENT_SCENE = 0
    SPATIAL_MEMORY = 1
    ROBOT_POSE = 2

    def __init__(self, hidden_size: int = 2048) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.token_type_embedding = nn.Embedding(3, hidden_size)
        self.position_projection = nn.Sequential(
            nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(
        self,
        current_scene: FusedSceneTokens,
        memory: SpatialMemorySnapshot,
        pose: TokenBatch,
        context: SceneGenerationContext,
        json_schema: Dict[str, Any],
    ) -> MultimodalBackboneRequest:
        if current_scene.features.shape[0] != 1:
            raise ValueError("scene generation supports one runtime stream")
        if current_scene.hidden_size != self.hidden_size:
            raise ValueError("current scene hidden size does not match backbone")
        if memory.features.shape[-1] != self.hidden_size:
            raise ValueError("memory hidden size does not match backbone")
        if pose.features.shape != (1, 1, self.hidden_size) or pose.modality != "pose":
            raise ValueError("pose must contain one correctly sized pose token")
        if memory.frame_id != context.frame_id:
            raise ValueError("memory and generation context frames must match")

        device = current_scene.features.device
        dtype = current_scene.features.dtype
        memory_features = memory.features.to(device=device, dtype=dtype).unsqueeze(0)
        memory_positions = memory.positions.to(device=device, dtype=dtype).unsqueeze(0)
        memory_confidence = memory.confidence.to(device=device, dtype=dtype).unsqueeze(0)
        memory_mask = torch.ones(
            1, memory.token_count, dtype=torch.bool, device=device
        )
        features = torch.cat(
            (current_scene.features, memory_features, pose.features.to(device, dtype)),
            dim=1,
        )
        positions = torch.cat(
            (current_scene.positions, memory_positions, pose.positions.to(device, dtype)),
            dim=1,
        )
        mask = torch.cat((current_scene.mask, memory_mask, pose.mask.to(device)), dim=1)
        confidence = torch.cat(
            (
                current_scene.confidence,
                memory_confidence,
                pose.confidence.to(device, dtype),
            ),
            dim=1,
        )
        token_type_ids = torch.cat(
            (
                torch.full_like(current_scene.mask, self.CURRENT_SCENE, dtype=torch.long),
                torch.full_like(memory_mask, self.SPATIAL_MEMORY, dtype=torch.long),
                torch.full_like(pose.mask.to(device), self.ROBOT_POSE, dtype=torch.long),
            ),
            dim=1,
        )
        embeddings = features + self.token_type_embedding(token_type_ids)
        embeddings = embeddings + self.position_projection(positions) * confidence.unsqueeze(-1)
        embeddings = embeddings * mask.unsqueeze(-1).to(embeddings.dtype)
        system_prompt, user_prompt = build_scene_prompts(
            instruction=context.instruction,
            timestamp_ms=context.timestamp_ms,
            frame_id=context.frame_id,
            observer_pose=context.observer_pose.to_dict(),
            memory_version=memory.version,
            observation_ids=context.observation_ids,
            token_summary={
                "current_scene_tokens": int(current_scene.mask.sum()),
                "memory_tokens": memory.token_count,
                "pose_tokens": int(pose.mask.sum()),
            },
        )
        return MultimodalBackboneRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="SceneDescription3D-v1",
            json_schema=json_schema,
            embeddings=embeddings,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            positions=positions,
            confidence=confidence,
        )
