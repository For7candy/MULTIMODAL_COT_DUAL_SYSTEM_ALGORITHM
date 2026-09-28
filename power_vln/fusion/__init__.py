"""Geometric alignment and multimodal scene-token fusion."""

from .cross_modal_fusion import (
    BidirectionalCrossModalFusion,
    FusedSceneTokens,
)
from .entity_tracker import (
    EntityObservation,
    EntityTracker,
    EntityTrackerSnapshot,
    TrackedEntity,
)
from .geometric_alignment import (
    AlignedTokenBatch,
    align_modalities_to_world,
    build_local_neighborhood_mask,
    lift_rgb_tokens_to_world,
    transform_token_positions,
)
from .scene_queries import SceneQueryAggregator, SceneQueryResult
from .scene_change_detector import SceneChangeDetector, SceneChangeEvent
from .spatial_memory import SpatialMemory, SpatialMemorySnapshot

__all__ = [
    "AlignedTokenBatch",
    "BidirectionalCrossModalFusion",
    "EntityObservation",
    "EntityTracker",
    "EntityTrackerSnapshot",
    "FusedSceneTokens",
    "SceneQueryAggregator",
    "SceneQueryResult",
    "SceneChangeDetector",
    "SceneChangeEvent",
    "SpatialMemory",
    "SpatialMemorySnapshot",
    "TrackedEntity",
    "align_modalities_to_world",
    "build_local_neighborhood_mask",
    "lift_rgb_tokens_to_world",
    "transform_token_positions",
]
