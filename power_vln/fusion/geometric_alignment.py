"""Tensor geometry for lifting image tokens and aligning sensor frames."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from power_vln.encoders import TokenBatch


@dataclass
class AlignedTokenBatch:
    """Tokens plus a mask stating which anchors have reliable 3D geometry."""

    tokens: TokenBatch
    geometry_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.geometry_mask.shape != self.tokens.mask.shape:
            raise ValueError("geometry_mask must have shape [B, N]")
        if self.geometry_mask.dtype != torch.bool:
            raise ValueError("geometry_mask must use torch.bool")
        if self.geometry_mask.device != self.tokens.features.device:
            raise ValueError("geometry_mask and tokens must share a device")
        if torch.any(self.geometry_mask & ~self.tokens.mask):
            raise ValueError("padded tokens cannot have valid geometry")


def _validate_transform(transform: torch.Tensor, batch_size: int) -> None:
    if transform.shape != (batch_size, 4, 4):
        raise ValueError("transform must have shape [B, 4, 4]")
    if not torch.isfinite(transform).all():
        raise ValueError("transform must be finite")


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply batched homogeneous transforms to ``points[B, N, 3]``."""

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B, N, 3]")
    _validate_transform(transform, points.shape[0])
    matrix = transform.to(device=points.device, dtype=points.dtype)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    transformed = torch.einsum("bij,bnj->bni", matrix, homogeneous)
    scale = transformed[..., 3:].clamp_min(1e-8)
    return transformed[..., :3] / scale


def transform_token_positions(
    tokens: TokenBatch,
    world_from_sensor: torch.Tensor,
) -> AlignedTokenBatch:
    """Transform metric token anchors from one sensor frame into world frame."""

    positions = transform_points(tokens.positions, world_from_sensor)
    transformed = TokenBatch(
        features=tokens.features,
        positions=positions,
        mask=tokens.mask,
        confidence=tokens.confidence,
        modality=tokens.modality,
    )
    return AlignedTokenBatch(transformed, tokens.mask.clone())


def lift_rgb_tokens_to_world(
    rgb_tokens: TokenBatch,
    depth_m: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    world_from_camera: torch.Tensor,
    image_size: Tuple[int, int],
    missing_depth_confidence: float = 0.25,
) -> AlignedTokenBatch:
    """Back-project normalized RGB anchors using per-token metric depth."""

    batch_size, token_count = rgb_tokens.mask.shape
    if rgb_tokens.positions.shape != (batch_size, token_count, 3):
        raise ValueError("RGB positions must have shape [B, N, 3]")
    if depth_m.shape != (batch_size, token_count):
        raise ValueError("depth_m must have shape [B, N]")
    if camera_intrinsics.shape != (batch_size, 3, 3):
        raise ValueError("camera_intrinsics must have shape [B, 3, 3]")
    _validate_transform(world_from_camera, batch_size)
    image_height, image_width = image_size
    if image_height <= 1 or image_width <= 1:
        raise ValueError("image_size must be greater than one pixel per axis")
    if not 0.0 <= missing_depth_confidence <= 1.0:
        raise ValueError("missing_depth_confidence must be within [0, 1]")

    intrinsics = camera_intrinsics.to(rgb_tokens.positions)
    fx = intrinsics[:, 0, 0:1]
    fy = intrinsics[:, 1, 1:2]
    cx = intrinsics[:, 0, 2:3]
    cy = intrinsics[:, 1, 2:3]
    if torch.any(fx.abs() < 1e-8) or torch.any(fy.abs() < 1e-8):
        raise ValueError("camera focal lengths must be non-zero")

    normalized_x = rgb_tokens.positions[..., 0]
    normalized_y = rgb_tokens.positions[..., 1]
    pixel_x = (normalized_x + 1.0) * 0.5 * (image_width - 1)
    pixel_y = (normalized_y + 1.0) * 0.5 * (image_height - 1)
    depth = depth_m.to(rgb_tokens.positions)
    geometry_mask = torch.isfinite(depth) & (depth > 0) & rgb_tokens.mask
    safe_depth = torch.where(geometry_mask, depth, torch.ones_like(depth))
    camera_points = torch.stack(
        (
            (pixel_x - cx) / fx * safe_depth,
            (pixel_y - cy) / fy * safe_depth,
            safe_depth,
        ),
        dim=-1,
    )
    world_points = transform_points(camera_points, world_from_camera)
    camera_origin = world_from_camera[:, :3, 3].to(rgb_tokens.positions)
    world_points = torch.where(
        geometry_mask.unsqueeze(-1),
        world_points,
        camera_origin.unsqueeze(1),
    )
    geometry_scale = torch.where(
        geometry_mask,
        torch.ones_like(depth),
        torch.full_like(depth, missing_depth_confidence),
    )
    confidence = rgb_tokens.confidence * geometry_scale.to(rgb_tokens.confidence)
    lifted = TokenBatch(
        features=rgb_tokens.features,
        positions=world_points,
        mask=rgb_tokens.mask,
        confidence=confidence,
        modality=rgb_tokens.modality,
    )
    return AlignedTokenBatch(lifted, geometry_mask)


def align_modalities_to_world(
    rgb_tokens: TokenBatch,
    point_tokens: TokenBatch,
    depth_m: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    world_from_camera: torch.Tensor,
    image_size: Tuple[int, int],
    world_from_point_cloud: Optional[torch.Tensor] = None,
) -> Tuple[AlignedTokenBatch, AlignedTokenBatch]:
    """Align image and point tokens to one metric world frame."""

    rgb_world = lift_rgb_tokens_to_world(
        rgb_tokens,
        depth_m,
        camera_intrinsics,
        world_from_camera,
        image_size,
    )
    if world_from_point_cloud is None:
        point_world = AlignedTokenBatch(point_tokens, point_tokens.mask.clone())
    else:
        point_world = transform_token_positions(point_tokens, world_from_point_cloud)
    return rgb_world, point_world


def build_local_neighborhood_mask(
    rgb: AlignedTokenBatch,
    point_cloud: AlignedTokenBatch,
    radius_m: float,
) -> torch.Tensor:
    """Build ``[B, N_rgb, N_pc]`` mask for geometrically local attention."""

    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if rgb.tokens.features.shape[0] != point_cloud.tokens.features.shape[0]:
        raise ValueError("modality batch sizes must match")
    distance = torch.cdist(rgb.tokens.positions.float(), point_cloud.tokens.positions.float())
    validity = rgb.geometry_mask.unsqueeze(-1) & point_cloud.geometry_mask.unsqueeze(1)
    return (distance <= radius_m) & validity
