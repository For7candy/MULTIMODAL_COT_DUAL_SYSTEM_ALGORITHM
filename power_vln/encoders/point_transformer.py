"""A compact Point Transformer implemented only with standard PyTorch ops."""

from typing import Optional, Tuple

import torch
from torch import nn

from .projector import TokenProjector
from .token_batch import TokenBatch


def _batched_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``values[B, N, ...]`` with ``indices[B, Q, K]``."""

    batch = torch.arange(values.shape[0], device=values.device)[:, None, None]
    return values[batch, indices]


def _select_valid_points(
    points: torch.Tensor,
    mask: torch.Tensor,
    limit: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministically select evenly spaced valid points and pad per batch."""

    selected_points = []
    selected_masks = []
    for batch_index in range(points.shape[0]):
        valid_indices = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        count = min(valid_indices.numel(), limit)
        if count:
            if valid_indices.numel() > limit:
                sample_positions = torch.linspace(
                    0,
                    valid_indices.numel() - 1,
                    limit,
                    device=points.device,
                ).round().long()
                chosen = valid_indices[sample_positions]
            else:
                chosen = valid_indices
            current = points[batch_index, chosen]
        else:
            current = points.new_zeros((0, points.shape[-1]))
        if count < limit:
            current = torch.cat(
                (current, points.new_zeros((limit - count, points.shape[-1]))), dim=0
            )
        selected_points.append(current)
        current_mask = torch.zeros(limit, dtype=torch.bool, device=points.device)
        current_mask[:count] = True
        selected_masks.append(current_mask)
    return torch.stack(selected_points), torch.stack(selected_masks)


def chunked_knn_indices(
    positions: torch.Tensor,
    mask: torch.Tensor,
    neighbors: int,
    query_chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return KNN indices and validity without materializing a huge distance cube."""

    if neighbors <= 0 or query_chunk_size <= 0:
        raise ValueError("neighbors and query_chunk_size must be positive")
    batch_size, point_count, _ = positions.shape
    actual_neighbors = min(neighbors, point_count)
    all_indices = []
    all_validity = []
    key_validity = mask[:, None, :]
    # Distance is intentionally FP32: CUDA cdist support for BF16 is incomplete.
    distance_positions = positions.float()
    for start in range(0, point_count, query_chunk_size):
        stop = min(start + query_chunk_size, point_count)
        distance = torch.cdist(distance_positions[:, start:stop], distance_positions)
        distance = distance.masked_fill(~key_validity, torch.inf)
        _, indices = torch.topk(
            distance, k=actual_neighbors, dim=-1, largest=False, sorted=True
        )
        neighbor_validity = _batched_gather(mask.unsqueeze(-1), indices).squeeze(-1)
        neighbor_validity = neighbor_validity & mask[:, start:stop, None]
        all_indices.append(indices)
        all_validity.append(neighbor_validity)
    return torch.cat(all_indices, dim=1), torch.cat(all_validity, dim=1)


class PointTransformerLayer(nn.Module):
    """Local multi-head point attention with relative-position encoding."""

    def __init__(
        self,
        hidden_size: int,
        attention_heads: int,
        neighbors: int,
        query_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.head_size = hidden_size // attention_heads
        self.neighbors = neighbors
        self.query_chunk_size = query_chunk_size
        self.norm1 = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.position_mlp = nn.Sequential(
            nn.Linear(3, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size)
        )
        self.output = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        neighbor_indices: Optional[torch.Tensor] = None,
        neighbor_validity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if neighbor_indices is None or neighbor_validity is None:
            neighbor_indices, neighbor_validity = chunked_knn_indices(
                positions, mask, self.neighbors, self.query_chunk_size
            )
        normalized = self.norm1(features)
        shape = (*normalized.shape[:2], self.attention_heads, self.head_size)
        query = self.query(normalized).reshape(shape)
        key = self.key(normalized).reshape(shape)
        value = self.value(normalized).reshape(shape)
        neighbor_key = _batched_gather(key, neighbor_indices)
        neighbor_value = _batched_gather(value, neighbor_indices)
        neighbor_position = _batched_gather(positions, neighbor_indices)
        relative_position = positions.unsqueeze(2) - neighbor_position
        position_embedding = self.position_mlp(relative_position).reshape(
            *relative_position.shape[:3], self.attention_heads, self.head_size
        )

        logits = (
            (query.unsqueeze(2) - neighbor_key + position_embedding).float()
            * (self.head_size ** -0.5)
        ).sum(dim=-1)
        logits = logits.masked_fill(~neighbor_validity.unsqueeze(-1), -1e4)
        attention = torch.softmax(logits, dim=2)
        attention = attention * neighbor_validity.unsqueeze(-1).to(attention.dtype)
        attention = attention / attention.sum(dim=2, keepdim=True).clamp_min(1e-6)
        attended = (
            (neighbor_value + position_embedding)
            * attention.to(neighbor_value.dtype).unsqueeze(-1)
        ).sum(dim=2)
        attended = attended.reshape_as(features)
        features = features + self.output(attended)
        features = features + self.feed_forward(self.norm2(features))
        return features * mask.unsqueeze(-1).to(features.dtype)


class PointTransformerSmall(nn.Module):
    """12GB-friendly point encoder that emits fixed-count spatial tokens."""

    def __init__(
        self,
        point_feature_size: int = 4,
        hidden_size: int = 256,
        output_size: int = 2048,
        layers: int = 4,
        attention_heads: int = 8,
        neighbors: int = 16,
        maximum_points: int = 2048,
        output_tokens: int = 128,
        query_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if point_feature_size < 3:
            raise ValueError("point_feature_size must include xyz")
        if layers <= 0 or maximum_points <= 0 or output_tokens <= 0:
            raise ValueError("layers and point counts must be positive")
        self.point_feature_size = point_feature_size
        self.maximum_points = maximum_points
        self.output_tokens = output_tokens
        self.input_encoder = nn.Sequential(
            nn.Linear(point_feature_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layers = nn.ModuleList(
            PointTransformerLayer(
                hidden_size,
                attention_heads,
                neighbors,
                query_chunk_size=query_chunk_size,
            )
            for _ in range(layers)
        )
        self.projector = TokenProjector(hidden_size, output_size)

    def forward(
        self,
        points: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> TokenBatch:
        if points.ndim != 3 or points.shape[-1] != self.point_feature_size:
            raise ValueError(
                "points must have shape [B, N, {}]".format(self.point_feature_size)
            )
        if points.shape[1] == 0:
            raise ValueError("points must contain at least one point")
        if not torch.isfinite(points).all():
            raise ValueError("points must be finite")
        if mask is None:
            mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        elif mask.shape != points.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be bool with shape [B, N]")

        sampled_points, sampled_mask = _select_valid_points(
            points, mask, min(self.maximum_points, points.shape[1])
        )
        positions = sampled_points[..., :3]
        features = self.input_encoder(sampled_points)
        neighbor_indices, neighbor_validity = chunked_knn_indices(
            positions,
            sampled_mask,
            self.layers[0].neighbors,
            self.layers[0].query_chunk_size,
        )
        for layer in self.layers:
            features = layer(
                features,
                positions,
                sampled_mask,
                neighbor_indices,
                neighbor_validity,
            )

        token_source = torch.cat((positions, features), dim=-1)
        selected, token_mask = _select_valid_points(
            token_source, sampled_mask, self.output_tokens
        )
        token_positions = selected[..., :3]
        token_features = self.projector(selected[..., 3:])
        token_features = token_features * token_mask.unsqueeze(-1).to(
            token_features.dtype
        )
        confidence = token_mask.to(dtype=token_features.dtype)
        return TokenBatch(
            token_features, token_positions, token_mask, confidence, "point_cloud"
        )
