"""Learnable scene queries that compress local fused tokens."""

from dataclasses import dataclass

import torch
from torch import nn

from .cross_modal_fusion import FusedSceneTokens, MaskedCrossAttention


@dataclass
class SceneQueryResult:
    """Fixed-count scene tokens and their source-token attribution."""

    tokens: FusedSceneTokens
    source_attention: torch.Tensor

    def __post_init__(self) -> None:
        batch_size, query_count = self.tokens.mask.shape
        if self.source_attention.ndim != 3:
            raise ValueError("source_attention must have shape [B, Q, N]")
        if self.source_attention.shape[:2] != (batch_size, query_count):
            raise ValueError("source_attention batch/query dimensions do not match")
        if not torch.isfinite(self.source_attention).all():
            raise ValueError("source_attention must be finite")


class SceneQueryAggregator(nn.Module):
    """Compress variable sensor tokens into fixed scene-level queries."""

    def __init__(
        self,
        source_size: int = 512,
        hidden_size: int = 512,
        output_size: int = 2048,
        scene_queries: int = 64,
        attention_heads: int = 16,
        feed_forward_ratio: int = 2,
    ) -> None:
        super().__init__()
        if scene_queries <= 0 or feed_forward_ratio <= 0:
            raise ValueError("scene_queries and feed_forward_ratio must be positive")
        self.source_size = source_size
        self.hidden_size = hidden_size
        self.query = nn.Parameter(torch.empty(1, scene_queries, hidden_size))
        nn.init.normal_(self.query, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.source_projection = nn.Sequential(
            nn.LayerNorm(source_size), nn.Linear(source_size, hidden_size)
        )
        self.attention = MaskedCrossAttention(hidden_size, attention_heads)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * feed_forward_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * feed_forward_ratio, hidden_size),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, output_size)
        )

    def forward(self, source: FusedSceneTokens) -> SceneQueryResult:
        if source.hidden_size != self.source_size:
            raise ValueError("source hidden size does not match scene-query source size")
        batch_size = source.features.shape[0]
        query = self.query.to(source.features).expand(batch_size, -1, -1)
        allowed = source.mask.unsqueeze(1).expand(-1, query.shape[1], -1)
        context, attention = self.attention(
            self.query_norm(query), self.source_projection(source.features), allowed
        )
        features = query + context
        features = features + self.feed_forward(self.output_norm(features))
        features = self.output_projection(features)
        positions = torch.matmul(attention.to(source.positions.dtype), source.positions)
        confidence = torch.matmul(
            attention.to(source.confidence.dtype), source.confidence.unsqueeze(-1)
        ).squeeze(-1)
        uncertainty = torch.matmul(
            attention.to(source.uncertainty.dtype), source.uncertainty.unsqueeze(-1)
        ).squeeze(-1)
        modality_weights = torch.matmul(
            attention.to(source.modality_weights.dtype), source.modality_weights
        )
        modality_weights = modality_weights / modality_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        has_source = source.mask.any(dim=-1, keepdim=True)
        mask = has_source.expand(-1, query.shape[1])
        features = features * mask.unsqueeze(-1).to(features.dtype)
        confidence = confidence.clamp(0.0, 1.0) * mask.to(confidence.dtype)
        uncertainty = uncertainty.clamp(0.0, 1.0)
        uncertainty = torch.where(mask, uncertainty, torch.ones_like(uncertainty))
        tokens = FusedSceneTokens(
            features=features,
            positions=positions,
            mask=mask,
            confidence=confidence,
            modality_weights=modality_weights,
            uncertainty=uncertainty,
        )
        return SceneQueryResult(tokens=tokens, source_attention=attention)
