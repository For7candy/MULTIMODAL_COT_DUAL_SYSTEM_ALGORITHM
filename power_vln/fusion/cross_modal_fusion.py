"""Local bidirectional RGB/point-cloud fusion with pose-conditioned gates."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from power_vln.encoders import TokenBatch

from .geometric_alignment import AlignedTokenBatch, build_local_neighborhood_mask


@dataclass
class FusedSceneTokens:
    """Spatial tokens with auditable RGB/point contribution weights."""

    features: torch.Tensor
    positions: torch.Tensor
    mask: torch.Tensor
    confidence: torch.Tensor
    modality_weights: torch.Tensor
    uncertainty: torch.Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must have shape [B, N, D]")
        batch_size, token_count, _ = self.features.shape
        if self.positions.shape != (batch_size, token_count, 3):
            raise ValueError("positions must have shape [B, N, 3]")
        if self.mask.shape != (batch_size, token_count) or self.mask.dtype != torch.bool:
            raise ValueError("mask must be bool with shape [B, N]")
        for name, value in (
            ("confidence", self.confidence),
            ("uncertainty", self.uncertainty),
        ):
            if value.shape != (batch_size, token_count):
                raise ValueError("{} must have shape [B, N]".format(name))
            if not torch.isfinite(value).all() or torch.any((value < 0) | (value > 1)):
                raise ValueError("{} must be finite within [0, 1]".format(name))
        if self.modality_weights.shape != (batch_size, token_count, 2):
            raise ValueError("modality_weights must have shape [B, N, 2]")
        if not torch.isfinite(self.features).all():
            raise ValueError("features must be finite")
        if not torch.isfinite(self.positions).all():
            raise ValueError("positions must be finite")
        if torch.any((self.modality_weights < 0) | (self.modality_weights > 1)):
            raise ValueError("modality_weights must be within [0, 1]")
        weight_sum = self.modality_weights.sum(dim=-1)
        if not torch.allclose(
            weight_sum[self.mask], torch.ones_like(weight_sum[self.mask]), atol=1e-4
        ):
            raise ValueError("valid modality weights must sum to one")

    @property
    def hidden_size(self) -> int:
        return self.features.shape[-1]


class MaskedCrossAttention(nn.Module):
    """Multi-head attention whose allowed query-key pairs are explicit."""

    def __init__(self, hidden_size: int, attention_heads: int) -> None:
        super().__init__()
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        self.attention_heads = attention_heads
        self.head_size = hidden_size // attention_heads
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        allowed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_count, hidden_size = query.shape
        key_count = key_value.shape[1]
        if allowed.shape != (batch_size, query_count, key_count):
            raise ValueError("allowed must have shape [B, N_query, N_key]")
        query_heads = self.query(query).reshape(
            batch_size, query_count, self.attention_heads, self.head_size
        ).transpose(1, 2)
        key_heads = self.key(key_value).reshape(
            batch_size, key_count, self.attention_heads, self.head_size
        ).transpose(1, 2)
        value_heads = self.value(key_value).reshape(
            batch_size, key_count, self.attention_heads, self.head_size
        ).transpose(1, 2)
        logits = torch.matmul(query_heads.float(), key_heads.float().transpose(-1, -2))
        logits = logits * (self.head_size ** -0.5)
        pair_mask = allowed.unsqueeze(1)
        logits = logits.masked_fill(~pair_mask, -1e4)
        attention = torch.softmax(logits, dim=-1) * pair_mask.to(logits.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        context = torch.matmul(attention.to(value_heads.dtype), value_heads)
        context = context.transpose(1, 2).reshape(batch_size, query_count, hidden_size)
        context = self.output(context)
        has_context = allowed.any(dim=-1, keepdim=True)
        context = context * has_context.to(context.dtype)
        return context, attention.mean(dim=1)


class _FusionLayer(nn.Module):
    def __init__(
        self, hidden_size: int, attention_heads: int, feed_forward_ratio: int
    ) -> None:
        super().__init__()
        self.rgb_norm = nn.LayerNorm(hidden_size)
        self.point_norm = nn.LayerNorm(hidden_size)
        self.rgb_from_point = MaskedCrossAttention(hidden_size, attention_heads)
        self.point_from_rgb = MaskedCrossAttention(hidden_size, attention_heads)
        self.rgb_ffn_norm = nn.LayerNorm(hidden_size)
        self.point_ffn_norm = nn.LayerNorm(hidden_size)
        self.rgb_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * feed_forward_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * feed_forward_ratio, hidden_size),
        )
        self.point_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * feed_forward_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * feed_forward_ratio, hidden_size),
        )

    def forward(self, rgb, point, local_mask):
        normalized_rgb = self.rgb_norm(rgb)
        normalized_point = self.point_norm(point)
        rgb_context, rgb_attention = self.rgb_from_point(
            normalized_rgb, normalized_point, local_mask
        )
        point_context, point_attention = self.point_from_rgb(
            normalized_point, normalized_rgb, local_mask.transpose(1, 2)
        )
        rgb = rgb + rgb_context
        point = point + point_context
        rgb = rgb + self.rgb_ffn(self.rgb_ffn_norm(rgb))
        point = point + self.point_ffn(self.point_ffn_norm(point))
        return rgb, point, rgb_context, point_context, rgb_attention, point_attention


class BidirectionalCrossModalFusion(nn.Module):
    """Fuse geometrically local RGB and point tokens, conditioned on robot pose."""

    def __init__(
        self,
        input_size: int = 2048,
        hidden_size: int = 512,
        layers: int = 4,
        attention_heads: int = 16,
        feed_forward_ratio: int = 2,
        neighborhood_radius_m: float = 0.5,
    ) -> None:
        super().__init__()
        if layers <= 0 or feed_forward_ratio <= 0 or neighborhood_radius_m <= 0:
            raise ValueError("layers, ratio, and neighborhood_radius_m must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.neighborhood_radius_m = neighborhood_radius_m
        self.rgb_input = nn.Sequential(nn.LayerNorm(input_size), nn.Linear(input_size, hidden_size))
        self.point_input = nn.Sequential(
            nn.LayerNorm(input_size), nn.Linear(input_size, hidden_size)
        )
        self.pose_input = nn.Sequential(
            nn.LayerNorm(input_size), nn.Linear(input_size, hidden_size)
        )
        self.layers = nn.ModuleList(
            _FusionLayer(hidden_size, attention_heads, feed_forward_ratio)
            for _ in range(layers)
        )
        self.rgb_gate = nn.Linear(hidden_size * 3, 1)
        self.point_gate = nn.Linear(hidden_size * 3, 1)
        self.missing_rgb = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.missing_point = nn.Parameter(torch.empty(1, 1, hidden_size))
        nn.init.normal_(self.missing_rgb, std=0.02)
        nn.init.normal_(self.missing_point, std=0.02)

    def _validate_pose(self, pose: TokenBatch, batch_size: int) -> torch.Tensor:
        if pose.modality != "pose" or pose.features.shape != (
            batch_size,
            1,
            self.input_size,
        ):
            raise ValueError("pose must contain one pose token per batch")
        return self.pose_input(pose.features[:, 0])

    def _single_modality(
        self,
        available: AlignedTokenBatch,
        pose: TokenBatch,
        missing_rgb: bool,
    ) -> FusedSceneTokens:
        tokens = available.tokens
        batch_size, token_count, input_size = tokens.features.shape
        if input_size != self.input_size:
            raise ValueError("token hidden size does not match fusion input size")
        self._validate_pose(pose, batch_size)
        input_projector = self.point_input if missing_rgb else self.rgb_input
        available_features = input_projector(tokens.features)
        missing_parameter = self.missing_rgb if missing_rgb else self.missing_point
        missing_feature = missing_parameter.to(available_features).expand(
            batch_size, -1, -1
        )
        missing_position = pose.positions[:, :1].to(tokens.positions)
        features = torch.cat((available_features, missing_feature), dim=1)
        positions = torch.cat((tokens.positions, missing_position), dim=1)
        mask = torch.cat(
            (
                tokens.mask,
                torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.mask.device),
            ),
            dim=1,
        )
        confidence = torch.cat(
            (tokens.confidence, tokens.confidence.new_zeros(batch_size, 1)), dim=1
        )
        if missing_rgb:
            available_weight = available_features.new_tensor((0.0, 1.0))
            missing_weight = available_features.new_tensor((1.0, 0.0))
        else:
            available_weight = available_features.new_tensor((1.0, 0.0))
            missing_weight = available_features.new_tensor((0.0, 1.0))
        modality_weights = torch.cat(
            (
                available_weight.view(1, 1, 2).expand(batch_size, token_count, 2),
                missing_weight.view(1, 1, 2).expand(batch_size, 1, 2),
            ),
            dim=1,
        )
        uncertainty = 1.0 - confidence
        features = features * mask.unsqueeze(-1).to(features.dtype)
        return FusedSceneTokens(
            features, positions, mask, confidence, modality_weights, uncertainty
        )

    def forward(
        self,
        rgb: Optional[AlignedTokenBatch],
        point_cloud: Optional[AlignedTokenBatch],
        pose: TokenBatch,
    ) -> FusedSceneTokens:
        if rgb is None and point_cloud is None:
            raise ValueError("at least one sensor modality is required")
        if rgb is None:
            return self._single_modality(point_cloud, pose, missing_rgb=True)
        if point_cloud is None:
            return self._single_modality(rgb, pose, missing_rgb=False)

        rgb_tokens = rgb.tokens
        point_tokens = point_cloud.tokens
        batch_size = rgb_tokens.features.shape[0]
        if point_tokens.features.shape[0] != batch_size:
            raise ValueError("modality batch sizes must match")
        if rgb_tokens.hidden_size != self.input_size:
            raise ValueError("RGB hidden size does not match fusion input size")
        if point_tokens.hidden_size != self.input_size:
            raise ValueError("point hidden size does not match fusion input size")
        pose_feature = self._validate_pose(pose, batch_size)
        local_mask = build_local_neighborhood_mask(
            rgb, point_cloud, self.neighborhood_radius_m
        )

        rgb_state = self.rgb_input(rgb_tokens.features)
        point_state = self.point_input(point_tokens.features)
        rgb_context = torch.zeros_like(rgb_state)
        point_context = torch.zeros_like(point_state)
        for layer in self.layers:
            (
                rgb_state,
                point_state,
                rgb_context,
                point_context,
                rgb_attention,
                point_attention,
            ) = layer(rgb_state, point_state, local_mask)

        rgb_has_context = local_mask.any(dim=-1)
        point_has_context = local_mask.any(dim=1)
        pose_for_rgb = pose_feature.unsqueeze(1).expand(-1, rgb_state.shape[1], -1)
        pose_for_point = pose_feature.unsqueeze(1).expand(-1, point_state.shape[1], -1)
        rgb_weight_for_rgb_tokens = torch.sigmoid(
            self.rgb_gate(torch.cat((rgb_state, rgb_context, pose_for_rgb), dim=-1))
        ).squeeze(-1)
        rgb_weight_for_point_tokens = torch.sigmoid(
            self.point_gate(
                torch.cat((point_context, point_state, pose_for_point), dim=-1)
            )
        ).squeeze(-1)
        rgb_weight_for_rgb_tokens = torch.where(
            rgb_has_context,
            rgb_weight_for_rgb_tokens,
            torch.ones_like(rgb_weight_for_rgb_tokens),
        )
        rgb_weight_for_point_tokens = torch.where(
            point_has_context,
            rgb_weight_for_point_tokens,
            torch.zeros_like(rgb_weight_for_point_tokens),
        )

        fused_rgb = (
            rgb_weight_for_rgb_tokens.unsqueeze(-1) * rgb_state
            + (1.0 - rgb_weight_for_rgb_tokens).unsqueeze(-1) * rgb_context
        )
        fused_point = (
            rgb_weight_for_point_tokens.unsqueeze(-1) * point_context
            + (1.0 - rgb_weight_for_point_tokens).unsqueeze(-1) * point_state
        )
        rgb_context_confidence = torch.matmul(
            rgb_attention.to(point_tokens.confidence.dtype),
            point_tokens.confidence.unsqueeze(-1),
        ).squeeze(-1)
        point_context_confidence = torch.matmul(
            point_attention.to(rgb_tokens.confidence.dtype),
            rgb_tokens.confidence.unsqueeze(-1),
        ).squeeze(-1)
        fused_rgb_confidence = (
            rgb_weight_for_rgb_tokens * rgb_tokens.confidence
            + (1.0 - rgb_weight_for_rgb_tokens) * rgb_context_confidence
        )
        fused_point_confidence = (
            rgb_weight_for_point_tokens * point_context_confidence
            + (1.0 - rgb_weight_for_point_tokens) * point_tokens.confidence
        )

        features = torch.cat((fused_rgb, fused_point), dim=1)
        positions = torch.cat((rgb_tokens.positions, point_tokens.positions), dim=1)
        mask = torch.cat((rgb_tokens.mask, point_tokens.mask), dim=1)
        confidence = torch.cat(
            (fused_rgb_confidence, fused_point_confidence), dim=1
        ).clamp(0.0, 1.0)
        rgb_modality = torch.stack(
            (rgb_weight_for_rgb_tokens, 1.0 - rgb_weight_for_rgb_tokens), dim=-1
        )
        point_modality = torch.stack(
            (rgb_weight_for_point_tokens, 1.0 - rgb_weight_for_point_tokens), dim=-1
        )
        modality_weights = torch.cat((rgb_modality, point_modality), dim=1)
        features = features * mask.unsqueeze(-1).to(features.dtype)
        confidence = confidence * mask.to(confidence.dtype)
        uncertainty = torch.where(mask, 1.0 - confidence, torch.ones_like(confidence))
        return FusedSceneTokens(
            features,
            positions,
            mask,
            confidence,
            modality_weights,
            uncertainty,
        )
