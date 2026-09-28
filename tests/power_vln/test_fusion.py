import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from power_vln.encoders import TokenBatch
    from power_vln.fusion import (
        AlignedTokenBatch,
        BidirectionalCrossModalFusion,
        SceneQueryAggregator,
        build_local_neighborhood_mask,
        lift_rgb_tokens_to_world,
        transform_token_positions,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def _tokens(self, features, positions, modality, mask=None, confidence=None):
        batch_size, token_count, _ = features.shape
        if mask is None:
            mask = torch.ones(
                batch_size, token_count, dtype=torch.bool, device=features.device
            )
        if confidence is None:
            confidence = torch.ones(
                batch_size, token_count, dtype=features.dtype, device=features.device
            )
        return TokenBatch(features, positions, mask, confidence, modality)

    def _pose(self, batch_size=1, hidden_size=16, device="cpu", dtype=torch.float32):
        return self._tokens(
            torch.zeros(batch_size, 1, hidden_size, device=device, dtype=dtype),
            torch.zeros(batch_size, 1, 3, device=device, dtype=dtype),
            "pose",
        )

    def test_lift_rgb_tokens_to_world_and_missing_depth(self):
        rgb = self._tokens(
            torch.randn(1, 2, 8),
            torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]),
            "rgb",
        )
        depth = torch.tensor([[2.0, float("nan")]])
        intrinsics = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])
        transform = torch.eye(4).unsqueeze(0)
        transform[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
        aligned = lift_rgb_tokens_to_world(
            rgb, depth, intrinsics, transform, image_size=(3, 3)
        )
        self.assertTrue(torch.allclose(
            aligned.tokens.positions[0, 0], torch.tensor([1.0, 2.0, 5.0])
        ))
        self.assertTrue(torch.allclose(
            aligned.tokens.positions[0, 1], torch.tensor([1.0, 2.0, 3.0])
        ))
        self.assertEqual(aligned.geometry_mask.tolist(), [[True, False]])
        self.assertAlmostEqual(aligned.tokens.confidence[0, 1].item(), 0.25)

    def test_transform_and_neighborhood_mask_are_metric(self):
        rgb_tokens = self._tokens(
            torch.randn(1, 2, 8),
            torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
            "rgb",
        )
        point_tokens = self._tokens(
            torch.randn(1, 2, 8),
            torch.tensor([[[0.1, 0.0, 0.0], [4.0, 0.0, 0.0]]]),
            "point_cloud",
        )
        transform = torch.eye(4).unsqueeze(0)
        transformed = transform_token_positions(point_tokens, transform)
        rgb = AlignedTokenBatch(rgb_tokens, rgb_tokens.mask.clone())
        local = build_local_neighborhood_mask(rgb, transformed, radius_m=0.5)
        self.assertEqual(local.tolist(), [[[True, False], [False, False]]])

    def test_far_point_cannot_change_rgb_through_local_attention(self):
        fusion = BidirectionalCrossModalFusion(
            input_size=16,
            hidden_size=16,
            layers=2,
            attention_heads=4,
            neighborhood_radius_m=0.5,
        ).eval()
        rgb_tokens = self._tokens(
            torch.randn(1, 2, 16),
            torch.tensor([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]),
            "rgb",
        )
        point_features = torch.randn(1, 2, 16)
        point_positions = torch.tensor([[[0.1, 0.0, 0.0], [20.0, 0.0, 0.0]]])
        point_a = self._tokens(point_features.clone(), point_positions, "point_cloud")
        changed = point_features.clone()
        changed[:, 1] = 1000.0
        point_b = self._tokens(changed, point_positions, "point_cloud")
        rgb = AlignedTokenBatch(rgb_tokens, rgb_tokens.mask.clone())
        output_a = fusion(
            rgb,
            AlignedTokenBatch(point_a, point_a.mask.clone()),
            self._pose(),
        )
        output_b = fusion(
            rgb,
            AlignedTokenBatch(point_b, point_b.mask.clone()),
            self._pose(),
        )
        self.assertTrue(torch.allclose(
            output_a.features[:, :2], output_b.features[:, :2], atol=1e-5
        ))
        self.assertTrue(torch.all(
            (output_a.modality_weights >= 0) & (output_a.modality_weights <= 1)
        ))

    def test_missing_rgb_emits_explicit_valid_marker(self):
        fusion = BidirectionalCrossModalFusion(
            input_size=16,
            hidden_size=16,
            layers=1,
            attention_heads=4,
            neighborhood_radius_m=0.5,
        )
        point_tokens = self._tokens(
            torch.randn(2, 3, 16), torch.randn(2, 3, 3), "point_cloud"
        )
        output = fusion(
            None,
            AlignedTokenBatch(point_tokens, point_tokens.mask.clone()),
            self._pose(batch_size=2),
        )
        self.assertEqual(output.features.shape, (2, 4, 16))
        self.assertTrue(output.mask[:, -1].all())
        self.assertTrue(torch.equal(output.confidence[:, -1], torch.zeros(2)))
        self.assertTrue(torch.equal(
            output.modality_weights[:, -1], torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        ))

    def test_scene_queries_preserve_attribution_and_gradients(self):
        fusion = BidirectionalCrossModalFusion(
            input_size=16,
            hidden_size=16,
            layers=1,
            attention_heads=4,
            neighborhood_radius_m=1.0,
        )
        aggregator = SceneQueryAggregator(
            source_size=16,
            hidden_size=16,
            output_size=16,
            scene_queries=5,
            attention_heads=4,
        )
        rgb_features = torch.randn(1, 3, 16, requires_grad=True)
        rgb_tokens = self._tokens(
            rgb_features, torch.randn(1, 3, 3) * 0.1, "rgb"
        )
        point_tokens = self._tokens(
            torch.randn(1, 4, 16), torch.randn(1, 4, 3) * 0.1, "point_cloud"
        )
        fused = fusion(
            AlignedTokenBatch(rgb_tokens, rgb_tokens.mask.clone()),
            AlignedTokenBatch(point_tokens, point_tokens.mask.clone()),
            self._pose(),
        )
        result = aggregator(fused)
        self.assertEqual(result.tokens.features.shape, (1, 5, 16))
        self.assertEqual(result.source_attention.shape, (1, 5, 7))
        self.assertTrue(torch.allclose(
            result.source_attention.sum(dim=-1), torch.ones(1, 5), atol=1e-5
        ))
        self.assertTrue(torch.allclose(
            result.tokens.modality_weights.sum(dim=-1), torch.ones(1, 5), atol=1e-5
        ))
        result.tokens.features.square().mean().backward()
        self.assertIsNotNone(rgb_features.grad)

    def test_production_fusion_parameter_budget(self):
        fusion = BidirectionalCrossModalFusion(
            input_size=2048,
            hidden_size=512,
            layers=4,
            attention_heads=16,
            feed_forward_ratio=2,
            neighborhood_radius_m=0.5,
        )
        aggregator = SceneQueryAggregator(
            source_size=512,
            hidden_size=512,
            output_size=2048,
            scene_queries=64,
            attention_heads=16,
            feed_forward_ratio=2,
        )
        parameter_count = sum(
            parameter.numel()
            for module in (fusion, aggregator)
            for parameter in module.parameters()
        )
        self.assertLess(parameter_count, 30_000_000)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA BF16 is unavailable",
    )
    def test_fusion_runs_on_gpu_in_bfloat16(self):
        device = torch.device("cuda")
        dtype = torch.bfloat16
        fusion = BidirectionalCrossModalFusion(
            input_size=32,
            hidden_size=32,
            layers=1,
            attention_heads=4,
            neighborhood_radius_m=1.0,
        ).to(device=device, dtype=dtype)
        rgb_tokens = self._tokens(
            torch.randn(1, 4, 32, device=device, dtype=dtype),
            torch.randn(1, 4, 3, device=device, dtype=dtype) * 0.1,
            "rgb",
        )
        point_tokens = self._tokens(
            torch.randn(1, 5, 32, device=device, dtype=dtype),
            torch.randn(1, 5, 3, device=device, dtype=dtype) * 0.1,
            "point_cloud",
        )
        output = fusion(
            AlignedTokenBatch(rgb_tokens, rgb_tokens.mask.clone()),
            AlignedTokenBatch(point_tokens, point_tokens.mask.clone()),
            self._pose(hidden_size=32, device=device, dtype=dtype),
        )
        self.assertEqual(output.features.device.type, "cuda")
        self.assertEqual(output.features.dtype, dtype)
        self.assertTrue(torch.isfinite(output.features).all())


if __name__ == "__main__":
    unittest.main()
