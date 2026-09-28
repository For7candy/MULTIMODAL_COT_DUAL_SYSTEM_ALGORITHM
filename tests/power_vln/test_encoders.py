import unittest

try:
    import torch
except ImportError:  # Dependency-free schema tests must still run without torch.
    torch = None

if torch is not None:
    from power_vln.encoders import (
        PointTransformerSmall,
        PoseTokenEncoder,
        RgbTokenEncoder,
        TokenBatch,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_token_batch_enforces_shared_shape_contract(self):
        with self.assertRaises(ValueError):
            TokenBatch(
                features=torch.zeros(2, 4, 8),
                positions=torch.zeros(2, 3, 3),
                mask=torch.ones(2, 4, dtype=torch.bool),
                confidence=torch.ones(2, 4),
                modality="rgb",
            )

    def test_rgb_encoder_outputs_spatial_tokens_and_mask(self):
        encoder = RgbTokenEncoder(
            source_feature_size=16, output_size=32, output_tokens=12
        )
        feature_map = torch.randn(2, 16, 9, 11, requires_grad=True)
        valid_mask = torch.ones(2, 9, 11, dtype=torch.bool)
        valid_mask[1] = False
        output = encoder(feature_map, valid_mask)
        self.assertEqual(output.features.shape, (2, 12, 32))
        self.assertEqual(output.positions.shape, (2, 12, 3))
        self.assertTrue(output.mask[0].all())
        self.assertFalse(output.mask[1].any())
        self.assertEqual(output.modality, "rgb")
        output.features.sum().backward()
        self.assertIsNotNone(feature_map.grad)

    def test_point_transformer_outputs_fixed_tokens_and_gradients(self):
        encoder = PointTransformerSmall(
            point_feature_size=4,
            hidden_size=32,
            output_size=64,
            layers=2,
            attention_heads=4,
            neighbors=4,
            maximum_points=24,
            output_tokens=8,
            query_chunk_size=8,
        )
        points = torch.randn(2, 32, 4, requires_grad=True)
        mask = torch.ones(2, 32, dtype=torch.bool)
        mask[1, 6:] = False
        output = encoder(points, mask)
        self.assertEqual(output.features.shape, (2, 8, 64))
        self.assertEqual(output.positions.shape, (2, 8, 3))
        self.assertEqual(output.mask[0].sum().item(), 8)
        self.assertEqual(output.mask[1].sum().item(), 6)
        self.assertTrue(torch.isfinite(output.features).all())
        output.features.square().mean().backward()
        self.assertIsNotNone(points.grad)

    def test_point_transformer_rejects_empty_cloud(self):
        encoder = PointTransformerSmall(
            point_feature_size=4,
            hidden_size=16,
            output_size=16,
            layers=1,
            attention_heads=4,
            neighbors=2,
            maximum_points=8,
            output_tokens=4,
        )
        with self.assertRaises(ValueError):
            encoder(torch.empty(1, 0, 4))

    def test_pose_encoder_is_invariant_to_quaternion_sign(self):
        encoder = PoseTokenEncoder(
            output_size=64,
            fourier_bands=3,
            include_timestamp=True,
            include_covariance=True,
        )
        translation = torch.tensor([[1.0, -2.0, 0.5]])
        quaternion = torch.tensor([[0.1, 0.2, 0.3, 0.9]])
        timestamp = torch.tensor([12.5])
        covariance = torch.eye(6).unsqueeze(0) * 0.01
        positive = encoder(translation, quaternion, timestamp, covariance)
        negative = encoder(translation, -quaternion, timestamp, covariance)
        self.assertEqual(positive.features.shape, (1, 1, 64))
        self.assertTrue(torch.allclose(positive.features, negative.features, atol=1e-6))
        self.assertGreater(positive.confidence.item(), 0.9)

    def test_pose_encoder_rejects_zero_quaternion(self):
        encoder = PoseTokenEncoder(
            output_size=16,
            fourier_bands=2,
            include_timestamp=False,
            include_covariance=False,
        )
        with self.assertRaises(ValueError):
            encoder(torch.zeros(1, 3), torch.zeros(1, 4))

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA BF16 is unavailable",
    )
    def test_point_transformer_runs_on_local_gpu_in_bfloat16(self):
        device = torch.device("cuda")
        encoder = PointTransformerSmall(
            point_feature_size=4,
            hidden_size=32,
            output_size=64,
            layers=1,
            attention_heads=4,
            neighbors=4,
            maximum_points=32,
            output_tokens=8,
            query_chunk_size=8,
        ).to(device=device, dtype=torch.bfloat16)
        points = torch.randn(1, 40, 4, device=device, dtype=torch.bfloat16)
        output = encoder(points)
        self.assertEqual(output.features.device.type, "cuda")
        self.assertEqual(output.features.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output.features).all())


if __name__ == "__main__":
    unittest.main()
