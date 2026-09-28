import unittest

from power_vln.adapters import (
    OpenNavBackboneAdapter,
    OpenNavRgbEncoderAdapter,
    OpenNavWaypointAdapter,
)


class FakeRgbEncoder:
    def __call__(self, observations):
        return ("encoded", observations["rgb"])


class FakeWaypointPredictor:
    def __call__(self, rgb_features, depth_features):
        return ("heatmap", rgb_features, depth_features)


class FakeLlmClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def gpt_infer(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.response


class OpenNavAdapterTests(unittest.TestCase):
    def test_rgb_encoder_accepts_injected_model_without_heavy_imports(self):
        adapter = OpenNavRgbEncoderAdapter(encoder=FakeRgbEncoder())
        self.assertEqual(adapter.encode("rgb-batch"), ("encoded", "rgb-batch"))
        self.assertTrue(adapter.is_loaded)

    def test_waypoint_predictor_accepts_injected_model(self):
        adapter = OpenNavWaypointAdapter(predictor=FakeWaypointPredictor())
        self.assertEqual(
            adapter.predict_heatmap("rgb-features", "depth-features"),
            ("heatmap", "rgb-features", "depth-features"),
        )

    def test_backbone_parses_fenced_json_and_adds_schema_instruction(self):
        client = FakeLlmClient('```json\n{"scene_id":"scene-001"}\n```')
        adapter = OpenNavBackboneAdapter(
            model_type="mock-model",
            client=client,
        )
        payload = adapter.generate_structured(
            system_prompt="Describe the scene.",
            user_prompt="input",
            schema_name="SceneDescription3D-v1",
        )
        self.assertEqual(payload, {"scene_id": "scene-001"})
        self.assertIn("SceneDescription3D-v1", client.calls[0][0])

    def test_backbone_rejects_non_object_json(self):
        adapter = OpenNavBackboneAdapter(
            model_type="mock-model",
            client=FakeLlmClient("[]"),
        )
        with self.assertRaises(ValueError):
            adapter.generate_structured("system", "user", "schema")

    def test_unconfigured_backbone_is_rejected(self):
        with self.assertRaises(ValueError):
            OpenNavBackboneAdapter(model_type="UNCONFIGURED")


if __name__ == "__main__":
    unittest.main()
