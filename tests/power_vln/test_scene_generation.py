import json
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from power_vln.encoders import TokenBatch
    from power_vln.fusion import FusedSceneTokens, SpatialMemory
    from power_vln.model import (
        MultimodalInputAssembler,
        SceneGenerationContext,
        SceneGenerationError,
        SceneGenerator,
        scene_description_json_schema,
    )
    from power_vln.schemas import Pose3D, Quaternion, Vector3


class FakeStructuredBackbone:
    def __init__(self, output):
        self.output = output
        self.requests = []

    def generate_structured(self, request):
        self.requests.append(request)
        return self.output


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SceneGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.hidden_size = 16
        self.observer_pose = Pose3D(
            position=Vector3(1.0, 2.0, 0.0),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        )
        self.context = SceneGenerationContext(
            instruction="Go to the red chair.",
            timestamp_ms=1200,
            frame_id="world",
            observer_pose=self.observer_pose,
            observation_ids=("rgb-001", "pc-001"),
        )

    def _current_scene(self):
        features = torch.randn(1, 2, self.hidden_size)
        positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        confidence = torch.tensor([[0.9, 0.8]])
        modality = torch.tensor([[[0.6, 0.4], [0.2, 0.8]]])
        return FusedSceneTokens(
            features,
            positions,
            mask,
            confidence,
            modality,
            1.0 - confidence,
        )

    def _pose_token(self):
        return TokenBatch(
            features=torch.randn(1, 1, self.hidden_size),
            positions=torch.tensor([[[1.0, 2.0, 0.0]]]),
            mask=torch.ones(1, 1, dtype=torch.bool),
            confidence=torch.ones(1, 1),
            modality="pose",
        )

    def _memory(self):
        memory = SpatialMemory(feature_size=self.hidden_size)
        return memory.update(self._current_scene(), timestamp_ms=1000)

    def _valid_payload(self):
        return {
            "schema_version": 1,
            "scene_id": "scene-0001",
            "timestamp_ms": 1200,
            "frame_id": "world",
            "observer_pose": self.observer_pose.to_dict(),
            "entities": [
                {
                    "entity_id": "chair-1",
                    "category": "chair",
                    "center": {"x": 2.0, "y": 1.0, "z": 0.5},
                    "size": {"x": 0.5, "y": 0.5, "z": 1.0},
                    "yaw": 0.0,
                    "attributes": ["red"],
                    "confidence": 0.9,
                    "source_observations": ["rgb-001", "pc-001"],
                }
            ],
            "relations": [],
            "free_space": [],
            "obstacles": [],
            "candidate_approaches": [
                {
                    "candidate_id": "approach-1",
                    "target_entity_id": "chair-1",
                    "position": {"x": 1.0, "y": 1.0, "z": 0.0},
                    "yaw": 0.0,
                    "clearance_m": 0.8,
                    "visibility_score": 0.9,
                    "confidence": 0.85,
                }
            ],
            "uncertainties": [],
        }

    def test_assembler_combines_current_memory_and_pose_tokens(self):
        assembler = MultimodalInputAssembler(hidden_size=self.hidden_size)
        memory = self._memory()
        request = assembler(
            self._current_scene(),
            memory,
            self._pose_token(),
            self.context,
            scene_description_json_schema(),
        )
        expected_tokens = 2 + memory.token_count + 1
        self.assertEqual(request.embeddings.shape, (1, expected_tokens, 16))
        self.assertEqual(request.token_type_ids[0, :2].tolist(), [0, 0])
        self.assertEqual(request.token_type_ids[0, -1].item(), 2)
        self.assertIn("Go to the red chair", request.user_prompt)
        self.assertTrue(request.json_schema["additionalProperties"] is False)

    def test_valid_json_decodes_to_typed_scene(self):
        backend = FakeStructuredBackbone(json.dumps(self._valid_payload()))
        generator = SceneGenerator(
            MultimodalInputAssembler(self.hidden_size), backend
        )
        memory = self._memory()
        result = generator.generate(
            self._current_scene(), memory, self._pose_token(), self.context
        )
        self.assertEqual(result.scene.scene_id, "scene-0001")
        self.assertEqual(result.scene.entities[0].entity_id, "chair-1")
        self.assertEqual(result.memory_version, memory.version)
        self.assertEqual(len(backend.requests), 1)

    def test_unknown_output_field_is_rejected_without_repair(self):
        payload = self._valid_payload()
        payload["invented"] = True
        generator = SceneGenerator(
            MultimodalInputAssembler(self.hidden_size),
            FakeStructuredBackbone(payload),
        )
        with self.assertRaises(SceneGenerationError) as caught:
            generator.generate(
                self._current_scene(), self._memory(), self._pose_token(), self.context
            )
        self.assertEqual(caught.exception.code, "unknown_fields")

    def test_invalid_cross_reference_is_rejected(self):
        payload = self._valid_payload()
        payload["candidate_approaches"][0]["target_entity_id"] = "missing"
        generator = SceneGenerator(
            MultimodalInputAssembler(self.hidden_size),
            FakeStructuredBackbone(payload),
        )
        with self.assertRaises(SceneGenerationError) as caught:
            generator.generate(
                self._current_scene(), self._memory(), self._pose_token(), self.context
            )
        self.assertEqual(caught.exception.code, "invalid_scene_geometry")

    def test_unknown_nested_field_is_rejected_by_json_schema(self):
        payload = self._valid_payload()
        payload["entities"][0]["invented_color_score"] = 1.0
        generator = SceneGenerator(
            MultimodalInputAssembler(self.hidden_size),
            FakeStructuredBackbone(payload),
        )
        with self.assertRaises(SceneGenerationError) as caught:
            generator.generate(
                self._current_scene(), self._memory(), self._pose_token(), self.context
            )
        self.assertEqual(caught.exception.code, "json_schema_violation")
        self.assertIn("unknown field", caught.exception.details[0])

    def test_unrecognized_source_observation_is_rejected(self):
        payload = self._valid_payload()
        payload["entities"][0]["source_observations"] = ["hallucinated-camera"]
        generator = SceneGenerator(
            MultimodalInputAssembler(self.hidden_size),
            FakeStructuredBackbone(payload),
        )
        with self.assertRaises(SceneGenerationError) as caught:
            generator.generate(
                self._current_scene(), self._memory(), self._pose_token(), self.context
            )
        self.assertEqual(caught.exception.code, "unknown_source_observation")


if __name__ == "__main__":
    unittest.main()
