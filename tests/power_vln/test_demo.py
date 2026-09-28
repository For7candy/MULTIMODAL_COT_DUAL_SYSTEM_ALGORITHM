from pathlib import Path
import sys
import unittest

# Direct execution sets sys.path to tests/power_vln instead of the repository
# root. Add the root explicitly while keeping unittest discovery unchanged.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.power_vln.demo_algorithm import run_demo


class AlgorithmDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_demo("mock")

    def test_demo_exercises_multimodal_model_outputs(self):
        self.assertEqual(self.result["inputs"]["rgb_shape"], [1, 3, 32, 32])
        self.assertEqual(self.result["inputs"]["point_cloud_shape"], [1, 64, 4])
        self.assertGreater(self.result["features_and_fusion"]["fused_tokens"], 0)
        self.assertEqual(
            self.result["structured_scene"]["entities"][0]["entity_id"],
            "cabinet-1",
        )
        self.assertEqual(
            self.result["structured_cot"]["decision"]["selected_candidate_id"],
            "cabinet-1-front",
        )
        self.assertEqual(self.result["navigation_signal"]["status"], "ready")

    def test_demo_exercises_slow_cache_and_fast_safety(self):
        self.assertFalse(self.result["slow_system"]["first_cache_hit"])
        self.assertTrue(self.result["slow_system"]["second_cache_hit"])
        self.assertEqual(
            self.result["fast_system"]["clear_path"]["feedback"], "active"
        )
        self.assertEqual(
            self.result["fast_system"]["goal_blocked"]["feedback"], "blocked"
        )
        self.assertEqual(
            self.result["fast_system"]["signal_expired"]["feedback"],
            "safety_stop",
        )
        self.assertTrue(
            self.result["fast_system"]["signal_expired"]["action"]["stop"]
        )


if __name__ == "__main__":
    unittest.main()
