from copy import deepcopy
import unittest

from power_vln.model import (
    NavigationPlanningContext,
    NavigationPlanningError,
    StructuredCoTPlanner,
)
from power_vln.schemas import (
    CandidateApproach,
    Entity3D,
    Pose3D,
    Quaternion,
    SceneDescription3D,
    Vector3,
)


class FakeBackbone:
    def __init__(self, output):
        self.output = output
        self.requests = []

    def generate_structured(self, request):
        self.requests.append(request)
        return deepcopy(self.output)


def sample_scene():
    pose = Pose3D(Vector3(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0))
    entity = Entity3D(
        entity_id="chair-1",
        category="chair",
        center=Vector3(2.0, 1.0, 0.5),
        size=Vector3(0.5, 0.5, 1.0),
        yaw=0.0,
        attributes=("red",),
        confidence=0.9,
        source_observations=("rgb-1", "pc-1"),
    )
    candidates = (
        CandidateApproach(
            "candidate-a", "chair-1", Vector3(1.0, 1.0, 0.0), 0.0, 0.8, 0.9, 0.85
        ),
        CandidateApproach(
            "candidate-b", "chair-1", Vector3(2.0, 0.0, 0.0), 1.57, 0.7, 0.8, 0.75
        ),
    )
    return SceneDescription3D(
        schema_version=1,
        scene_id="scene-1",
        timestamp_ms=1000,
        frame_id="world",
        observer_pose=pose,
        entities=(entity,),
        candidate_approaches=candidates,
    )


def ready_payload():
    return {
        "reasoning": {
            "task_parse": {
                "action": "approach",
                "target": "red chair",
                "constraints": [],
            },
            "grounding": {"entity_id": "chair-1", "confidence": 0.95},
            "candidate_ids": ["candidate-a", "candidate-b"],
            "evaluations": [
                {
                    "candidate_id": "candidate-a",
                    "reachable": True,
                    "score": 0.9,
                    "reason_codes": ["clear", "visible"],
                },
                {
                    "candidate_id": "candidate-b",
                    "reachable": True,
                    "score": 0.7,
                    "reason_codes": ["longer_path"],
                },
            ],
            "decision": {
                "selected_candidate_id": "candidate-a",
                "reason_codes": ["highest_reachable_score"],
            },
        },
        "signal": {
            "schema_version": 1,
            "mission_id": "mission-1",
            "scene_id": "scene-1",
            "subtask_id": "step-1",
            "intent": "approach",
            "status": "ready",
            "constraints": {
                "minimum_clearance_m": 0.5,
                "maximum_speed_mps": 0.3,
                "forbidden_region_ids": [],
            },
            "confidence": 0.9,
            "generated_at_ms": 1200,
            "valid_for_ms": 5000,
            "replan_triggers": ["scene_changed", "path_blocked", "target_missing"],
            "target_entity_id": "chair-1",
            "candidate_id": "candidate-a",
            "goal": {
                "frame_id": "world",
                "position": {"x": 1.0, "y": 1.0, "z": 0.0},
                "yaw": 0.0,
                "tolerance_m": 0.2,
            },
        },
    }


def planning_context():
    return NavigationPlanningContext(
        instruction="Approach the red chair.",
        mission_id="mission-1",
        subtask_id="step-1",
        generated_at_ms=1200,
        valid_for_ms=5000,
        task_history=("entered room",),
    )


class StructuredCoTPlannerTests(unittest.TestCase):
    def test_ready_plan_is_traceable_to_highest_ranked_scene_candidate(self):
        backend = FakeBackbone(ready_payload())
        planner = StructuredCoTPlanner(backend, hidden_size=16)
        result = planner.plan(sample_scene(), planning_context())
        self.assertEqual(result.reasoning.grounding.entity_id, "chair-1")
        self.assertEqual(result.signal.candidate_id, "candidate-a")
        self.assertEqual(backend.requests[0].embeddings.shape, (1, 0, 16))
        self.assertIn("task_history", backend.requests[0].user_prompt)

    def test_same_input_and_output_are_deterministic(self):
        planner = StructuredCoTPlanner(FakeBackbone(ready_payload()), hidden_size=16)
        first = planner.plan(sample_scene(), planning_context())
        second = planner.plan(sample_scene(), planning_context())
        self.assertEqual(first, second)

    def test_ambiguous_target_has_no_executable_goal(self):
        payload = ready_payload()
        payload["reasoning"]["grounding"] = None
        payload["reasoning"]["candidate_ids"] = []
        payload["reasoning"]["evaluations"] = []
        payload["reasoning"]["decision"] = {
            "selected_candidate_id": None,
            "reason_codes": ["multiple_matching_entities"],
        }
        payload["signal"].update(
            {
                "intent": "observe",
                "status": "ambiguous_target",
                "confidence": 0.4,
                "target_entity_id": None,
                "candidate_id": None,
                "goal": None,
            }
        )
        result = StructuredCoTPlanner(
            FakeBackbone(payload), hidden_size=16
        ).plan(sample_scene(), planning_context())
        self.assertEqual(result.signal.status.value, "ambiguous_target")
        self.assertIsNone(result.signal.goal)

    def test_non_ready_signal_cannot_smuggle_executable_target(self):
        payload = ready_payload()
        payload["signal"]["status"] = "need_observation"
        planner = StructuredCoTPlanner(FakeBackbone(payload), hidden_size=16)
        with self.assertRaises(NavigationPlanningError) as caught:
            planner.plan(sample_scene(), planning_context())
        self.assertEqual(caught.exception.code, "unsafe_non_ready_signal")

    def test_lower_ranked_candidate_is_rejected(self):
        payload = ready_payload()
        payload["reasoning"]["decision"]["selected_candidate_id"] = "candidate-b"
        payload["signal"]["candidate_id"] = "candidate-b"
        payload["signal"]["goal"] = {
            "frame_id": "world",
            "position": {"x": 2.0, "y": 0.0, "z": 0.0},
            "yaw": 1.57,
            "tolerance_m": 0.2,
        }
        planner = StructuredCoTPlanner(FakeBackbone(payload), hidden_size=16)
        with self.assertRaises(NavigationPlanningError) as caught:
            planner.plan(sample_scene(), planning_context())
        self.assertEqual(
            caught.exception.code, "non_deterministic_candidate_selection"
        )

    def test_unknown_candidate_is_rejected_before_signal_execution(self):
        payload = ready_payload()
        payload["reasoning"]["candidate_ids"][0] = "invented"
        payload["reasoning"]["evaluations"][0]["candidate_id"] = "invented"
        payload["reasoning"]["decision"]["selected_candidate_id"] = "invented"
        planner = StructuredCoTPlanner(FakeBackbone(payload), hidden_size=16)
        with self.assertRaises(NavigationPlanningError) as caught:
            planner.plan(sample_scene(), planning_context())
        self.assertEqual(caught.exception.code, "invalid_reasoning_references")


if __name__ == "__main__":
    unittest.main()
