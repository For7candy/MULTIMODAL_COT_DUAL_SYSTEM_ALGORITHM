from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from power_vln.encoders import TokenBatch
    from power_vln.fusion import FusedSceneTokens, SpatialMemory
    from power_vln.model import (
        NavigationPlanResult,
        NavigationPlanningContext,
        NavigationPlanningError,
        SceneGenerationContext,
    )
    from power_vln.schemas import (
        CandidateApproach,
        CandidateEvaluation,
        Entity3D,
        GroundingResult,
        NavigationConstraints,
        NavigationDecision,
        NavigationGoal3D,
        NavigationIntent,
        NavigationReasoning,
        NavigationSignal,
        NavigationStatus,
        Pose3D,
        Quaternion,
        SceneDescription3D,
        TaskParse,
        Vector3,
    )
    from power_vln.slow_system import (
        SlowEvent,
        SlowEventType,
        SlowSystem,
        SlowSystemInput,
        SlowSystemStatus,
    )


class FakeSceneGenerator:
    def __init__(self, scene=None, error=None):
        self.scene = scene
        self.error = error
        self.calls = 0

    def generate(self, *args):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(scene=self.scene)


class FakePlanner:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def plan(self, scene, context):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SlowSystemTests(unittest.TestCase):
    def setUp(self):
        self.pose = Pose3D(
            Vector3(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)
        )
        self.entity = Entity3D(
            "chair-1",
            "chair",
            Vector3(2.0, 0.0, 0.5),
            Vector3(0.5, 0.5, 1.0),
            0.0,
            ("red",),
            0.9,
            ("rgb-1", "pc-1"),
        )
        self.candidate = CandidateApproach(
            "candidate-1",
            "chair-1",
            Vector3(1.0, 0.0, 0.0),
            0.0,
            0.8,
            0.9,
            0.9,
        )
        self.scene = SceneDescription3D(
            1,
            "scene-1",
            1200,
            "world",
            self.pose,
            entities=(self.entity,),
            candidate_approaches=(self.candidate,),
        )
        reasoning = NavigationReasoning(
            task_parse=TaskParse("approach", "red chair"),
            grounding=GroundingResult("chair-1", 0.95),
            candidate_ids=("candidate-1",),
            evaluations=(
                CandidateEvaluation("candidate-1", True, 0.9, ("clear",)),
            ),
            decision=NavigationDecision("candidate-1", ("highest_score",)),
        )
        signal = NavigationSignal(
            schema_version=1,
            mission_id="mission-1",
            scene_id="scene-1",
            subtask_id="step-1",
            intent=NavigationIntent.APPROACH,
            status=NavigationStatus.READY,
            constraints=NavigationConstraints(0.5, 0.3),
            confidence=0.9,
            generated_at_ms=1200,
            valid_for_ms=5000,
            replan_triggers=("scene_changed", "path_blocked", "target_missing"),
            target_entity_id="chair-1",
            candidate_id="candidate-1",
            goal=NavigationGoal3D("world", Vector3(1.0, 0.0, 0.0), 0.0, 0.2),
        )
        self.plan_result = NavigationPlanResult(reasoning, signal)

    def _fused_tokens(self):
        return FusedSceneTokens(
            features=torch.ones(1, 1, 4),
            positions=torch.zeros(1, 1, 3),
            mask=torch.ones(1, 1, dtype=torch.bool),
            confidence=torch.ones(1, 1),
            modality_weights=torch.tensor([[[0.5, 0.5]]]),
            uncertainty=torch.zeros(1, 1),
        )

    def _inputs(self, memory=None):
        if memory is None:
            spatial_memory = SpatialMemory(feature_size=4)
            memory = spatial_memory.update(self._fused_tokens(), 1200)
        pose_token = TokenBatch(
            torch.ones(1, 1, 4),
            torch.zeros(1, 1, 3),
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones(1, 1),
            "pose",
        )
        return SlowSystemInput(
            current_scene_tokens=self._fused_tokens(),
            memory=memory,
            pose=pose_token,
            scene_context=SceneGenerationContext(
                "Approach the red chair.",
                1200,
                "world",
                self.pose,
                ("rgb-1", "pc-1"),
            ),
            planning_context=NavigationPlanningContext(
                "Approach the red chair.",
                "mission-1",
                "step-1",
                1200,
            ),
        )

    def _event(self, event_id, event_type=SlowEventType.NEW_TASK, timestamp=1200):
        return SlowEvent(event_id, event_type, timestamp, "mission-1", "step-1")

    def test_same_memory_version_reuses_scene_but_reruns_cot(self):
        scene_generator = FakeSceneGenerator(self.scene)
        planner = FakePlanner(self.plan_result)
        system = SlowSystem(scene_generator, planner, "qwen-test")
        inputs = self._inputs()
        first = system.handle(self._event("event-1"), inputs)
        second = system.handle(
            self._event("event-2", SlowEventType.SUBGOAL_COMPLETED), inputs
        )
        self.assertTrue(first.executable)
        self.assertTrue(second.trace.cache_hit)
        self.assertEqual(scene_generator.calls, 1)
        self.assertEqual(planner.calls, 2)

    def test_duplicate_event_does_not_replan(self):
        scene_generator = FakeSceneGenerator(self.scene)
        planner = FakePlanner(self.plan_result)
        system = SlowSystem(scene_generator, planner, "qwen-test")
        event = self._event("event-1")
        inputs = self._inputs()
        system.handle(event, inputs)
        duplicate = system.handle(event, inputs)
        self.assertEqual(duplicate.status, SlowSystemStatus.DUPLICATE_EVENT)
        self.assertEqual(scene_generator.calls, 1)
        self.assertEqual(planner.calls, 1)

    def test_new_memory_version_cannot_reuse_old_scene(self):
        scene_generator = FakeSceneGenerator(self.scene)
        planner = FakePlanner(self.plan_result)
        system = SlowSystem(scene_generator, planner, "qwen-test")
        spatial_memory = SpatialMemory(feature_size=4)
        spatial_memory.update(self._fused_tokens(), 1100)
        memory_v2 = spatial_memory.update(self._fused_tokens(), 1200)
        system.handle(self._event("event-1"), self._inputs())
        result = system.handle(
            self._event("event-2", SlowEventType.SCENE_CHANGED),
            self._inputs(memory_v2),
        )
        self.assertFalse(result.trace.cache_hit)
        self.assertEqual(scene_generator.calls, 2)

    def test_planner_error_never_produces_executable_result(self):
        planner = FakePlanner(
            error=NavigationPlanningError("bad_plan", "invalid plan")
        )
        system = SlowSystem(FakeSceneGenerator(self.scene), planner, "qwen-test")
        result = system.handle(self._event("event-1"), self._inputs())
        self.assertEqual(result.status, SlowSystemStatus.FAILED)
        self.assertEqual(result.trace.error_code, "bad_plan")
        self.assertFalse(result.executable)

    def test_failed_new_task_does_not_return_previous_signal(self):
        planner = FakePlanner(self.plan_result)
        system = SlowSystem(FakeSceneGenerator(self.scene), planner, "qwen-test")
        inputs = self._inputs()
        first = system.handle(self._event("event-1"), inputs)
        self.assertTrue(first.executable)

        planner.error = NavigationPlanningError("bad_plan", "invalid plan")
        failed = system.handle(self._event("event-2"), inputs)
        self.assertEqual(failed.status, SlowSystemStatus.FAILED)
        self.assertIsNone(failed.signal)
        self.assertIsNone(system.active_signal(1200))

    def test_signal_expiry_is_not_extended_by_duplicate_event(self):
        short_signal = NavigationSignal(
            **{
                **self.plan_result.signal.__dict__,
                "valid_for_ms": 100,
            }
        )
        system = SlowSystem(
            FakeSceneGenerator(self.scene),
            FakePlanner(NavigationPlanResult(self.plan_result.reasoning, short_signal)),
            "qwen-test",
        )
        inputs = self._inputs()
        event = self._event("event-1")
        system.handle(event, inputs)
        duplicate = system.handle(
            self._event("event-1", timestamp=1300), inputs
        )
        self.assertIsNone(duplicate.signal)
        self.assertIsNone(system.active_signal(1300))


if __name__ == "__main__":
    unittest.main()
