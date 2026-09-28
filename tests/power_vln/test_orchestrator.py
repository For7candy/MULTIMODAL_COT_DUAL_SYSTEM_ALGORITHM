from threading import Event
import unittest

from power_vln.fast_system import FastFeedback
from power_vln.orchestrator import (
    AtomicSignalStore,
    DualSystemStepInput,
    M3DCoTDualSystem,
    SlowPollStatus,
)
from power_vln.schemas import (
    NavigationConstraints,
    NavigationGoal3D,
    NavigationIntent,
    NavigationSignal,
    NavigationStatus,
    Pose3D,
    Quaternion,
    StampedPose,
    Vector3,
)
from power_vln.slow_system import SlowEvent, SlowEventType
from power_vln.state_machine import SystemEvent


class BlockingSlowSystem:
    def __init__(self):
        self.release = Event()
        self.started = Event()

    def handle(self, event, inputs):
        self.started.set()
        self.release.wait(2.0)
        return None


class MemoryInput:
    def __init__(self, version):
        self.memory = type("Memory", (), {"version": version})()


def ready_signal(generated_at_ms=1000):
    return NavigationSignal(
        schema_version=1,
        mission_id="mission-1",
        scene_id="scene-1",
        subtask_id="step-1",
        intent=NavigationIntent.APPROACH,
        status=NavigationStatus.READY,
        constraints=NavigationConstraints(0.25, 0.3),
        confidence=0.9,
        generated_at_ms=generated_at_ms,
        valid_for_ms=1000,
        replan_triggers=("path_blocked",),
        target_entity_id="chair-1",
        candidate_id="candidate-1",
        goal=NavigationGoal3D("world", Vector3(2.0, 0.0, 0.0), 0.0, 0.1),
    )


class SignalStoreTests(unittest.TestCase):
    def test_read_requires_scene_task_and_unexpired_timestamp(self):
        store = AtomicSignalStore()
        signal = ready_signal()
        store.publish(signal, 7, "mission-1", "step-1")
        self.assertIsNotNone(store.read(7, "mission-1", "step-1", 1100))
        self.assertIsNone(store.read(8, "mission-1", "step-1", 1100))
        self.assertIsNone(store.read(7, "other", "step-1", 1100))
        self.assertIsNone(store.read(7, "mission-1", "step-1", 2000))

    def test_older_signal_cannot_replace_newer_signal(self):
        store = AtomicSignalStore()
        store.publish(ready_signal(1100), 7, "mission-1", "step-1")
        with self.assertRaises(ValueError):
            store.publish(ready_signal(1000), 7, "mission-1", "step-1")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.slow = BlockingSlowSystem()
        self.system = M3DCoTDualSystem(self.slow)
        self.pose = StampedPose(
            1000,
            "world",
            Pose3D(Vector3(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
            "base_link",
        )

    def tearDown(self):
        self.slow.release.set()
        self.system.close()

    def _step_input(self, **changes):
        values = dict(
            now_ms=1100,
            scene_version=7,
            mission_id="mission-1",
            subtask_id="step-1",
            point_cloud=((4.0, 4.0, 0.5),),
            point_timestamp_ms=1000,
            point_frame_id="base_link",
            robot_pose=self.pose,
        )
        values.update(changes)
        return DualSystemStepInput(**values)

    def test_blocked_slow_worker_does_not_block_valid_fast_signal(self):
        self.system.signal_store.publish(
            ready_signal(), 7, "mission-1", "step-1"
        )
        self.system.state_machine.dispatch(SystemEvent.MISSION_RECEIVED)
        self.system.state_machine.dispatch(SystemEvent.PLAN_ACCEPTED)
        event = SlowEvent(
            "refresh-1", SlowEventType.SCENE_CHANGED, 1000, "mission-1", "step-1"
        )
        self.system.scheduler.submit(event, MemoryInput(7), 1000)
        self.assertTrue(self.slow.started.wait(1.0))

        result = self.system.step(self._step_input())
        self.assertEqual(result.fast_result.feedback, FastFeedback.ACTIVE)
        self.assertFalse(result.fast_result.action.stop)

    def test_step_rejects_mismatched_slow_scene_version(self):
        event = SlowEvent(
            "event-1", SlowEventType.NEW_TASK, 1000, "mission-1", "step-1"
        )
        with self.assertRaises(ValueError):
            self.system.step(
                self._step_input(slow_event=event, slow_inputs=MemoryInput(8))
            )

    def test_duplicate_slow_event_is_merged_and_timeout_is_reported(self):
        event = SlowEvent(
            "event-1", SlowEventType.NEW_TASK, 1000, "mission-1", "step-1"
        )
        self.assertTrue(self.system.scheduler.submit(event, MemoryInput(7), 1000))
        self.assertFalse(self.system.scheduler.submit(event, MemoryInput(7), 1000))
        self.assertTrue(self.slow.started.wait(1.0))
        poll = self.system.scheduler.poll(2000)
        self.assertEqual(poll.status, SlowPollStatus.TIMED_OUT)
        self.assertEqual(poll.event_id, "event-1")
        self.assertEqual(poll.input_version, 7)

    def test_expired_active_signal_stops_and_requires_replan(self):
        self.system.signal_store.publish(
            ready_signal(), 7, "mission-1", "step-1"
        )
        self.system.state_machine.dispatch(SystemEvent.MISSION_RECEIVED)
        self.system.state_machine.dispatch(SystemEvent.PLAN_ACCEPTED)
        result = self.system.step(self._step_input(now_ms=2000))
        self.assertEqual(result.fast_result.feedback, FastFeedback.SAFETY_STOP)
        self.assertTrue(result.fast_result.action.stop)
        self.assertIsNone(result.active_signal)
        self.assertEqual(result.state.value, "replan_required")


if __name__ == "__main__":
    unittest.main()
