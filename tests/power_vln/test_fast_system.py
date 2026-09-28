import math
import unittest

from power_vln.fast_system import FastFeedback, FastSystem, FastSystemInput
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


class FastSystemTests(unittest.TestCase):
    def setUp(self):
        self.pose = StampedPose(
            1000,
            "world",
            Pose3D(Vector3(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
            "base_link",
        )
        self.signal = NavigationSignal(
            schema_version=1,
            mission_id="mission-1",
            scene_id="scene-1",
            subtask_id="step-1",
            intent=NavigationIntent.APPROACH,
            status=NavigationStatus.READY,
            constraints=NavigationConstraints(0.25, 0.3),
            confidence=0.9,
            generated_at_ms=1000,
            valid_for_ms=1000,
            replan_triggers=("path_blocked",),
            target_entity_id="chair-1",
            candidate_id="candidate-1",
            goal=NavigationGoal3D("world", Vector3(2.0, 0.0, 0.0), 0.0, 0.1),
        )

    def _inputs(self, **changes):
        values = dict(
            signal=self.signal,
            point_cloud=((4.0, 4.0, 0.5),),
            point_timestamp_ms=1000,
            point_frame_id="base_link",
            robot_pose=self.pose,
            now_ms=1100,
        )
        values.update(changes)
        return FastSystemInput(**values)

    def test_clear_geometry_produces_bounded_active_action(self):
        result = FastSystem().step(self._inputs())
        self.assertEqual(result.feedback, FastFeedback.ACTIVE)
        self.assertFalse(result.action.stop)
        self.assertLessEqual(result.action.linear_velocity_mps, 0.3)
        self.assertAlmostEqual(result.action.angular_velocity_rps, 0.0)
        self.assertEqual(result.local_path.source, "geometry")

    def test_expired_signal_only_returns_stop(self):
        result = FastSystem().step(self._inputs(now_ms=2000))
        self.assertEqual(result.feedback, FastFeedback.SAFETY_STOP)
        self.assertTrue(result.action.stop)
        self.assertEqual(result.reason_codes, ("signal_expired",))

    def test_stale_point_cloud_only_returns_stop(self):
        result = FastSystem().step(self._inputs(now_ms=1300))
        self.assertEqual(result.feedback, FastFeedback.INPUT_STALE)
        self.assertTrue(result.action.stop)

    def test_empty_point_cloud_only_returns_stop(self):
        result = FastSystem().step(self._inputs(point_cloud=()))
        self.assertEqual(result.feedback, FastFeedback.SAFETY_STOP)
        self.assertEqual(result.reason_codes, ("invalid_point_cloud",))
        self.assertTrue(result.action.stop)

    def test_goal_in_obstacle_is_blocked(self):
        result = FastSystem().step(
            self._inputs(point_cloud=((2.0, 0.0, 0.5),))
        )
        self.assertEqual(result.feedback, FastFeedback.BLOCKED)
        self.assertTrue(result.action.stop)
        self.assertIn("goal_occupied", result.reason_codes)

    def test_all_candidate_headings_blocked(self):
        obstacles = []
        for index in range(-20, 21):
            angle = math.radians(index * 3)
            obstacles.append((0.35 * math.cos(angle), 0.35 * math.sin(angle), 0.5))
        result = FastSystem().step(self._inputs(point_cloud=tuple(obstacles)))
        self.assertIn(result.feedback, (FastFeedback.BLOCKED, FastFeedback.SAFETY_STOP))
        self.assertTrue(result.action.stop)

    def test_goal_transform_uses_robot_pose_yaw(self):
        half = math.sqrt(0.5)
        rotated_pose = StampedPose(
            1000,
            "world",
            Pose3D(Vector3(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, half, half)),
            "base_link",
        )
        rotated_signal = NavigationSignal(
            **{
                **self.signal.__dict__,
                "goal": NavigationGoal3D(
                    "world", Vector3(0.0, 2.0, 0.0), math.pi / 2.0, 0.1
                ),
            }
        )
        result = FastSystem().step(
            self._inputs(signal=rotated_signal, robot_pose=rotated_pose)
        )
        self.assertEqual(result.feedback, FastFeedback.ACTIVE)
        self.assertAlmostEqual(result.action.angular_velocity_rps, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
