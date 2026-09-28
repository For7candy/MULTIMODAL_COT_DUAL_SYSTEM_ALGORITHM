import json
import unittest
from dataclasses import replace

from power_vln.schemas import (
    CameraIntrinsics,
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
    PointCloudObservation,
    Pose3D,
    Quaternion,
    Relation3D,
    RgbObservation,
    RigidTransform,
    SceneDescription3D,
    SensorPacket,
    StampedPose,
    TaskParse,
    Vector3,
)
from power_vln.validators import (
    validate_navigation_signal,
    validate_reasoning,
    validate_scene,
    validate_sensor_packet,
)


def pose(x=0.0, y=0.0, z=0.0):
    return Pose3D(
        position=Vector3(x, y, z),
        orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
    )


def sample_packet():
    return SensorPacket(
        schema_version=1,
        packet_id="packet-001",
        instruction="approach cabinet 07",
        rgb=RgbObservation(
            observation_id="rgb-001",
            timestamp_ms=1000,
            frame_id="camera_rgb",
            height=224,
            width=224,
        ),
        point_cloud=PointCloudObservation(
            observation_id="pc-001",
            timestamp_ms=1020,
            frame_id="lidar",
            point_count=4096,
            feature_size=4,
        ),
        robot_pose=StampedPose(
            timestamp_ms=1010,
            frame_id="world",
            pose=pose(),
            covariance=(0.0,) * 36,
        ),
        camera_intrinsics=CameraIntrinsics(
            width=224,
            height=224,
            fx=180.0,
            fy=180.0,
            cx=112.0,
            cy=112.0,
        ),
        sensor_extrinsics=(
            RigidTransform("base_link", "camera_rgb", pose()),
            RigidTransform("base_link", "lidar", pose()),
        ),
    )


def sample_scene():
    entity = Entity3D(
        entity_id="cabinet-07",
        category="switch_cabinet",
        center=Vector3(4.2, 1.1, 0.9),
        size=Vector3(0.8, 0.6, 1.8),
        yaw=1.57,
        attributes=("closed",),
        confidence=0.94,
        source_observations=("rgb-001", "pc-001"),
    )
    candidate = CandidateApproach(
        candidate_id="approach-a",
        target_entity_id=entity.entity_id,
        position=Vector3(3.3, 1.1, 0.0),
        yaw=1.57,
        clearance_m=0.8,
        visibility_score=0.9,
        confidence=0.88,
    )
    return SceneDescription3D(
        schema_version=1,
        scene_id="scene-001",
        timestamp_ms=1100,
        frame_id="world",
        observer_pose=pose(),
        entities=(entity,),
        candidate_approaches=(candidate,),
    )


def sample_reasoning():
    return NavigationReasoning(
        task_parse=TaskParse(action="approach", target="cabinet 07"),
        grounding=GroundingResult(entity_id="cabinet-07", confidence=0.95),
        candidate_ids=("approach-a",),
        evaluations=(
            CandidateEvaluation(
                candidate_id="approach-a",
                reachable=True,
                score=0.87,
                reason_codes=("reachable", "visible"),
            ),
        ),
        decision=NavigationDecision(
            selected_candidate_id="approach-a",
            reason_codes=("highest_score",),
        ),
    )


def sample_signal():
    return NavigationSignal(
        schema_version=1,
        mission_id="inspection-001",
        scene_id="scene-001",
        subtask_id="step-02",
        intent=NavigationIntent.APPROACH,
        status=NavigationStatus.READY,
        target_entity_id="cabinet-07",
        candidate_id="approach-a",
        goal=NavigationGoal3D(
            frame_id="world",
            position=Vector3(3.3, 1.1, 0.0),
            yaw=1.57,
            tolerance_m=0.2,
        ),
        constraints=NavigationConstraints(
            minimum_clearance_m=0.5,
            maximum_speed_mps=0.3,
        ),
        confidence=0.87,
        generated_at_ms=1200,
        valid_for_ms=5000,
        replan_triggers=("target_missing", "path_blocked", "scene_changed"),
    )


class SensorPacketTests(unittest.TestCase):
    def test_metadata_json_round_trip_excludes_runtime_payload(self):
        packet = sample_packet()
        decoded = SensorPacket.from_dict(json.loads(packet.to_json()))
        self.assertEqual(decoded, packet)
        self.assertNotIn("payload", packet.to_json())

    def test_sensor_validator_detects_time_skew_and_missing_extrinsic(self):
        packet = sample_packet()
        invalid = replace(
            packet,
            point_cloud=replace(
                packet.point_cloud, timestamp_ms=1500, frame_id="uncalibrated_lidar"
            ),
        )
        result = validate_sensor_packet(invalid, maximum_sensor_skew_ms=100)
        self.assertFalse(result.is_valid)
        self.assertEqual(
            {issue.code for issue in result.issues},
            {"sensor_time_skew", "missing_extrinsic"},
        )


class SceneSchemaTests(unittest.TestCase):
    def test_scene_json_round_trip_and_validation(self):
        scene = sample_scene()
        decoded = SceneDescription3D.from_dict(json.loads(scene.to_json()))
        self.assertEqual(decoded, scene)
        self.assertTrue(validate_scene(decoded).is_valid)

    def test_scene_validator_detects_duplicate_and_unknown_reference(self):
        scene = sample_scene()
        invalid = replace(
            scene,
            entities=(scene.entities[0], scene.entities[0]),
            relations=(
                Relation3D(
                    subject_id="cabinet-07",
                    predicate="left_of",
                    object_id="missing-door",
                    confidence=0.8,
                ),
            ),
        )
        result = validate_scene(invalid)
        self.assertEqual(
            {issue.code for issue in result.issues},
            {"duplicate_entity_id", "unknown_relation_entity"},
        )


class NavigationSchemaTests(unittest.TestCase):
    def test_reasoning_and_signal_round_trip(self):
        reasoning = sample_reasoning()
        signal = sample_signal()
        self.assertEqual(
            NavigationReasoning.from_dict(json.loads(reasoning.to_json())), reasoning
        )
        self.assertEqual(NavigationSignal.from_dict(json.loads(signal.to_json())), signal)
        self.assertTrue(validate_reasoning(reasoning, sample_scene()).is_valid)
        self.assertTrue(
            validate_navigation_signal(signal, sample_scene(), now_ms=1300).is_valid
        )

    def test_ready_signal_requires_target_candidate_and_goal(self):
        with self.assertRaises(ValueError):
            NavigationSignal(
                schema_version=1,
                mission_id="mission",
                scene_id="scene",
                subtask_id="step",
                intent=NavigationIntent.APPROACH,
                status=NavigationStatus.READY,
                constraints=NavigationConstraints(0.5, 0.3),
                confidence=0.8,
                generated_at_ms=0,
                valid_for_ms=1000,
                replan_triggers=(),
            )

    def test_navigation_validator_collects_cross_contract_failures(self):
        signal = replace(
            sample_signal(),
            scene_id="old-scene",
            target_entity_id="missing-target",
            candidate_id="missing-candidate",
            goal=replace(sample_signal().goal, frame_id="odom"),
            confidence=0.2,
        )
        result = validate_navigation_signal(signal, sample_scene(), now_ms=7000)
        self.assertEqual(
            {issue.code for issue in result.issues},
            {
                "scene_version_mismatch",
                "low_confidence",
                "expired",
                "unknown_target_entity",
                "unknown_candidate",
                "frame_mismatch",
            },
        )


if __name__ == "__main__":
    unittest.main()
