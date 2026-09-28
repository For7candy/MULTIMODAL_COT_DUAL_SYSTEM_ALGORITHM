import unittest
from dataclasses import replace

from power_vln.data import (
    IDENTITY_MATRIX,
    TransformGraph,
    prepare_inputs,
    project_point_cloud_to_depth,
    transform_point,
)
from power_vln.schemas import (
    CameraIntrinsics,
    PointCloudObservation,
    Pose3D,
    Quaternion,
    RgbObservation,
    RigidTransform,
    SchemaValidationError,
    SensorPacket,
    StampedPose,
    Vector3,
)


def pose(x=0.0, y=0.0, z=0.0):
    return Pose3D(
        position=Vector3(x, y, z),
        orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
    )


def packet():
    return SensorPacket(
        schema_version=1,
        packet_id="packet-001",
        instruction="approach cabinet 07",
        rgb=RgbObservation("rgb-001", 1000, "camera_rgb", 5, 5),
        point_cloud=PointCloudObservation("pc-001", 1020, "lidar", 4, 3),
        robot_pose=StampedPose(1010, "world", pose(1.0, 2.0, 0.0)),
        camera_intrinsics=CameraIntrinsics(5, 5, 1.0, 1.0, 2.0, 2.0),
        sensor_extrinsics=(
            RigidTransform("base_link", "camera_rgb", pose()),
            RigidTransform("base_link", "lidar", pose()),
        ),
    )


class CalibrationTests(unittest.TestCase):
    def test_transform_graph_composes_sensor_pose_into_world(self):
        bundle = prepare_inputs(packet(), maximum_sensor_skew_ms=100)
        camera_origin_world = transform_point(
            bundle.world_from_rgb, Vector3(0.0, 0.0, 0.0)
        )
        self.assertEqual(camera_origin_world, Vector3(1.0, 2.0, 0.0))
        self.assertEqual(bundle.rgb_from_point_cloud, IDENTITY_MATRIX)
        self.assertEqual(bundle.reference_timestamp_ms, 1010)

    def test_transform_graph_reports_disconnected_frames(self):
        graph = TransformGraph((RigidTransform("a", "b", pose()),))
        with self.assertRaises(KeyError):
            graph.resolve("b", "missing")


class SynchronizationTests(unittest.TestCase):
    def test_prepare_inputs_rejects_excessive_time_skew(self):
        original = packet()
        invalid = replace(
            original,
            point_cloud=replace(original.point_cloud, timestamp_ms=1500),
        )
        with self.assertRaises(SchemaValidationError):
            prepare_inputs(invalid, maximum_sensor_skew_ms=100)


class ProjectionTests(unittest.TestCase):
    def test_projection_uses_nearest_positive_depth(self):
        intrinsics = CameraIntrinsics(5, 5, 1.0, 1.0, 2.0, 2.0)
        depth = project_point_cloud_to_depth(
            points=(
                (0.0, 0.0, 2.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
                (100.0, 100.0, 1.0),
            ),
            camera_from_point_cloud=IDENTITY_MATRIX,
            intrinsics=intrinsics,
        )
        self.assertEqual(depth[2][2], 1.0)
        self.assertEqual(sum(value > 0.0 for row in depth for value in row), 1)

    def test_projection_rejects_points_without_xyz(self):
        with self.assertRaises(ValueError):
            project_point_cloud_to_depth(
                points=((1.0, 2.0),),
                camera_from_point_cloud=IDENTITY_MATRIX,
                intrinsics=CameraIntrinsics(5, 5, 1.0, 1.0, 2.0, 2.0),
            )


if __name__ == "__main__":
    unittest.main()
