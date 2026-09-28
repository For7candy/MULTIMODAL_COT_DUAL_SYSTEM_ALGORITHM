"""Build a normalized input bundle with resolved coordinate transforms."""

from dataclasses import dataclass

from power_vln.schemas.sensor_packet import RigidTransform, SensorPacket

from .calibration import Matrix4, TransformGraph
from .synchronizer import synchronized_timestamp_ms


@dataclass(frozen=True)
class AlignedInputBundle:
    packet: SensorPacket
    reference_timestamp_ms: int
    world_from_rgb: Matrix4
    world_from_point_cloud: Matrix4
    rgb_from_point_cloud: Matrix4


def prepare_inputs(
    packet: SensorPacket,
    maximum_sensor_skew_ms: int = 100,
) -> AlignedInputBundle:
    reference_timestamp = synchronized_timestamp_ms(packet, maximum_sensor_skew_ms)
    pose_transform = RigidTransform(
        parent_frame=packet.robot_pose.frame_id,
        child_frame=packet.robot_pose.child_frame_id,
        pose=packet.robot_pose.pose,
    )
    graph = TransformGraph((pose_transform,) + packet.sensor_extrinsics)
    world_frame = packet.robot_pose.frame_id
    return AlignedInputBundle(
        packet=packet,
        reference_timestamp_ms=reference_timestamp,
        world_from_rgb=graph.resolve(packet.rgb.frame_id, world_frame),
        world_from_point_cloud=graph.resolve(
            packet.point_cloud.frame_id, world_frame
        ),
        rgb_from_point_cloud=graph.resolve(
            packet.point_cloud.frame_id, packet.rgb.frame_id
        ),
    )
