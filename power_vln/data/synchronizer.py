"""Sensor timestamp synchronization without framework dependencies."""

from power_vln.schemas.sensor_packet import SensorPacket
from power_vln.validators import validate_sensor_packet


def synchronized_timestamp_ms(packet: SensorPacket, maximum_skew_ms: int) -> int:
    result = validate_sensor_packet(
        packet, maximum_sensor_skew_ms=maximum_skew_ms
    )
    result.raise_for_errors()
    timestamps = sorted(
        (
            packet.rgb.timestamp_ms,
            packet.point_cloud.timestamp_ms,
            packet.robot_pose.timestamp_ms,
        )
    )
    return timestamps[1]
