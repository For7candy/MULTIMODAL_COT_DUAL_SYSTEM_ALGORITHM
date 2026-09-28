"""Cross-field validation for synchronized multimodal input."""

from typing import Tuple

from power_vln.schemas.common import ValidationIssue, ValidationResult
from power_vln.schemas.sensor_packet import SensorPacket


def validate_sensor_packet(
    packet: SensorPacket,
    maximum_sensor_skew_ms: int = 100,
) -> ValidationResult:
    if maximum_sensor_skew_ms < 0:
        raise ValueError("maximum_sensor_skew_ms must be non-negative")

    issues = []
    timestamps = (
        packet.rgb.timestamp_ms,
        packet.point_cloud.timestamp_ms,
        packet.robot_pose.timestamp_ms,
    )
    if max(timestamps) - min(timestamps) > maximum_sensor_skew_ms:
        issues.append(
            ValidationIssue(
                code="sensor_time_skew",
                path="SensorPacket",
                message="RGB, point cloud and pose timestamps exceed allowed skew",
            )
        )

    if (
        packet.camera_intrinsics.width != packet.rgb.width
        or packet.camera_intrinsics.height != packet.rgb.height
    ):
        issues.append(
            ValidationIssue(
                code="intrinsics_shape_mismatch",
                path="camera_intrinsics",
                message="camera intrinsics dimensions must match RGB dimensions",
            )
        )

    calibrated_frames = {item.child_frame for item in packet.sensor_extrinsics}
    for path, frame_id in (
        ("rgb.frame_id", packet.rgb.frame_id),
        ("point_cloud.frame_id", packet.point_cloud.frame_id),
    ):
        if frame_id not in calibrated_frames:
            issues.append(
                ValidationIssue(
                    code="missing_extrinsic",
                    path=path,
                    message="sensor frame has no declared extrinsic transform",
                )
            )

    return ValidationResult(issues=tuple(issues))
