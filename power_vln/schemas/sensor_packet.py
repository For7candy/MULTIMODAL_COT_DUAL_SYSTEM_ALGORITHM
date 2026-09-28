"""Versioned metadata and runtime payload contracts for multimodal input."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .common import JsonSchemaMixin, Pose3D, require_finite, require_text


@dataclass(frozen=True)
class CameraIntrinsics(JsonSchemaMixin):
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        require_finite((self.fx, self.fy, self.cx, self.cy), "camera intrinsics")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CameraIntrinsics":
        return cls(**payload)


@dataclass(frozen=True)
class RigidTransform(JsonSchemaMixin):
    parent_frame: str
    child_frame: str
    pose: Pose3D

    def __post_init__(self) -> None:
        require_text(self.parent_frame, "parent_frame")
        require_text(self.child_frame, "child_frame")
        if self.parent_frame == self.child_frame:
            raise ValueError("rigid transform frames must be different")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RigidTransform":
        return cls(
            parent_frame=payload["parent_frame"],
            child_frame=payload["child_frame"],
            pose=Pose3D.from_dict(payload["pose"]),
        )


@dataclass(frozen=True)
class RgbObservation(JsonSchemaMixin):
    observation_id: str
    timestamp_ms: int
    frame_id: str
    height: int
    width: int
    channels: int = 3
    data_reference: str = "runtime://rgb"
    payload: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        require_text(self.observation_id, "observation_id")
        require_text(self.frame_id, "frame_id")
        require_text(self.data_reference, "data_reference")
        if self.timestamp_ms < 0:
            raise ValueError("RGB timestamp_ms must be non-negative")
        if self.height <= 0 or self.width <= 0 or self.channels != 3:
            raise ValueError("RGB shape must be positive HxWx3")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "timestamp_ms": self.timestamp_ms,
            "frame_id": self.frame_id,
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "data_reference": self.data_reference,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RgbObservation":
        return cls(**payload)


@dataclass(frozen=True)
class PointCloudObservation(JsonSchemaMixin):
    observation_id: str
    timestamp_ms: int
    frame_id: str
    point_count: int
    feature_size: int
    data_reference: str = "runtime://point_cloud"
    payload: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        require_text(self.observation_id, "observation_id")
        require_text(self.frame_id, "frame_id")
        require_text(self.data_reference, "data_reference")
        if self.timestamp_ms < 0:
            raise ValueError("point cloud timestamp_ms must be non-negative")
        if self.point_count <= 0 or self.feature_size < 3:
            raise ValueError("point cloud must contain N>0 points with at least xyz")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "timestamp_ms": self.timestamp_ms,
            "frame_id": self.frame_id,
            "point_count": self.point_count,
            "feature_size": self.feature_size,
            "data_reference": self.data_reference,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PointCloudObservation":
        return cls(**payload)


@dataclass(frozen=True)
class StampedPose(JsonSchemaMixin):
    timestamp_ms: int
    frame_id: str
    pose: Pose3D
    child_frame_id: str = "base_link"
    covariance: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.frame_id, "frame_id")
        require_text(self.child_frame_id, "child_frame_id")
        if self.frame_id == self.child_frame_id:
            raise ValueError("pose parent and child frames must be different")
        if self.timestamp_ms < 0:
            raise ValueError("pose timestamp_ms must be non-negative")
        if self.covariance and len(self.covariance) != 36:
            raise ValueError("pose covariance must contain 36 values")
        require_finite(self.covariance, "pose covariance")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StampedPose":
        return cls(
            timestamp_ms=payload["timestamp_ms"],
            frame_id=payload["frame_id"],
            pose=Pose3D.from_dict(payload["pose"]),
            child_frame_id=payload.get("child_frame_id", "base_link"),
            covariance=tuple(payload.get("covariance", ())),
        )


@dataclass(frozen=True)
class SensorPacket(JsonSchemaMixin):
    schema_version: int
    packet_id: str
    instruction: str
    rgb: RgbObservation
    point_cloud: PointCloudObservation
    robot_pose: StampedPose
    camera_intrinsics: CameraIntrinsics
    sensor_extrinsics: Tuple[RigidTransform, ...]
    memory_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        require_text(self.packet_id, "packet_id")
        require_text(self.instruction, "instruction")
        if not self.sensor_extrinsics:
            raise ValueError("sensor_extrinsics must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "packet_id": self.packet_id,
            "instruction": self.instruction,
            "rgb": self.rgb.to_dict(),
            "point_cloud": self.point_cloud.to_dict(),
            "robot_pose": self.robot_pose.to_dict(),
            "camera_intrinsics": self.camera_intrinsics.to_dict(),
            "sensor_extrinsics": [item.to_dict() for item in self.sensor_extrinsics],
            "memory_id": self.memory_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SensorPacket":
        return cls(
            schema_version=payload["schema_version"],
            packet_id=payload["packet_id"],
            instruction=payload["instruction"],
            rgb=RgbObservation.from_dict(payload["rgb"]),
            point_cloud=PointCloudObservation.from_dict(payload["point_cloud"]),
            robot_pose=StampedPose.from_dict(payload["robot_pose"]),
            camera_intrinsics=CameraIntrinsics.from_dict(payload["camera_intrinsics"]),
            sensor_extrinsics=tuple(
                RigidTransform.from_dict(item) for item in payload["sensor_extrinsics"]
            ),
            memory_id=payload.get("memory_id"),
        )
