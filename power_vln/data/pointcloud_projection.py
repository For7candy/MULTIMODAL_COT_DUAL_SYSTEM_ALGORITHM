"""Project calibrated point-cloud xyz values into a camera depth image."""

from typing import Iterable, Sequence, Tuple

from power_vln.schemas.common import Vector3
from power_vln.schemas.sensor_packet import CameraIntrinsics

from .calibration import Matrix4, transform_point


def project_point_cloud_to_depth(
    points: Iterable[Sequence[float]],
    camera_from_point_cloud: Matrix4,
    intrinsics: CameraIntrinsics,
) -> Tuple[Tuple[float, ...], ...]:
    depth = [
        [0.0 for _ in range(intrinsics.width)]
        for _ in range(intrinsics.height)
    ]
    for index, point in enumerate(points):
        if len(point) < 3:
            raise ValueError("point {} has fewer than xyz coordinates".format(index))
        camera_point = transform_point(
            camera_from_point_cloud,
            Vector3(float(point[0]), float(point[1]), float(point[2])),
        )
        if camera_point.z <= 0:
            continue
        u = int(round(intrinsics.fx * camera_point.x / camera_point.z + intrinsics.cx))
        v = int(round(intrinsics.fy * camera_point.y / camera_point.z + intrinsics.cy))
        if not (0 <= u < intrinsics.width and 0 <= v < intrinsics.height):
            continue
        old_depth = depth[v][u]
        if old_depth == 0.0 or camera_point.z < old_depth:
            depth[v][u] = camera_point.z
    return tuple(tuple(row) for row in depth)
