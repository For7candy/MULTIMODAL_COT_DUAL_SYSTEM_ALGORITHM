"""Pure-Python rigid transforms and a small coordinate-frame graph."""

from collections import deque
from typing import Dict, Iterable, List, Tuple

from power_vln.schemas.common import Pose3D, Vector3
from power_vln.schemas.sensor_packet import RigidTransform

Matrix4 = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]


IDENTITY_MATRIX: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def pose_to_matrix(pose: Pose3D) -> Matrix4:
    q = pose.orientation
    norm = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) ** 0.5
    x, y, z, w = q.x / norm, q.y / norm, q.z / norm, q.w / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    p = pose.position
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), p.x),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), p.y),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), p.z),
        (0.0, 0.0, 0.0, 1.0),
    )


def multiply(left: Matrix4, right: Matrix4) -> Matrix4:
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(4))
            for column in range(4)
        )
        for row in range(4)
    )  # type: ignore


def invert_rigid(matrix: Matrix4) -> Matrix4:
    rotation_t = tuple(
        tuple(matrix[column][row] for column in range(3)) for row in range(3)
    )
    translation = (matrix[0][3], matrix[1][3], matrix[2][3])
    inverse_translation = tuple(
        -sum(rotation_t[row][index] * translation[index] for index in range(3))
        for row in range(3)
    )
    return (
        (rotation_t[0][0], rotation_t[0][1], rotation_t[0][2], inverse_translation[0]),
        (rotation_t[1][0], rotation_t[1][1], rotation_t[1][2], inverse_translation[1]),
        (rotation_t[2][0], rotation_t[2][1], rotation_t[2][2], inverse_translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def transform_point(matrix: Matrix4, point: Vector3) -> Vector3:
    values = (point.x, point.y, point.z, 1.0)
    output = tuple(
        sum(matrix[row][index] * values[index] for index in range(4))
        for row in range(3)
    )
    return Vector3(*output)


class TransformGraph:
    """Resolve transforms as T_target_source matrices."""

    def __init__(self, transforms: Iterable[RigidTransform] = ()) -> None:
        self._edges: Dict[str, List[Tuple[str, Matrix4]]] = {}
        for transform in transforms:
            self.add(transform)

    def add(self, transform: RigidTransform) -> None:
        parent_from_child = pose_to_matrix(transform.pose)
        child_from_parent = invert_rigid(parent_from_child)
        self._edges.setdefault(transform.child_frame, []).append(
            (transform.parent_frame, parent_from_child)
        )
        self._edges.setdefault(transform.parent_frame, []).append(
            (transform.child_frame, child_from_parent)
        )

    def resolve(self, source_frame: str, target_frame: str) -> Matrix4:
        if source_frame == target_frame:
            return IDENTITY_MATRIX
        queue = deque([(source_frame, IDENTITY_MATRIX)])
        visited = {source_frame}
        while queue:
            current_frame, current_from_source = queue.popleft()
            for next_frame, next_from_current in self._edges.get(current_frame, ()):
                if next_frame in visited:
                    continue
                next_from_source = multiply(next_from_current, current_from_source)
                if next_frame == target_frame:
                    return next_from_source
                visited.add(next_frame)
                queue.append((next_frame, next_from_source))
        raise KeyError(
            "no transform path from {} to {}".format(source_frame, target_frame)
        )
