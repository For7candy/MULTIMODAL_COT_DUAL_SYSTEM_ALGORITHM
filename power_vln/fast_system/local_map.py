"""Lightweight robot-local occupancy map built from the latest point cloud."""

from dataclasses import dataclass
import math
from typing import Any, FrozenSet, Iterable, Tuple


@dataclass(frozen=True)
class LocalMapConfig:
    resolution_m: float = 0.10
    radius_m: float = 5.0
    minimum_obstacle_height_m: float = 0.05
    maximum_obstacle_height_m: float = 1.80
    maximum_input_age_ms: int = 250

    def __post_init__(self) -> None:
        if self.resolution_m <= 0 or self.radius_m <= 0:
            raise ValueError("map resolution and radius must be positive")
        if self.minimum_obstacle_height_m > self.maximum_obstacle_height_m:
            raise ValueError("obstacle height interval is invalid")
        if self.maximum_input_age_ms <= 0:
            raise ValueError("maximum_input_age_ms must be positive")


@dataclass(frozen=True)
class LocalOccupancySnapshot:
    timestamp_ms: int
    frame_id: str
    resolution_m: float
    radius_m: float
    occupied_cells: FrozenSet[Tuple[int, int]]

    def contains(self, x: float, y: float) -> bool:
        return math.hypot(x, y) <= self.radius_m

    def cell(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(math.floor(x / self.resolution_m)),
            int(math.floor(y / self.resolution_m)),
        )

    def is_occupied(self, x: float, y: float, clearance_m: float = 0.0) -> bool:
        if clearance_m < 0:
            raise ValueError("clearance_m must be non-negative")
        center_x, center_y = self.cell(x, y)
        radius_cells = int(math.ceil(clearance_m / self.resolution_m))
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if math.hypot(dx, dy) * self.resolution_m > clearance_m + self.resolution_m:
                    continue
                if (center_x + dx, center_y + dy) in self.occupied_cells:
                    return True
        return False


def _point_rows(points: Any) -> Iterable[Tuple[float, float, float]]:
    if points is None:
        raise ValueError("point cloud payload is missing")
    if hasattr(points, "detach"):
        points = points.detach().cpu().tolist()
    elif hasattr(points, "tolist"):
        points = points.tolist()
    for row in points:
        if len(row) < 3:
            raise ValueError("each point must contain at least xyz")
        xyz = (float(row[0]), float(row[1]), float(row[2]))
        if not all(math.isfinite(value) for value in xyz):
            continue
        yield xyz


class LocalOccupancyMap:
    def __init__(self, config: LocalMapConfig = LocalMapConfig()) -> None:
        self.config = config
        self._snapshot = None

    @property
    def snapshot(self) -> LocalOccupancySnapshot:
        if self._snapshot is None:
            raise RuntimeError("local map has not been initialized")
        return self._snapshot

    def update(
        self,
        points: Any,
        timestamp_ms: int,
        frame_id: str,
    ) -> LocalOccupancySnapshot:
        if timestamp_ms < 0 or not frame_id:
            raise ValueError("point cloud timestamp and frame are invalid")
        occupied = set()
        valid_points = 0
        for x, y, z in _point_rows(points):
            valid_points += 1
            if math.hypot(x, y) > self.config.radius_m:
                continue
            if not (
                self.config.minimum_obstacle_height_m
                <= z
                <= self.config.maximum_obstacle_height_m
            ):
                continue
            occupied.add(
                (
                    int(math.floor(x / self.config.resolution_m)),
                    int(math.floor(y / self.config.resolution_m)),
                )
            )
        if valid_points == 0:
            raise ValueError("point cloud contains no finite xyz samples")
        self._snapshot = LocalOccupancySnapshot(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            resolution_m=self.config.resolution_m,
            radius_m=self.config.radius_m,
            occupied_cells=frozenset(occupied),
        )
        return self._snapshot
