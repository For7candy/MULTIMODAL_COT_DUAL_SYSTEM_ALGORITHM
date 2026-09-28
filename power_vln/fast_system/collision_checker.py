"""Deterministic collision, clearance, and forbidden-region checks."""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple

from .local_map import LocalOccupancySnapshot


@dataclass(frozen=True)
class ForbiddenRegion:
    region_id: str
    center_x: float
    center_y: float
    radius_m: float
    frame_id: str = "base_link"

    def __post_init__(self) -> None:
        if not self.region_id or not self.frame_id or self.radius_m < 0:
            raise ValueError("forbidden region is invalid")


@dataclass(frozen=True)
class CollisionCheck:
    safe: bool
    reason_codes: Tuple[str, ...] = ()


def _samples(
    points: Sequence[Tuple[float, float]], step_m: float
) -> Iterable[Tuple[float, float]]:
    if not points:
        return
    yield points[0]
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        count = max(1, int(math.ceil(math.hypot(dx, dy) / step_m)))
        for index in range(1, count + 1):
            ratio = index / count
            yield start[0] + ratio * dx, start[1] + ratio * dy


class CollisionChecker:
    def check_path(
        self,
        path: Sequence[Tuple[float, float]],
        occupancy: LocalOccupancySnapshot,
        minimum_clearance_m: float,
        forbidden_region_ids: Sequence[str] = (),
        forbidden_regions: Sequence[ForbiddenRegion] = (),
    ) -> CollisionCheck:
        if minimum_clearance_m < 0:
            raise ValueError("minimum_clearance_m must be non-negative")
        reasons = []
        region_by_id = {region.region_id: region for region in forbidden_regions}
        missing = sorted(set(forbidden_region_ids) - set(region_by_id))
        if missing:
            reasons.append("unknown_forbidden_region")
        selected_regions = tuple(
            region_by_id[region_id]
            for region_id in forbidden_region_ids
            if region_id in region_by_id
        )
        if any(region.frame_id != occupancy.frame_id for region in selected_regions):
            reasons.append("forbidden_region_frame_mismatch")
        for x, y in _samples(path, occupancy.resolution_m / 2.0):
            if not occupancy.contains(x, y):
                reasons.append("outside_local_map")
                break
            if occupancy.is_occupied(x, y, minimum_clearance_m):
                reasons.append("insufficient_clearance")
                break
            if any(
                math.hypot(x - region.center_x, y - region.center_y)
                <= region.radius_m + minimum_clearance_m
                for region in selected_regions
            ):
                reasons.append("forbidden_region")
                break
        return CollisionCheck(not reasons, tuple(dict.fromkeys(reasons)))
