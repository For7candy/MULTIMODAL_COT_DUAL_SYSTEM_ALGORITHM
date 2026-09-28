"""Robot-local goal transformation and deterministic geometric path proposals."""

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

from power_vln.schemas import NavigationGoal3D, Pose3D

from .collision_checker import CollisionChecker, ForbiddenRegion
from .local_map import LocalOccupancySnapshot


def planar_yaw(pose: Pose3D) -> float:
    q = pose.orientation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    x, y, z, w = q.x / norm, q.y / norm, q.z / norm, q.w / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(frozen=True)
class LocalGoal:
    x: float
    y: float
    yaw: float
    tolerance_m: float

    @property
    def distance_m(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass(frozen=True)
class LocalPath:
    points: Tuple[Tuple[float, float], ...]
    score: float
    source: str = "geometry"


@dataclass(frozen=True)
class PlanningResult:
    selected_path: LocalPath | None
    reason_codes: Tuple[str, ...] = ()


def goal_to_robot_frame(goal: NavigationGoal3D, robot_pose: Pose3D) -> LocalGoal:
    world_yaw = planar_yaw(robot_pose)
    dx = goal.position.x - robot_pose.position.x
    dy = goal.position.y - robot_pose.position.y
    cosine = math.cos(world_yaw)
    sine = math.sin(world_yaw)
    return LocalGoal(
        x=cosine * dx + sine * dy,
        y=-sine * dx + cosine * dy,
        yaw=math.atan2(math.sin(goal.yaw - world_yaw), math.cos(goal.yaw - world_yaw)),
        tolerance_m=goal.tolerance_m,
    )


class GeometricPlanner:
    def __init__(
        self,
        maximum_step_m: float = 1.0,
        heading_offsets_deg: Sequence[float] = (0.0, -20.0, 20.0, -40.0, 40.0),
    ) -> None:
        if maximum_step_m <= 0 or not heading_offsets_deg:
            raise ValueError("geometric planner configuration is invalid")
        self.maximum_step_m = maximum_step_m
        self.heading_offsets = tuple(math.radians(value) for value in heading_offsets_deg)

    def candidates(
        self,
        goal: LocalGoal,
        supplemental_waypoints: Sequence[Tuple[float, float]] = (),
    ) -> Tuple[LocalPath, ...]:
        bearing = math.atan2(goal.y, goal.x)
        distance = min(goal.distance_m, self.maximum_step_m)
        candidates = []
        for index, offset in enumerate(self.heading_offsets):
            heading = bearing + offset
            endpoint = (distance * math.cos(heading), distance * math.sin(heading))
            remaining = math.hypot(goal.x - endpoint[0], goal.y - endpoint[1])
            score = -remaining - 0.05 * abs(offset)
            candidates.append(LocalPath(((0.0, 0.0), endpoint), score, "geometry"))
        for x, y in supplemental_waypoints:
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            length = math.hypot(x, y)
            if length <= 0 or length > self.maximum_step_m:
                continue
            remaining = math.hypot(goal.x - x, goal.y - y)
            candidates.append(LocalPath(((0.0, 0.0), (x, y)), -remaining, "open_nav"))
        return tuple(candidates)

    def plan(
        self,
        goal: LocalGoal,
        occupancy: LocalOccupancySnapshot,
        checker: CollisionChecker,
        minimum_clearance_m: float,
        forbidden_region_ids: Sequence[str] = (),
        forbidden_regions: Sequence[ForbiddenRegion] = (),
        supplemental_waypoints: Sequence[Tuple[float, float]] = (),
    ) -> PlanningResult:
        safe = []
        rejected_reasons = []
        for candidate in self.candidates(goal, supplemental_waypoints):
            result = checker.check_path(
                candidate.points,
                occupancy,
                minimum_clearance_m,
                forbidden_region_ids,
                forbidden_regions,
            )
            if result.safe:
                safe.append(candidate)
            else:
                rejected_reasons.extend(result.reason_codes)
        if not safe:
            return PlanningResult(
                None, tuple(dict.fromkeys(rejected_reasons or ["no_safe_candidate"]))
            )
        selected = max(safe, key=lambda item: (item.score, item.source == "geometry"))
        return PlanningResult(selected)
