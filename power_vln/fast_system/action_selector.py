"""Fail-closed high-frequency fast system with no model dependency."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Optional, Tuple

from power_vln.schemas import NavigationSignal, NavigationStatus, StampedPose

from .collision_checker import CollisionChecker, ForbiddenRegion
from .geometric_planner import GeometricPlanner, LocalPath, goal_to_robot_frame
from .local_map import LocalMapConfig, LocalOccupancyMap


class FastFeedback(str, Enum):
    ACTIVE = "active"
    REACHED = "reached"
    BLOCKED = "blocked"
    INPUT_STALE = "input_stale"
    SAFETY_STOP = "safety_stop"


@dataclass(frozen=True)
class FastAction:
    linear_velocity_mps: float
    angular_velocity_rps: float
    stop: bool

    @classmethod
    def stopped(cls) -> "FastAction":
        return cls(0.0, 0.0, True)


@dataclass(frozen=True)
class FastSystemInput:
    signal: Optional[NavigationSignal]
    point_cloud: Any
    point_timestamp_ms: int
    point_frame_id: str
    robot_pose: StampedPose
    now_ms: int
    forbidden_regions: Tuple[ForbiddenRegion, ...] = ()
    supplemental_waypoints: Tuple[Tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.point_timestamp_ms < 0 or self.now_ms < 0 or not self.point_frame_id:
            raise ValueError("fast-system input metadata is invalid")


@dataclass(frozen=True)
class FastSystemResult:
    action: FastAction
    feedback: FastFeedback
    local_path: Optional[LocalPath] = None
    reason_codes: Tuple[str, ...] = ()


class FastSystem:
    """Convert one approved semantic signal into a safe local motion command."""

    def __init__(
        self,
        map_config: LocalMapConfig = LocalMapConfig(),
        planner: Optional[GeometricPlanner] = None,
        checker: Optional[CollisionChecker] = None,
        robot_radius_m: float = 0.25,
        maximum_angular_velocity_rps: float = 1.0,
    ) -> None:
        if robot_radius_m < 0 or maximum_angular_velocity_rps <= 0:
            raise ValueError("fast-system safety limits are invalid")
        self.local_map = LocalOccupancyMap(map_config)
        self.planner = planner or GeometricPlanner()
        self.checker = checker or CollisionChecker()
        self.robot_radius_m = robot_radius_m
        self.maximum_angular_velocity_rps = maximum_angular_velocity_rps

    @staticmethod
    def _stop(feedback: FastFeedback, *reasons: str) -> FastSystemResult:
        return FastSystemResult(FastAction.stopped(), feedback, reason_codes=tuple(reasons))

    def step(self, inputs: FastSystemInput) -> FastSystemResult:
        signal = inputs.signal
        if signal is None or signal.status != NavigationStatus.READY or signal.goal is None:
            return self._stop(FastFeedback.SAFETY_STOP, "no_executable_signal")
        if signal.is_expired(inputs.now_ms):
            return self._stop(FastFeedback.SAFETY_STOP, "signal_expired")
        maximum_age = self.local_map.config.maximum_input_age_ms
        if (
            inputs.now_ms - inputs.point_timestamp_ms > maximum_age
            or inputs.now_ms - inputs.robot_pose.timestamp_ms > maximum_age
            or inputs.point_timestamp_ms > inputs.now_ms
            or inputs.robot_pose.timestamp_ms > inputs.now_ms
        ):
            return self._stop(FastFeedback.INPUT_STALE, "sensor_input_stale")
        if inputs.point_frame_id != inputs.robot_pose.child_frame_id:
            return self._stop(FastFeedback.SAFETY_STOP, "point_frame_mismatch")
        if signal.goal.frame_id != inputs.robot_pose.frame_id:
            return self._stop(FastFeedback.SAFETY_STOP, "goal_frame_mismatch")

        try:
            occupancy = self.local_map.update(
                inputs.point_cloud, inputs.point_timestamp_ms, inputs.point_frame_id
            )
        except (TypeError, ValueError):
            return self._stop(FastFeedback.SAFETY_STOP, "invalid_point_cloud")
        if occupancy.is_occupied(0.0, 0.0, self.robot_radius_m):
            return self._stop(FastFeedback.SAFETY_STOP, "robot_collision_risk")

        goal = goal_to_robot_frame(signal.goal, inputs.robot_pose.pose)
        if goal.distance_m <= goal.tolerance_m:
            return self._stop(FastFeedback.REACHED, "goal_tolerance_reached")
        clearance = max(signal.constraints.minimum_clearance_m, self.robot_radius_m)
        if occupancy.contains(goal.x, goal.y) and occupancy.is_occupied(
            goal.x, goal.y, clearance
        ):
            return self._stop(FastFeedback.BLOCKED, "goal_occupied")

        plan = self.planner.plan(
            goal,
            occupancy,
            self.checker,
            clearance,
            signal.constraints.forbidden_region_ids,
            inputs.forbidden_regions,
            inputs.supplemental_waypoints,
        )
        if plan.selected_path is None:
            return self._stop(FastFeedback.BLOCKED, *plan.reason_codes)

        endpoint = plan.selected_path.points[-1]
        heading = math.atan2(endpoint[1], endpoint[0])
        angular = max(
            -self.maximum_angular_velocity_rps,
            min(self.maximum_angular_velocity_rps, 1.5 * heading),
        )
        linear = min(signal.constraints.maximum_speed_mps, math.hypot(*endpoint))
        if abs(heading) > math.radians(35.0):
            linear = 0.0
        return FastSystemResult(
            FastAction(linear, angular, False),
            FastFeedback.ACTIVE,
            local_path=plan.selected_path,
        )
