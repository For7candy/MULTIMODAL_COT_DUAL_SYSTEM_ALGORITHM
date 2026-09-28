"""Offline-callable integration of event-driven reasoning and local control."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from power_vln.fast_system import (
    FastFeedback,
    FastSystem,
    FastSystemInput,
    FastSystemResult,
    ForbiddenRegion,
)
from power_vln.schemas import StampedPose
from power_vln.slow_system import (
    SlowEvent,
    SlowEventType,
    SlowSystemInput,
    SlowSystemResult,
    SlowSystemStatus,
)
from power_vln.state_machine import DualSystemStateMachine, SystemEvent, SystemState

from .scheduler import SlowPoll, SlowPollStatus, SlowScheduler
from .signal_store import AtomicSignalStore, SignalEnvelope


@dataclass(frozen=True)
class DualSystemStepInput:
    now_ms: int
    scene_version: int
    mission_id: str
    subtask_id: str
    point_cloud: Any
    point_timestamp_ms: int
    point_frame_id: str
    robot_pose: StampedPose
    slow_event: Optional[SlowEvent] = None
    slow_inputs: Optional[SlowSystemInput] = None
    forbidden_regions: Tuple[ForbiddenRegion, ...] = ()
    supplemental_waypoints: Tuple[Tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.now_ms < 0 or self.scene_version < 0:
            raise ValueError("step time and scene version must be non-negative")
        if not self.mission_id or not self.subtask_id:
            raise ValueError("mission_id and subtask_id must not be empty")
        if (self.slow_event is None) != (self.slow_inputs is None):
            raise ValueError("slow_event and slow_inputs must be provided together")


@dataclass(frozen=True)
class DualSystemStepResult:
    state: SystemState
    slow_poll: SlowPoll
    slow_result: Optional[SlowSystemResult]
    active_signal: Optional[SignalEnvelope]
    fast_result: FastSystemResult


class M3DCoTDualSystem:
    def __init__(
        self,
        slow_system,
        fast_system: Optional[FastSystem] = None,
        slow_timeout_ms: int = 15000,
    ) -> None:
        if slow_timeout_ms <= 0:
            raise ValueError("slow_timeout_ms must be positive")
        self.fast_system = fast_system or FastSystem()
        self.signal_store = AtomicSignalStore()
        self.scheduler = SlowScheduler(slow_system)
        self.state_machine = DualSystemStateMachine()
        self.slow_timeout_ms = slow_timeout_ms

    def _safe_dispatch(self, event: SystemEvent) -> None:
        try:
            self.state_machine.dispatch(event)
        except ValueError:
            pass

    def _submit_if_requested(self, inputs: DualSystemStepInput) -> None:
        event = inputs.slow_event
        if event is None:
            return
        if event.mission_id != inputs.mission_id or event.subtask_id != inputs.subtask_id:
            raise ValueError("slow event and step task identity must match")
        if inputs.slow_inputs.memory.version != inputs.scene_version:
            raise ValueError("slow input and step scene versions must match")
        if event.event_type in {
            SlowEventType.NEW_TASK,
            SlowEventType.SUBGOAL_COMPLETED,
            SlowEventType.SCENE_CHANGED,
            SlowEventType.PATH_BLOCKED,
            SlowEventType.SIGNAL_EXPIRED,
        }:
            self.signal_store.clear()
        if self.state_machine.state == SystemState.IDLE:
            self._safe_dispatch(SystemEvent.MISSION_RECEIVED)
        elif self.state_machine.state == SystemState.FAST_EXECUTING:
            trigger = {
                SlowEventType.SUBGOAL_COMPLETED: SystemEvent.SUBGOAL_REACHED,
                SlowEventType.PATH_BLOCKED: SystemEvent.LOCAL_BLOCKED,
                SlowEventType.SIGNAL_EXPIRED: SystemEvent.PLAN_EXPIRED,
            }.get(event.event_type, SystemEvent.SEMANTIC_ANOMALY)
            self._safe_dispatch(trigger)
        self.scheduler.submit(event, inputs.slow_inputs, self.slow_timeout_ms)

    def _consume_slow_result(
        self, poll: SlowPoll, inputs: DualSystemStepInput
    ) -> Optional[SlowSystemResult]:
        result = poll.result
        if poll.status != SlowPollStatus.COMPLETED or result is None:
            if poll.status in {SlowPollStatus.TIMED_OUT, SlowPollStatus.FAILED}:
                self._safe_dispatch(SystemEvent.PLAN_REJECTED)
            return result
        if (
            poll.input_version != inputs.scene_version
            or poll.mission_id != inputs.mission_id
            or poll.subtask_id != inputs.subtask_id
        ):
            self._safe_dispatch(SystemEvent.PLAN_REJECTED)
            return result
        if result.status == SlowSystemStatus.PLANNED and result.signal is not None:
            self.signal_store.publish(
                result.signal,
                poll.input_version,
                inputs.mission_id,
                inputs.subtask_id,
            )
            self._safe_dispatch(SystemEvent.PLAN_ACCEPTED)
        else:
            self._safe_dispatch(SystemEvent.PLAN_REJECTED)
        return result

    def step(self, inputs: DualSystemStepInput) -> DualSystemStepResult:
        self._submit_if_requested(inputs)
        poll = self.scheduler.poll(inputs.now_ms)
        slow_result = self._consume_slow_result(poll, inputs)
        envelope = self.signal_store.read(
            inputs.scene_version,
            inputs.mission_id,
            inputs.subtask_id,
            inputs.now_ms,
        )
        if envelope is None and self.state_machine.state == SystemState.FAST_EXECUTING:
            self.signal_store.clear()
            self._safe_dispatch(SystemEvent.PLAN_EXPIRED)
        fast_result = self.fast_system.step(
            FastSystemInput(
                signal=envelope.signal if envelope else None,
                point_cloud=inputs.point_cloud,
                point_timestamp_ms=inputs.point_timestamp_ms,
                point_frame_id=inputs.point_frame_id,
                robot_pose=inputs.robot_pose,
                now_ms=inputs.now_ms,
                forbidden_regions=inputs.forbidden_regions,
                supplemental_waypoints=inputs.supplemental_waypoints,
            )
        )
        if fast_result.feedback == FastFeedback.REACHED:
            self.signal_store.clear()
            self._safe_dispatch(SystemEvent.SUBGOAL_REACHED)
        elif fast_result.feedback == FastFeedback.BLOCKED:
            self.signal_store.clear()
            self._safe_dispatch(SystemEvent.LOCAL_BLOCKED)
        elif fast_result.feedback == FastFeedback.INPUT_STALE:
            self.signal_store.clear()
            self._safe_dispatch(SystemEvent.SEMANTIC_ANOMALY)
        elif fast_result.feedback == FastFeedback.SAFETY_STOP and envelope is not None:
            self.signal_store.clear()
            self._safe_dispatch(SystemEvent.EMERGENCY)
        return DualSystemStepResult(
            state=self.state_machine.state,
            slow_poll=poll,
            slow_result=slow_result,
            active_signal=self.signal_store.read(
                inputs.scene_version,
                inputs.mission_id,
                inputs.subtask_id,
                inputs.now_ms,
            ),
            fast_result=fast_result,
        )

    def close(self) -> None:
        self.scheduler.close()
