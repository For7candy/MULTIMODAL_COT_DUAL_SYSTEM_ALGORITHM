"""Deterministic coordination state machine for the fast-slow architecture."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class SystemState(str, Enum):
    IDLE = "idle"
    SLOW_REASONING = "slow_reasoning"
    SLOW_PLANNING = "slow_reasoning"
    FAST_EXECUTING = "fast_executing"
    REPLAN_REQUIRED = "replan_required"
    REPLANNING = "replan_required"
    STOPPED = "stopped"
    SAFETY_OVERRIDE = "stopped"
    COMPLETE = "complete"
    FAULT = "fault"


class SystemEvent(str, Enum):
    MISSION_RECEIVED = "mission_received"
    PLAN_ACCEPTED = "plan_accepted"
    PLAN_REJECTED = "plan_rejected"
    SUBGOAL_REACHED = "subgoal_reached"
    MISSION_COMPLETED = "mission_completed"
    SEMANTIC_ANOMALY = "semantic_anomaly"
    LOCAL_BLOCKED = "local_blocked"
    PLAN_EXPIRED = "plan_expired"
    EMERGENCY = "emergency"
    EMERGENCY_CLEARED = "emergency_cleared"
    SYSTEM_FAULT = "system_fault"
    RESET = "reset"


@dataclass(frozen=True)
class Transition:
    previous: SystemState
    event: SystemEvent
    current: SystemState


class DualSystemStateMachine:
    """Fail-closed state machine; safety and fault events preempt normal work."""

    _TRANSITIONS: Dict[Tuple[SystemState, SystemEvent], SystemState] = {
        (SystemState.IDLE, SystemEvent.MISSION_RECEIVED): SystemState.SLOW_REASONING,
        (SystemState.SLOW_REASONING, SystemEvent.PLAN_ACCEPTED): SystemState.FAST_EXECUTING,
        (SystemState.SLOW_REASONING, SystemEvent.PLAN_REJECTED): SystemState.REPLAN_REQUIRED,
        (SystemState.FAST_EXECUTING, SystemEvent.SUBGOAL_REACHED): SystemState.SLOW_REASONING,
        (SystemState.FAST_EXECUTING, SystemEvent.MISSION_COMPLETED): SystemState.COMPLETE,
        (SystemState.FAST_EXECUTING, SystemEvent.SEMANTIC_ANOMALY): SystemState.REPLAN_REQUIRED,
        (SystemState.FAST_EXECUTING, SystemEvent.LOCAL_BLOCKED): SystemState.REPLAN_REQUIRED,
        (SystemState.FAST_EXECUTING, SystemEvent.PLAN_EXPIRED): SystemState.REPLAN_REQUIRED,
        (SystemState.REPLAN_REQUIRED, SystemEvent.PLAN_ACCEPTED): SystemState.FAST_EXECUTING,
        (SystemState.REPLAN_REQUIRED, SystemEvent.PLAN_REJECTED): SystemState.REPLAN_REQUIRED,
        (SystemState.STOPPED, SystemEvent.EMERGENCY_CLEARED): SystemState.REPLAN_REQUIRED,
        (SystemState.COMPLETE, SystemEvent.RESET): SystemState.IDLE,
        (SystemState.FAULT, SystemEvent.RESET): SystemState.IDLE,
    }

    def __init__(self) -> None:
        self._state = SystemState.IDLE

    @property
    def state(self) -> SystemState:
        return self._state

    def dispatch(self, event: SystemEvent) -> Transition:
        previous = self._state

        if event == SystemEvent.SYSTEM_FAULT:
            self._state = SystemState.FAULT
        elif event == SystemEvent.EMERGENCY and self._state != SystemState.FAULT:
            self._state = SystemState.STOPPED
        else:
            key = (self._state, event)
            if key not in self._TRANSITIONS:
                raise ValueError(
                    "invalid transition: {} + {}".format(self._state.value, event.value)
                )
            self._state = self._TRANSITIONS[key]

        return Transition(previous=previous, event=event, current=self._state)
