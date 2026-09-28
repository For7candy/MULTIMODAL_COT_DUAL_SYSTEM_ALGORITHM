"""Typed slow-system triggers and deterministic event de-duplication."""

from dataclasses import dataclass
from enum import Enum


class SlowEventType(str, Enum):
    NEW_TASK = "new_task"
    SUBGOAL_COMPLETED = "subgoal_completed"
    SCENE_CHANGED = "scene_changed"
    PATH_BLOCKED = "path_blocked"
    SIGNAL_EXPIRED = "signal_expired"


@dataclass(frozen=True)
class SlowEvent:
    event_id: str
    event_type: SlowEventType
    timestamp_ms: int
    mission_id: str
    subtask_id: str

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.mission_id.strip():
            raise ValueError("event_id and mission_id must not be empty")
        if not self.subtask_id.strip() or self.timestamp_ms < 0:
            raise ValueError("subtask_id and timestamp_ms are invalid")


class EventDeduplicator:
    def __init__(self, maximum_events: int = 4096) -> None:
        if maximum_events <= 0:
            raise ValueError("maximum_events must be positive")
        self.maximum_events = maximum_events
        self._ordered_ids = []
        self._ids = set()

    def contains(self, event_id: str) -> bool:
        return event_id in self._ids

    def add(self, event_id: str) -> None:
        if event_id in self._ids:
            return
        self._ids.add(event_id)
        self._ordered_ids.append(event_id)
        if len(self._ordered_ids) > self.maximum_events:
            removed = self._ordered_ids.pop(0)
            self._ids.remove(removed)
