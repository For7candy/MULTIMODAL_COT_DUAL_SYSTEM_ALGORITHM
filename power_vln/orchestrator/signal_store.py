"""Thread-safe atomic publication of immutable navigation signals."""

from dataclasses import dataclass
from threading import Lock
from typing import Optional

from power_vln.schemas import NavigationSignal, NavigationStatus


@dataclass(frozen=True)
class SignalEnvelope:
    signal: NavigationSignal
    scene_version: int

    def __post_init__(self) -> None:
        if self.scene_version < 0:
            raise ValueError("scene_version must be non-negative")


class AtomicSignalStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._value: Optional[SignalEnvelope] = None

    def publish(
        self,
        signal: NavigationSignal,
        scene_version: int,
        mission_id: str,
        subtask_id: str,
    ) -> SignalEnvelope:
        if signal.status != NavigationStatus.READY or signal.goal is None:
            raise ValueError("only executable ready signals may be published")
        if signal.mission_id != mission_id or signal.subtask_id != subtask_id:
            raise ValueError("signal task identity mismatch")
        envelope = SignalEnvelope(signal, scene_version)
        with self._lock:
            current = self._value
            if current is not None and signal.generated_at_ms < current.signal.generated_at_ms:
                raise ValueError("signal timestamp is older than the active signal")
            self._value = envelope
        return envelope

    def read(
        self,
        scene_version: int,
        mission_id: str,
        subtask_id: str,
        now_ms: int,
    ) -> Optional[SignalEnvelope]:
        with self._lock:
            value = self._value
        if value is None:
            return None
        if value.scene_version != scene_version:
            return None
        signal = value.signal
        if signal.mission_id != mission_id or signal.subtask_id != subtask_id:
            return None
        if signal.is_expired(now_ms):
            return None
        return value

    def clear(self) -> None:
        with self._lock:
            self._value = None
