"""Non-blocking slow-system scheduling with logical cancellation and timeouts."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Optional

from power_vln.slow_system import SlowEvent, SlowSystemInput, SlowSystemResult


class SlowPollStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


@dataclass(frozen=True)
class SlowPoll:
    status: SlowPollStatus
    event_id: Optional[str] = None
    result: Optional[SlowSystemResult] = None
    error_code: Optional[str] = None
    input_version: Optional[int] = None
    mission_id: Optional[str] = None
    subtask_id: Optional[str] = None


@dataclass(frozen=True)
class _SlowJob:
    generation: int
    event_id: str
    deadline_ms: int
    input_version: int
    mission_id: str
    subtask_id: str
    future: Future


class SlowScheduler:
    def __init__(self, slow_system, maximum_workers: int = 2) -> None:
        if maximum_workers <= 0:
            raise ValueError("maximum_workers must be positive")
        self.slow_system = slow_system
        self._executor = ThreadPoolExecutor(
            max_workers=maximum_workers, thread_name_prefix="power-vln-slow"
        )
        self._lock = Lock()
        self._generation = 0
        self._job: Optional[_SlowJob] = None
        self._seen_event_ids = set()

    def submit(
        self,
        event: SlowEvent,
        inputs: SlowSystemInput,
        timeout_ms: int,
    ) -> bool:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        with self._lock:
            if event.event_id in self._seen_event_ids:
                return False
            self._seen_event_ids.add(event.event_id)
            self._generation += 1
            generation = self._generation
            old_job = self._job
            if old_job is not None:
                old_job.future.cancel()
            future = self._executor.submit(self.slow_system.handle, event, inputs)
            self._job = _SlowJob(
                generation,
                event.event_id,
                event.timestamp_ms + timeout_ms,
                inputs.memory.version,
                event.mission_id,
                event.subtask_id,
                future,
            )
        return True

    def poll(self, now_ms: int) -> SlowPoll:
        with self._lock:
            job = self._job
        if job is None:
            return SlowPoll(SlowPollStatus.IDLE)
        metadata = {
            "input_version": job.input_version,
            "mission_id": job.mission_id,
            "subtask_id": job.subtask_id,
        }
        if now_ms >= job.deadline_ms:
            job.future.cancel()
            with self._lock:
                if self._job is job:
                    self._job = None
            return SlowPoll(
                SlowPollStatus.TIMED_OUT,
                job.event_id,
                error_code="slow_timeout",
                **metadata,
            )
        if not job.future.done():
            return SlowPoll(SlowPollStatus.RUNNING, job.event_id, **metadata)
        with self._lock:
            if self._job is job:
                self._job = None
            current_generation = self._generation
        if job.generation != current_generation:
            return SlowPoll(SlowPollStatus.IDLE)
        try:
            return SlowPoll(
                SlowPollStatus.COMPLETED,
                job.event_id,
                result=job.future.result(),
                **metadata,
            )
        except Exception:
            return SlowPoll(
                SlowPollStatus.FAILED,
                job.event_id,
                error_code="slow_worker_error",
                **metadata,
            )

    def cancel(self) -> None:
        with self._lock:
            self._generation += 1
            job = self._job
            self._job = None
        if job is not None:
            job.future.cancel()

    def close(self, wait: bool = False) -> None:
        self.cancel()
        self._executor.shutdown(wait=wait, cancel_futures=True)
