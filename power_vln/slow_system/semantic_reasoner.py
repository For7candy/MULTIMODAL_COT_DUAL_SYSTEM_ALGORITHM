"""Synchronous event-driven slow-system orchestration with fail-closed outputs."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from power_vln.encoders import TokenBatch
from power_vln.fusion import FusedSceneTokens, SpatialMemorySnapshot
from power_vln.model import (
    NavigationPlanResult,
    NavigationPlanningContext,
    NavigationPlanningError,
    SceneGenerationContext,
    SceneGenerationError,
)
from power_vln.schemas import (
    NavigationReasoning,
    NavigationSignal,
    NavigationStatus,
    SceneDescription3D,
)

from .event_handler import EventDeduplicator, SlowEvent, SlowEventType
from .plan_cache import ScenePlanCache


class SlowSystemStatus(str, Enum):
    PLANNED = "planned"
    NON_EXECUTABLE = "non_executable"
    DUPLICATE_EVENT = "duplicate_event"
    FAILED = "failed"


@dataclass(frozen=True)
class SlowSystemTrace:
    event_id: str
    input_version: int
    model_version: str
    scene_prompt_version: str
    cot_prompt_version: str
    cache_hit: bool
    error_code: Optional[str] = None


@dataclass(frozen=True)
class SlowSystemInput:
    current_scene_tokens: FusedSceneTokens
    memory: SpatialMemorySnapshot
    pose: TokenBatch
    scene_context: SceneGenerationContext
    planning_context: NavigationPlanningContext

    def __post_init__(self) -> None:
        if self.memory.frame_id != self.scene_context.frame_id:
            raise ValueError("memory and scene context frames must match")
        if self.planning_context.generated_at_ms != self.scene_context.timestamp_ms:
            raise ValueError("scene and planning timestamps must match")


@dataclass(frozen=True)
class SlowSystemResult:
    status: SlowSystemStatus
    trace: SlowSystemTrace
    scene: Optional[SceneDescription3D] = None
    reasoning: Optional[NavigationReasoning] = None
    signal: Optional[NavigationSignal] = None

    @property
    def executable(self) -> bool:
        return (
            self.status == SlowSystemStatus.PLANNED
            and self.signal is not None
            and self.signal.status == NavigationStatus.READY
        )


class SlowSystem:
    """Cache scene generation, run CoT, and atomically publish valid signals."""

    _INVALIDATING_EVENTS = {
        SlowEventType.NEW_TASK,
        SlowEventType.SUBGOAL_COMPLETED,
        SlowEventType.SCENE_CHANGED,
        SlowEventType.PATH_BLOCKED,
        SlowEventType.SIGNAL_EXPIRED,
    }

    def __init__(
        self,
        scene_generator,
        cot_planner,
        model_version: str,
        cache: Optional[ScenePlanCache] = None,
        scene_prompt_version: str = "SceneDescription3D-v1",
        cot_prompt_version: str = "NavigationPlan-v1",
    ) -> None:
        if not model_version.strip():
            raise ValueError("model_version must not be empty")
        self.scene_generator = scene_generator
        self.cot_planner = cot_planner
        self.model_version = model_version
        self.cache = cache or ScenePlanCache()
        self.scene_prompt_version = scene_prompt_version
        self.cot_prompt_version = cot_prompt_version
        self._events = EventDeduplicator()
        self._active_signal: Optional[NavigationSignal] = None

    def active_signal(self, now_ms: int) -> Optional[NavigationSignal]:
        signal = self._active_signal
        if signal is None or signal.is_expired(now_ms):
            return None
        return signal

    def _trace(
        self,
        event: SlowEvent,
        input_version: int,
        cache_hit: bool,
        error_code: Optional[str] = None,
    ) -> SlowSystemTrace:
        return SlowSystemTrace(
            event_id=event.event_id,
            input_version=input_version,
            model_version=self.model_version,
            scene_prompt_version=self.scene_prompt_version,
            cot_prompt_version=self.cot_prompt_version,
            cache_hit=cache_hit,
            error_code=error_code,
        )

    def handle(
        self,
        event: SlowEvent,
        inputs: SlowSystemInput,
    ) -> SlowSystemResult:
        if event.mission_id != inputs.planning_context.mission_id:
            raise ValueError("event and planning mission IDs must match")
        if event.subtask_id != inputs.planning_context.subtask_id:
            raise ValueError("event and planning subtask IDs must match")
        if self._events.contains(event.event_id):
            return SlowSystemResult(
                status=SlowSystemStatus.DUPLICATE_EVENT,
                trace=self._trace(
                    event,
                    inputs.memory.version,
                    cache_hit=False,
                    error_code="duplicate_event",
                ),
                signal=self.active_signal(event.timestamp_ms),
            )
        self._events.add(event.event_id)
        # Every supported trigger changes the task, subgoal, scene, or safety
        # assumptions under which the previous command was produced. Retire it
        # before replanning so a failed slow pass can never leak a stale command.
        if event.event_type in self._INVALIDATING_EVENTS:
            self._active_signal = None

        scene = self.cache.get(inputs.memory.version, inputs.memory.frame_id)
        cache_hit = scene is not None
        try:
            if scene is None:
                scene_result = self.scene_generator.generate(
                    inputs.current_scene_tokens,
                    inputs.memory,
                    inputs.pose,
                    inputs.scene_context,
                )
                scene = scene_result.scene
                self.cache.put(inputs.memory.version, inputs.memory.frame_id, scene)
            plan_result: NavigationPlanResult = self.cot_planner.plan(
                scene, inputs.planning_context
            )
        except (SceneGenerationError, NavigationPlanningError, ValueError) as error:
            error_code = getattr(error, "code", "slow_system_error")
            return SlowSystemResult(
                status=SlowSystemStatus.FAILED,
                trace=self._trace(
                    event, inputs.memory.version, cache_hit, error_code=error_code
                ),
                scene=scene,
                signal=self.active_signal(event.timestamp_ms),
            )

        if plan_result.signal.status == NavigationStatus.READY:
            self._active_signal = plan_result.signal
            status = SlowSystemStatus.PLANNED
        else:
            self._active_signal = None
            status = SlowSystemStatus.NON_EXECUTABLE
        return SlowSystemResult(
            status=status,
            trace=self._trace(event, inputs.memory.version, cache_hit),
            scene=scene,
            reasoning=plan_result.reasoning,
            signal=plan_result.signal,
        )
