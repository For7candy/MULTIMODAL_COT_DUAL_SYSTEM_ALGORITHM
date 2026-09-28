"""Scene-description cache that cannot cross spatial-memory versions."""

from collections import OrderedDict
from typing import Optional, Tuple

from power_vln.schemas import SceneDescription3D


class ScenePlanCache:
    def __init__(self, maximum_entries: int = 8) -> None:
        if maximum_entries <= 0:
            raise ValueError("maximum_entries must be positive")
        self.maximum_entries = maximum_entries
        self._entries = OrderedDict()

    def _key(self, memory_version: int, frame_id: str) -> Tuple[int, str]:
        if memory_version < 0 or not frame_id:
            raise ValueError("cache key is invalid")
        return memory_version, frame_id

    def get(
        self, memory_version: int, frame_id: str
    ) -> Optional[SceneDescription3D]:
        key = self._key(memory_version, frame_id)
        scene = self._entries.get(key)
        if scene is not None:
            self._entries.move_to_end(key)
        return scene

    def put(
        self,
        memory_version: int,
        frame_id: str,
        scene: SceneDescription3D,
    ) -> None:
        key = self._key(memory_version, frame_id)
        if scene.frame_id != frame_id:
            raise ValueError("cached scene frame does not match the cache key")
        self._entries[key] = scene
        self._entries.move_to_end(key)
        while len(self._entries) > self.maximum_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
