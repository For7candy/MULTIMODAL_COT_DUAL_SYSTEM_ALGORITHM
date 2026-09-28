"""Lightweight interfaces that isolate optional model dependencies."""

from typing import Any, Dict, Protocol


class RgbEncoder(Protocol):
    def encode(self, rgb_batch: Any) -> Any:
        ...


class WaypointPredictor(Protocol):
    def predict_heatmap(self, rgb_features: Any, depth_features: Any) -> Any:
        ...


class StructuredBackbone(Protocol):
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
    ) -> Dict[str, Any]:
        ...
