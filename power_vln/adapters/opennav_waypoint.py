"""Lazy wrapper around Open-Nav's candidate waypoint predictor."""

from contextlib import nullcontext
from typing import Any, Callable, Optional


class OpenNavWaypointAdapter:
    def __init__(
        self,
        predictor: Any = None,
        predictor_factory: Optional[Callable[[], Any]] = None,
        checkpoint_path: Optional[str] = None,
        device: Any = None,
    ) -> None:
        self._predictor = predictor
        self._predictor_factory = predictor_factory
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._uses_torch = predictor is None

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def _load(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        if self._predictor_factory is not None:
            self._predictor = self._predictor_factory()
            self._uses_torch = False
            return self._predictor

        if not self._checkpoint_path:
            raise ValueError("checkpoint_path is required for the Open-Nav predictor")

        import torch
        from waypoint_prediction.TRM_net import BinaryDistPredictor_TRM

        predictor = BinaryDistPredictor_TRM(device=self._device)
        checkpoint = torch.load(self._checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("predictor", checkpoint).get(
            "state_dict", checkpoint.get("predictor", checkpoint)
        )
        predictor.load_state_dict(state_dict)
        predictor.to(self._device)
        predictor.eval()
        for parameter in predictor.parameters():
            parameter.requires_grad = False
        self._predictor = predictor
        return predictor

    def predict_heatmap(self, rgb_features: Any, depth_features: Any) -> Any:
        predictor = self._load()
        context = nullcontext()
        if self._uses_torch:
            import torch

            context = torch.no_grad()
        with context:
            return predictor(rgb_features, depth_features)

    def suppress(self, heatmap: Any, max_predictions: int = 10) -> Any:
        if max_predictions <= 0:
            raise ValueError("max_predictions must be positive")
        from waypoint_prediction.utils import nms

        return nms(heatmap, max_predictions=max_predictions)
