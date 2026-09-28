"""Lazy wrapper around Open-Nav's RGB encoder."""

from typing import Any, Callable, Dict, Optional


class OpenNavRgbEncoderAdapter:
    def __init__(
        self,
        encoder: Any = None,
        encoder_factory: Optional[Callable[[], Any]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._encoder = encoder
        self._encoder_factory = encoder_factory
        self._encoder_kwargs = dict(encoder_kwargs or {})

    @property
    def is_loaded(self) -> bool:
        return self._encoder is not None

    def _load(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        if self._encoder_factory is not None:
            self._encoder = self._encoder_factory()
            return self._encoder

        from vlnce_baselines.models.encoders.resnet_encoders import (
            TorchVisionResNet50,
        )

        required = {"observation_space", "output_size", "device"}
        missing = required.difference(self._encoder_kwargs)
        if missing:
            raise ValueError(
                "missing Open-Nav RGB encoder arguments: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        self._encoder = TorchVisionResNet50(**self._encoder_kwargs)
        self._encoder.eval()
        return self._encoder

    def encode(self, rgb_batch: Any) -> Any:
        encoder = self._load()
        return encoder({"rgb": rgb_batch})
