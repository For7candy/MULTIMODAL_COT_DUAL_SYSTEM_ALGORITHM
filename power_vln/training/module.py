"""Fine-tuning policies for a 12 GB local GPU training workflow."""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple

from torch import nn


class FineTuneMode(str, Enum):
    FROZEN_BACKBONE = "frozen_backbone"
    ADAPTER_ONLY = "adapter_only"
    LORA = "lora"


@dataclass(frozen=True)
class LoraSpec:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.dropout < 1.0 or not self.target_modules:
            raise ValueError("LoRA dropout or target modules are invalid")


@dataclass(frozen=True)
class TrainableSummary:
    trainable_parameters: int
    total_parameters: int

    @property
    def trainable_fraction(self) -> float:
        if self.total_parameters == 0:
            return 0.0
        return self.trainable_parameters / self.total_parameters


_ADAPTER_KEYWORDS = (
    "adapter",
    "projector",
    "point_encoder",
    "fusion",
    "scene_query",
    "lora_",
)


def configure_trainable_parameters(
    model: nn.Module,
    mode: FineTuneMode,
    backbone_keywords: Iterable[str] = ("backbone", "foundation_model"),
    adapter_keywords: Iterable[str] = _ADAPTER_KEYWORDS,
) -> TrainableSummary:
    backbone_keywords = tuple(backbone_keywords)
    adapter_keywords = tuple(adapter_keywords)
    for name, parameter in model.named_parameters():
        if mode == FineTuneMode.FROZEN_BACKBONE:
            parameter.requires_grad = not any(key in name for key in backbone_keywords)
        elif mode == FineTuneMode.ADAPTER_ONLY:
            parameter.requires_grad = any(key in name for key in adapter_keywords)
        elif mode == FineTuneMode.LORA:
            parameter.requires_grad = "lora_" in name or any(
                key in name for key in adapter_keywords if key != "lora_"
            )
        else:
            raise ValueError("unsupported fine-tune mode")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return TrainableSummary(trainable, total)


def attach_lora(model: nn.Module, spec: LoraSpec = LoraSpec()) -> nn.Module:
    """Attach PEFT LoRA lazily; importing this module does not require PEFT."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:
        raise RuntimeError("LoRA mode requires the optional 'peft' package") from error
    config = LoraConfig(
        r=spec.rank,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=list(spec.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)
