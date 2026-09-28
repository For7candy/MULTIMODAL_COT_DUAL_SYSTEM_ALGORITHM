"""Dependency-free primitives for versioned power-VLN schemas."""

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from typing import Any, Dict, Iterable, Optional, Tuple


def require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must not be empty".format(field_name))


def require_confidence(value: float, field_name: str = "confidence") -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("{} must be within [0, 1]".format(field_name))


def require_finite(values: Iterable[float], field_name: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("{} must contain finite values".format(field_name))


def to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    return value


class JsonSchemaMixin:
    """Common deterministic JSON serialization for immutable contracts."""

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(asdict(self))

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )


@dataclass(frozen=True)
class Vector3(JsonSchemaMixin):
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        require_finite((self.x, self.y, self.z), "Vector3")

    @classmethod
    def from_value(cls, value: Any) -> "Vector3":
        if isinstance(value, dict):
            return cls(x=value["x"], y=value["y"], z=value["z"])
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return cls(x=value[0], y=value[1], z=value[2])
        raise ValueError("Vector3 requires an xyz mapping or three-item sequence")

    def to_list(self) -> list:
        return [self.x, self.y, self.z]


@dataclass(frozen=True)
class Quaternion(JsonSchemaMixin):
    x: float
    y: float
    z: float
    w: float

    def __post_init__(self) -> None:
        require_finite((self.x, self.y, self.z, self.w), "Quaternion")
        norm = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2 + self.w ** 2)
        if norm < 1e-8:
            raise ValueError("Quaternion norm must be non-zero")

    @classmethod
    def from_value(cls, value: Any) -> "Quaternion":
        if isinstance(value, dict):
            return cls(x=value["x"], y=value["y"], z=value["z"], w=value["w"])
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return cls(x=value[0], y=value[1], z=value[2], w=value[3])
        raise ValueError("Quaternion requires an xyzw mapping or four-item sequence")

    def to_list(self) -> list:
        return [self.x, self.y, self.z, self.w]


@dataclass(frozen=True)
class Pose3D(JsonSchemaMixin):
    position: Vector3
    orientation: Quaternion

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Pose3D":
        return cls(
            position=Vector3.from_value(payload["position"]),
            orientation=Quaternion.from_value(payload["orientation"]),
        )


@dataclass(frozen=True)
class ValidationIssue(JsonSchemaMixin):
    code: str
    path: str
    message: str

    def __post_init__(self) -> None:
        require_text(self.code, "code")
        require_text(self.path, "path")
        require_text(self.message, "message")


@dataclass(frozen=True)
class ValidationResult(JsonSchemaMixin):
    issues: Tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def raise_for_errors(self) -> None:
        if self.issues:
            detail = "; ".join(
                "{}@{}: {}".format(item.code, item.path, item.message)
                for item in self.issues
            )
            raise SchemaValidationError(detail, self.issues)


class SchemaValidationError(ValueError):
    def __init__(self, message: str, issues: Optional[Tuple[ValidationIssue, ...]] = None):
        super().__init__(message)
        self.issues = issues or ()
