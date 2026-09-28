"""Optional Open-Nav and model adapters with lazy dependency loading."""

from .backbone_client import OpenNavBackboneAdapter
from .opennav_rgb import OpenNavRgbEncoderAdapter
from .opennav_waypoint import OpenNavWaypointAdapter
from .protocols import RgbEncoder, StructuredBackbone, WaypointPredictor
from .qwen25_vl import Qwen25VLBackboneAdapter

__all__ = [
    "OpenNavBackboneAdapter",
    "OpenNavRgbEncoderAdapter",
    "OpenNavWaypointAdapter",
    "Qwen25VLBackboneAdapter",
    "RgbEncoder",
    "StructuredBackbone",
    "WaypointPredictor",
]
