"""End-to-end offline demonstration of the M3D-CoT-DS algorithm.

The default mock backend is deterministic and exercises every interface without
requiring model weights. ``--backend qwen`` replaces only the structured model
backend with the local Qwen2.5-VL checkpoint.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.nn import functional as functional

from power_vln.adapters import Qwen25VLBackboneAdapter
from power_vln.data import prepare_inputs
from power_vln.encoders import PointTransformerSmall, PoseTokenEncoder, RgbTokenEncoder
from power_vln.fast_system import FastSystem, FastSystemInput
from power_vln.fusion import (
    BidirectionalCrossModalFusion,
    SceneQueryAggregator,
    SpatialMemory,
    align_modalities_to_world,
)
from power_vln.model import (
    MultimodalInputAssembler,
    NavigationPlanningContext,
    SceneGenerationContext,
    SceneGenerator,
    StructuredCoTPlanner,
)
from power_vln.schemas import (
    CameraIntrinsics,
    PointCloudObservation,
    Pose3D,
    Quaternion,
    RgbObservation,
    RigidTransform,
    SensorPacket,
    StampedPose,
    Vector3,
)
from power_vln.slow_system import SlowEvent, SlowEventType, SlowSystem, SlowSystemInput


class DemoStructuredBackbone:
    """Deterministic stand-in for an untrained structured foundation model."""

    def generate_structured(self, request):
        if request.schema_name == "SceneDescription3D-v1":
            return {
                "schema_version": 1,
                "scene_id": "demo-scene-1",
                "timestamp_ms": 1000,
                "frame_id": "world",
                "observer_pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "entities": [
                    {
                        "entity_id": "cabinet-1",
                        "category": "electrical_cabinet",
                        "center": {"x": 2.0, "y": 0.0, "z": 1.0},
                        "size": {"x": 0.8, "y": 0.4, "z": 2.0},
                        "yaw": 0.0,
                        "attributes": ["closed", "red_indicator_off"],
                        "confidence": 0.95,
                        "source_observations": ["rgb-demo-1", "pc-demo-1"],
                    }
                ],
                "relations": [],
                "free_space": [
                    {
                        "region_id": "corridor-1",
                        "vertices": [
                            {"x": 0.0, "y": -0.8, "z": 0.0},
                            {"x": 2.0, "y": -0.8, "z": 0.0},
                            {"x": 2.0, "y": 0.8, "z": 0.0},
                            {"x": 0.0, "y": 0.8, "z": 0.0},
                        ],
                        "clearance_height_m": 2.0,
                        "confidence": 0.9,
                    }
                ],
                "obstacles": [],
                "candidate_approaches": [
                    {
                        "candidate_id": "cabinet-1-front",
                        "target_entity_id": "cabinet-1",
                        "position": {"x": 1.2, "y": 0.0, "z": 0.0},
                        "yaw": 0.0,
                        "clearance_m": 0.8,
                        "visibility_score": 0.92,
                        "confidence": 0.91,
                    }
                ],
                "uncertainties": [],
            }
        if request.schema_name == "NavigationPlan-v1":
            return {
                "reasoning": {
                    "task_parse": {
                        "action": "approach",
                        "target": "electrical cabinet",
                        "constraints": ["maintain_clearance"],
                    },
                    "grounding": {"entity_id": "cabinet-1", "confidence": 0.95},
                    "candidate_ids": ["cabinet-1-front"],
                    "evaluations": [
                        {
                            "candidate_id": "cabinet-1-front",
                            "reachable": True,
                            "score": 0.91,
                            "reason_codes": ["clear", "target_visible"],
                        }
                    ],
                    "decision": {
                        "selected_candidate_id": "cabinet-1-front",
                        "reason_codes": ["highest_score"],
                    },
                },
                "signal": {
                    "schema_version": 1,
                    "mission_id": "demo-mission-1",
                    "scene_id": "demo-scene-1",
                    "subtask_id": "approach-cabinet",
                    "intent": "approach",
                    "status": "ready",
                    "constraints": {
                        "minimum_clearance_m": 0.5,
                        "maximum_speed_mps": 0.3,
                        "forbidden_region_ids": [],
                    },
                    "confidence": 0.91,
                    "generated_at_ms": 1000,
                    "valid_for_ms": 5000,
                    "replan_triggers": [
                        "scene_changed",
                        "path_blocked",
                        "target_missing",
                    ],
                    "target_entity_id": "cabinet-1",
                    "candidate_id": "cabinet-1-front",
                    "goal": {
                        "frame_id": "world",
                        "position": {"x": 1.2, "y": 0.0, "z": 0.0},
                        "yaw": 0.0,
                        "tolerance_m": 0.2,
                    },
                },
            }
        raise ValueError("unsupported demo schema: {}".format(request.schema_name))


def _synthetic_inputs():
    axis = torch.linspace(0.0, 1.0, 32)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    rgb = torch.stack((grid_x, 1.0 - grid_y, 0.25 + 0.5 * grid_x * grid_y), dim=0)
    rgb = rgb.unsqueeze(0)

    point_rows = []
    for y in torch.linspace(-0.7, 0.7, 8):
        for x in torch.linspace(-0.7, 0.7, 8):
            point_rows.append((float(x), float(y), 2.0, 0.8))
    point_cloud = torch.tensor(point_rows, dtype=torch.float32).unsqueeze(0)
    return rgb, point_cloud


def _rgb_features(rgb: torch.Tensor) -> torch.Tensor:
    pooled = functional.adaptive_avg_pool2d(rgb, (8, 8))
    batch, _, height, width = pooled.shape
    y = torch.linspace(-1.0, 1.0, height, dtype=pooled.dtype)
    x = torch.linspace(-1.0, 1.0, width, dtype=pooled.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_x = grid_x.expand(batch, 1, -1, -1)
    grid_y = grid_y.expand(batch, 1, -1, -1)
    luminance = pooled.mean(dim=1, keepdim=True)
    return torch.cat(
        (
            pooled,
            luminance,
            grid_x,
            grid_y,
            pooled[:, 0:1] - pooled[:, 1:2],
            pooled[:, 2:3] - pooled[:, 1:2],
        ),
        dim=1,
    )


def _identity_pose() -> Pose3D:
    return Pose3D(Vector3(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0))


def _sensor_packet(rgb, point_cloud) -> SensorPacket:
    identity = _identity_pose()
    return SensorPacket(
        schema_version=1,
        packet_id="packet-demo-1",
        instruction="Approach the electrical cabinet while maintaining clearance.",
        rgb=RgbObservation(
            "rgb-demo-1", 1000, "camera_rgb", 32, 32, payload=rgb
        ),
        point_cloud=PointCloudObservation(
            "pc-demo-1", 1000, "lidar", point_cloud.shape[1], 4,
            payload=point_cloud,
        ),
        robot_pose=StampedPose(1000, "world", identity, "base_link", (0.0,) * 36),
        camera_intrinsics=CameraIntrinsics(32, 32, 24.0, 24.0, 15.5, 15.5),
        sensor_extrinsics=(
            RigidTransform("base_link", "camera_rgb", identity),
            RigidTransform("base_link", "lidar", identity),
        ),
    )


def _tensor_transform(matrix, dtype=torch.float32):
    return torch.tensor(matrix, dtype=dtype).unsqueeze(0)


def _action_dict(result):
    return {
        "feedback": result.feedback.value,
        "action": {
            "linear_velocity_mps": result.action.linear_velocity_mps,
            "angular_velocity_rps": result.action.angular_velocity_rps,
            "stop": result.action.stop,
        },
        "reason_codes": list(result.reason_codes),
        "local_path": (
            [list(point) for point in result.local_path.points]
            if result.local_path is not None
            else None
        ),
    }


def run_demo(backend_name: str = "mock", model_path: str = "") -> Dict[str, Any]:
    torch.manual_seed(7)
    model_width = 2048 if backend_name == "qwen" else 32
    fusion_width = 64 if backend_name == "qwen" else 16
    rgb, point_cloud = _synthetic_inputs()
    packet = _sensor_packet(rgb, point_cloud)
    aligned = prepare_inputs(packet)

    rgb_encoder = RgbTokenEncoder(8, model_width, output_tokens=16).eval()
    point_encoder = PointTransformerSmall(
        point_feature_size=4,
        hidden_size=32,
        output_size=model_width,
        layers=1,
        attention_heads=4,
        neighbors=8,
        maximum_points=64,
        output_tokens=16,
    ).eval()
    pose_encoder = PoseTokenEncoder(
        output_size=model_width,
        fourier_bands=2,
        include_timestamp=True,
        include_covariance=True,
    ).eval()
    fusion = BidirectionalCrossModalFusion(
        input_size=model_width,
        hidden_size=fusion_width,
        layers=1,
        attention_heads=4,
        neighborhood_radius_m=0.75,
    ).eval()
    scene_queries = SceneQueryAggregator(
        source_size=fusion_width,
        hidden_size=fusion_width,
        output_size=model_width,
        scene_queries=8,
        attention_heads=4,
    ).eval()

    with torch.no_grad():
        rgb_tokens = rgb_encoder(_rgb_features(rgb))
        point_tokens = point_encoder(point_cloud)
        pose_tokens = pose_encoder(
            torch.zeros(1, 3),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            torch.tensor([1.0]),
            torch.zeros(1, 36),
        )
        depth = torch.full((1, rgb_tokens.features.shape[1]), 2.0)
        intrinsics = torch.tensor(
            [[[24.0, 0.0, 15.5], [0.0, 24.0, 15.5], [0.0, 0.0, 1.0]]]
        )
        rgb_world, point_world = align_modalities_to_world(
            rgb_tokens,
            point_tokens,
            depth,
            intrinsics,
            _tensor_transform(aligned.world_from_rgb),
            (32, 32),
            _tensor_transform(aligned.world_from_point_cloud),
        )
        fused = fusion(rgb_world, point_world, pose_tokens)
        scene_token_result = scene_queries(fused)

    memory = SpatialMemory(model_width, frame_id="world")
    memory_snapshot = memory.update(scene_token_result.tokens, 1000)
    if backend_name == "mock":
        backbone = DemoStructuredBackbone()
    elif backend_name == "qwen":
        if not model_path:
            raise ValueError("model_path is required for the qwen backend")
        backbone = Qwen25VLBackboneAdapter(model_path)
    else:
        raise ValueError("backend must be 'mock' or 'qwen'")

    scene_generator = SceneGenerator(MultimodalInputAssembler(model_width), backbone)
    cot_planner = StructuredCoTPlanner(backbone, hidden_size=model_width)
    slow_system = SlowSystem(scene_generator, cot_planner, "demo-{}".format(backend_name))
    scene_context = SceneGenerationContext(
        packet.instruction,
        1000,
        "world",
        packet.robot_pose.pose,
        (packet.rgb.observation_id, packet.point_cloud.observation_id),
    )
    planning_context = NavigationPlanningContext(
        packet.instruction,
        "demo-mission-1",
        "approach-cabinet",
        1000,
        valid_for_ms=5000,
    )
    slow_inputs = SlowSystemInput(
        scene_token_result.tokens,
        memory_snapshot,
        pose_tokens,
        scene_context,
        planning_context,
    )
    first = slow_system.handle(
        SlowEvent(
            "demo-event-1", SlowEventType.NEW_TASK, 1000,
            "demo-mission-1", "approach-cabinet",
        ),
        slow_inputs,
    )
    second = slow_system.handle(
        SlowEvent(
            "demo-event-2", SlowEventType.PATH_BLOCKED, 1000,
            "demo-mission-1", "approach-cabinet",
        ),
        slow_inputs,
    )
    if not first.executable or first.signal is None:
        raise RuntimeError("demo slow system did not produce an executable signal")

    fast_system = FastSystem()
    clear_points = ((4.0, 4.0, 0.5),)
    common_fast = dict(
        signal=first.signal,
        point_timestamp_ms=1000,
        point_frame_id="base_link",
        robot_pose=packet.robot_pose,
    )
    active = fast_system.step(
        FastSystemInput(point_cloud=clear_points, now_ms=1050, **common_fast)
    )
    blocked = fast_system.step(
        FastSystemInput(point_cloud=((1.2, 0.0, 0.5),), now_ms=1050, **common_fast)
    )
    expired = fast_system.step(
        FastSystemInput(point_cloud=clear_points, now_ms=6000, **common_fast)
    )

    return {
        "demo": {"backend": backend_name, "deterministic_seed": 7},
        "inputs": {
            "instruction": packet.instruction,
            "rgb_shape": list(rgb.shape),
            "point_cloud_shape": list(point_cloud.shape),
            "pose_frame": packet.robot_pose.frame_id,
            "sensor_frames": [packet.rgb.frame_id, packet.point_cloud.frame_id],
            "reference_timestamp_ms": aligned.reference_timestamp_ms,
        },
        "features_and_fusion": {
            "rgb_tokens": int(rgb_tokens.mask.sum()),
            "point_tokens": int(point_tokens.mask.sum()),
            "fused_tokens": int(fused.mask.sum()),
            "scene_query_tokens": int(scene_token_result.tokens.mask.sum()),
            "foundation_model_width": model_width,
            "memory_version": memory_snapshot.version,
            "memory_tokens": memory_snapshot.token_count,
        },
        "structured_scene": first.scene.to_dict(),
        "structured_cot": first.reasoning.to_dict(),
        "navigation_signal": first.signal.to_dict(),
        "slow_system": {
            "first_status": first.status.value,
            "first_cache_hit": first.trace.cache_hit,
            "second_status": second.status.value,
            "second_cache_hit": second.trace.cache_hit,
        },
        "fast_system": {
            "clear_path": _action_dict(active),
            "goal_blocked": _action_dict(blocked),
            "signal_expired": _action_dict(expired),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mock", "qwen"), default="mock")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_demo(arguments.backend, arguments.model_path)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
        print("demo_output={}".format(arguments.output.resolve()))
    else:
        print(serialized)


if __name__ == "__main__":
    main()
