"""Run a minimal local Qwen structured-CoT smoke test without a simulator."""

import argparse
import json

from power_vln.adapters import Qwen25VLBackboneAdapter
from power_vln.model import (
    NavigationPlanningContext,
    NavigationPlanningError,
    StructuredCoTPlanner,
)
from power_vln.schemas import (
    CandidateApproach,
    Entity3D,
    Pose3D,
    Quaternion,
    SceneDescription3D,
    Vector3,
)


def build_scene() -> SceneDescription3D:
    observer = Pose3D(
        Vector3(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)
    )
    entity = Entity3D(
        entity_id="chair-1",
        category="chair",
        center=Vector3(2.0, 1.0, 0.5),
        size=Vector3(0.5, 0.5, 1.0),
        yaw=0.0,
        attributes=("red",),
        confidence=0.95,
        source_observations=("rgb-1", "pc-1"),
    )
    candidate = CandidateApproach(
        candidate_id="candidate-a",
        target_entity_id="chair-1",
        position=Vector3(1.0, 1.0, 0.0),
        yaw=0.0,
        clearance_m=0.8,
        visibility_score=0.9,
        confidence=0.9,
    )
    return SceneDescription3D(
        schema_version=1,
        scene_id="scene-smoke-1",
        timestamp_ms=1000,
        frame_id="world",
        observer_pose=observer,
        entities=(entity,),
        candidate_approaches=(candidate,),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        required=True,
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    backend = Qwen25VLBackboneAdapter(
        args.model_path, max_new_tokens=args.max_new_tokens
    )
    planner = StructuredCoTPlanner(backend)
    try:
        result = planner.plan(
            build_scene(),
            NavigationPlanningContext(
                instruction="Approach the red chair.",
                mission_id="mission-smoke-1",
                subtask_id="step-1",
                generated_at_ms=1200,
                valid_for_ms=5000,
            ),
        )
    except NavigationPlanningError as error:
        print("planning_error={}".format(error.code))
        print("details={}".format(json.dumps(error.details, ensure_ascii=False)))
        print(
            "model_payload={}".format(
                json.dumps(backend.last_payload, ensure_ascii=False, sort_keys=True)
            )
        )
        raise
    print(result.reasoning.to_json())
    print(result.signal.to_json())


if __name__ == "__main__":
    main()
