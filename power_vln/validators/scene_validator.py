"""Cross-reference and geometry validation for 3D scene descriptions."""

from power_vln.schemas.common import ValidationIssue, ValidationResult
from power_vln.schemas.scene_description import SceneDescription3D


def _duplicate_values(values):
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _inside_axis_aligned_box(position, center, size):
    return (
        abs(position.x - center.x) <= size.x / 2.0
        and abs(position.y - center.y) <= size.y / 2.0
        and abs(position.z - center.z) <= size.z / 2.0
    )


def validate_scene(scene: SceneDescription3D) -> ValidationResult:
    issues = []
    entity_ids = [item.entity_id for item in scene.entities]
    entity_id_set = set(entity_ids)

    for duplicate in sorted(_duplicate_values(entity_ids)):
        issues.append(
            ValidationIssue(
                code="duplicate_entity_id",
                path="entities",
                message="duplicate entity id: {}".format(duplicate),
            )
        )

    for index, relation in enumerate(scene.relations):
        for role, entity_id in (
            ("subject_id", relation.subject_id),
            ("object_id", relation.object_id),
        ):
            if entity_id not in entity_id_set:
                issues.append(
                    ValidationIssue(
                        code="unknown_relation_entity",
                        path="relations[{}].{}".format(index, role),
                        message="relation references unknown entity: {}".format(entity_id),
                    )
                )

    candidate_ids = [item.candidate_id for item in scene.candidate_approaches]
    for duplicate in sorted(_duplicate_values(candidate_ids)):
        issues.append(
            ValidationIssue(
                code="duplicate_candidate_id",
                path="candidate_approaches",
                message="duplicate candidate id: {}".format(duplicate),
            )
        )

    for index, candidate in enumerate(scene.candidate_approaches):
        if candidate.target_entity_id not in entity_id_set:
            issues.append(
                ValidationIssue(
                    code="unknown_candidate_target",
                    path="candidate_approaches[{}].target_entity_id".format(index),
                    message="candidate references unknown target entity",
                )
            )
        for obstacle in scene.obstacles:
            if _inside_axis_aligned_box(candidate.position, obstacle.center, obstacle.size):
                issues.append(
                    ValidationIssue(
                        code="candidate_inside_obstacle",
                        path="candidate_approaches[{}].position".format(index),
                        message="candidate lies inside obstacle {}".format(
                            obstacle.obstacle_id
                        ),
                    )
                )

    for index, uncertainty in enumerate(scene.uncertainties):
        if uncertainty.entity_id and uncertainty.entity_id not in entity_id_set:
            issues.append(
                ValidationIssue(
                    code="unknown_uncertainty_entity",
                    path="uncertainties[{}].entity_id".format(index),
                    message="uncertainty references unknown entity",
                )
            )

    for collection_name, values in (
        ("free_space", [item.region_id for item in scene.free_space]),
        ("obstacles", [item.obstacle_id for item in scene.obstacles]),
    ):
        for duplicate in sorted(_duplicate_values(values)):
            issues.append(
                ValidationIssue(
                    code="duplicate_{}_id".format(collection_name),
                    path=collection_name,
                    message="duplicate id: {}".format(duplicate),
                )
            )

    return ValidationResult(issues=tuple(issues))
