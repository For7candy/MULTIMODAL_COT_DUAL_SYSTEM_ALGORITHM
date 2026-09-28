"""Small dependency-free validator for the JSON-Schema subset used at runtime."""

from typing import Any, Dict, Tuple


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def validate_json_schema_subset(
    value: Any,
    schema: Dict[str, Any],
    path: str = "$",
) -> Tuple[str, ...]:
    """Validate types, fields, arrays, enums, constants, and numeric bounds."""

    issues = []
    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = (
            expected_type if isinstance(expected_type, list) else [expected_type]
        )
        if not any(_matches_json_type(value, item) for item in expected_types):
            return ("{}: expected {}".format(path, expected_type),)
    if "const" in schema and value != schema["const"]:
        issues.append("{}: expected constant {}".format(path, schema["const"]))
    if "enum" in schema and value not in schema["enum"]:
        issues.append("{}: value is outside the enum".format(path))
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", ()):
            if required not in value:
                issues.append("{}.{}: missing required field".format(path, required))
        if schema.get("additionalProperties") is False:
            for key in sorted(set(value) - set(properties)):
                issues.append("{}.{}: unknown field".format(path, key))
        for key, item in value.items():
            if key in properties:
                issues.extend(
                    validate_json_schema_subset(
                        item, properties[key], "{}.{}".format(path, key)
                    )
                )
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            issues.append("{}: too few items".format(path))
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                issues.extend(
                    validate_json_schema_subset(
                        item, item_schema, "{}[{}]".format(path, index)
                    )
                )
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            issues.append("{}: string is too short".format(path))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            issues.append("{}: below minimum".format(path))
        if "maximum" in schema and value > schema["maximum"]:
            issues.append("{}: above maximum".format(path))
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            issues.append("{}: below exclusive minimum".format(path))
    return tuple(issues)
