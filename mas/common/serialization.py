"""Shared serialization and numeric parsing helpers."""

from __future__ import annotations

from typing import Any, Callable, Optional

ScalarSerializer = Callable[[Any], Any]


def serialize_value_attr(value: Any) -> Any:
    """Serialize enum-like values using `.value` when available."""
    scalar = getattr(value, "value", None)

    if isinstance(scalar, (str, int, float, bool)) or scalar is None:
        if scalar is not None:
            return scalar

    return value


def serialize_enum_like(value: Any) -> Any:
    """Serialize enum-like values using `.value`/`.name` best effort."""
    if hasattr(value, "value"):
        return value.value

    if hasattr(value, "name"):
        return value.name

    return value


def to_json_safe(
    value: Any,
    scalar_serializer: Optional[ScalarSerializer] = None,
) -> Any:
    """Recursively convert values into JSON-serializable structures."""
    serializer = scalar_serializer or serialize_enum_like

    if isinstance(value, dict):
        return {
            str(key): to_json_safe(item, scalar_serializer=serializer)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_json_safe(item, scalar_serializer=serializer) for item in value]

    if isinstance(value, set):
        return [
            to_json_safe(item, scalar_serializer=serializer)
            for item in sorted(value, key=str)
        ]

    serialized = serializer(value)

    if isinstance(serialized, (str, int, float, bool)) or serialized is None:
        return serialized

    if isinstance(serialized, (dict, list, tuple, set)):
        return to_json_safe(serialized, scalar_serializer=serializer)

    return str(serialized)


def as_non_negative_int(value: Any, fallback: int = 0) -> int:
    """Parse a non-negative integer with fallback on invalid input."""
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else fallback

    except (TypeError, ValueError):
        return fallback
