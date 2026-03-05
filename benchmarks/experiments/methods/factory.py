"""Factory for constructing experiment method registry skeletons."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

MethodRunner = Callable[..., dict[str, Any]]


def _build_placeholder_runner(method_name: str) -> MethodRunner:
    def _runner(*, case: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        case_uid = None

        if case is not None:
            case_uid = case.get("uid")

        return {
            "method": method_name,
            "status": "placeholder",
            "case_uid": case_uid,
            "output": {
                "claims": [],
                "status": {},
                "trace": [],
                "token_usage": {},
                "latency": None,
            },
        }

    return _runner


def build_default_registry() -> dict[str, MethodRunner]:
    """Build the default method registry used by experiment orchestrators."""
    return {
        "main_system": _build_placeholder_runner("main_system"),
        "baseline_b1_structured_rag": _build_placeholder_runner(
            "baseline_b1_structured_rag"
        ),
        "baseline_b2_vanilla_mad": _build_placeholder_runner("baseline_b2_vanilla_mad"),
        "baseline_b3_stateful_no_axioms": _build_placeholder_runner(
            "baseline_b3_stateful_no_axioms"
        ),
    }
