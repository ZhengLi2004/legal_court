"""Minimal orchestration skeleton for Claim 1-4 experiment runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _build_skeleton_result(
    *,
    claim_id: int,
    cases: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    case_count = len(cases) if cases is not None else 0
    method_names = sorted(registry.keys()) if registry is not None else []

    return {
        "claim_id": claim_id,
        "status": "skeleton",
        "case_count": case_count,
        "methods": method_names,
        "metrics": {},
        "artifacts": [],
        "errors": [],
    }


def run_claim1_experiment(
    *,
    cases: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Claim 1 experiment skeleton and return structured result."""
    return _build_skeleton_result(claim_id=1, cases=cases, registry=registry)


def run_claim2_experiment(
    *,
    cases: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Claim 2 experiment skeleton and return structured result."""
    return _build_skeleton_result(claim_id=2, cases=cases, registry=registry)


def run_claim3_experiment(
    *,
    cases: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Claim 3 experiment skeleton and return structured result."""
    return _build_skeleton_result(claim_id=3, cases=cases, registry=registry)


def run_claim4_experiment(
    *,
    cases: Sequence[Mapping[str, Any]] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Claim 4 experiment skeleton and return structured result."""
    return _build_skeleton_result(claim_id=4, cases=cases, registry=registry)
