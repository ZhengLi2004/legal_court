"""Vanilla MAD proxy baseline runner for Step 09."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.experiments.methods.base import execute_method


def run_baseline_b2_vanilla_mad(
    *,
    case: Mapping[str, Any],
    storage_root_dir: str | None = None,
    budget: Mapping[str, Any] | None = None,
    seed: int | None = None,
    retrieval_config: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run debate baseline without recall worker or initial insights."""
    return execute_method(
        method_name="baseline_b2_vanilla_mad",
        case=case,
        storage_root_dir=storage_root_dir,
        budget=budget,
        seed=seed,
        retrieval_config=retrieval_config,
        verbose=verbose,
        disable_recall_worker=True,
        disable_initial_insights=True,
    )
