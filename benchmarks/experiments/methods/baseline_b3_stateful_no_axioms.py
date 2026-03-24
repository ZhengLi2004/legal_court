"""Stateful no-axioms baseline runner for Step 09."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.experiments.methods.base import execute_method


def run_baseline_b3_stateful_no_axioms(
    *,
    case: Mapping[str, Any],
    storage_root_dir: str | None = None,
    budget: Mapping[str, Any] | None = None,
    seed: int | None = None,
    retrieval_config: Mapping[str, Any] | None = None,
    verbose: bool = False,
    test_mode_no_learning: bool = False,
) -> dict[str, Any]:
    """Run the stateful baseline with validate-step gating disabled."""
    return execute_method(
        method_name="baseline_b3_stateful_no_axioms",
        case=case,
        storage_root_dir=storage_root_dir,
        budget=budget,
        seed=seed,
        retrieval_config=retrieval_config,
        verbose=verbose,
        test_mode_no_learning=test_mode_no_learning,
        skip_validate_step=True,
    )
