"""Structured RAG proxy baseline runner for Step 09."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.experiments.methods.base import execute_method


def run_baseline_b1_structured_rag(
    *,
    case: Mapping[str, Any],
    budget: Mapping[str, Any] | None = None,
    seed: int | None = None,
    retrieval_config: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one structured single-pass baseline without debate turns."""
    return execute_method(
        method_name="baseline_b1_structured_rag",
        case=case,
        budget=budget,
        seed=seed,
        retrieval_config=retrieval_config,
        verbose=verbose,
        direct_adjudication=True,
    )
