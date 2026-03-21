"""Factory for constructing Step 09 method registry."""

from __future__ import annotations

from benchmarks.experiments.methods.base import MethodRunner
from benchmarks.experiments.methods.baseline_b1_structured_rag import (
    run_baseline_b1_structured_rag,
)
from benchmarks.experiments.methods.baseline_b2_vanilla_mad import (
    run_baseline_b2_vanilla_mad,
)
from benchmarks.experiments.methods.baseline_b3_stateful_no_axioms import (
    run_baseline_b3_stateful_no_axioms,
)
from benchmarks.experiments.methods.main_system import run_main_system



def build_default_registry() -> dict[str, MethodRunner]:
    """Build the default method registry used by experiment orchestrators."""
    return {
        "main_system": run_main_system,
        "baseline_b1_structured_rag": run_baseline_b1_structured_rag,
        "baseline_b2_vanilla_mad": run_baseline_b2_vanilla_mad,
        "baseline_b3_stateful_no_axioms": run_baseline_b3_stateful_no_axioms,
    }
