"""Shared statistical wrappers for Claim 2 metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from benchmarks.experiments.eval._metric_utils import validate_case_score_mapping
from benchmarks.experiments.stats.bootstrap import BootstrapConfig, bootstrap_case_mean
from benchmarks.experiments.stats.hypothesis import (
    holm_bonferroni,
    mcnemar_test,
    stuart_maxwell_test,
)


def evaluate_claim2_metrics(
    *,
    per_case_scores: Mapping[str, float],
    bootstrap_config: BootstrapConfig | None = None,
    family_name: str = "claim2",
    p_values: Mapping[str, float] | None = None,
    mcnemar_inputs: Mapping[str, Sequence[str]] | None = None,
    stuart_maxwell_inputs: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Wrap Claim 2 per-case scores with bootstrap and optional tests.

    Args:
        per_case_scores: Per-case Claim 2 scores keyed by uid.
        bootstrap_config: Optional bootstrap configuration override.
        family_name: Family label written into the metric payload.
        p_values: Optional family of precomputed p-values for correction.
        mcnemar_inputs: Optional McNemar input payload.
        stuart_maxwell_inputs: Optional Stuart-Maxwell input payload.

    Returns:
        Claim 2 metric payload with bootstrap summary and optional tests.
    """

    config = bootstrap_config or BootstrapConfig()
    normalized = validate_case_score_mapping(per_case_scores, name="per_case_scores")
    tests: dict[str, Any] = {}

    if p_values is not None:
        tests["holm_bonferroni"] = holm_bonferroni(p_values).to_dict()

    if mcnemar_inputs is not None:
        tests["mcnemar"] = mcnemar_test(
            mcnemar_inputs["y_true"],
            mcnemar_inputs["pred_a"],
            mcnemar_inputs["pred_b"],
        ).to_dict()

    if stuart_maxwell_inputs is not None:
        tests["stuart_maxwell"] = stuart_maxwell_test(
            stuart_maxwell_inputs["labels_a"],
            stuart_maxwell_inputs["labels_b"],
            classes=stuart_maxwell_inputs.get(
                "classes",
                ("VALIDATED", "DEFEATED", "HYPOTHETICAL"),
            ),
        ).to_dict()

    return {
        "family_name": family_name,
        "bootstrap": asdict(config),
        "metric": asdict(bootstrap_case_mean(normalized, config=config)),
        "case_scores": normalized,
        "tests": tests,
    }
