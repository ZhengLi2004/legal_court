"""Budget-grid summaries for Claim 4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from benchmarks.experiments.eval._metric_utils import validate_nested_case_score_mapping
from benchmarks.experiments.stats.bootstrap import (
    BootstrapConfig,
    bootstrap_case_mean,
    paired_case_bootstrap,
)


def summarize_claim4_budget_grid(
    *,
    budget_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    prereg_points: Mapping[str, Sequence[str]] | None = None,
    bootstrap_config: BootstrapConfig | None = None,
    family_name: str = "claim4",
    reference_method: str = "main_system",
) -> dict[str, Any]:
    config = bootstrap_config or BootstrapConfig()
    normalized = validate_nested_case_score_mapping(budget_scores, name="budget_scores")
    summaries: dict[str, dict[str, Any]] = {}

    for method_name, points in normalized.items():
        summaries[method_name] = {}

        for point_name, case_scores in points.items():
            summaries[method_name][point_name] = asdict(
                bootstrap_case_mean(case_scores, config=config)
            )

    pairwise_deltas: dict[str, dict[str, Any]] = {}

    if reference_method in normalized:
        ref_points = normalized[reference_method]

        for method_name, points in normalized.items():
            if method_name == reference_method:
                continue

            method_deltas: dict[str, Any] = {}

            for point_name, case_scores in points.items():
                if point_name in ref_points:
                    method_deltas[point_name] = asdict(
                        paired_case_bootstrap(
                            ref_points[point_name], case_scores, config=config
                        )
                    )

            pairwise_deltas[method_name] = method_deltas

    return {
        "family_name": family_name,
        "bootstrap": asdict(config),
        "budget_scores": normalized,
        "budget_summaries": summaries,
        "pairwise_deltas_vs_reference": pairwise_deltas,
        "prereg_points": {
            str(k): [str(v) for v in values]
            for k, values in (prereg_points or {}).items()
        },
        "reference_method": reference_method,
    }
