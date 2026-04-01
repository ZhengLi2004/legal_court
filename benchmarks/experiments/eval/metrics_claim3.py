"""Curve-oriented statistical summaries for Claim 3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from benchmarks.experiments.eval._metric_utils import validate_nested_case_score_mapping
from benchmarks.experiments.stats.bootstrap import BootstrapConfig, bootstrap_case_mean


def summarize_claim3_curve(
    *,
    curve_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    bootstrap_config: BootstrapConfig | None = None,
    family_name: str = "claim3",
) -> dict[str, Any]:
    """Bootstrap every point on one or more Claim 3 score curves.

    Args:
        curve_scores: Nested ``series -> point -> case`` score mapping.
        bootstrap_config: Optional bootstrap configuration override.
        family_name: Family label written into the summary payload.

    Returns:
        Claim 3 curve payload with normalized scores and bootstrap summaries.
    """

    config = bootstrap_config or BootstrapConfig()
    normalized = validate_nested_case_score_mapping(curve_scores, name="curve_scores")
    summaries: dict[str, dict[str, Any]] = {}

    for series_name, points in normalized.items():
        summaries[series_name] = {}

        for point_name, case_scores in points.items():
            summaries[series_name][point_name] = asdict(
                bootstrap_case_mean(case_scores, config=config)
            )

    return {
        "family_name": family_name,
        "bootstrap": asdict(config),
        "curve_scores": normalized,
        "curve_summaries": summaries,
    }
