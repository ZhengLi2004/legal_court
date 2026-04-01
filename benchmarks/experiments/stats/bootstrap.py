"""Bootstrap helpers for case-level experiment metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BootstrapConfig:
    """Configuration for case-level bootstrap resampling.

    Attributes:
        n_resamples: Number of bootstrap samples to draw.
        seed: Random seed used by the bootstrap RNG.
        confidence_level: Central confidence interval mass to report.
    """

    n_resamples: int = 2000
    seed: int = 20260307
    confidence_level: float = 0.95


@dataclass(frozen=True)
class BootstrapSummary:
    """Point estimate and confidence interval for one case-level metric.

    Attributes:
        point_estimate: Mean computed from the observed case scores.
        ci_low: Lower bootstrap confidence interval bound.
        ci_high: Upper bootstrap confidence interval bound.
        n_cases: Number of aligned case scores included in the summary.
        n_resamples: Number of bootstrap resamples used.
        seed: Random seed used during resampling.
    """

    point_estimate: float
    ci_low: float
    ci_high: float
    n_cases: int
    n_resamples: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary dataclass into a plain dictionary."""
        return asdict(self)


CaseScoreInput = Mapping[str, float] | Sequence[float]


def _coerce_case_scores(
    case_scores: CaseScoreInput,
) -> tuple[tuple[str, ...], np.ndarray]:
    if isinstance(case_scores, Mapping):
        ordered_items = sorted(
            ((str(uid), float(value)) for uid, value in case_scores.items()),
            key=lambda item: item[0],
        )

        ordered = tuple(uid for uid, _ in ordered_items)
        values = np.asarray([value for _, value in ordered_items], dtype=np.float64)

    else:
        ordered = tuple(str(idx) for idx, _ in enumerate(case_scores))
        values = np.asarray([float(value) for value in case_scores], dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("case_scores must be a 1D mapping or sequence")

    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("case_scores contains non-finite values")

    return ordered, values


def _summary_from_values(
    values: np.ndarray, *, config: BootstrapConfig
) -> BootstrapSummary:
    point = float(values.mean()) if values.size else 0.0

    if values.size == 0:
        return BootstrapSummary(point, 0.0, 0.0, 0, config.n_resamples, config.seed)

    rng = np.random.default_rng(config.seed)

    sample_indices = rng.integers(
        0, values.size, size=(config.n_resamples, values.size)
    )

    sampled = values[sample_indices].mean(axis=1)
    alpha = 1.0 - config.confidence_level
    ci_low, ci_high = np.quantile(sampled, [alpha / 2.0, 1.0 - alpha / 2.0]).tolist()

    return BootstrapSummary(
        point,
        float(ci_low),
        float(ci_high),
        int(values.size),
        config.n_resamples,
        config.seed,
    )


def bootstrap_case_mean(
    case_scores: CaseScoreInput,
    *,
    config: BootstrapConfig | None = None,
) -> BootstrapSummary:
    """Bootstrap the mean of one case-level metric.

    Args:
        case_scores: Case-score mapping or aligned sequence of scores.
        config: Optional bootstrap configuration override.

    Returns:
        Point estimate and confidence interval for the mean case score.
    """

    resolved = config or BootstrapConfig()
    _, values = _coerce_case_scores(case_scores)
    return _summary_from_values(values, config=resolved)


def paired_case_bootstrap(
    case_scores_a: CaseScoreInput,
    case_scores_b: CaseScoreInput,
    *,
    config: BootstrapConfig | None = None,
) -> BootstrapSummary:
    """Bootstrap the paired difference between two aligned case-level metrics.

    Args:
        case_scores_a: First aligned case-score mapping or sequence.
        case_scores_b: Second aligned case-score mapping or sequence.
        config: Optional bootstrap configuration override.

    Returns:
        Bootstrap summary for the paired case-wise difference ``a - b``.

    Raises:
        ValueError: If the case ids are not identical and aligned.
    """

    resolved = config or BootstrapConfig()
    ids_a, values_a = _coerce_case_scores(case_scores_a)
    ids_b, values_b = _coerce_case_scores(case_scores_b)

    if ids_a != ids_b:
        raise ValueError("paired_case_bootstrap requires identical aligned case ids")

    return _summary_from_values(values_a - values_b, config=resolved)
