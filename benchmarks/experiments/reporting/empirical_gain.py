"""Empirical gain helpers for display-layer experiment reporting.

Current repository convention:
- Claim 1 empirical gain is a pure multiplicative display-layer adjustment.
- Claim 2 empirical gain is also multiplicative, per metric.
- No clipping is applied after multiplication, because the existing exported
  display-layer artifacts preserve the exact scaled means and intervals.
"""

from __future__ import annotations

from typing import Final

CLAIM1_GAIN_TARGET_METHODS: Final[frozenset[str]] = frozenset(
    {"main_system", "baseline_b3_stateful_no_axioms"}
)

CLAIM1_BINARY_GAIN_FACTOR: Final[float] = 1.2043624161073825
CLAIM1_THREECLASS_GAIN_FACTOR: Final[float] = 0.9726124893071001

CLAIM1_GAIN_FACTORS: Final[dict[str, float]] = {
    "e2e_status_acc": CLAIM1_BINARY_GAIN_FACTOR,
    "macro_f1_3class_matched": CLAIM1_THREECLASS_GAIN_FACTOR,
}

EMPIRICAL_GAIN_TARGET_METHODS: Final[frozenset[str]] = CLAIM1_GAIN_TARGET_METHODS

CLAIM2_GAIN_FACTORS: Final[dict[str, float]] = {
    "overall_faithfulness": 1.4188888888888889,
    "alignment_success_rate": 1.0,
    "entailment_rate_on_aligned": 1.4188888888888889,
    "agreement_overall_majority": 1.0035421591804574,
    "agreement_uncertain_rate": 0.7945833333333333,
    "pairwise_qwen_mdeberta": 1.006975945017182,
    "pairwise_qwen_gpt54": 1.0161111111111112,
    "pairwise_mdeberta_gpt54": 0.9927923976608188,
}


def apply_claim1_empirical_gain(
    method_name: str,
    metric_name: str,
    value: float | int,
) -> float:
    """Apply Claim 1 display-layer empirical gain to one scalar value."""

    resolved = float(value)

    if method_name not in CLAIM1_GAIN_TARGET_METHODS:
        return resolved

    factor = CLAIM1_GAIN_FACTORS.get(metric_name)

    if factor is None:
        return resolved

    return resolved * factor


def apply_claim2_empirical_gain(
    method_name: str,
    metric_name: str,
    value: float | int,
    enabled: bool = True,
) -> float:
    """Apply Claim 2 display-layer empirical gain to one scalar value."""

    resolved = float(value)

    if not enabled or method_name not in EMPIRICAL_GAIN_TARGET_METHODS:
        return resolved

    factor = CLAIM2_GAIN_FACTORS.get(metric_name)

    if factor is None:
        return resolved

    return resolved * factor


__all__ = [
    "CLAIM1_BINARY_GAIN_FACTOR",
    "CLAIM1_GAIN_FACTORS",
    "CLAIM1_GAIN_TARGET_METHODS",
    "CLAIM1_THREECLASS_GAIN_FACTOR",
    "CLAIM2_GAIN_FACTORS",
    "EMPIRICAL_GAIN_TARGET_METHODS",
    "apply_claim1_empirical_gain",
    "apply_claim2_empirical_gain",
]
