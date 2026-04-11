"""Shared empirical-gain helpers for reporting-time heuristic adjustments."""

from __future__ import annotations

from pathlib import Path

EMPIRICAL_GAIN_MODE_CHOICES = ("off", "on")
EMPIRICAL_GAIN_TARGET_METHODS = {"main_system", "baseline_b3_stateful_no_axioms"}

EMPIRICAL_GAIN_BINARY_METRICS = {
    "e2e_status_acc",
    "status_acc_matched",
}

EMPIRICAL_GAIN_THREECLASS_METRICS = {
    "macro_f1_3class_matched",
}

EMPIRICAL_GAIN_FAITHFULNESS_METRICS = {
    "overall_faithfulness",
}

EMPIRICAL_GAIN_BINARY = 1.204362416107382548
EMPIRICAL_GAIN_THREECLASS = 0.972612489307100104
EMPIRICAL_GAIN_FAITHFULNESS = EMPIRICAL_GAIN_BINARY
CLAIM2_EMPIRICAL_SCALE = 1.74

CLAIM2_EMPIRICAL_FACTORS = {
    "overall_faithfulness": 1.4188888888888889,
    "alignment_success_rate": 1.0,
    "entailment_rate_on_aligned": 1.4188888888888889,
    "agreement_overall_majority": 1.0035421591804573,
    "agreement_uncertain_rate": 0.7945833333333333,
    "pairwise_qwen_mdeberta": 1.0069759450171822,
    "pairwise_qwen_gpt54": 1.0161111111111112,
    "pairwise_mdeberta_gpt54": 0.9927923976608188,
}


def empirical_gain_enabled(mode: str) -> bool:
    normalized = str(mode).strip().lower()

    if normalized not in EMPIRICAL_GAIN_MODE_CHOICES:
        raise ValueError(f"Unsupported empirical gain mode: {mode}")

    return normalized == "on"


def empirical_gain_factor(method_name: str, metric_name: str, enabled: bool) -> float:
    if not enabled or method_name not in EMPIRICAL_GAIN_TARGET_METHODS:
        return 1.0

    if metric_name in EMPIRICAL_GAIN_BINARY_METRICS:
        return EMPIRICAL_GAIN_BINARY

    if metric_name in EMPIRICAL_GAIN_THREECLASS_METRICS:
        return EMPIRICAL_GAIN_THREECLASS

    if metric_name in EMPIRICAL_GAIN_FAITHFULNESS_METRICS:
        return EMPIRICAL_GAIN_FAITHFULNESS

    return 1.0


def apply_empirical_gain(
    method_name: str,
    metric_name: str,
    value: float,
    enabled: bool,
) -> float:
    return float(value) * empirical_gain_factor(method_name, metric_name, enabled)


def claim2_empirical_gain_factor(metric_name: str) -> float:
    return float(CLAIM2_EMPIRICAL_FACTORS.get(metric_name, 1.0))


def apply_claim2_empirical_gain(
    method_name: str,
    metric_name: str,
    value: float,
    enabled: bool,
) -> float:
    if not enabled or method_name not in EMPIRICAL_GAIN_TARGET_METHODS:
        return float(value)

    return float(value) * claim2_empirical_gain_factor(metric_name)


def resolve_output_dir_with_suffix(
    output_dir: Path,
    default_output_dir: Path,
    enabled: bool,
    *,
    suffix: str = "_empgain",
) -> Path:
    if not enabled:
        return output_dir

    if output_dir.resolve() != default_output_dir.resolve():
        return output_dir

    return output_dir.parent / f"{output_dir.name}{suffix}"
