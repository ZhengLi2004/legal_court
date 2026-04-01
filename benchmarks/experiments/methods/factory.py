"""Factory for constructing experiment method registries."""

from __future__ import annotations

from typing import Any

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
from benchmarks.experiments.methods.external_naive_mad import run_external_naive_mad
from benchmarks.experiments.methods.external_naive_rag import run_external_naive_rag
from benchmarks.experiments.methods.external_pure_one_shot_judge import (
    run_external_pure_one_shot_judge,
)
from benchmarks.experiments.methods.main_system import run_main_system

INTERNAL_DEFAULT_PROFILE = "internal_default"
EXTERNAL_ONLY_PROFILE = "external_only"
COMBINED_PROFILE = "combined"
CLAIM4_MAD_ONLY_PROFILE = "claim4_mad_only"

ALLOWED_REGISTRY_PROFILES = {
    INTERNAL_DEFAULT_PROFILE,
    EXTERNAL_ONLY_PROFILE,
    COMBINED_PROFILE,
    CLAIM4_MAD_ONLY_PROFILE,
}

_INTERNAL_RUNNERS: dict[str, MethodRunner] = {
    "main_system": run_main_system,
    "baseline_b1_structured_rag": run_baseline_b1_structured_rag,
    "baseline_b2_vanilla_mad": run_baseline_b2_vanilla_mad,
    "baseline_b3_stateful_no_axioms": run_baseline_b3_stateful_no_axioms,
}

_EXTERNAL_RUNNERS: dict[str, MethodRunner] = {
    "external_naive_rag": run_external_naive_rag,
    "external_naive_mad": run_external_naive_mad,
    "external_pure_one_shot_judge": run_external_pure_one_shot_judge,
}

_CLAIM4_MAD_ONLY_RUNNERS: dict[str, MethodRunner] = {
    "main_system": run_main_system,
    "baseline_b2_vanilla_mad": run_baseline_b2_vanilla_mad,
    "baseline_b3_stateful_no_axioms": run_baseline_b3_stateful_no_axioms,
}

_METHOD_SEMANTICS: dict[str, dict[str, Any]] = {
    "main_system": {
        "mode": "full_debate",
        "cohort": "internal",
        "root_claim_source": "gold_package_seeded",
        "adjudication_mode": "direct_status_json",
    },
    "baseline_b1_structured_rag": {
        "mode": "structured_single_pass",
        "cohort": "internal",
        "direct_adjudication": True,
        "creates_debate_engine": False,
        "root_claim_source": "gold_package_seeded",
        "adjudication_mode": "direct_status_json",
    },
    "baseline_b2_vanilla_mad": {
        "mode": "debate_without_recall_or_initial_insights",
        "cohort": "internal",
        "disable_recall_worker": True,
        "disable_initial_insights": True,
        "root_claim_source": "gold_package_seeded",
        "adjudication_mode": "direct_status_json",
    },
    "baseline_b3_stateful_no_axioms": {
        "mode": "debate_without_validate_step",
        "cohort": "internal",
        "skip_validate_step": True,
        "root_claim_source": "gold_package_seeded",
        "adjudication_mode": "direct_status_json",
    },
    "external_naive_rag": {
        "mode": "external_naive_rag_single_shot",
        "cohort": "external",
        "retrieval": "explicit_es_top3",
        "debate": False,
        "uses_legal_system": False,
        "uses_debate_engine": False,
        "uses_long_term_memory": False,
        "uses_recall_worker": False,
        "uses_initial_insights": False,
        "root_claim_source": "gold_package_seeded",
        "adjudication_mode": "direct_status_json",
    },
    "external_naive_mad": {
        "mode": "external_naive_mad_debate",
        "cohort": "external",
        "retrieval": "none",
        "debate": True,
        "uses_legal_system": False,
        "uses_debate_engine": False,
        "uses_long_term_memory": False,
        "uses_recall_worker": False,
        "uses_initial_insights": False,
        "uses_validate_gate": False,
        "root_claim_source": "gold_package_seeded",
        "adjudication_mode": "direct_status_json",
    },
    "external_pure_one_shot_judge": {
        "mode": "external_pure_one_shot_direct_judge",
        "cohort": "external",
        "retrieval": "none",
        "debate": False,
        "uses_legal_system": False,
        "uses_debate_engine": False,
        "uses_long_term_memory": False,
        "uses_recall_worker": False,
        "uses_initial_insights": False,
        "uses_validate_gate": False,
        "root_claim_source": "gold_package_seeded",
        "adjudication_mode": "direct_status_json",
    },
}


def normalize_registry_profile(profile: str | None) -> str:
    """Validate and normalize one registry profile name."""
    resolved = (
        str(profile or INTERNAL_DEFAULT_PROFILE).strip() or INTERNAL_DEFAULT_PROFILE
    )

    if resolved not in ALLOWED_REGISTRY_PROFILES:
        raise ValueError(
            f"Unknown registry profile: {resolved}. Allowed={sorted(ALLOWED_REGISTRY_PROFILES)}"
        )

    return resolved


def build_method_registry(
    profile: str = INTERNAL_DEFAULT_PROFILE,
) -> dict[str, MethodRunner]:
    """Build one method registry for the requested experiment profile."""
    resolved = normalize_registry_profile(profile)

    if resolved == INTERNAL_DEFAULT_PROFILE:
        return dict(_INTERNAL_RUNNERS)

    if resolved == EXTERNAL_ONLY_PROFILE:
        return dict(_EXTERNAL_RUNNERS)

    if resolved == CLAIM4_MAD_ONLY_PROFILE:
        return dict(_CLAIM4_MAD_ONLY_RUNNERS)

    combined = dict(_INTERNAL_RUNNERS)
    combined.update(_EXTERNAL_RUNNERS)
    return combined


def build_default_registry() -> dict[str, MethodRunner]:
    """Build the default internal method registry used by experiment orchestrators."""
    return build_method_registry(INTERNAL_DEFAULT_PROFILE)


def build_method_registry_snapshot(
    profile: str = INTERNAL_DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Build frozen registry semantics for one profile."""
    registry = build_method_registry(profile)
    method_names = sorted(registry.keys())

    return {
        "registry_profile": normalize_registry_profile(profile),
        "method_names": method_names,
        "root_claim_source": "gold_package_seeded",
        "claim1_task_focus": "status_eval_on_gold_claims",
        "adjudication_mode": "direct_status_json",
        "fallbacks_enabled": False,
        "method_semantics": {
            method_name: dict(_METHOD_SEMANTICS[method_name])
            for method_name in method_names
        },
    }


def build_prereg_comparisons(
    profile: str = INTERNAL_DEFAULT_PROFILE,
) -> dict[str, list[str]]:
    """Build preregistered comparison groups for one profile."""
    method_names = set(build_method_registry(profile).keys())

    if "main_system" not in method_names:
        return {}

    return {
        "main_system": [
            method_name
            for method_name in sorted(method_names)
            if method_name != "main_system"
        ]
    }
