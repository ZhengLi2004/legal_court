"""Evaluation helpers for benchmark experiments."""

from benchmarks.experiments.eval.matching import (
    CaseMatchBundle,
    DedupCluster,
    MatchingConfig,
    MatchPair,
    SentenceTransformerTextEncoder,
    TextEncoder,
    build_method_case_matches,
    claim_text,
    deduplicate_prediction_rows,
    match_case_claims,
    require_claim_text,
)
from benchmarks.experiments.eval.matching_robustness import (
    FROZEN_MATCHING_PROTOCOL_VERSION,
    MatchingProtocolRun,
    MatchingRobustnessGateError,
    MatchingScenario,
    PrimaryMetricResult,
    ScenarioRobustnessResult,
    assert_matching_gate,
    default_matching_scenarios,
    load_dev_case_uids,
    run_frozen_matching_protocol,
    run_matching_protocol,
    write_matching_artifacts,
)
from benchmarks.experiments.eval.metrics_claim1 import evaluate_claim1_metrics
from benchmarks.experiments.eval.metrics_claim2 import evaluate_claim2_metrics
from benchmarks.experiments.eval.metrics_claim3 import summarize_claim3_curve
from benchmarks.experiments.eval.metrics_claim4 import summarize_claim4_budget_grid
from benchmarks.experiments.stats.bootstrap import (
    BootstrapConfig,
    BootstrapSummary,
    bootstrap_case_mean,
    paired_case_bootstrap,
)
from benchmarks.experiments.stats.hypothesis import (
    HolmBonferroniResult,
    McNemarResult,
    StuartMaxwellResult,
    holm_bonferroni,
    mcnemar_test,
    stuart_maxwell_test,
)

__all__ = [
    "CaseMatchBundle",
    "DedupCluster",
    "FROZEN_MATCHING_PROTOCOL_VERSION",
    "MatchPair",
    "MatchingConfig",
    "MatchingProtocolRun",
    "MatchingRobustnessGateError",
    "MatchingScenario",
    "McNemarResult",
    "PrimaryMetricResult",
    "ScenarioRobustnessResult",
    "SentenceTransformerTextEncoder",
    "StuartMaxwellResult",
    "TextEncoder",
    "BootstrapConfig",
    "BootstrapSummary",
    "HolmBonferroniResult",
    "assert_matching_gate",
    "bootstrap_case_mean",
    "build_method_case_matches",
    "claim_text",
    "deduplicate_prediction_rows",
    "default_matching_scenarios",
    "evaluate_claim1_metrics",
    "evaluate_claim2_metrics",
    "holm_bonferroni",
    "load_dev_case_uids",
    "match_case_claims",
    "mcnemar_test",
    "paired_case_bootstrap",
    "require_claim_text",
    "run_frozen_matching_protocol",
    "run_matching_protocol",
    "stuart_maxwell_test",
    "summarize_claim3_curve",
    "summarize_claim4_budget_grid",
    "write_matching_artifacts",
]
