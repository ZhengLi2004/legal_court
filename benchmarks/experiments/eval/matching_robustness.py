"""Robustness gating and artifact writing for Step 07 matching protocol."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from benchmarks.experiments.eval.matching import (
    CaseMatchBundle,
    MatchingConfig,
    TextEncoder,
    build_method_case_matches,
    require_claim_text,
)

FROZEN_MATCHING_PROTOCOL_VERSION = "step07_v2"
FROZEN_DEV_IDS_PATH = Path("benchmarks/experiments/artifacts/splits/dev_ids.json")
FROZEN_BASE_CONFIG = MatchingConfig()
FROZEN_GATE_RHO_THRESHOLD = 0.8
FROZEN_GATE_FLIP_THRESHOLD = 0.1


@dataclass(frozen=True)
class PrimaryMetricResult:
    case_scores: dict[str, float]
    mean_dev: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchingScenario:
    name: str
    description: str = ""
    config_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodScenarioResult:
    method_name: str
    scenario_name: str
    config: MatchingConfig
    case_bundles: tuple[CaseMatchBundle, ...]
    metric: PrimaryMetricResult


@dataclass(frozen=True)
class ScenarioRobustnessResult:
    scenario_name: str
    passed: bool
    spearman_rho: float
    pairwise_flip_rate: float
    ranking: tuple[str, ...]
    flipped_pairs: tuple[tuple[str, str], ...]
    top_score_deltas: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class MatchingProtocolRun:
    protocol_version: str
    base_config: MatchingConfig
    scenarios: tuple[MatchingScenario, ...]
    protocol_metadata: dict[str, Any]
    method_names: tuple[str, ...]
    case_uids: tuple[str, ...]
    method_prediction_counts: dict[str, int]
    base_scenario_name: str
    scenario_results: dict[str, dict[str, MethodScenarioResult]]
    robustness_results: tuple[ScenarioRobustnessResult, ...]
    gate: dict[str, Any]


class MatchingRobustnessGateError(RuntimeError):
    """Raised when the frozen matching gate fails."""


PrimaryMetricFn = Callable[
    [Sequence[CaseMatchBundle]], PrimaryMetricResult | Mapping[str, Any]
]


def default_matching_scenarios(
    base_config: MatchingConfig | None = None,
) -> list[MatchingScenario]:
    resolved = base_config or MatchingConfig()

    return [
        MatchingScenario(
            name="dedup_minus_0p03",
            description="dedup_similarity_threshold - 0.03",
            config_overrides={
                "dedup_similarity_threshold": max(
                    0.0, resolved.dedup_similarity_threshold - 0.03
                )
            },
        ),
        MatchingScenario(
            name="dedup_plus_0p03",
            description="dedup_similarity_threshold + 0.03",
            config_overrides={
                "dedup_similarity_threshold": min(
                    1.0, resolved.dedup_similarity_threshold + 0.03
                )
            },
        ),
        MatchingScenario(
            name="match_minus_0p05",
            description="match_similarity_threshold - 0.05",
            config_overrides={
                "match_similarity_threshold": max(
                    0.0, resolved.match_similarity_threshold - 0.05
                )
            },
        ),
        MatchingScenario(
            name="match_plus_0p05",
            description="match_similarity_threshold + 0.05",
            config_overrides={
                "match_similarity_threshold": min(
                    1.0, resolved.match_similarity_threshold + 0.05
                )
            },
        ),
        MatchingScenario(
            name="semantic_only",
            description="remove length penalty from the cost function",
            config_overrides={"semantic_weight": 1.0, "length_penalty_weight": 0.0},
        ),
        MatchingScenario(
            name="raw_text_feature",
            description="switch from normalized claim text to raw claim text",
            config_overrides={"text_field": "claim_text_raw"},
        ),
    ]


def _coerce_metric_result(
    value: PrimaryMetricResult | Mapping[str, Any],
) -> PrimaryMetricResult:
    if isinstance(value, PrimaryMetricResult):
        return value

    if isinstance(value, Mapping):
        return PrimaryMetricResult(
            case_scores={
                str(key): float(score)
                for key, score in value.get("case_scores", {}).items()
            },
            mean_dev=(
                None if value.get("mean_dev") is None else float(value.get("mean_dev"))
            ),
            details=dict(value.get("details", {}) or {}),
        )

    raise TypeError("primary_metric_fn must return PrimaryMetricResult or mapping")


def _finalize_metric_result(
    metric: PrimaryMetricResult,
    case_uids: Sequence[str],
) -> PrimaryMetricResult:
    expected_case_uids = tuple(str(uid) for uid in case_uids)
    actual_case_uids = tuple(sorted(metric.case_scores.keys()))
    expected_case_uid_set = set(expected_case_uids)
    missing = sorted(expected_case_uid_set - set(actual_case_uids))
    extras = sorted(set(actual_case_uids) - expected_case_uid_set)

    if missing or extras:
        raise ValueError(
            "primary_metric_fn case_scores must exactly cover the provided case_uids"
        )

    ordered_scores: dict[str, float] = {}

    for uid in expected_case_uids:
        score = float(metric.case_scores[uid])

        if not np.isfinite(score):
            raise ValueError(f"Non-finite case score for uid={uid}")

        ordered_scores[uid] = score

    recomputed_mean = (
        sum(ordered_scores.values()) / len(ordered_scores) if ordered_scores else 0.0
    )

    if metric.mean_dev is not None and not np.isclose(metric.mean_dev, recomputed_mean):
        raise ValueError("primary_metric_fn mean_dev does not match mean(case_scores)")

    return PrimaryMetricResult(
        case_scores=ordered_scores,
        mean_dev=recomputed_mean,
        details=dict(metric.details),
    )


def _ranking_from_metrics(
    results: Mapping[str, MethodScenarioResult],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            results.keys(),
            key=lambda method_name: (
                -float(results[method_name].metric.mean_dev),
                method_name,
            ),
        )
    )


def _spearman_rho(base_ranking: Sequence[str], other_ranking: Sequence[str]) -> float:
    if len(base_ranking) != len(other_ranking):
        raise ValueError("rankings must have the same length")

    count = len(base_ranking)

    if count < 2:
        return 1.0

    other_pos = {method_name: idx for idx, method_name in enumerate(other_ranking)}
    diff_sum = 0

    for idx, method_name in enumerate(base_ranking):
        diff = idx - other_pos[method_name]
        diff_sum += diff * diff

    return 1.0 - (6.0 * diff_sum) / (count * (count * count - 1))


def _pairwise_flips(
    base_ranking: Sequence[str],
    other_ranking: Sequence[str],
) -> tuple[float, tuple[tuple[str, str], ...]]:
    base_pos = {method_name: idx for idx, method_name in enumerate(base_ranking)}
    other_pos = {method_name: idx for idx, method_name in enumerate(other_ranking)}
    method_names = list(base_ranking)
    flipped_pairs: list[tuple[str, str]] = []
    total_pairs = 0

    for left_index, left in enumerate(method_names):
        for right in method_names[left_index + 1 :]:
            total_pairs += 1
            base_order = base_pos[left] < base_pos[right]
            other_order = other_pos[left] < other_pos[right]

            if base_order != other_order:
                flipped_pairs.append((left, right))

    flip_rate = float(len(flipped_pairs) / total_pairs) if total_pairs else 0.0
    return flip_rate, tuple(flipped_pairs)


def _score_deltas(
    base_results: Mapping[str, MethodScenarioResult],
    scenario_results: Mapping[str, MethodScenarioResult],
) -> tuple[tuple[str, float], ...]:
    deltas = []

    for method_name in sorted(base_results):
        delta = float(scenario_results[method_name].metric.mean_dev) - float(
            base_results[method_name].metric.mean_dev
        )

        deltas.append((method_name, delta))

    deltas.sort(key=lambda item: (-abs(item[1]), item[0]))
    return tuple(deltas)


def _scenario_config(
    base_config: MatchingConfig, scenario: MatchingScenario
) -> MatchingConfig:
    return replace(base_config, **scenario.config_overrides)


def load_dev_case_uids(
    dev_ids_path: str | Path = FROZEN_DEV_IDS_PATH,
) -> tuple[str, ...]:
    payload = json.loads(Path(dev_ids_path).read_text(encoding="utf-8"))
    raw_ids = payload.get("ids")

    if not isinstance(raw_ids, list):
        raise ValueError("dev_ids.json must contain an `ids` list")

    ordered: list[str] = []
    seen: set[str] = set()

    for raw_uid in raw_ids:
        uid = str(raw_uid)

        if uid not in seen:
            ordered.append(uid)
            seen.add(uid)

    return tuple(ordered)


def run_matching_protocol(
    *,
    case_uids: Sequence[str],
    gold_rows: Sequence[Mapping[str, Any]],
    method_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    primary_metric_fn: PrimaryMetricFn,
    encoder: TextEncoder,
    base_config: MatchingConfig | None = None,
    scenarios: Sequence[MatchingScenario] | None = None,
    gate_rho_threshold: float = 0.8,
    gate_flip_threshold: float = 0.1,
    artifact_dir: str | Path | None = None,
    protocol_version: str = "step07_generic",
    protocol_metadata: Mapping[str, Any] | None = None,
) -> MatchingProtocolRun:
    normalized_case_uids = tuple(str(uid) for uid in case_uids)
    resolved_base = base_config or MatchingConfig()

    scenario_list = (
        list(scenarios)
        if scenarios is not None
        else default_matching_scenarios(resolved_base)
    )

    method_names = tuple(sorted(method_predictions.keys()))

    if not method_names:
        raise ValueError("method_predictions must contain at least one method")

    scenario_results: dict[str, dict[str, MethodScenarioResult]] = {}
    all_scenarios: list[tuple[str, MatchingConfig]] = [("base", resolved_base)]

    all_scenarios.extend(
        (scenario.name, _scenario_config(resolved_base, scenario))
        for scenario in scenario_list
    )

    for scenario_name, config in all_scenarios:
        per_method: dict[str, MethodScenarioResult] = {}

        for method_name in method_names:
            case_bundles = tuple(
                build_method_case_matches(
                    case_uids=normalized_case_uids,
                    gold_rows=gold_rows,
                    prediction_rows=method_predictions[method_name],
                    method_name=method_name,
                    encoder=encoder,
                    config=config,
                    scenario_name=scenario_name,
                )
            )

            metric = _finalize_metric_result(
                _coerce_metric_result(primary_metric_fn(case_bundles)),
                normalized_case_uids,
            )

            per_method[method_name] = MethodScenarioResult(
                method_name=method_name,
                scenario_name=scenario_name,
                config=config,
                case_bundles=case_bundles,
                metric=metric,
            )

        scenario_results[scenario_name] = per_method

    base_results = scenario_results["base"]
    base_ranking = _ranking_from_metrics(base_results)
    robustness_results: list[ScenarioRobustnessResult] = []

    for scenario in scenario_list:
        results = scenario_results[scenario.name]
        ranking = _ranking_from_metrics(results)
        rho = _spearman_rho(base_ranking, ranking)
        flip_rate, flipped_pairs = _pairwise_flips(base_ranking, ranking)
        passed = rho >= gate_rho_threshold and flip_rate <= gate_flip_threshold

        robustness_results.append(
            ScenarioRobustnessResult(
                scenario_name=scenario.name,
                passed=passed,
                spearman_rho=rho,
                pairwise_flip_rate=flip_rate,
                ranking=ranking,
                flipped_pairs=flipped_pairs,
                top_score_deltas=_score_deltas(base_results, results),
            )
        )

    failed_scenarios = [
        result.scenario_name for result in robustness_results if not result.passed
    ]

    pass_rate = (
        sum(result.passed for result in robustness_results) / len(robustness_results)
        if robustness_results
        else 1.0
    )

    worst_rho = min((result.spearman_rho for result in robustness_results), default=1.0)

    worst_flip_rate = max(
        (result.pairwise_flip_rate for result in robustness_results), default=0.0
    )

    major_flip_sources = [
        {
            "scenario_name": result.scenario_name,
            "flipped_pairs": list(result.flipped_pairs),
            "top_score_deltas": list(result.top_score_deltas[:3]),
        }
        for result in sorted(
            robustness_results,
            key=lambda item: (-item.pairwise_flip_rate, item.scenario_name),
        )
        if not result.passed
    ]

    gate = {
        "passed": not failed_scenarios,
        "gate_rho_threshold": gate_rho_threshold,
        "gate_flip_threshold": gate_flip_threshold,
        "base_ranking": list(base_ranking),
        "pass_rate": float(pass_rate),
        "worst_rho": float(worst_rho),
        "worst_flip_rate": float(worst_flip_rate),
        "failed_scenarios": failed_scenarios,
        "major_flip_sources": major_flip_sources,
        "scenario_summaries": [
            {
                "scenario_name": result.scenario_name,
                "passed": result.passed,
                "spearman_rho": result.spearman_rho,
                "pairwise_flip_rate": result.pairwise_flip_rate,
                "ranking": list(result.ranking),
            }
            for result in robustness_results
        ],
    }

    run = MatchingProtocolRun(
        protocol_version=protocol_version,
        base_config=resolved_base,
        scenarios=tuple(scenario_list),
        protocol_metadata=dict(protocol_metadata or {}),
        method_names=method_names,
        case_uids=normalized_case_uids,
        method_prediction_counts={
            method_name: len(method_predictions[method_name])
            for method_name in method_names
        },
        base_scenario_name="base",
        scenario_results=scenario_results,
        robustness_results=tuple(robustness_results),
        gate=gate,
    )

    if artifact_dir is not None:
        write_matching_artifacts(run, artifact_dir)

    return run


def run_frozen_matching_protocol(
    *,
    gold_rows: Sequence[Mapping[str, Any]],
    method_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    primary_metric_fn: PrimaryMetricFn,
    encoder: TextEncoder,
    artifact_dir: str | Path | None = None,
    dev_ids_path: str | Path = FROZEN_DEV_IDS_PATH,
    enforce_gate: bool = True,
) -> MatchingProtocolRun:
    case_uids = load_dev_case_uids(dev_ids_path)

    metadata = {
        "dev_ids_path": str(Path(dev_ids_path).resolve()),
        "seeded_claim_matching_mode": "claim_id_direct_when_gold_seeded",
        "gold_claim_id_prefix": "GOLD_",
        "claim1_task_focus": "status_eval_on_gold_claims",
    }

    run = run_matching_protocol(
        case_uids=case_uids,
        gold_rows=gold_rows,
        method_predictions=method_predictions,
        primary_metric_fn=primary_metric_fn,
        encoder=encoder,
        base_config=FROZEN_BASE_CONFIG,
        scenarios=default_matching_scenarios(FROZEN_BASE_CONFIG),
        gate_rho_threshold=FROZEN_GATE_RHO_THRESHOLD,
        gate_flip_threshold=FROZEN_GATE_FLIP_THRESHOLD,
        artifact_dir=artifact_dir,
        protocol_version=FROZEN_MATCHING_PROTOCOL_VERSION,
        protocol_metadata=metadata,
    )

    if enforce_gate:
        assert_matching_gate(run)

    return run


def assert_matching_gate(run: MatchingProtocolRun) -> None:
    if bool(run.gate.get("passed")):
        return

    failed = ", ".join(run.gate.get("failed_scenarios") or [])
    worst_rho = float(run.gate.get("worst_rho", 0.0))
    worst_flip = float(run.gate.get("worst_flip_rate", 1.0))

    raise MatchingRobustnessGateError(
        f"Matching robustness gate failed: scenarios={failed}; worst_rho={worst_rho:.4f}; worst_flip_rate={worst_flip:.4f}"
    )


def _serialize_bundle(bundle: CaseMatchBundle) -> dict[str, Any]:
    return {
        "uid": bundle.uid,
        "method_name": bundle.method_name,
        "scenario_name": bundle.scenario_name,
        "gold_count": len(bundle.gold_rows),
        "prediction_raw_count": len(bundle.prediction_rows_raw),
        "prediction_deduped_count": len(bundle.prediction_rows_deduped),
        "matched_pair_count": len(bundle.matched_pairs),
        "unmatched_gold_count": len(bundle.unmatched_gold_rows),
        "unmatched_prediction_count": len(bundle.unmatched_prediction_rows),
        "dedup_clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "representative_claim_id": cluster.representative_claim_id,
                "member_claim_ids": list(cluster.member_claim_ids),
                "centroid_similarity": cluster.centroid_similarity,
            }
            for cluster in bundle.dedup_clusters
        ],
        "matched_pairs": [
            {
                "gold_claim_id": str(pair.gold_row.get("claim_id", "") or ""),
                "prediction_claim_id": str(
                    pair.prediction_row.get("claim_id", "") or ""
                ),
                "matching_mode": pair.matching_mode,
                "direct_claim_id": pair.direct_claim_id,
                "semantic_similarity": pair.semantic_similarity,
                "length_gap": pair.length_gap,
                "cost": pair.cost,
            }
            for pair in bundle.matched_pairs
        ],
        "unmatched_gold_claim_ids": [
            str(row.get("claim_id", "") or "") for row in bundle.unmatched_gold_rows
        ],
        "unmatched_prediction_claim_ids": [
            str(row.get("claim_id", "") or "")
            for row in bundle.unmatched_prediction_rows
        ],
        "matching_mode": bundle.matching_mode,
    }


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _length_quantiles(run: MatchingProtocolRun) -> tuple[float, float, float]:
    base_results = run.scenario_results[run.base_scenario_name]
    sample_method = run.method_names[0]
    values: list[float] = []

    for bundle in base_results[sample_method].case_bundles:
        for row in bundle.gold_rows:
            text = require_claim_text(row, run.base_config)
            values.append(float(len("".join(text.split()))))

    if not values:
        return 0.0, 0.0, 0.0

    arr = np.asarray(values, dtype=np.float32)
    q1, q2, q3 = np.quantile(arr, [0.25, 0.5, 0.75]).tolist()
    return float(q1), float(q2), float(q3)


def _bucket_length(length: float, quantiles: tuple[float, float, float]) -> str:
    q1, q2, q3 = quantiles

    if length <= q1:
        return "q1"

    if length <= q2:
        return "q2"

    if length <= q3:
        return "q3"

    return "q4"


def write_matching_artifacts(
    run: MatchingProtocolRun, artifact_dir: str | Path
) -> None:
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario_name in sorted(run.scenario_results):
        per_method = run.scenario_results[scenario_name]

        for method_name, result in sorted(per_method.items()):
            path = (
                output_dir
                / f"match_pairs_{_safe_name(scenario_name)}_{_safe_name(method_name)}.jsonl"
            )

            with path.open("w", encoding="utf-8") as handle:
                for bundle in result.case_bundles:
                    handle.write(
                        json.dumps(_serialize_bundle(bundle), ensure_ascii=False) + "\n"
                    )

    gate_payload = dict(run.gate)

    gate_payload.update(
        {
            "protocol_version": run.protocol_version,
            "base_config": asdict(run.base_config),
            "scenarios": [asdict(scenario) for scenario in run.scenarios],
            "protocol_metadata": dict(run.protocol_metadata),
            "method_names": list(run.method_names),
            "case_count": len(run.case_uids),
            "method_prediction_counts": dict(run.method_prediction_counts),
        }
    )

    (output_dir / "robustness_gate.json").write_text(
        json.dumps(gate_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    quantiles = _length_quantiles(run)
    aggregate: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for scenario_name, per_method in sorted(run.scenario_results.items()):
        for method_name, result in sorted(per_method.items()):
            matched_gold_by_key = {
                (bundle.uid, str(pair.gold_row.get("claim_id", "") or "")): pair
                for bundle in result.case_bundles
                for pair in bundle.matched_pairs
            }

            matched_prediction_by_key = {
                (bundle.uid, str(pair.prediction_row.get("claim_id", "") or "")): pair
                for bundle in result.case_bundles
                for pair in bundle.matched_pairs
            }

            for bundle in result.case_bundles:
                for row in bundle.gold_rows:
                    text = require_claim_text(row, result.config)

                    bucket = _bucket_length(
                        float(len("".join(text.split()))), quantiles
                    )

                    key = (scenario_name, method_name, "gold", bucket)

                    item = aggregate.setdefault(
                        key,
                        {
                            "scenario_name": scenario_name,
                            "method_name": method_name,
                            "bucket_side": "gold",
                            "length_bucket": bucket,
                            "gold_claim_count": 0,
                            "matched_gold_count": 0,
                            "unmatched_gold_count": 0,
                            "prediction_claim_count": 0,
                            "matched_prediction_count": 0,
                            "unmatched_prediction_count": 0,
                            "semantic_scores": [],
                        },
                    )

                    item["gold_claim_count"] += 1

                    pair = matched_gold_by_key.get(
                        (bundle.uid, str(row.get("claim_id", "") or ""))
                    )

                    if pair is None:
                        item["unmatched_gold_count"] += 1

                    else:
                        item["matched_gold_count"] += 1
                        item["semantic_scores"].append(float(pair.semantic_similarity))

                for row in bundle.prediction_rows_deduped:
                    text = require_claim_text(row, result.config)

                    bucket = _bucket_length(
                        float(len("".join(text.split()))), quantiles
                    )

                    key = (scenario_name, method_name, "prediction", bucket)

                    item = aggregate.setdefault(
                        key,
                        {
                            "scenario_name": scenario_name,
                            "method_name": method_name,
                            "bucket_side": "prediction",
                            "length_bucket": bucket,
                            "gold_claim_count": 0,
                            "matched_gold_count": 0,
                            "unmatched_gold_count": 0,
                            "prediction_claim_count": 0,
                            "matched_prediction_count": 0,
                            "unmatched_prediction_count": 0,
                            "semantic_scores": [],
                        },
                    )

                    item["prediction_claim_count"] += 1

                    pair = matched_prediction_by_key.get(
                        (bundle.uid, str(row.get("claim_id", "") or ""))
                    )

                    if pair is None:
                        item["unmatched_prediction_count"] += 1

                    else:
                        item["matched_prediction_count"] += 1
                        item["semantic_scores"].append(float(pair.semantic_similarity))

    csv_path = output_dir / "error_buckets_by_length.csv"

    fieldnames = [
        "scenario_name",
        "method_name",
        "bucket_side",
        "length_bucket",
        "gold_claim_count",
        "matched_gold_count",
        "unmatched_gold_count",
        "prediction_claim_count",
        "matched_prediction_count",
        "unmatched_prediction_count",
        "matched_rate",
        "mean_semantic_similarity",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for _, item in sorted(aggregate.items()):
            scores = item.pop("semantic_scores")

            if item["bucket_side"] == "gold":
                denom = item["gold_claim_count"]
                matched = item["matched_gold_count"]

            else:
                denom = item["prediction_claim_count"]
                matched = item["matched_prediction_count"]

            item["matched_rate"] = float(matched / denom) if denom else 0.0
            item["mean_semantic_similarity"] = float(np.mean(scores)) if scores else 0.0
            writer.writerow(item)


__all__ = [
    "MatchingProtocolRun",
    "MatchingRobustnessGateError",
    "MatchingScenario",
    "PrimaryMetricResult",
    "ScenarioRobustnessResult",
    "FROZEN_MATCHING_PROTOCOL_VERSION",
    "assert_matching_gate",
    "default_matching_scenarios",
    "load_dev_case_uids",
    "run_frozen_matching_protocol",
    "run_matching_protocol",
    "write_matching_artifacts",
]
