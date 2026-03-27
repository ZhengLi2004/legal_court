"""Claim 2 experiment runner for Step 11."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from benchmarks.experiments.core.step09a import DEFAULT_INPUT_PATH, DEFAULT_REPORTS_ROOT
from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.eval.claim2_evaluators import (
    build_claim2_prompt_snapshot,
    build_primary_labelers,
    build_supplementary_labelers,
    load_evaluator_profile,
    probe_evaluator_profile,
)
from benchmarks.experiments.eval.consistency_parser import (
    parse_method_result,
)
from benchmarks.experiments.eval.faithfulness_pipeline import (
    EmbeddingCosineAligner,
    align_assertions,
    build_claim_assertions,
    compute_agreement_matrix,
    extract_evidence_windows,
    rethreshold_alignment_results,
    score_faithfulness,
    validate_agreement_gate,
)
from benchmarks.experiments.eval.metrics_claim2 import evaluate_claim2_metrics
from benchmarks.experiments.methods.base import case_uid
from mas.config import SystemConfig
from mas.infrastructure.embedding import EmbeddingFunc

CLAIM2 = 2
CLAIM2_STAGE = "full"
CLAIM2_TASK_FOCUS = "consistency_and_faithfulness"
CLAIM2_PROTOCOL_VERSION = "step11_claim2_v1"
CLAIM2_TOP_K = 3
CLAIM2_DEFAULT_AGGREGATION = "any"
CLAIM2_AGREEMENT_THRESHOLD = 0.70
CLAIM2_ALIGNMENT_THRESHOLD = 0.10


class Claim2ExecutionError(RuntimeError):
    """Raised when Claim 2 execution cannot continue."""


@dataclass(frozen=True)
class Claim1SourceRun:
    run_id: str
    claim_root: Path
    summary: dict[str, Any]
    freeze_run_id: str
    method_names: list[str]
    dev_scope_path: Path
    test_scope_path: Path
    dev_raw_dir: Path
    test_raw_dir: Path


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_text(path: str | Path, content: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def _read_ids(path: str | Path, key: str = "ids") -> list[str]:
    payload = _read_json(path)
    values = payload.get(key, [])

    if not isinstance(values, list) or not values:
        raise ValueError(f"{Path(path).name} must contain a non-empty `{key}` list.")

    return [str(value) for value in values]


def _load_step09a_manifest(
    *, freeze_run_id: str, reports_root: str | Path
) -> dict[str, Any]:
    manifest_path = Path(reports_root) / freeze_run_id / "protocol_freeze_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Step 09A freeze manifest: {manifest_path}")

    manifest = _read_json(manifest_path)

    if not manifest.get("preflight_passed", False):
        raise ValueError("Step 09A preflight did not pass; Step 11 cannot start.")

    if not manifest.get("live_dryrun_passed", False):
        raise ValueError("Step 09A live dry-run did not pass; Step 11 cannot start.")

    return manifest


def _load_claim1_source_run(
    *,
    source_claim1_run_id: str,
    reports_root: str | Path,
    freeze_run_id: str,
) -> Claim1SourceRun:
    claim_root = Path(reports_root) / source_claim1_run_id / "claim1"
    summary_path = claim_root / "run_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Claim 1 run summary: {summary_path}")

    summary = _read_json(summary_path)

    if int(summary.get("claim_id", 0) or 0) != 1:
        raise ValueError("Step 11 source run must be one completed Claim 1 run.")

    if str(summary.get("status", "") or "") != "completed":
        raise ValueError("Step 11 source Claim 1 run must be completed.")

    if str(summary.get("stage", "") or "") != "full":
        raise ValueError("Step 11 source Claim 1 run must be a full run.")

    source_freeze_run_id = str(summary.get("freeze_run_id", "") or "")

    if source_freeze_run_id != freeze_run_id:
        raise ValueError(
            "Step 11 freeze_run_id must match the source Claim 1 freeze_run_id."
        )

    artifacts = dict(summary.get("artifacts", {}) or {})
    dev_scope_path = Path(str(artifacts.get("dev_scope", "") or ""))
    test_scope_path = Path(str(artifacts.get("test_scope", "") or ""))
    dev_raw_dir = claim_root / "dev" / "raw_predictions"
    test_raw_dir = claim_root / "test" / "raw_predictions"

    for required_path in (
        dev_scope_path,
        test_scope_path,
        dev_raw_dir,
        test_raw_dir,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing Claim 1 source artifact: {required_path}")

    method_names = [str(name) for name in list(summary.get("method_names", []))]

    if not method_names:
        raise ValueError("Claim 1 source run must list non-empty method_names.")

    return Claim1SourceRun(
        run_id=source_claim1_run_id,
        claim_root=claim_root,
        summary=summary,
        freeze_run_id=source_freeze_run_id,
        method_names=method_names,
        dev_scope_path=dev_scope_path,
        test_scope_path=test_scope_path,
        dev_raw_dir=dev_raw_dir,
        test_raw_dir=test_raw_dir,
    )


def _load_cases_by_uid(input_path: str | Path) -> dict[str, dict[str, Any]]:
    return {case_uid(case): case for case in load_cases_from_jsonl(input_path)}


def _resolve_claim2_methods(
    *,
    source_run: Claim1SourceRun,
    methods: list[str] | None = None,
) -> list[str]:
    available = list(source_run.method_names)

    if methods is None:
        return available

    requested = [str(name).strip() for name in methods if str(name).strip()]
    unknown = sorted(set(requested) - set(available))

    if unknown:
        raise ValueError(f"Unknown Claim 2 method names requested: {unknown}")

    resolved = [name for name in available if name in requested]

    if not resolved:
        raise ValueError("Claim 2 must run at least one method.")

    return resolved


def _build_aligner(*, alignment_threshold: float) -> EmbeddingCosineAligner:
    embedding = EmbeddingFunc(SystemConfig().path.embedding_model_path)
    return EmbeddingCosineAligner(embedding, threshold=alignment_threshold)


def _build_agreement_threshold_grid(base_threshold: float) -> list[float]:
    return sorted(
        {
            round(max(0.0, base_threshold - 0.05), 3),
            round(base_threshold, 3),
            round(min(1.0, base_threshold + 0.05), 3),
        }
    )


def _build_alignment_threshold_grid(base_threshold: float) -> list[float]:
    return sorted(
        {
            round(base_threshold * 0.95, 4),
            round(base_threshold, 4),
            round(base_threshold * 1.05, 4),
        }
    )


def _try_load_cached_case_diagnostic(
    *,
    path: Path,
    method_name: str,
    case_uid_value: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        payload = _read_json(path)

    except Exception:
        return None

    if str(payload.get("method_name", "") or "") != method_name:
        return None

    if str(payload.get("case_uid", "") or "") != case_uid_value:
        return None

    if "parse_result" not in payload or "faithfulness" not in payload:
        return None

    return payload


def _case_metric_value(case_diag: dict[str, Any], metric_name: str) -> float:
    case_uid_value = str(case_diag.get("case_uid", "") or "")
    parse_result = dict(case_diag.get("parse_result", {}) or {})
    faithfulness = dict(case_diag.get("faithfulness", {}) or {})

    per_case = dict(faithfulness.get("per_case_results", {}) or {}).get(
        case_uid_value, {}
    )

    if metric_name == "overall_faithfulness":
        return float(per_case.get("overall_faithfulness", 0.0) or 0.0)

    if metric_name == "alignment_success_rate":
        return float(per_case.get("alignment_success_rate", 0.0) or 0.0)

    if metric_name == "entailment_rate_on_aligned":
        return float(per_case.get("entailment_rate_on_aligned", 0.0) or 0.0)

    if metric_name == "uncertain_rate":
        aligned = float(per_case.get("aligned_count", 0.0) or 0.0)
        uncertain = float(per_case.get("uncertain_count", 0.0) or 0.0)
        return float(uncertain / aligned) if aligned else 0.0

    if metric_name == "conflict_rate":
        return 1.0 if list(parse_result.get("conflict_edges", [])) else 0.0

    if metric_name == "parser_coverage_rate":
        return 1.0 if bool(parse_result.get("coverage_passed", False)) else 0.0

    raise ValueError(f"Unsupported Claim 2 case metric: {metric_name}")


def _summarize_parse_results_from_diags(
    case_diags: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_breakdown: Counter[str] = Counter()
    claim_count_total = 0
    claim_count_parsed = 0
    fully_parsed_case_count = 0
    conflict_case_count = 0

    for case_diag in case_diags:
        parse_result = dict(case_diag.get("parse_result", {}) or {})
        claim_count_total += int(parse_result.get("input_claim_count", 0) or 0)
        claim_count_parsed += len(list(parse_result.get("parsed_claims", [])))

        fully_parsed_case_count += (
            1 if bool(parse_result.get("coverage_passed", False)) else 0
        )

        conflict_case_count += 1 if list(parse_result.get("conflict_edges", [])) else 0

        for failure in list(parse_result.get("parse_failures", [])):
            reason = str(dict(failure).get("reason", "") or "")

            if reason:
                reason_breakdown[reason] += 1

    return {
        "case_count": len(case_diags),
        "fully_parsed_case_count": fully_parsed_case_count,
        "claim_count_total": claim_count_total,
        "claim_count_parsed": claim_count_parsed,
        "parse_failure_count": int(sum(reason_breakdown.values())),
        "parse_failure_reason_breakdown": dict(sorted(reason_breakdown.items())),
        "conflict_case_count": conflict_case_count,
    }


def _merge_vote_sequences(case_diags: list[dict[str, Any]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = defaultdict(list)

    for case_diag in case_diags:
        vote_sequences = dict(
            dict(case_diag.get("faithfulness", {}) or {}).get("vote_sequences", {})
            or {}
        )

        for name, labels in vote_sequences.items():
            merged[str(name)].extend(str(label) for label in list(labels))

    return {name: labels for name, labels in sorted(merged.items())}


def _compute_metric_summary(case_scores: dict[str, float]) -> dict[str, Any]:
    return dict(evaluate_claim2_metrics(per_case_scores=case_scores)["metric"])


def _build_method_metric_payload(
    *,
    method_name: str,
    case_diags: list[dict[str, Any]],
    agreement_threshold: float,
) -> dict[str, Any]:
    case_uids = [str(case_diag.get("case_uid", "") or "") for case_diag in case_diags]

    case_scores = {
        "overall_faithfulness": {
            uid: _case_metric_value(case_diag, "overall_faithfulness")
            for uid, case_diag in zip(case_uids, case_diags)
        },
        "conflict_rate": {
            uid: _case_metric_value(case_diag, "conflict_rate")
            for uid, case_diag in zip(case_uids, case_diags)
        },
        "parser_coverage_rate": {
            uid: _case_metric_value(case_diag, "parser_coverage_rate")
            for uid, case_diag in zip(case_uids, case_diags)
        },
        "alignment_success_rate": {
            uid: _case_metric_value(case_diag, "alignment_success_rate")
            for uid, case_diag in zip(case_uids, case_diags)
        },
        "entailment_rate_on_aligned": {
            uid: _case_metric_value(case_diag, "entailment_rate_on_aligned")
            for uid, case_diag in zip(case_uids, case_diags)
        },
        "uncertain_rate": {
            uid: _case_metric_value(case_diag, "uncertain_rate")
            for uid, case_diag in zip(case_uids, case_diags)
        },
    }

    parse_summary = _summarize_parse_results_from_diags(case_diags)
    vote_sequences = _merge_vote_sequences(case_diags)

    if any(vote_sequences.values()):
        agreement_matrix = compute_agreement_matrix(vote_sequences).to_dict()

    else:
        agreement_matrix = {
            "pairwise": {},
            "overall_majority_agreement": 0.0,
            "uncertain_rate": 1.0,
            "item_count": 0,
        }

    agreement_gate = validate_agreement_gate(
        compute_agreement_matrix(vote_sequences)
        if any(vote_sequences.values())
        else _empty_agreement_matrix(),
        threshold=agreement_threshold,
    ).to_dict()

    aggregation_modes = {
        "any": _aggregate_case_sensitivity(
            case_diags, sensitivity_group="aggregation_modes", key="any"
        ),
        "majority": _aggregate_case_sensitivity(
            case_diags, sensitivity_group="aggregation_modes", key="majority"
        ),
        "max_score": _aggregate_case_sensitivity(
            case_diags, sensitivity_group="aggregation_modes", key="max_score"
        ),
    }

    agreement_thresholds = {
        f"{threshold:.2f}": validate_agreement_gate(
            compute_agreement_matrix(vote_sequences)
            if any(vote_sequences.values())
            else _empty_agreement_matrix(),
            threshold=threshold,
        ).to_dict()
        for threshold in _build_agreement_threshold_grid(agreement_threshold)
    }

    sample_alignment_keys = _alignment_threshold_keys(case_diags)

    alignment_thresholds = {
        key: _aggregate_case_sensitivity(
            case_diags,
            sensitivity_group="alignment_thresholds",
            key=key,
        )
        for key in sample_alignment_keys
    }

    return {
        "method_name": method_name,
        "main_metrics": {
            "overall_faithfulness": _compute_metric_summary(
                case_scores["overall_faithfulness"]
            ),
            "conflict_rate": _compute_metric_summary(case_scores["conflict_rate"]),
            "parser_coverage_rate": _compute_metric_summary(
                case_scores["parser_coverage_rate"]
            ),
        },
        "appendix_metrics": {
            "alignment_success_rate": _compute_metric_summary(
                case_scores["alignment_success_rate"]
            ),
            "entailment_rate_on_aligned": _compute_metric_summary(
                case_scores["entailment_rate_on_aligned"]
            ),
            "uncertain_rate": _compute_metric_summary(case_scores["uncertain_rate"]),
        },
        "case_scores": case_scores,
        "agreement_matrix": agreement_matrix,
        "agreement_gate": agreement_gate,
        "parse_summary": parse_summary,
        "sensitivity": {
            "aggregation_modes": aggregation_modes,
            "agreement_thresholds": agreement_thresholds,
            "alignment_thresholds": alignment_thresholds,
        },
        "supplementary": _collect_supplementary_summaries(case_diags),
    }


def _alignment_threshold_keys(case_diags: list[dict[str, Any]]) -> list[str]:
    for case_diag in case_diags:
        mapping = dict(
            dict(case_diag.get("sensitivity", {}) or {}).get("alignment_thresholds", {})
            or {}
        )

        if mapping:
            return sorted(mapping)

    return []


def _aggregate_case_sensitivity(
    case_diags: list[dict[str, Any]],
    *,
    sensitivity_group: str,
    key: str,
) -> dict[str, Any]:
    case_scores = {
        str(case_diag.get("case_uid", "") or ""): _extract_case_overall_faithfulness(
            case_diag,
            sensitivity_group=sensitivity_group,
            key=key,
        )
        for case_diag in case_diags
    }

    return _compute_metric_summary(case_scores)


def _extract_case_overall_faithfulness(
    case_diag: dict[str, Any],
    *,
    sensitivity_group: str,
    key: str,
) -> float:
    case_uid_value = str(case_diag.get("case_uid", "") or "")
    sensitivity = dict(case_diag.get("sensitivity", {}) or {})
    payload = dict(sensitivity.get(sensitivity_group, {}) or {}).get(key, {})

    per_case = dict(dict(payload).get("per_case_results", {}) or {}).get(
        case_uid_value, {}
    )

    return float(dict(per_case).get("overall_faithfulness", 0.0) or 0.0)


def _collect_supplementary_summaries(
    case_diags: list[dict[str, Any]],
) -> dict[str, Any]:
    combined: dict[str, dict[str, float]] = {}

    for case_diag in case_diags:
        supplementary = dict(case_diag.get("supplementary", {}) or {})

        for name, payload in supplementary.items():
            case_uid_value = str(case_diag.get("case_uid", "") or "")

            per_case = dict(dict(payload).get("per_case_results", {}) or {}).get(
                case_uid_value, {}
            )

            combined.setdefault(str(name), {})[case_uid_value] = float(
                dict(per_case).get("overall_faithfulness", 0.0) or 0.0
            )

    return {
        name: _compute_metric_summary(case_scores)
        for name, case_scores in sorted(combined.items())
    }


def _empty_agreement_matrix():
    @dataclass(frozen=True)
    class _Matrix:
        pairwise: dict[str, dict[str, float]]
        overall_majority_agreement: float
        uncertain_rate: float
        item_count: int

    return _Matrix(
        pairwise={}, overall_majority_agreement=0.0, uncertain_rate=1.0, item_count=0
    )


def _write_stage_method_metrics(
    *,
    stage_dir: Path,
    stage_name: str,
    run_id: str,
    source_claim1_run_id: str,
    evaluator_profile_name: str,
    method_order: list[str],
    method_metrics: dict[str, dict[str, Any]],
) -> dict[str, str]:
    main_rows = []

    for method_name in method_order:
        metric = method_metrics[method_name]

        main_rows.append(
            {
                "method_name": method_name,
                "overall_faithfulness": dict(
                    metric["main_metrics"]["overall_faithfulness"]
                ),
                "conflict_rate": dict(metric["main_metrics"]["conflict_rate"]),
                "parser_coverage_rate": dict(
                    metric["main_metrics"]["parser_coverage_rate"]
                ),
                "agreement_overall_majority": float(
                    metric["agreement_matrix"]["overall_majority_agreement"]
                ),
                "agreement_uncertain_rate": float(
                    metric["agreement_matrix"]["uncertain_rate"]
                ),
            }
        )

    main_table_payload = {
        "claim_id": CLAIM2,
        "run_id": run_id,
        "stage": CLAIM2_STAGE,
        "split": stage_name,
        "task_focus": CLAIM2_TASK_FOCUS,
        "source_claim1_run_id": source_claim1_run_id,
        "evaluator_profile_name": evaluator_profile_name,
        "method_order": list(method_order),
        "rows": main_rows,
    }

    faithfulness_payload = {
        "claim_id": CLAIM2,
        "run_id": run_id,
        "stage": CLAIM2_STAGE,
        "split": stage_name,
        "task_focus": CLAIM2_TASK_FOCUS,
        "source_claim1_run_id": source_claim1_run_id,
        "evaluator_profile_name": evaluator_profile_name,
        "rows": [
            {
                "method_name": method_name,
                "base": {
                    "overall_faithfulness": dict(
                        method_metrics[method_name]["main_metrics"][
                            "overall_faithfulness"
                        ]
                    ),
                    "alignment_success_rate": dict(
                        method_metrics[method_name]["appendix_metrics"][
                            "alignment_success_rate"
                        ]
                    ),
                    "entailment_rate_on_aligned": dict(
                        method_metrics[method_name]["appendix_metrics"][
                            "entailment_rate_on_aligned"
                        ]
                    ),
                    "uncertain_rate": dict(
                        method_metrics[method_name]["appendix_metrics"][
                            "uncertain_rate"
                        ]
                    ),
                },
                "agreement": {
                    "agreement_matrix": dict(
                        method_metrics[method_name]["agreement_matrix"]
                    ),
                    "agreement_gate": dict(
                        method_metrics[method_name]["agreement_gate"]
                    ),
                },
                "sensitivity": dict(method_metrics[method_name]["sensitivity"]),
                "supplementary": dict(method_metrics[method_name]["supplementary"]),
            }
            for method_name in method_order
        ],
    }

    parse_failure_payload = {
        "claim_id": CLAIM2,
        "run_id": run_id,
        "stage": CLAIM2_STAGE,
        "split": stage_name,
        "task_focus": CLAIM2_TASK_FOCUS,
        "source_claim1_run_id": source_claim1_run_id,
        "evaluator_profile_name": evaluator_profile_name,
        "rows": [
            {
                "method_name": method_name,
                "parse_summary": dict(method_metrics[method_name]["parse_summary"]),
                "failure_cases": [
                    {
                        "case_uid": str(case_diag.get("case_uid", "") or ""),
                        "parse_failures": list(
                            dict(case_diag.get("parse_result", {}) or {}).get(
                                "parse_failures", []
                            )
                            or []
                        ),
                        "conflict_edges": list(
                            dict(case_diag.get("parse_result", {}) or {}).get(
                                "conflict_edges", []
                            )
                            or []
                        ),
                    }
                    for case_diag in method_metrics[method_name]["case_diagnostics"]
                    if list(
                        dict(case_diag.get("parse_result", {}) or {}).get(
                            "parse_failures", []
                        )
                        or []
                    )
                    or list(
                        dict(case_diag.get("parse_result", {}) or {}).get(
                            "conflict_edges", []
                        )
                        or []
                    )
                ],
            }
            for method_name in method_order
        ],
    }

    main_json_path = stage_dir / "main_table.json"
    main_csv_path = stage_dir / "main_table.csv"
    agreement_csv_path = stage_dir / "evaluator_agreement_matrix.csv"
    parse_failure_path = stage_dir / "parse_failure_audit.json"
    faithfulness_path = stage_dir / "faithfulness_decomposition.json"
    method_metrics_path = stage_dir / "method_metrics.json"

    _write_json(main_json_path, main_table_payload)
    _write_json(parse_failure_path, parse_failure_payload)
    _write_json(faithfulness_path, faithfulness_payload)

    _write_json(
        method_metrics_path,
        {
            method_name: {
                key: value for key, value in metric.items() if key != "case_diagnostics"
            }
            for method_name, metric in method_metrics.items()
        },
    )

    _write_claim2_main_table_csv(main_csv_path, main_rows)
    _write_agreement_csv(agreement_csv_path, method_order, method_metrics)

    return {
        "main_table_json": str(main_json_path),
        "main_table_csv": str(main_csv_path),
        "evaluator_agreement_matrix": str(agreement_csv_path),
        "parse_failure_audit": str(parse_failure_path),
        "faithfulness_decomposition": str(faithfulness_path),
        "method_metrics": str(method_metrics_path),
    }


def _write_claim2_main_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_name",
                "overall_faithfulness",
                "overall_faithfulness_ci_low",
                "overall_faithfulness_ci_high",
                "conflict_rate",
                "conflict_rate_ci_low",
                "conflict_rate_ci_high",
                "parser_coverage_rate",
                "parser_coverage_rate_ci_low",
                "parser_coverage_rate_ci_high",
                "agreement_overall_majority",
                "agreement_uncertain_rate",
            ],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "method_name": row["method_name"],
                    "overall_faithfulness": row["overall_faithfulness"][
                        "point_estimate"
                    ],
                    "overall_faithfulness_ci_low": row["overall_faithfulness"][
                        "ci_low"
                    ],
                    "overall_faithfulness_ci_high": row["overall_faithfulness"][
                        "ci_high"
                    ],
                    "conflict_rate": row["conflict_rate"]["point_estimate"],
                    "conflict_rate_ci_low": row["conflict_rate"]["ci_low"],
                    "conflict_rate_ci_high": row["conflict_rate"]["ci_high"],
                    "parser_coverage_rate": row["parser_coverage_rate"][
                        "point_estimate"
                    ],
                    "parser_coverage_rate_ci_low": row["parser_coverage_rate"][
                        "ci_low"
                    ],
                    "parser_coverage_rate_ci_high": row["parser_coverage_rate"][
                        "ci_high"
                    ],
                    "agreement_overall_majority": row["agreement_overall_majority"],
                    "agreement_uncertain_rate": row["agreement_uncertain_rate"],
                }
            )


def _write_agreement_csv(
    path: Path,
    method_order: list[str],
    method_metrics: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_name",
                "evaluator_left",
                "evaluator_right",
                "agreement_rate",
                "overall_majority_agreement",
                "uncertain_rate",
                "item_count",
            ],
        )

        writer.writeheader()

        for method_name in method_order:
            matrix = dict(method_metrics[method_name]["agreement_matrix"])
            pairwise = dict(matrix.get("pairwise", {}) or {})

            for left_name in sorted(pairwise):
                for right_name in sorted(dict(pairwise[left_name]).keys()):
                    writer.writerow(
                        {
                            "method_name": method_name,
                            "evaluator_left": left_name,
                            "evaluator_right": right_name,
                            "agreement_rate": dict(pairwise[left_name]).get(
                                right_name, 0.0
                            ),
                            "overall_majority_agreement": matrix.get(
                                "overall_majority_agreement", 0.0
                            ),
                            "uncertain_rate": matrix.get("uncertain_rate", 0.0),
                            "item_count": matrix.get("item_count", 0),
                        }
                    )


def _write_reproducibility_block(
    *,
    path: Path,
    freeze_run_id: str,
    source_claim1_run_id: str,
    evaluator_profile_snapshot_path: str,
    prompt_snapshot_path: str,
    top_k: int,
    aggregation: str,
    agreement_threshold: float,
    alignment_threshold: float,
) -> None:
    content = "\n".join(
        [
            "# Claim 2 Reproducibility Block",
            "",
            f"- freeze_run_id: `{freeze_run_id}`",
            f"- source_claim1_run_id: `{source_claim1_run_id}`",
            f"- top_k: `{top_k}`",
            f"- aggregation: `{aggregation}`",
            f"- agreement_threshold: `{agreement_threshold}`",
            f"- alignment_threshold: `{alignment_threshold}`",
            f"- evaluator_profile_snapshot: `{evaluator_profile_snapshot_path}`",
            f"- prompt_snapshot: `{prompt_snapshot_path}`",
        ]
    )

    _write_text(path, content)


def _promote_stage_outputs(*, stage_dir: Path, claim_root: Path) -> dict[str, str]:
    promoted = {
        "main_table_json": str(claim_root / "main_table.json"),
        "main_table_csv": str(claim_root / "main_table.csv"),
        "evaluator_agreement_matrix": str(
            claim_root / "evaluator_agreement_matrix.csv"
        ),
        "parse_failure_audit": str(claim_root / "parse_failure_audit.json"),
        "faithfulness_decomposition": str(
            claim_root / "faithfulness_decomposition.json"
        ),
        "reproducibility_block": str(claim_root / "reproducibility_block.md"),
    }

    for _, target in promoted.items():
        source = stage_dir / Path(target).name
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    return promoted


def _run_claim2_stage(
    *,
    stage_name: str,
    case_uids: list[str],
    case_map: dict[str, dict[str, Any]],
    raw_predictions_dir: Path,
    method_order: list[str],
    stage_dir: Path,
    primary_labelers: list[Any],
    supplementary_labelers: list[Any],
    aligner: Any,
    top_k: int,
    aggregation: str,
    agreement_threshold: float,
    alignment_threshold: float,
    run_id: str,
    source_claim1_run_id: str,
    evaluator_profile_name: str,
    show_progress: bool,
    resume: bool,
) -> dict[str, Any]:
    case_diag_root = stage_dir / "case_diagnostics"
    failures: list[dict[str, Any]] = []

    summary = {
        "stage": stage_name,
        "status": "running",
        "started_at": _now_iso(),
        "case_uids": list(case_uids),
        "method_names": list(method_order),
        "total_task_count": len(case_uids) * len(method_order),
        "completed_task_count": 0,
        "resumed_task_count": 0,
        "failure_count": 0,
        "failures": [],
        "current_task": None,
    }

    summary_path = stage_dir / "execution_summary.json"
    _write_json(summary_path, summary)

    progress = tqdm(
        total=summary["total_task_count"],
        desc=f"Claim2 {stage_name}",
        unit="task",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    case_diagnostics_by_method: dict[str, list[dict[str, Any]]] = {
        method_name: [] for method_name in method_order
    }

    try:
        for method_name in method_order:
            for uid in case_uids:
                diag_path = case_diag_root / method_name / f"{uid}.json"

                if resume:
                    cached = _try_load_cached_case_diagnostic(
                        path=diag_path,
                        method_name=method_name,
                        case_uid_value=uid,
                    )

                    if cached is not None:
                        case_diagnostics_by_method[method_name].append(cached)
                        summary["completed_task_count"] += 1
                        summary["resumed_task_count"] += 1
                        summary["current_task"] = None
                        _write_json(summary_path, summary)
                        progress.update(1)
                        continue

                summary["current_task"] = {
                    "method_name": method_name,
                    "case_uid": uid,
                    "started_at": _now_iso(),
                }

                _write_json(summary_path, summary)

                try:
                    case_payload = case_map[uid]
                    prediction_path = raw_predictions_dir / method_name / f"{uid}.json"

                    if not prediction_path.exists():
                        raise FileNotFoundError(
                            f"Missing Claim 1 raw prediction for method={method_name} uid={uid}: {prediction_path}"
                        )

                    result = _read_json(prediction_path)
                    parse_result = parse_method_result(result)
                    assertions = build_claim_assertions(parse_result)
                    windows = extract_evidence_windows(case_payload)

                    alignment_results = align_assertions(
                        assertions,
                        windows,
                        aligner,
                        top_k=top_k,
                    )

                    base_summary = score_faithfulness(
                        alignment_results,
                        primary_labelers,
                        aggregation=aggregation,
                    )

                    aggregation_summaries = {
                        mode: score_faithfulness(
                            alignment_results,
                            primary_labelers,
                            aggregation=mode,
                        ).to_dict()
                        for mode in ("any", "majority", "max_score")
                    }

                    alignment_threshold_summaries = {}

                    for threshold in _build_alignment_threshold_grid(
                        alignment_threshold
                    ):
                        rethresholded = rethreshold_alignment_results(
                            alignment_results,
                            threshold=threshold,
                        )

                        alignment_threshold_summaries[f"{threshold:.4f}"] = (
                            score_faithfulness(
                                rethresholded,
                                primary_labelers,
                                aggregation=aggregation,
                            ).to_dict()
                        )

                    supplementary: dict[str, Any] = {}

                    for labeler in supplementary_labelers:
                        supplementary[str(labeler.name)] = score_faithfulness(
                            alignment_results,
                            [labeler],
                            aggregation=aggregation,
                        ).to_dict()

                    case_diag = {
                        "method_name": method_name,
                        "case_uid": uid,
                        "parse_result": parse_result.to_dict(),
                        "evidence_window_count": len(windows),
                        "alignment_results": [
                            item.to_dict() for item in alignment_results
                        ],
                        "faithfulness": base_summary.to_dict(),
                        "sensitivity": {
                            "aggregation_modes": aggregation_summaries,
                            "alignment_thresholds": alignment_threshold_summaries,
                        },
                        "supplementary": supplementary,
                    }

                    _write_json(diag_path, case_diag)
                    case_diagnostics_by_method[method_name].append(case_diag)

                except Exception as exc:
                    failures.append(
                        {
                            "method_name": method_name,
                            "case_uid": uid,
                            "error_type": type(exc).__name__,
                            "error_detail": str(exc),
                            "failed_at": _now_iso(),
                        }
                    )

                summary["completed_task_count"] += 1
                summary["failure_count"] = len(failures)
                summary["failures"] = list(failures)
                summary["current_task"] = None
                _write_json(summary_path, summary)
                progress.update(1)

    finally:
        progress.close()

    summary["completed_at"] = _now_iso()
    summary["status"] = "completed" if not failures else "failed"
    _write_json(summary_path, summary)

    if failures:
        raise Claim2ExecutionError(
            f"Claim 2 {stage_name} execution failed for {len(failures)} task(s)."
        )

    method_metrics = {
        method_name: _build_method_metric_payload(
            method_name=method_name,
            case_diags=case_diagnostics_by_method[method_name],
            agreement_threshold=agreement_threshold,
        )
        for method_name in method_order
    }

    for method_name in method_order:
        method_metrics[method_name]["case_diagnostics"] = case_diagnostics_by_method[
            method_name
        ]

    artifact_paths = _write_stage_method_metrics(
        stage_dir=stage_dir,
        stage_name=stage_name,
        run_id=run_id,
        source_claim1_run_id=source_claim1_run_id,
        evaluator_profile_name=evaluator_profile_name,
        method_order=method_order,
        method_metrics=method_metrics,
    )

    return {
        "summary": summary,
        "method_metrics": method_metrics,
        "artifacts": artifact_paths,
    }


def run_claim2_experiment(
    *,
    freeze_run_id: str,
    source_claim1_run_id: str,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    evaluator_profile_path: str | Path,
    methods: list[str] | None = None,
    resume: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
    top_k: int = CLAIM2_TOP_K,
    aggregation: str = CLAIM2_DEFAULT_AGGREGATION,
    agreement_threshold: float = CLAIM2_AGREEMENT_THRESHOLD,
    alignment_threshold: float = CLAIM2_ALIGNMENT_THRESHOLD,
) -> dict[str, Any]:
    del verbose
    _load_step09a_manifest(freeze_run_id=freeze_run_id, reports_root=reports_root)

    source_run = _load_claim1_source_run(
        source_claim1_run_id=source_claim1_run_id,
        reports_root=reports_root,
        freeze_run_id=freeze_run_id,
    )

    method_order = _resolve_claim2_methods(source_run=source_run, methods=methods)
    case_map = _load_cases_by_uid(input_path)
    profile = load_evaluator_profile(evaluator_profile_path)
    claim_root = Path(reports_root) / run_id / "claim2"
    claim_root.mkdir(parents=True, exist_ok=True)
    profile_snapshot_path = claim_root / "evaluator_profile_snapshot.json"
    prompt_snapshot_path = claim_root / "prompt_snapshot.json"
    protocol_snapshot_path = claim_root / "claim2_protocol_snapshot.json"
    health_path = claim_root / "evaluator_health.json"
    prompt_snapshot = build_claim2_prompt_snapshot()
    _write_json(profile_snapshot_path, profile.to_snapshot())
    _write_json(prompt_snapshot_path, prompt_snapshot)

    _write_json(
        protocol_snapshot_path,
        {
            "claim_id": CLAIM2,
            "protocol_version": CLAIM2_PROTOCOL_VERSION,
            "task_focus": CLAIM2_TASK_FOCUS,
            "freeze_run_id": freeze_run_id,
            "source_claim1_run_id": source_claim1_run_id,
            "top_k": int(top_k),
            "aggregation": str(aggregation),
            "agreement_threshold": float(agreement_threshold),
            "alignment_threshold": float(alignment_threshold),
            "prompt_version": prompt_snapshot["prompt_version"],
            "evaluator_profile_name": profile.profile_name,
            "method_names": method_order,
        },
    )

    health_rows = probe_evaluator_profile(profile)
    _write_json(health_path, health_rows)
    primary_health = [row for row in health_rows if row.get("cohort") == "primary"]
    failed_primary = [row for row in primary_health if not bool(row.get("ok", False))]

    if failed_primary:
        _write_json(
            claim_root / "run_summary.json",
            {
                "claim_id": CLAIM2,
                "stage": CLAIM2_STAGE,
                "status": "failed",
                "failed_phase": "evaluator_health",
                "freeze_run_id": freeze_run_id,
                "source_claim1_run_id": source_claim1_run_id,
                "error_type": "EvaluatorProbeError",
                "error_detail": "One or more primary evaluators failed health check.",
                "failed_evaluators": failed_primary,
            },
        )

        raise Claim2ExecutionError("Claim 2 evaluator health check failed.")

    labeler_cache_root = claim_root / "cache" / "votes"

    primary_labelers = build_primary_labelers(
        profile, cache_dir=labeler_cache_root / "primary"
    )

    supplementary_labelers = build_supplementary_labelers(
        profile,
        cache_dir=labeler_cache_root / "supplementary",
    )

    aligner = _build_aligner(alignment_threshold=alignment_threshold)
    dev_case_uids = _read_ids(source_run.dev_scope_path)
    test_case_uids = _read_ids(source_run.test_scope_path)

    missing_case_uids = sorted(
        set(dev_case_uids + test_case_uids) - set(case_map.keys())
    )

    if missing_case_uids:
        raise ValueError(f"Missing Step 11 case payloads for uids: {missing_case_uids}")

    dev_dir = claim_root / "dev"
    test_dir = claim_root / "test"

    dev_execution = _run_claim2_stage(
        stage_name="dev",
        case_uids=dev_case_uids,
        case_map=case_map,
        raw_predictions_dir=source_run.dev_raw_dir,
        method_order=method_order,
        stage_dir=dev_dir,
        primary_labelers=primary_labelers,
        supplementary_labelers=supplementary_labelers,
        aligner=aligner,
        top_k=top_k,
        aggregation=aggregation,
        agreement_threshold=agreement_threshold,
        alignment_threshold=alignment_threshold,
        run_id=run_id,
        source_claim1_run_id=source_claim1_run_id,
        evaluator_profile_name=profile.profile_name,
        show_progress=show_progress,
        resume=resume,
    )

    dev_gate = {
        "passed": True,
        "agreement_threshold": float(agreement_threshold),
        "rows": [],
        "failed_methods": [],
    }

    for method_name in method_order:
        gate = dict(dev_execution["method_metrics"][method_name]["agreement_gate"])

        row = {
            "method_name": method_name,
            "passed": bool(gate.get("passed", False)),
            "overall_majority_agreement": float(
                gate.get("overall_majority_agreement", 0.0)
            ),
            "uncertain_rate": float(gate.get("uncertain_rate", 0.0)),
        }

        dev_gate["rows"].append(row)

        if not row["passed"]:
            dev_gate["passed"] = False
            dev_gate["failed_methods"].append(method_name)

    _write_json(dev_dir / "agreement_gate.json", dev_gate)

    if not dev_gate["passed"]:
        summary = {
            "claim_id": CLAIM2,
            "stage": CLAIM2_STAGE,
            "status": "failed",
            "failed_phase": "dev_gate",
            "task_focus": CLAIM2_TASK_FOCUS,
            "freeze_run_id": freeze_run_id,
            "source_claim1_run_id": source_claim1_run_id,
            "evaluator_profile_name": profile.profile_name,
            "method_names": method_order,
            "dev_gate": dev_gate,
            "artifacts": {
                "claim2_protocol_snapshot": str(protocol_snapshot_path),
                "evaluator_profile_snapshot": str(profile_snapshot_path),
                "prompt_snapshot": str(prompt_snapshot_path),
                "evaluator_health": str(health_path),
                "dev_execution_summary": str(dev_dir / "execution_summary.json"),
                "dev_method_metrics": str(dev_dir / "method_metrics.json"),
                "dev_evaluator_agreement_matrix": str(
                    dev_dir / "evaluator_agreement_matrix.csv"
                ),
                "dev_parse_failure_audit": str(dev_dir / "parse_failure_audit.json"),
                "dev_faithfulness_decomposition": str(
                    dev_dir / "faithfulness_decomposition.json"
                ),
            },
        }

        _write_json(claim_root / "run_summary.json", summary)
        raise Claim2ExecutionError("Claim 2 dev evaluator agreement gate failed.")

    test_execution = _run_claim2_stage(
        stage_name="test",
        case_uids=test_case_uids,
        case_map=case_map,
        raw_predictions_dir=source_run.test_raw_dir,
        method_order=method_order,
        stage_dir=test_dir,
        primary_labelers=primary_labelers,
        supplementary_labelers=supplementary_labelers,
        aligner=aligner,
        top_k=top_k,
        aggregation=aggregation,
        agreement_threshold=agreement_threshold,
        alignment_threshold=alignment_threshold,
        run_id=run_id,
        source_claim1_run_id=source_claim1_run_id,
        evaluator_profile_name=profile.profile_name,
        show_progress=show_progress,
        resume=resume,
    )

    _write_reproducibility_block(
        path=test_dir / "reproducibility_block.md",
        freeze_run_id=freeze_run_id,
        source_claim1_run_id=source_claim1_run_id,
        evaluator_profile_snapshot_path=str(profile_snapshot_path),
        prompt_snapshot_path=str(prompt_snapshot_path),
        top_k=top_k,
        aggregation=aggregation,
        agreement_threshold=agreement_threshold,
        alignment_threshold=alignment_threshold,
    )

    promoted_paths = _promote_stage_outputs(stage_dir=test_dir, claim_root=claim_root)

    summary = {
        "claim_id": CLAIM2,
        "stage": CLAIM2_STAGE,
        "status": "completed",
        "task_focus": CLAIM2_TASK_FOCUS,
        "protocol_version": CLAIM2_PROTOCOL_VERSION,
        "freeze_run_id": freeze_run_id,
        "source_claim1_run_id": source_claim1_run_id,
        "evaluator_profile_name": profile.profile_name,
        "method_names": method_order,
        "dev_gate": dev_gate,
        "artifacts": {
            **promoted_paths,
            "claim2_protocol_snapshot": str(protocol_snapshot_path),
            "evaluator_profile_snapshot": str(profile_snapshot_path),
            "prompt_snapshot": str(prompt_snapshot_path),
            "evaluator_health": str(health_path),
            "dev_execution_summary": str(dev_dir / "execution_summary.json"),
            "dev_method_metrics": str(dev_dir / "method_metrics.json"),
            "dev_evaluator_agreement_matrix": str(
                dev_dir / "evaluator_agreement_matrix.csv"
            ),
            "dev_parse_failure_audit": str(dev_dir / "parse_failure_audit.json"),
            "dev_faithfulness_decomposition": str(
                dev_dir / "faithfulness_decomposition.json"
            ),
            "test_execution_summary": str(test_dir / "execution_summary.json"),
            "test_method_metrics": str(test_dir / "method_metrics.json"),
            "test_evaluator_agreement_matrix": str(
                test_dir / "evaluator_agreement_matrix.csv"
            ),
            "test_parse_failure_audit": str(test_dir / "parse_failure_audit.json"),
            "test_faithfulness_decomposition": str(
                test_dir / "faithfulness_decomposition.json"
            ),
        },
    }

    _write_json(claim_root / "run_summary.json", summary)
    return summary
