"""Claim 4 extension runner for adaptive stopping on internal baselines.

The official Claim 4 adaptive overlay only reruns the full system.  This
extension keeps the two learned/internal baselines outside the official artifact
tree, while reusing the same frozen dev scope, full budget, matching protocol,
and scoring contract.  Its target comparison is adaptive/full versus fixed/full
for each baseline.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from benchmarks.experiments.core.claim4_runner import (
    CLAIM4,
    CLAIM4_ADAPTIVE_POLICY,
    CLAIM4_DEV_REPEATS,
    CLAIM4_FIXED_POLICY,
    CLAIM4_PRIMARY_CASE_SCORE_KEY,
    TASK_MAX_ATTEMPTS,
    _build_claim4_context,
    _build_empty_prediction_result,
    _build_primary_score_payload,
    _filter_rows_by_uids,
    _flatten_prediction_rows,
    _is_retryable_task_error,
    _load_cases_by_uid,
    _read_json,
    _summarize_tokens,
    _try_load_valid_result,
    _write_json,
)
from benchmarks.experiments.core.claim4_runner import (
    _slice_metrics_path as _official_slice_metrics_path,
)
from benchmarks.experiments.core.claim4_runner import (
    _slice_raw_root as _official_slice_raw_root,
)
from benchmarks.experiments.core.claim4_runner import (
    _slice_runtime_path as _official_slice_runtime_path,
)
from benchmarks.experiments.core.step09a import DEFAULT_REPORTS_ROOT
from benchmarks.experiments.eval import (
    SentenceTransformerTextEncoder,
    build_method_case_matches,
    evaluate_claim1_metrics,
)
from benchmarks.experiments.methods.base import case_uid, validate_method_result
from benchmarks.experiments.methods.factory import (
    CLAIM4_MAD_ONLY_PROFILE,
    build_method_registry,
)
from mas.config import SystemConfig

EXTENSION_NAME = "claim4_internal_adaptive_baselines"
DEFAULT_STAGE = "dev"
DEFAULT_POINT = "full"
DEFAULT_REPEATS = CLAIM4_DEV_REPEATS
DEFAULT_METHODS = ("baseline_b2_vanilla_mad", "baseline_b3_stateful_no_axioms")
METHOD_LABELS = {
    "baseline_b2_vanilla_mad": "无学习辩论基线",
    "baseline_b3_stateful_no_axioms": "无图校验基线",
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_csv(
    path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0

    return float(statistics.stdev(values))


def _extension_root(*, reports_root: str | Path, run_id: str) -> Path:
    return Path(reports_root) / run_id / EXTENSION_NAME


def _load_claim4_context(*, reports_root: str | Path, run_id: str):
    manifest = _read_json(Path(reports_root) / run_id / "claim4" / "manifest.json")

    return _build_claim4_context(
        freeze_run_id=str(manifest["freeze_run_id"]),
        source_claim1_run_id=str(manifest["source_claim1_run_id"]),
        run_id=run_id,
        reports_root=reports_root,
        input_path=str(manifest["input_path"]),
    )


def _validate_methods(methods: tuple[str, ...]) -> tuple[str, ...]:
    invalid = [method for method in methods if method not in DEFAULT_METHODS]

    if invalid:
        raise ValueError(f"Unsupported method(s): {invalid}")

    return methods


def _slice_raw_root(
    *,
    reports_root: str | Path,
    run_id: str,
    stage: str,
    repeat: int,
) -> Path:
    return (
        _extension_root(reports_root=reports_root, run_id=run_id)
        / stage
        / "raw_predictions"
        / CLAIM4_ADAPTIVE_POLICY
        / DEFAULT_POINT
        / f"repeat_{repeat}"
    )


def _slice_metrics_path(
    *,
    reports_root: str | Path,
    run_id: str,
    stage: str,
    repeat: int,
) -> Path:
    return (
        _extension_root(reports_root=reports_root, run_id=run_id)
        / stage
        / "repeat_metrics"
        / CLAIM4_ADAPTIVE_POLICY
        / DEFAULT_POINT
        / f"repeat_{repeat}.json"
    )


def _slice_runtime_path(
    *,
    reports_root: str | Path,
    run_id: str,
    stage: str,
    repeat: int,
) -> Path:
    return (
        _extension_root(reports_root=reports_root, run_id=run_id)
        / stage
        / "repeat_runtime"
        / CLAIM4_ADAPTIVE_POLICY
        / DEFAULT_POINT
        / f"repeat_{repeat}.json"
    )


def _slice_execution_summary_path(
    *,
    reports_root: str | Path,
    run_id: str,
    stage: str,
    repeat: int,
) -> Path:
    return (
        _extension_root(reports_root=reports_root, run_id=run_id)
        / stage
        / "execution_summary"
        / CLAIM4_ADAPTIVE_POLICY
        / DEFAULT_POINT
        / f"repeat_{repeat}.json"
    )


def _runtime_memory_dir(
    *,
    reports_root: str | Path,
    run_id: str,
    stage: str,
    repeat: int,
    method_name: str,
) -> Path:
    return (
        _extension_root(reports_root=reports_root, run_id=run_id)
        / "runtime_memory"
        / stage
        / CLAIM4_ADAPTIVE_POLICY
        / DEFAULT_POINT
        / f"repeat_{repeat}"
        / method_name
    )


def _turn_count(result: dict[str, Any]) -> float:
    return float(len(result.get("trace", {}).get("turn_artifacts", []) or []))


def run_slice(
    *,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    stage: str = DEFAULT_STAGE,
    repeat: int,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    resume: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run one adaptive/full slice for the selected internal baselines."""
    if stage != DEFAULT_STAGE:
        raise ValueError(
            f"{EXTENSION_NAME} currently supports only stage={DEFAULT_STAGE}."
        )

    if repeat not in DEFAULT_REPEATS:
        raise ValueError(f"Unsupported repeat: {repeat}")

    method_names = _validate_methods(methods)
    ctx = _load_claim4_context(reports_root=reports_root, run_id=run_id)
    registry = build_method_registry(CLAIM4_MAD_ONLY_PROFILE)
    missing = [method for method in method_names if method not in registry]

    if missing:
        raise ValueError(f"Method registry is missing: {missing}")

    case_map, _ = _load_cases_by_uid(ctx.input_path)
    case_uids = list(ctx.stage_uids_by_name[stage])
    cases = [dict(case_map[uid]) for uid in case_uids]
    slice_budget = dict(ctx.budget_points[DEFAULT_POINT])
    slice_budget["temperature"] = 0.0
    slice_budget["disable_adaptive_termination"] = False
    extension_root = _extension_root(reports_root=reports_root, run_id=run_id)

    raw_root = _slice_raw_root(
        reports_root=reports_root,
        run_id=run_id,
        stage=stage,
        repeat=repeat,
    )

    summary_path = _slice_execution_summary_path(
        reports_root=reports_root,
        run_id=run_id,
        stage=stage,
        repeat=repeat,
    )

    results_by_method: dict[str, list[dict[str, Any]]] = {
        method_name: [] for method_name in method_names
    }

    flattened_predictions: dict[str, list[dict[str, Any]]] = {
        method_name: [] for method_name in method_names
    }

    validation_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    retry_events: list[dict[str, Any]] = []

    summary: dict[str, Any] = {
        "claim_id": CLAIM4,
        "extension": EXTENSION_NAME,
        "run_id": run_id,
        "stage": stage,
        "policy": CLAIM4_ADAPTIVE_POLICY,
        "point": DEFAULT_POINT,
        "repeat": repeat,
        "status": "running",
        "started_at": _now_iso(),
        "method_names": list(method_names),
        "case_uids": case_uids,
        "total_task_count": len(case_uids) * len(method_names),
        "completed_task_count": 0,
        "validation_rows": [],
        "failure_count": 0,
        "failures": [],
        "retry_event_count": 0,
        "retry_events": [],
        "resume_enabled": bool(resume),
        "resumed_task_count": 0,
        "current_task": None,
        "budget": dict(slice_budget),
        "extension_root": str(extension_root),
    }

    _write_json(summary_path, summary)

    progress = tqdm(
        total=summary["total_task_count"],
        desc=f"Claim4 adaptive baselines r{repeat}",
        unit="task",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    try:
        for method_name in method_names:
            runner = registry[method_name]

            storage_root_dir = _runtime_memory_dir(
                reports_root=reports_root,
                run_id=run_id,
                stage=stage,
                repeat=repeat,
                method_name=method_name,
            )

            storage_root_dir.mkdir(parents=True, exist_ok=True)

            for case in cases:
                uid = case_uid(case)
                raw_result_path = raw_root / method_name / f"{uid}.json"

                if resume:
                    cached_result = _try_load_valid_result(raw_result_path)

                    if cached_result is not None:
                        results_by_method[method_name].append(cached_result)

                        flattened_predictions[method_name].extend(
                            _flatten_prediction_rows(cached_result)
                        )

                        validation_rows.append(
                            {
                                "method": method_name,
                                "case_uid": uid,
                                "claim_count": len(cached_result["claims"]),
                                "status_count": len(cached_result["status"]),
                                "warning_count": len(cached_result["warnings"]),
                            }
                        )

                        summary["completed_task_count"] += 1
                        summary["resumed_task_count"] += 1
                        summary["validation_rows"] = list(validation_rows)
                        summary["current_task"] = None
                        _write_json(summary_path, summary)
                        progress.update(1)
                        continue

                summary["current_task"] = {
                    "method": method_name,
                    "case_uid": uid,
                    "started_at": _now_iso(),
                }

                _write_json(summary_path, summary)

                for attempt in range(1, TASK_MAX_ATTEMPTS + 1):
                    try:
                        result = runner(
                            case=case,
                            storage_root_dir=str(storage_root_dir),
                            budget=dict(slice_budget),
                            seed=ctx.seed,
                            retrieval_config=dict(ctx.retrieval_config),
                            test_mode_no_learning=True,
                            verbose=verbose,
                        )

                        validate_method_result(result)
                        results_by_method[method_name].append(result)

                        flattened_predictions[method_name].extend(
                            _flatten_prediction_rows(result)
                        )

                        validation_rows.append(
                            {
                                "method": method_name,
                                "case_uid": uid,
                                "claim_count": len(result["claims"]),
                                "status_count": len(result["status"]),
                                "warning_count": len(result["warnings"]),
                            }
                        )

                        _write_json(raw_result_path, result)
                        break

                    except Exception as exc:
                        error_detail = str(exc)

                        if "No root claims found in graph metadata." in error_detail:
                            result = _build_empty_prediction_result(
                                method_name=method_name,
                                case_uid_value=uid,
                                budget=slice_budget,
                                retrieval_config=dict(ctx.retrieval_config),
                                warning="degraded_empty_output:no_root_claims_found",
                            )

                            validate_method_result(result)
                            results_by_method[method_name].append(result)

                            flattened_predictions[method_name].extend(
                                _flatten_prediction_rows(result)
                            )

                            validation_rows.append(
                                {
                                    "method": method_name,
                                    "case_uid": uid,
                                    "claim_count": len(result["claims"]),
                                    "status_count": len(result["status"]),
                                    "warning_count": len(result["warnings"]),
                                }
                            )

                            _write_json(raw_result_path, result)

                            retry_events.append(
                                {
                                    "method": method_name,
                                    "case_uid": uid,
                                    "attempt": attempt,
                                    "retryable": False,
                                    "error_type": type(exc).__name__,
                                    "error_detail": error_detail,
                                    "recorded_at": _now_iso(),
                                    "recovered_as": "empty_prediction_result",
                                }
                            )

                            break

                        retryable = (
                            attempt < TASK_MAX_ATTEMPTS
                            and _is_retryable_task_error(exc)
                        )

                        retry_events.append(
                            {
                                "method": method_name,
                                "case_uid": uid,
                                "attempt": attempt,
                                "retryable": retryable,
                                "error_type": type(exc).__name__,
                                "error_detail": error_detail,
                                "recorded_at": _now_iso(),
                            }
                        )

                        if retryable:
                            time.sleep(float(attempt))
                            continue

                        failures.append(
                            {
                                "method": method_name,
                                "case_uid": uid,
                                "error_type": type(exc).__name__,
                                "error_detail": error_detail,
                                "failed_at": _now_iso(),
                                "attempts": attempt,
                            }
                        )

                        break

                summary["completed_task_count"] += 1
                summary["validation_rows"] = list(validation_rows)
                summary["failure_count"] = len(failures)
                summary["failures"] = list(failures)
                summary["retry_event_count"] = len(retry_events)
                summary["retry_events"] = list(retry_events)
                summary["current_task"] = None
                _write_json(summary_path, summary)
                progress.update(1)

    finally:
        progress.close()

    summary["completed_at"] = _now_iso()
    summary["validation_row_count"] = len(validation_rows)
    summary["status"] = "completed" if not failures else "failed"
    _write_json(summary_path, summary)

    if failures:
        raise RuntimeError(
            f"{EXTENSION_NAME} {stage} adaptive full repeat {repeat} failed for "
            f"{len(failures)} task(s)."
        )

    encoder = SentenceTransformerTextEncoder(SystemConfig().path.embedding_model_path)
    gold_claim_rows = _filter_rows_by_uids(ctx.gold_claim_rows, case_uids)
    gold_status_rows = _filter_rows_by_uids(ctx.gold_status_rows, case_uids)
    method_metrics: dict[str, dict[str, Any]] = {}
    runtime_metrics: dict[str, dict[str, float]] = {}

    for method_name in method_names:
        bundles = build_method_case_matches(
            case_uids=case_uids,
            gold_rows=gold_claim_rows,
            prediction_rows=flattened_predictions[method_name],
            method_name=method_name,
            encoder=encoder,
            config=ctx.matching_config,
        )

        method_metrics[method_name] = evaluate_claim1_metrics(bundles, gold_status_rows)
        runtime_metrics[method_name] = _summarize_tokens(results_by_method[method_name])

    repeat_metrics_payload = {
        "claim_id": CLAIM4,
        "extension": EXTENSION_NAME,
        "run_id": run_id,
        "stage": stage,
        "policy": CLAIM4_ADAPTIVE_POLICY,
        "point": DEFAULT_POINT,
        "repeat": repeat,
        "method_names": list(method_names),
        "primary_metric_key": CLAIM4_PRIMARY_CASE_SCORE_KEY,
        "budget": dict(slice_budget),
        "methods": {
            method_name: {
                "primary_metric": _build_primary_score_payload(
                    method_metrics[method_name]
                ),
                "case_scores": dict(
                    method_metrics[method_name]["case_scores"][
                        CLAIM4_PRIMARY_CASE_SCORE_KEY
                    ]
                ),
            }
            for method_name in method_names
        },
    }

    runtime_payload = {
        "claim_id": CLAIM4,
        "extension": EXTENSION_NAME,
        "run_id": run_id,
        "stage": stage,
        "policy": CLAIM4_ADAPTIVE_POLICY,
        "point": DEFAULT_POINT,
        "repeat": repeat,
        "methods": runtime_metrics,
    }

    _write_json(
        _slice_metrics_path(
            reports_root=reports_root,
            run_id=run_id,
            stage=stage,
            repeat=repeat,
        ),
        repeat_metrics_payload,
    )

    _write_json(
        _slice_runtime_path(
            reports_root=reports_root,
            run_id=run_id,
            stage=stage,
            repeat=repeat,
        ),
        runtime_payload,
    )

    return {
        "execution_summary": summary,
        "repeat_metrics": repeat_metrics_payload,
        "runtime_metrics": runtime_payload,
    }


def summarize_extension(
    *,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    stage: str = DEFAULT_STAGE,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    repeats: tuple[int, ...] = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Summarize adaptive/full baseline results against fixed/full anchors."""
    if stage != DEFAULT_STAGE:
        raise ValueError(
            f"{EXTENSION_NAME} currently supports only stage={DEFAULT_STAGE}."
        )

    method_names = _validate_methods(methods)
    ctx = _load_claim4_context(reports_root=reports_root, run_id=run_id)
    extension_root = _extension_root(reports_root=reports_root, run_id=run_id)
    stage_uids = list(ctx.stage_uids_by_name[stage])
    repeat_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for method_name in method_names:
        adaptive_scores: list[float] = []
        fixed_scores: list[float] = []
        adaptive_tokens: list[float] = []
        fixed_tokens: list[float] = []
        adaptive_turns: list[float] = []
        fixed_turns: list[float] = []

        for repeat in repeats:
            adaptive_metrics = _read_json(
                _slice_metrics_path(
                    reports_root=reports_root,
                    run_id=run_id,
                    stage=stage,
                    repeat=repeat,
                )
            )

            adaptive_runtime = _read_json(
                _slice_runtime_path(
                    reports_root=reports_root,
                    run_id=run_id,
                    stage=stage,
                    repeat=repeat,
                )
            )

            fixed_metrics = _read_json(
                _official_slice_metrics_path(
                    ctx,
                    stage=stage,
                    policy=CLAIM4_FIXED_POLICY,
                    point=DEFAULT_POINT,
                    repeat=repeat,
                )
            )

            fixed_runtime = _read_json(
                _official_slice_runtime_path(
                    ctx,
                    stage=stage,
                    policy=CLAIM4_FIXED_POLICY,
                    point=DEFAULT_POINT,
                    repeat=repeat,
                )
            )

            adaptive_score = float(
                adaptive_metrics["methods"][method_name]["primary_metric"][
                    "point_estimate"
                ]
            )

            fixed_score = float(
                fixed_metrics["methods"][method_name]["primary_metric"][
                    "point_estimate"
                ]
            )

            adaptive_token = float(
                adaptive_runtime["methods"][method_name]["mean_total_tokens_per_case"]
            )

            fixed_token = float(
                fixed_runtime["methods"][method_name]["mean_total_tokens_per_case"]
            )

            adaptive_turn = _mean(
                [
                    _turn_count(
                        _read_json(
                            _slice_raw_root(
                                reports_root=reports_root,
                                run_id=run_id,
                                stage=stage,
                                repeat=repeat,
                            )
                            / method_name
                            / f"{uid}.json"
                        )
                    )
                    for uid in stage_uids
                ]
            )

            fixed_turn = _mean(
                [
                    _turn_count(
                        _read_json(
                            _official_slice_raw_root(
                                ctx,
                                stage=stage,
                                policy=CLAIM4_FIXED_POLICY,
                                point=DEFAULT_POINT,
                                repeat=repeat,
                            )
                            / method_name
                            / f"{uid}.json"
                        )
                    )
                    for uid in stage_uids
                ]
            )

            adaptive_scores.append(adaptive_score)
            fixed_scores.append(fixed_score)
            adaptive_tokens.append(adaptive_token)
            fixed_tokens.append(fixed_token)
            adaptive_turns.append(adaptive_turn)
            fixed_turns.append(fixed_turn)

            repeat_rows.append(
                {
                    "stage": stage,
                    "method_name": method_name,
                    "method_label": METHOD_LABELS[method_name],
                    "point": DEFAULT_POINT,
                    "repeat": repeat,
                    "adaptive_primary_score": adaptive_score,
                    "fixed_primary_score": fixed_score,
                    "delta_primary_score_vs_fixed": adaptive_score - fixed_score,
                    "adaptive_mean_total_tokens_per_case": adaptive_token,
                    "fixed_mean_total_tokens_per_case": fixed_token,
                    "delta_tokens_vs_fixed": adaptive_token - fixed_token,
                    "adaptive_mean_realized_turns_per_case": adaptive_turn,
                    "fixed_mean_realized_turns_per_case": fixed_turn,
                    "delta_realized_turns_vs_fixed": adaptive_turn - fixed_turn,
                }
            )

        summary_rows.append(
            {
                "stage": stage,
                "method_name": method_name,
                "method_label": METHOD_LABELS[method_name],
                "point": DEFAULT_POINT,
                "repeat_count": len(repeats),
                "mean_adaptive_primary_score": _mean(adaptive_scores),
                "std_adaptive_primary_score": _std(adaptive_scores),
                "mean_fixed_primary_score": _mean(fixed_scores),
                "mean_delta_primary_score_vs_fixed": _mean(adaptive_scores)
                - _mean(fixed_scores),
                "mean_adaptive_total_tokens_per_case": _mean(adaptive_tokens),
                "mean_fixed_total_tokens_per_case": _mean(fixed_tokens),
                "mean_delta_tokens_vs_fixed": _mean(adaptive_tokens)
                - _mean(fixed_tokens),
                "mean_adaptive_realized_turns_per_case": _mean(adaptive_turns),
                "mean_fixed_realized_turns_per_case": _mean(fixed_turns),
                "mean_delta_realized_turns_vs_fixed": _mean(adaptive_turns)
                - _mean(fixed_turns),
            }
        )

    _write_csv(
        extension_root / "adaptive_overlay_repeats.csv",
        [
            "stage",
            "method_name",
            "method_label",
            "point",
            "repeat",
            "adaptive_primary_score",
            "fixed_primary_score",
            "delta_primary_score_vs_fixed",
            "adaptive_mean_total_tokens_per_case",
            "fixed_mean_total_tokens_per_case",
            "delta_tokens_vs_fixed",
            "adaptive_mean_realized_turns_per_case",
            "fixed_mean_realized_turns_per_case",
            "delta_realized_turns_vs_fixed",
        ],
        repeat_rows,
    )

    _write_csv(
        extension_root / "adaptive_overlay.csv",
        [
            "stage",
            "method_name",
            "method_label",
            "point",
            "repeat_count",
            "mean_adaptive_primary_score",
            "std_adaptive_primary_score",
            "mean_fixed_primary_score",
            "mean_delta_primary_score_vs_fixed",
            "mean_adaptive_total_tokens_per_case",
            "mean_fixed_total_tokens_per_case",
            "mean_delta_tokens_vs_fixed",
            "mean_adaptive_realized_turns_per_case",
            "mean_fixed_realized_turns_per_case",
            "mean_delta_realized_turns_vs_fixed",
        ],
        summary_rows,
    )

    summary = {
        "claim_id": CLAIM4,
        "extension": EXTENSION_NAME,
        "run_id": run_id,
        "status": "completed",
        "stage": stage,
        "policy": CLAIM4_ADAPTIVE_POLICY,
        "point": DEFAULT_POINT,
        "method_names": list(method_names),
        "method_labels": {name: METHOD_LABELS[name] for name in method_names},
        "repeats": list(repeats),
        "primary_metric_key": CLAIM4_PRIMARY_CASE_SCORE_KEY,
        "compare_against": "fixed/full",
        "artifacts": {
            "adaptive_overlay_csv": str(extension_root / "adaptive_overlay.csv"),
            "adaptive_overlay_repeats_csv": str(
                extension_root / "adaptive_overlay_repeats.csv"
            ),
        },
        "updated_at": _now_iso(),
    }

    _write_json(extension_root / "run_summary.json", summary)
    return summary


def _parse_methods(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_METHODS

    return _validate_methods(
        tuple(item.strip() for item in raw.split(",") if item.strip())
    )


def _parse_repeats(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return DEFAULT_REPEATS

    repeats = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    invalid = [repeat for repeat in repeats if repeat not in DEFAULT_REPEATS]

    if invalid:
        raise ValueError(f"Unsupported repeat(s): {invalid}")

    return repeats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run adaptive stopping for Claim 4 internal baselines."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-slice")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    run_parser.add_argument("--stage", default=DEFAULT_STAGE)

    run_parser.add_argument(
        "--repeat", required=True, type=int, choices=list(DEFAULT_REPEATS)
    )

    run_parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method names.",
    )

    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--verbose", action="store_true")
    run_parser.add_argument("--no-progress", action="store_true")
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--run-id", required=True)
    summarize_parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    summarize_parser.add_argument("--stage", default=DEFAULT_STAGE)

    summarize_parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method names.",
    )

    summarize_parser.add_argument(
        "--repeats",
        default=",".join(str(item) for item in DEFAULT_REPEATS),
        help="Comma-separated repeats; default: 1,2,3.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "run-slice":
        run_slice(
            run_id=args.run_id,
            reports_root=args.reports_root,
            stage=args.stage,
            repeat=args.repeat,
            methods=_parse_methods(args.methods),
            resume=args.resume,
            verbose=args.verbose,
            show_progress=not args.no_progress,
        )

        return

    if args.command == "summarize":
        summarize_extension(
            run_id=args.run_id,
            reports_root=args.reports_root,
            stage=args.stage,
            methods=_parse_methods(args.methods),
            repeats=_parse_repeats(args.repeats),
        )

        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
