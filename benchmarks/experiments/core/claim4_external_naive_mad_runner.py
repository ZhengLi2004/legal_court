"""Claim 4 extension runner for the external naive MAD budget curve.

This module intentionally keeps the external MAD budget experiment outside the
official Claim 4 step13/step14 artifact tree.  It reuses the frozen Claim 4
case scope, budgets, matching protocol, and scoring contract, but writes results
under a separate extension root so existing Claim 4 summaries are not invalidated
or rebuilt accidentally.
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
    CLAIM4_DEV_REPEATS,
    CLAIM4_FIXED_BUDGET_POINTS,
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
from benchmarks.experiments.core.step09a import DEFAULT_REPORTS_ROOT
from benchmarks.experiments.eval import (
    SentenceTransformerTextEncoder,
    build_method_case_matches,
    evaluate_claim1_metrics,
)
from benchmarks.experiments.methods.base import case_uid, validate_method_result
from benchmarks.experiments.methods.factory import (
    COMBINED_PROFILE,
    build_method_registry,
)
from mas.config import SystemConfig

EXTENSION_NAME = "claim4_external_naive_mad_budget"
METHOD_NAME = "external_naive_mad"
METHOD_LABEL = "简化多轮基线"
DEFAULT_STAGE = "dev"
DEFAULT_REPEATS = CLAIM4_DEV_REPEATS
DEFAULT_POINTS = CLAIM4_FIXED_BUDGET_POINTS


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


def _slice_raw_root(
    *,
    reports_root: str | Path,
    run_id: str,
    stage: str,
    point: str,
    repeat: int,
) -> Path:
    return (
        _extension_root(reports_root=reports_root, run_id=run_id)
        / stage
        / "raw_predictions"
        / CLAIM4_FIXED_POLICY
        / point
        / f"repeat_{repeat}"
    )


def _slice_metrics_path(
    *,
    reports_root: str | Path,
    run_id: str,
    stage: str,
    point: str,
    repeat: int,
) -> Path:
    return (
        _extension_root(reports_root=reports_root, run_id=run_id)
        / stage
        / "repeat_metrics"
        / CLAIM4_FIXED_POLICY
        / point
        / f"repeat_{repeat}.json"
    )


def _slice_runtime_path(
    *,
    reports_root: str | Path,
    run_id: str,
    stage: str,
    point: str,
    repeat: int,
) -> Path:
    return (
        _extension_root(reports_root=reports_root, run_id=run_id)
        / stage
        / "repeat_runtime"
        / CLAIM4_FIXED_POLICY
        / point
        / f"repeat_{repeat}.json"
    )


def _slice_execution_summary_path(
    *,
    reports_root: str | Path,
    run_id: str,
    stage: str,
    point: str,
    repeat: int,
) -> Path:
    return (
        _extension_root(reports_root=reports_root, run_id=run_id)
        / stage
        / "execution_summary"
        / CLAIM4_FIXED_POLICY
        / point
        / f"repeat_{repeat}.json"
    )


def _runtime_memory_dir(
    *,
    reports_root: str | Path,
    run_id: str,
    stage: str,
    point: str,
    repeat: int,
) -> Path:
    return (
        _extension_root(reports_root=reports_root, run_id=run_id)
        / "runtime_memory"
        / stage
        / CLAIM4_FIXED_POLICY
        / point
        / f"repeat_{repeat}"
        / METHOD_NAME
    )


def run_slice(
    *,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    stage: str = DEFAULT_STAGE,
    point: str,
    repeat: int,
    resume: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run one fixed-budget external MAD slice and score it."""
    if stage != DEFAULT_STAGE:
        raise ValueError(
            f"{EXTENSION_NAME} currently supports only stage={DEFAULT_STAGE}."
        )

    if point not in DEFAULT_POINTS:
        raise ValueError(f"Unsupported budget point: {point}")

    if repeat not in DEFAULT_REPEATS:
        raise ValueError(f"Unsupported repeat: {repeat}")

    ctx = _load_claim4_context(reports_root=reports_root, run_id=run_id)
    registry = build_method_registry(COMBINED_PROFILE)

    if METHOD_NAME not in registry:
        raise ValueError(f"Method registry is missing {METHOD_NAME}.")

    case_map, _ = _load_cases_by_uid(ctx.input_path)
    case_uids = list(ctx.stage_uids_by_name[stage])
    cases = [dict(case_map[uid]) for uid in case_uids]
    runner = registry[METHOD_NAME]
    slice_budget = dict(ctx.budget_points[point])
    slice_budget["temperature"] = 0.0
    slice_budget["disable_adaptive_termination"] = True
    extension_root = _extension_root(reports_root=reports_root, run_id=run_id)

    raw_root = _slice_raw_root(
        reports_root=reports_root,
        run_id=run_id,
        stage=stage,
        point=point,
        repeat=repeat,
    )

    summary_path = _slice_execution_summary_path(
        reports_root=reports_root,
        run_id=run_id,
        stage=stage,
        point=point,
        repeat=repeat,
    )

    storage_root_dir = _runtime_memory_dir(
        reports_root=reports_root,
        run_id=run_id,
        stage=stage,
        point=point,
        repeat=repeat,
    )

    storage_root_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    flattened_predictions: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    retry_events: list[dict[str, Any]] = []

    summary: dict[str, Any] = {
        "claim_id": CLAIM4,
        "extension": EXTENSION_NAME,
        "run_id": run_id,
        "stage": stage,
        "policy": CLAIM4_FIXED_POLICY,
        "point": point,
        "repeat": repeat,
        "status": "running",
        "started_at": _now_iso(),
        "method_names": [METHOD_NAME],
        "case_uids": case_uids,
        "total_task_count": len(case_uids),
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
        total=len(cases),
        desc=f"Claim4 external MAD {point} r{repeat}",
        unit="case",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    try:
        for case in cases:
            uid = case_uid(case)
            raw_result_path = raw_root / METHOD_NAME / f"{uid}.json"

            if resume:
                cached_result = _try_load_valid_result(raw_result_path)

                if cached_result is not None:
                    results.append(cached_result)

                    flattened_predictions.extend(
                        _flatten_prediction_rows(cached_result)
                    )

                    validation_rows.append(
                        {
                            "method": METHOD_NAME,
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
                "method": METHOD_NAME,
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
                    results.append(result)
                    flattened_predictions.extend(_flatten_prediction_rows(result))

                    validation_rows.append(
                        {
                            "method": METHOD_NAME,
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
                            method_name=METHOD_NAME,
                            case_uid_value=uid,
                            budget=slice_budget,
                            retrieval_config=dict(ctx.retrieval_config),
                            warning="degraded_empty_output:no_root_claims_found",
                        )

                        validate_method_result(result)
                        results.append(result)
                        flattened_predictions.extend(_flatten_prediction_rows(result))

                        validation_rows.append(
                            {
                                "method": METHOD_NAME,
                                "case_uid": uid,
                                "claim_count": len(result["claims"]),
                                "status_count": len(result["status"]),
                                "warning_count": len(result["warnings"]),
                            }
                        )

                        _write_json(raw_result_path, result)

                        retry_events.append(
                            {
                                "method": METHOD_NAME,
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
                        attempt < TASK_MAX_ATTEMPTS and _is_retryable_task_error(exc)
                    )

                    retry_events.append(
                        {
                            "method": METHOD_NAME,
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
                            "method": METHOD_NAME,
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
            f"{EXTENSION_NAME} {stage} {point} repeat {repeat} failed for "
            f"{len(failures)} case(s)."
        )

    encoder = SentenceTransformerTextEncoder(SystemConfig().path.embedding_model_path)
    gold_claim_rows = _filter_rows_by_uids(ctx.gold_claim_rows, case_uids)
    gold_status_rows = _filter_rows_by_uids(ctx.gold_status_rows, case_uids)

    bundles = build_method_case_matches(
        case_uids=case_uids,
        gold_rows=gold_claim_rows,
        prediction_rows=flattened_predictions,
        method_name=METHOD_NAME,
        encoder=encoder,
        config=ctx.matching_config,
    )

    method_metric = evaluate_claim1_metrics(bundles, gold_status_rows)
    runtime_metric = _summarize_tokens(results)

    repeat_metrics_payload = {
        "claim_id": CLAIM4,
        "extension": EXTENSION_NAME,
        "run_id": run_id,
        "stage": stage,
        "policy": CLAIM4_FIXED_POLICY,
        "point": point,
        "repeat": repeat,
        "method_names": [METHOD_NAME],
        "primary_metric_key": CLAIM4_PRIMARY_CASE_SCORE_KEY,
        "budget": dict(slice_budget),
        "methods": {
            METHOD_NAME: {
                "primary_metric": _build_primary_score_payload(method_metric),
                "case_scores": dict(
                    method_metric["case_scores"][CLAIM4_PRIMARY_CASE_SCORE_KEY]
                ),
            }
        },
    }

    runtime_payload = {
        "claim_id": CLAIM4,
        "extension": EXTENSION_NAME,
        "run_id": run_id,
        "stage": stage,
        "policy": CLAIM4_FIXED_POLICY,
        "point": point,
        "repeat": repeat,
        "methods": {METHOD_NAME: runtime_metric},
    }

    _write_json(
        _slice_metrics_path(
            reports_root=reports_root,
            run_id=run_id,
            stage=stage,
            point=point,
            repeat=repeat,
        ),
        repeat_metrics_payload,
    )

    _write_json(
        _slice_runtime_path(
            reports_root=reports_root,
            run_id=run_id,
            stage=stage,
            point=point,
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
    points: tuple[str, ...] = DEFAULT_POINTS,
    repeats: tuple[int, ...] = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Build a compact budget-curve summary for completed external MAD slices."""
    if stage != DEFAULT_STAGE:
        raise ValueError(
            f"{EXTENSION_NAME} currently supports only stage={DEFAULT_STAGE}."
        )

    ctx = _load_claim4_context(reports_root=reports_root, run_id=run_id)
    extension_root = _extension_root(reports_root=reports_root, run_id=run_id)
    rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []

    for point in points:
        score_values: list[float] = []
        token_values: list[float] = []
        prompt_values: list[float] = []
        completion_values: list[float] = []
        latency_values: list[float] = []

        for repeat in repeats:
            metric_payload = _read_json(
                _slice_metrics_path(
                    reports_root=reports_root,
                    run_id=run_id,
                    stage=stage,
                    point=point,
                    repeat=repeat,
                )
            )

            runtime_payload = _read_json(
                _slice_runtime_path(
                    reports_root=reports_root,
                    run_id=run_id,
                    stage=stage,
                    point=point,
                    repeat=repeat,
                )
            )

            metric = metric_payload["methods"][METHOD_NAME]["primary_metric"]
            runtime = runtime_payload["methods"][METHOD_NAME]
            score = float(metric["point_estimate"])
            tokens = float(runtime["mean_total_tokens_per_case"])
            prompt_tokens = float(runtime["mean_prompt_tokens_per_case"])
            completion_tokens = float(runtime["mean_completion_tokens_per_case"])
            latency = float(runtime["mean_latency_seconds_per_case"])
            score_values.append(score)
            token_values.append(tokens)
            prompt_values.append(prompt_tokens)
            completion_values.append(completion_tokens)
            latency_values.append(latency)

            repeat_rows.append(
                {
                    "stage": stage,
                    "policy": CLAIM4_FIXED_POLICY,
                    "method_name": METHOD_NAME,
                    "method_label": METHOD_LABEL,
                    "point": point,
                    "repeat": repeat,
                    "budget_max_turns": int(ctx.budget_points[point]["max_turns"]),
                    "primary_score": score,
                    "mean_total_tokens_per_case": tokens,
                    "mean_prompt_tokens_per_case": prompt_tokens,
                    "mean_completion_tokens_per_case": completion_tokens,
                    "mean_latency_seconds_per_case": latency,
                }
            )

        rows.append(
            {
                "stage": stage,
                "policy": CLAIM4_FIXED_POLICY,
                "method_name": METHOD_NAME,
                "method_label": METHOD_LABEL,
                "point": point,
                "budget_max_turns": int(ctx.budget_points[point]["max_turns"]),
                "repeat_count": len(repeats),
                "mean_primary_score": _mean(score_values),
                "std_primary_score": _std(score_values),
                "mean_total_tokens_per_case": _mean(token_values),
                "mean_prompt_tokens_per_case": _mean(prompt_values),
                "mean_completion_tokens_per_case": _mean(completion_values),
                "mean_latency_seconds_per_case": _mean(latency_values),
                "passed_noise_gate": True,
                "is_pareto_frontier": False,
            }
        )

    _write_csv(
        extension_root / "budget_curve_repeats.csv",
        [
            "stage",
            "policy",
            "method_name",
            "method_label",
            "point",
            "repeat",
            "budget_max_turns",
            "primary_score",
            "mean_total_tokens_per_case",
            "mean_prompt_tokens_per_case",
            "mean_completion_tokens_per_case",
            "mean_latency_seconds_per_case",
        ],
        repeat_rows,
    )

    _write_csv(
        extension_root / "budget_curve.csv",
        [
            "stage",
            "policy",
            "method_name",
            "method_label",
            "point",
            "budget_max_turns",
            "repeat_count",
            "mean_primary_score",
            "std_primary_score",
            "mean_total_tokens_per_case",
            "mean_prompt_tokens_per_case",
            "mean_completion_tokens_per_case",
            "mean_latency_seconds_per_case",
            "passed_noise_gate",
            "is_pareto_frontier",
        ],
        rows,
    )

    summary = {
        "claim_id": CLAIM4,
        "extension": EXTENSION_NAME,
        "run_id": run_id,
        "status": "completed",
        "stage": stage,
        "policy": CLAIM4_FIXED_POLICY,
        "method_names": [METHOD_NAME],
        "method_label": METHOD_LABEL,
        "points": list(points),
        "repeats": list(repeats),
        "primary_metric_key": CLAIM4_PRIMARY_CASE_SCORE_KEY,
        "budget_axis": "max_turns",
        "artifacts": {
            "budget_curve_csv": str(extension_root / "budget_curve.csv"),
            "budget_curve_repeats_csv": str(
                extension_root / "budget_curve_repeats.csv"
            ),
        },
        "updated_at": _now_iso(),
    }

    _write_json(extension_root / "run_summary.json", summary)
    return summary


def _parse_points(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_POINTS

    points = tuple(item.strip() for item in raw.split(",") if item.strip())
    invalid = [point for point in points if point not in DEFAULT_POINTS]

    if invalid:
        raise ValueError(f"Unsupported point(s): {invalid}")

    return points


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
        description="Run external naive MAD as a Claim 4 budget extension."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-slice")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    run_parser.add_argument("--stage", default=DEFAULT_STAGE)
    run_parser.add_argument("--point", required=True, choices=list(DEFAULT_POINTS))

    run_parser.add_argument(
        "--repeat", required=True, type=int, choices=list(DEFAULT_REPEATS)
    )

    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--verbose", action="store_true")
    run_parser.add_argument("--no-progress", action="store_true")
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--run-id", required=True)
    summarize_parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    summarize_parser.add_argument("--stage", default=DEFAULT_STAGE)

    summarize_parser.add_argument(
        "--points",
        default=",".join(DEFAULT_POINTS),
        help="Comma-separated budget points; default: q25,q50,q75,full.",
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
            point=args.point,
            repeat=args.repeat,
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
            points=_parse_points(args.points),
            repeats=_parse_repeats(args.repeats),
        )

        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
