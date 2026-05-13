"""Claim 4 experiment runner for Step 13."""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from benchmarks.experiments.core.step09a import DEFAULT_INPUT_PATH, DEFAULT_REPORTS_ROOT
from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.eval import (
    FROZEN_MATCHING_PROTOCOL_VERSION,
    MatchingConfig,
    SentenceTransformerTextEncoder,
    build_method_case_matches,
    evaluate_claim1_metrics,
)
from benchmarks.experiments.eval._metric_utils import normalize_status_eval
from benchmarks.experiments.eval.consistency_parser import parse_method_result
from benchmarks.experiments.methods.base import case_uid, validate_method_result
from benchmarks.experiments.methods.factory import (
    CLAIM4_MAD_ONLY_PROFILE,
    build_method_registry,
    build_prereg_comparisons,
)
from mas.config import SystemConfig

CLAIM4 = 4
CLAIM4_PROTOCOL_VERSION = "step13_claim4_v1"
CLAIM4_TASK_FOCUS = "controlled_budget_pareto_internal_mad"
CLAIM4_PRIMARY_CASE_SCORE_KEY = "e2e_status_acc"

CLAIM4_METHOD_NAMES = (
    "main_system",
    "baseline_b2_vanilla_mad",
    "baseline_b3_stateful_no_axioms",
)

CLAIM4_FIXED_POLICY = "fixed"
CLAIM4_ADAPTIVE_POLICY = "adaptive"
CLAIM4_FIXED_METHOD_NAMES = CLAIM4_METHOD_NAMES
CLAIM4_ADAPTIVE_METHOD_NAMES = ("main_system",)
CLAIM4_FIXED_BUDGET_POINTS = ("q25", "q50", "q75", "full")
CLAIM4_ADAPTIVE_POINTS = ("full",)
CLAIM4_DEV_REPEATS = (1, 2, 3)
CLAIM4_TEST_REPEATS = (1,)
CLAIM4_ROUND_POINTS = (1, 3, 5)
CLAIM4_STAGES = ("dev", "test")
CLAIM4_OFFICIAL_STAGES = ("dev",)
TASK_MAX_ATTEMPTS = 2


class Claim4ExecutionError(RuntimeError):
    """Raised when Claim 4 execution cannot continue.

    Attributes:
        args: Standard runtime-error payload describing the failure.
    """


@dataclass(frozen=True)
class Claim4Context:
    """Frozen inputs, paths, and scoring references for one Claim 4 run.

    Attributes:
        freeze_run_id: Step 09A freeze bundle reused by the run.
        source_claim1_run_id: Claim 1 source run used for full-budget anchors.
        run_id: Destination Step 13 run identifier.
        reports_root: Root directory storing experiment outputs.
        input_path: Original case JSONL used to resolve evaluation cases.
        claim_root: Root output directory for Claim 4 artifacts.
        source_claim_root: Root directory of the source Claim 1 artifacts.
        metric_root: Claim 1 metric directory used as the reference anchor.
        manifest_path: Path to the prepared Claim 4 manifest.
        budget_grid_path: Path to the serialized budget grid.
        prereg_points_path: Path to preregistered analysis points.
        reference_full_budget_path: Path to full-budget reference metrics.
        noise_audit_path: Path to the consolidated noise audit JSON.
        method_names: Ordered MAD-style methods considered by the experiment.
        stage_uids_by_name: Mapping from stage name to ordered case ids.
        gold_claim_rows: Gold claim rows used for per-case comparisons.
        gold_status_rows: Gold status rows used for scoring.
        matching_config: Frozen matching configuration reused from Claim 1.
        retrieval_config: Frozen retrieval configuration snapshot.
        seed: Shared deterministic seed for slice execution.
        full_budget: Full-budget configuration used as the reference ceiling.
        budget_points: Mapping from named budget points to runtime budgets.
    """

    freeze_run_id: str
    source_claim1_run_id: str
    run_id: str
    reports_root: Path
    input_path: Path
    claim_root: Path
    source_claim_root: Path
    metric_root: Path
    manifest_path: Path
    budget_grid_path: Path
    prereg_points_path: Path
    reference_full_budget_path: Path
    noise_audit_path: Path
    method_names: tuple[str, ...]
    stage_uids_by_name: dict[str, list[str]]
    gold_claim_rows: list[dict[str, Any]]
    gold_status_rows: list[dict[str, Any]]
    matching_config: MatchingConfig
    retrieval_config: dict[str, Any]
    seed: int
    full_budget: dict[str, Any]
    budget_points: dict[str, dict[str, Any]]


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


def _write_csv(
    path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_ids(path: str | Path, key: str = "ids") -> list[str]:
    payload = _read_json(path)
    values = payload.get(key, [])

    if not isinstance(values, list) or not values:
        raise ValueError(f"{Path(path).name} must contain a non-empty `{key}` list.")

    return [str(value) for value in values]


def _load_cases_by_uid(
    input_path: str | Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    ordered_cases = load_cases_from_jsonl(input_path)
    ordered_uids = [case_uid(case) for case in ordered_cases]
    return {case_uid(case): case for case in ordered_cases}, ordered_uids


def _load_gold_rows(path: str | Path) -> list[dict[str, Any]]:
    return [dict(row) for row in load_cases_from_jsonl(path)]


def _resolve_metric_root(*, claim_root: Path) -> Path:
    candidates = []

    for child in claim_root.iterdir():
        if child.is_dir() and (child / "recomputed_summary.json").exists():
            summary = _read_json(child / "recomputed_summary.json")

            if (
                str(summary.get("metric_contract_version", "") or "")
                == "step08_claim1_v2_missing_as_unmentioned"
            ):
                candidates.append(child)

    if candidates:
        return sorted(candidates)[-1]

    return claim_root


def _filter_rows_by_uids(
    rows: list[dict[str, Any]],
    case_uids: list[str],
) -> list[dict[str, Any]]:
    uid_set = set(case_uids)
    return [dict(row) for row in rows if str(row.get("uid", "") or "") in uid_set]


def _load_freeze_bundle(
    *,
    freeze_run_id: str,
    reports_root: str | Path,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    run_root = Path(reports_root) / freeze_run_id
    manifest = _read_json(run_root / "protocol_freeze_manifest.json")

    if not manifest.get("preflight_passed", False):
        raise ValueError("Step 09A preflight did not pass; Step 13 cannot start.")

    if not manifest.get("live_dryrun_passed", False):
        raise ValueError("Step 09A live dry-run did not pass; Step 13 cannot start.")

    matching_protocol_snapshot = _read_json(
        run_root / "matching_protocol_snapshot.json"
    )

    if (
        str(matching_protocol_snapshot.get("protocol_version", "") or "")
        != FROZEN_MATCHING_PROTOCOL_VERSION
    ):
        raise ValueError("Current matching protocol does not match Step 09A freeze.")

    return (
        manifest,
        _read_json(run_root / "split_refs.json"),
        _read_json(run_root / "gold_refs.json"),
        _read_json(run_root / "budget_grid.json"),
        matching_protocol_snapshot,
    )


def _load_claim1_source_summary(
    *,
    source_claim1_run_id: str,
    reports_root: str | Path,
    freeze_run_id: str,
) -> tuple[dict[str, Any], Path]:
    claim_root = Path(reports_root) / source_claim1_run_id / "claim1"
    summary = _read_json(claim_root / "run_summary.json")

    if int(summary.get("claim_id", 0) or 0) != 1:
        raise ValueError("Step 13 source run must be one Claim 1 run.")

    if str(summary.get("status", "") or "") != "completed":
        raise ValueError("Step 13 source Claim 1 run must be completed.")

    if str(summary.get("freeze_run_id", "") or "") != freeze_run_id:
        raise ValueError(
            "Step 13 freeze_run_id must match source Claim 1 freeze_run_id."
        )

    method_names = {str(name) for name in list(summary.get("method_names", []))}
    missing = [name for name in CLAIM4_METHOD_NAMES if name not in method_names]

    if missing:
        raise ValueError(
            f"Step 13 source Claim 1 run is missing required MAD methods: {missing}"
        )

    return summary, claim_root


def _build_claim4_context(
    *,
    freeze_run_id: str,
    source_claim1_run_id: str,
    run_id: str,
    reports_root: str | Path,
    input_path: str | Path,
) -> Claim4Context:
    manifest, _split_refs, gold_refs, freeze_budget_grid, matching_protocol_snapshot = (
        _load_freeze_bundle(freeze_run_id=freeze_run_id, reports_root=reports_root)
    )

    source_summary, source_claim_root = _load_claim1_source_summary(
        source_claim1_run_id=source_claim1_run_id,
        reports_root=reports_root,
        freeze_run_id=freeze_run_id,
    )

    metric_root = _resolve_metric_root(claim_root=source_claim_root)
    case_map, _ = _load_cases_by_uid(input_path)
    gold_claim_rows = _load_gold_rows(gold_refs["gold_claims_path"])
    gold_status_rows = _load_gold_rows(gold_refs["gold_status_path"])

    gold_uid_set = {
        str(row.get("uid", "") or "").strip()
        for row in gold_claim_rows
        if str(row.get("uid", "") or "").strip()
    }

    stage_uids_by_name: dict[str, list[str]] = {}

    for stage_name, artifact_key in (("dev", "dev_scope"), ("test", "test_scope")):
        stage_uids = [
            uid
            for uid in _read_ids(source_summary["artifacts"][artifact_key])
            if uid in case_map and uid in gold_uid_set
        ]

        if not stage_uids:
            raise ValueError(f"Step 13 {stage_name} scope is empty after filtering.")

        stage_uids_by_name[stage_name] = stage_uids

    budget_points = {
        "q25": dict(freeze_budget_grid.get("budget_points", {}).get("q25", {}) or {}),
        "q50": dict(freeze_budget_grid.get("budget_points", {}).get("q50", {}) or {}),
        "q75": dict(freeze_budget_grid.get("budget_points", {}).get("q75", {}) or {}),
        "full": dict(freeze_budget_grid.get("full_budget", {}) or {}),
    }

    missing_budget_points = [
        point_name
        for point_name in CLAIM4_FIXED_BUDGET_POINTS
        if not budget_points.get(point_name)
    ]

    if missing_budget_points:
        raise ValueError(
            f"Step 13 freeze budget grid is missing points: {missing_budget_points}"
        )

    claim_root = Path(reports_root) / run_id / "claim4"
    claim_root.mkdir(parents=True, exist_ok=True)

    return Claim4Context(
        freeze_run_id=freeze_run_id,
        source_claim1_run_id=source_claim1_run_id,
        run_id=run_id,
        reports_root=Path(reports_root),
        input_path=Path(input_path),
        claim_root=claim_root,
        source_claim_root=source_claim_root,
        metric_root=metric_root,
        manifest_path=claim_root / "manifest.json",
        budget_grid_path=claim_root / "budget_grid.json",
        prereg_points_path=claim_root / "prereg_points.json",
        reference_full_budget_path=claim_root / "reference_full_budget.json",
        noise_audit_path=claim_root / "noise_audit.json",
        method_names=tuple(CLAIM4_METHOD_NAMES),
        stage_uids_by_name=stage_uids_by_name,
        gold_claim_rows=gold_claim_rows,
        gold_status_rows=gold_status_rows,
        matching_config=MatchingConfig(
            **dict(matching_protocol_snapshot.get("base_config", {}) or {})
        ),
        retrieval_config=dict(
            manifest.get("shared_constraints", {}).get("retrieval_config_snapshot", {})
            or {}
        ),
        seed=int(manifest.get("shared_constraints", {}).get("seed", 20260307)),
        full_budget=dict(freeze_budget_grid.get("full_budget", {}) or {}),
        budget_points=budget_points,
    )


def _stage_dir(ctx: Claim4Context, stage: str) -> Path:
    return ctx.claim_root / stage


def _stage_repeats(stage: str) -> tuple[int, ...]:
    if stage == "dev":
        return CLAIM4_DEV_REPEATS

    if stage == "test":
        return CLAIM4_TEST_REPEATS

    raise ValueError(f"Unsupported Claim 4 stage: {stage}")


def _policy_method_names(policy: str) -> tuple[str, ...]:
    if policy == CLAIM4_FIXED_POLICY:
        return CLAIM4_FIXED_METHOD_NAMES

    if policy == CLAIM4_ADAPTIVE_POLICY:
        return CLAIM4_ADAPTIVE_METHOD_NAMES

    raise ValueError(f"Unsupported Claim 4 policy: {policy}")


def _policy_points(policy: str) -> tuple[str, ...]:
    if policy == CLAIM4_FIXED_POLICY:
        return CLAIM4_FIXED_BUDGET_POINTS

    if policy == CLAIM4_ADAPTIVE_POLICY:
        return CLAIM4_ADAPTIVE_POINTS

    raise ValueError(f"Unsupported Claim 4 policy: {policy}")


def _slice_raw_root(
    ctx: Claim4Context,
    *,
    stage: str,
    policy: str,
    point: str,
    repeat: int,
) -> Path:
    return (
        _stage_dir(ctx, stage) / "raw_predictions" / policy / point / f"repeat_{repeat}"
    )


def _slice_metrics_path(
    ctx: Claim4Context,
    *,
    stage: str,
    policy: str,
    point: str,
    repeat: int,
) -> Path:
    return (
        _stage_dir(ctx, stage)
        / "repeat_metrics"
        / policy
        / point
        / f"repeat_{repeat}.json"
    )


def _slice_runtime_path(
    ctx: Claim4Context,
    *,
    stage: str,
    policy: str,
    point: str,
    repeat: int,
) -> Path:
    return (
        _stage_dir(ctx, stage)
        / "repeat_runtime"
        / policy
        / point
        / f"repeat_{repeat}.json"
    )


def _slice_execution_summary_path(
    ctx: Claim4Context,
    *,
    stage: str,
    policy: str,
    point: str,
    repeat: int,
) -> Path:
    return (
        _stage_dir(ctx, stage)
        / "execution_summary"
        / policy
        / point
        / f"repeat_{repeat}.json"
    )


def _slice_runtime_memory_dir(
    ctx: Claim4Context,
    *,
    stage: str,
    policy: str,
    point: str,
    repeat: int,
    method_name: str,
) -> Path:
    return (
        ctx.claim_root
        / "runtime_memory"
        / stage
        / policy
        / point
        / f"repeat_{repeat}"
        / method_name
    )


def _reference_raw_result_path(
    ctx: Claim4Context,
    *,
    stage: str,
    method_name: str,
    uid: str,
) -> Path:
    return (
        ctx.source_claim_root / stage / "raw_predictions" / method_name / f"{uid}.json"
    )


def _job_plan_path(ctx: Claim4Context, stage: str) -> Path:
    return ctx.claim_root / f"job_plan_{stage}.json"


def _flatten_prediction_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    validate_method_result(result)

    status_lookup = {
        str(row.get("claim_id", "") or ""): str(row.get("status_eval", "") or "")
        for row in result["status"]
    }

    flattened: list[dict[str, Any]] = []

    for claim_row in result["claims"]:
        claim_id = str(claim_row.get("claim_id", "") or "")
        status_eval = status_lookup.get(claim_id, "")

        if not status_eval:
            raise ValueError(
                f"Prediction result missing status row for claim_id={claim_id}"
            )

        merged = dict(claim_row)
        merged["status_eval"] = status_eval
        flattened.append(merged)

    return flattened


def _try_load_valid_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        result = _read_json(path)
        validate_method_result(result)
        return result

    except Exception:
        return None


def _is_retryable_task_error(exc: Exception) -> bool:
    message = str(exc).lower()

    retryable_markers = (
        "error code: 503",
        "error code: 429",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "json decode failed",
        "failed under strict json",
        "connection error",
        "missing adjudication claim_ids",
    )

    return any(marker in message for marker in retryable_markers)


def _build_empty_prediction_result(
    *,
    method_name: str,
    case_uid_value: str,
    budget: dict[str, Any],
    retrieval_config: dict[str, Any],
    warning: str,
) -> dict[str, Any]:
    """Build one schema-valid empty prediction result for degraded budget runs."""
    return {
        "method": method_name,
        "case_uid": case_uid_value,
        "claims": [],
        "status": [],
        "trace": {
            "adjudication_mode": "direct_status_json",
            "full_transcript": [],
            "turn_artifacts": [],
            "requested_budget": dict(budget),
            "requested_retrieval_config": dict(retrieval_config),
            "degraded_empty_output": True,
        },
        "warnings": [warning],
        "token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "latency": 0.0,
    }


def _summarize_tokens(results: list[dict[str, Any]]) -> dict[str, float]:
    prompt = [
        float(item.get("token_usage", {}).get("prompt_tokens", 0) or 0)
        for item in results
    ]

    completion = [
        float(item.get("token_usage", {}).get("completion_tokens", 0) or 0)
        for item in results
    ]

    total = [
        float(item.get("token_usage", {}).get("total_tokens", 0) or 0)
        for item in results
    ]

    latency = [float(item.get("latency", 0.0) or 0.0) for item in results]

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "case_count": len(results),
        "mean_prompt_tokens_per_case": _mean(prompt),
        "mean_completion_tokens_per_case": _mean(completion),
        "mean_total_tokens_per_case": _mean(total),
        "mean_latency_seconds_per_case": _mean(latency),
    }


def _build_reference_full_budget(ctx: Claim4Context) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "claim_id": CLAIM4,
        "source_claim1_run_id": ctx.source_claim1_run_id,
        "metric_root": str(ctx.metric_root),
        "primary_metric_key": CLAIM4_PRIMARY_CASE_SCORE_KEY,
        "method_names": list(ctx.method_names),
        "stages": {},
    }

    for stage_name in CLAIM4_STAGES:
        method_metrics = _read_json(
            ctx.metric_root / stage_name / "method_metrics.json"
        )

        stage_rows: dict[str, Any] = {}

        for method_name in ctx.method_names:
            metric = method_metrics[method_name]

            result_dir = (
                ctx.source_claim_root / stage_name / "raw_predictions" / method_name
            )

            token_means = []

            for uid in ctx.stage_uids_by_name[stage_name]:
                result = _read_json(result_dir / f"{uid}.json")

                token_means.append(
                    float(result.get("token_usage", {}).get("total_tokens", 0) or 0)
                )

            stage_rows[method_name] = {
                "primary_metric": dict(
                    metric["main_metrics"][CLAIM4_PRIMARY_CASE_SCORE_KEY]
                ),
                "case_scores": dict(
                    metric["case_scores"][CLAIM4_PRIMARY_CASE_SCORE_KEY]
                ),
                "raw_predictions_dir": str(result_dir),
                "mean_total_tokens_per_case": (
                    float(sum(token_means) / len(token_means)) if token_means else 0.0
                ),
            }

        payload["stages"][stage_name] = {
            "case_uids": list(ctx.stage_uids_by_name[stage_name]),
            "methods": stage_rows,
        }

    _write_json(ctx.reference_full_budget_path, payload)
    return payload


def _write_manifest_and_job_plans(ctx: Claim4Context) -> dict[str, Any]:
    budget_grid = {
        "budget_axis": "max_turns",
        "full_budget": dict(ctx.full_budget),
        "budget_points": {
            name: dict(value) for name, value in ctx.budget_points.items()
        },
    }

    prereg_points = {
        "comparisons": build_prereg_comparisons(CLAIM4_MAD_ONLY_PROFILE),
        "budget_points": list(CLAIM4_FIXED_BUDGET_POINTS),
        "round_points": list(CLAIM4_ROUND_POINTS),
        "temperature": 0,
        "dev_repeats_per_budget_point": len(CLAIM4_DEV_REPEATS),
        "test_repeats_per_budget_point": len(CLAIM4_TEST_REPEATS),
        "registry_profile": CLAIM4_MAD_ONLY_PROFILE,
        "disable_adaptive_termination": True,
        "adaptive_overlay": {
            "methods": list(CLAIM4_ADAPTIVE_METHOD_NAMES),
            "points": list(CLAIM4_ADAPTIVE_POINTS),
            "compare_against": "fixed/full",
        },
    }

    manifest = {
        "claim_id": CLAIM4,
        "protocol_version": CLAIM4_PROTOCOL_VERSION,
        "task_focus": CLAIM4_TASK_FOCUS,
        "run_id": ctx.run_id,
        "freeze_run_id": ctx.freeze_run_id,
        "source_claim1_run_id": ctx.source_claim1_run_id,
        "input_path": str(ctx.input_path),
        "method_names": list(ctx.method_names),
        "fixed_policy_method_names": list(CLAIM4_FIXED_METHOD_NAMES),
        "adaptive_policy_method_names": list(CLAIM4_ADAPTIVE_METHOD_NAMES),
        "registry_profile": CLAIM4_MAD_ONLY_PROFILE,
        "official_stages": list(CLAIM4_OFFICIAL_STAGES),
        "stages": {
            stage_name: list(ctx.stage_uids_by_name[stage_name])
            for stage_name in CLAIM4_STAGES
        },
        "primary_metric_key": CLAIM4_PRIMARY_CASE_SCORE_KEY,
        "budget_grid_path": str(ctx.budget_grid_path),
        "prereg_points_path": str(ctx.prereg_points_path),
        "reference_full_budget_path": str(ctx.reference_full_budget_path),
        "controlled_rerun_root": str(ctx.claim_root / "controlled_rerun"),
        "seed": int(ctx.seed),
        "retrieval_config": dict(ctx.retrieval_config),
        "disable_adaptive_termination": True,
        "policies": {
            CLAIM4_FIXED_POLICY: {
                "points": list(CLAIM4_FIXED_BUDGET_POINTS),
                "methods": list(CLAIM4_FIXED_METHOD_NAMES),
            },
            CLAIM4_ADAPTIVE_POLICY: {
                "points": list(CLAIM4_ADAPTIVE_POINTS),
                "methods": list(CLAIM4_ADAPTIVE_METHOD_NAMES),
            },
        },
    }

    _write_json(ctx.manifest_path, manifest)
    _write_json(ctx.budget_grid_path, budget_grid)
    _write_json(ctx.prereg_points_path, prereg_points)
    stage_job_plans: dict[str, Any] = {}

    for stage_name in CLAIM4_OFFICIAL_STAGES:
        jobs = []
        repeats = _stage_repeats(stage_name)

        for point_name in CLAIM4_FIXED_BUDGET_POINTS:
            for repeat in repeats:
                jobs.append(
                    {
                        "stage": stage_name,
                        "policy": CLAIM4_FIXED_POLICY,
                        "point": point_name,
                        "repeat": repeat,
                        "methods": list(CLAIM4_FIXED_METHOD_NAMES),
                        "case_count": len(ctx.stage_uids_by_name[stage_name]),
                        "budget": {
                            **dict(ctx.budget_points[point_name]),
                            "temperature": 0.0,
                            "disable_adaptive_termination": True,
                        },
                    }
                )

        for point_name in CLAIM4_ADAPTIVE_POINTS:
            for repeat in repeats:
                jobs.append(
                    {
                        "stage": stage_name,
                        "policy": CLAIM4_ADAPTIVE_POLICY,
                        "point": point_name,
                        "repeat": repeat,
                        "methods": list(CLAIM4_ADAPTIVE_METHOD_NAMES),
                        "case_count": len(ctx.stage_uids_by_name[stage_name]),
                        "budget": {
                            **dict(ctx.budget_points[point_name]),
                            "temperature": 0.0,
                        },
                    }
                )

        payload = {
            "claim_id": CLAIM4,
            "run_id": ctx.run_id,
            "stage": stage_name,
            "generated_at": _now_iso(),
            "job_count": len(jobs),
            "jobs": jobs,
        }

        _write_json(_job_plan_path(ctx, stage_name), payload)
        stage_job_plans[stage_name] = payload

    return {
        "manifest": manifest,
        "budget_grid": budget_grid,
        "prereg_points": prereg_points,
        "stage_job_plans": stage_job_plans,
    }


def prepare_claim4_experiment(
    *,
    freeze_run_id: str,
    source_claim1_run_id: str,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    input_path: str | Path = DEFAULT_INPUT_PATH,
) -> dict[str, Any]:
    """Write manifests, budget grids, and reference anchors for Claim 4.

    Args:
        freeze_run_id: Step 09A freeze bundle that defines the shared protocol.
        source_claim1_run_id: Completed Claim 1 run reused as the full-budget anchor.
        run_id: Destination Step 13 run identifier.
        reports_root: Root directory that stores experiment outputs.
        input_path: Original case JSONL used to resolve evaluation cases.

    Returns:
        A preparation summary with manifest, budget grid, and anchor artifacts.

    Raises:
        FileNotFoundError: If prerequisite Claim 1 artifacts are missing.
        ValueError: If the frozen protocol or source run is incompatible.
    """

    ctx = _build_claim4_context(
        freeze_run_id=freeze_run_id,
        source_claim1_run_id=source_claim1_run_id,
        run_id=run_id,
        reports_root=reports_root,
        input_path=input_path,
    )

    reference_payload = _build_reference_full_budget(ctx)
    prepared_payload = _write_manifest_and_job_plans(ctx)
    (ctx.claim_root / "controlled_rerun" / "dev").mkdir(parents=True, exist_ok=True)
    (ctx.claim_root / "controlled_rerun" / "test").mkdir(parents=True, exist_ok=True)

    summary = {
        "claim_id": CLAIM4,
        "status": "prepared",
        "run_id": run_id,
        "freeze_run_id": freeze_run_id,
        "source_claim1_run_id": source_claim1_run_id,
        "method_names": list(ctx.method_names),
        "official_stages": list(CLAIM4_OFFICIAL_STAGES),
        "stages": {
            stage_name: {
                "case_count": len(ctx.stage_uids_by_name[stage_name]),
                "job_plan": str(_job_plan_path(ctx, stage_name)),
            }
            for stage_name in CLAIM4_OFFICIAL_STAGES
        },
        "artifacts": {
            "manifest": str(ctx.manifest_path),
            "budget_grid": str(ctx.budget_grid_path),
            "prereg_points": str(ctx.prereg_points_path),
            "reference_full_budget": str(ctx.reference_full_budget_path),
        },
        "reference_full_budget": reference_payload,
        "prepared": prepared_payload,
    }

    _write_json(ctx.claim_root / "run_summary.json", summary)
    return summary


def _build_primary_score_payload(metric: dict[str, Any]) -> dict[str, Any]:
    return dict(metric["main_metrics"][CLAIM4_PRIMARY_CASE_SCORE_KEY])


def _run_claim4_slice(
    *,
    ctx: Claim4Context,
    stage: str,
    policy: str,
    point: str,
    repeat: int,
    resume: bool,
    verbose: bool,
    show_progress: bool,
) -> dict[str, Any]:
    if stage not in CLAIM4_STAGES:
        raise ValueError(f"Unsupported Claim 4 stage: {stage}")

    if policy not in {CLAIM4_FIXED_POLICY, CLAIM4_ADAPTIVE_POLICY}:
        raise ValueError(f"Unsupported Claim 4 policy: {policy}")

    if point not in _policy_points(policy):
        raise ValueError(f"Unsupported Claim 4 budget point: {point}")

    if repeat not in _stage_repeats(stage):
        raise ValueError(f"Unsupported Claim 4 repeat: {repeat}")

    case_map, _ = _load_cases_by_uid(ctx.input_path)
    case_uids = list(ctx.stage_uids_by_name[stage])
    cases = [dict(case_map[uid]) for uid in case_uids]
    registry = build_method_registry(CLAIM4_MAD_ONLY_PROFILE)
    method_names = tuple(_policy_method_names(policy))
    slice_budget = dict(ctx.budget_points[point])
    slice_budget["temperature"] = 0.0

    if policy == CLAIM4_FIXED_POLICY:
        slice_budget["disable_adaptive_termination"] = True

    summary_path = _slice_execution_summary_path(
        ctx, stage=stage, policy=policy, point=point, repeat=repeat
    )

    raw_root = _slice_raw_root(
        ctx, stage=stage, policy=policy, point=point, repeat=repeat
    )

    validation_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    retry_events: list[dict[str, Any]] = []

    results_by_method: dict[str, list[dict[str, Any]]] = {
        method_name: [] for method_name in method_names
    }

    flattened_predictions: dict[str, list[dict[str, Any]]] = {
        method_name: [] for method_name in method_names
    }

    summary: dict[str, Any] = {
        "claim_id": CLAIM4,
        "run_id": ctx.run_id,
        "stage": stage,
        "policy": policy,
        "point": point,
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
    }

    _write_json(summary_path, summary)

    progress = tqdm(
        total=summary["total_task_count"],
        desc=f"Claim4 {stage} {policy} {point} r{repeat}",
        unit="task",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    try:
        for method_name in method_names:
            runner = registry[method_name]

            storage_root_dir = _slice_runtime_memory_dir(
                ctx,
                stage=stage,
                policy=policy,
                point=point,
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
        raise Claim4ExecutionError(
            f"Claim 4 {stage} {policy} {point} repeat {repeat} failed for {len(failures)} task(s)."
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
        "run_id": ctx.run_id,
        "stage": stage,
        "policy": policy,
        "point": point,
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
        "run_id": ctx.run_id,
        "stage": stage,
        "policy": policy,
        "point": point,
        "repeat": repeat,
        "methods": runtime_metrics,
    }

    _write_json(
        _slice_metrics_path(
            ctx, stage=stage, policy=policy, point=point, repeat=repeat
        ),
        repeat_metrics_payload,
    )

    _write_json(
        _slice_runtime_path(
            ctx, stage=stage, policy=policy, point=point, repeat=repeat
        ),
        runtime_payload,
    )

    return {
        "execution_summary": summary,
        "repeat_metrics": repeat_metrics_payload,
        "runtime_metrics": runtime_payload,
    }


def run_claim4_slice(
    *,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    stage: str,
    policy: str,
    point: str,
    repeat: int,
    resume: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Execute one Claim 4 slice for a single stage, policy, point, and repeat.

    Args:
        run_id: Prepared Claim 4 run identifier.
        reports_root: Root directory that stores experiment outputs.
        stage: Split name to evaluate, such as `dev` or `test`.
        policy: Budget policy, either fixed or adaptive.
        point: Named budget point such as `q25`, `q50`, `q75`, or `full`.
        repeat: Repeat index within the selected stage.
        resume: Whether to reuse already-written case outputs.
        verbose: Whether to enable verbose method execution logs.
        show_progress: Whether to render progress bars.

    Returns:
        A payload containing execution, scoring, and runtime summaries.

    Raises:
        FileNotFoundError: If prepared Claim 4 artifacts are missing.
        ValueError: If the stage, policy, point, or repeat is unsupported.
        Claim4ExecutionError: If one or more case tasks fail irrecoverably.
    """

    manifest = _read_json(Path(reports_root) / run_id / "claim4" / "manifest.json")

    ctx = _build_claim4_context(
        freeze_run_id=str(manifest["freeze_run_id"]),
        source_claim1_run_id=str(manifest["source_claim1_run_id"]),
        run_id=run_id,
        reports_root=reports_root,
        input_path=str(manifest["input_path"]),
    )

    return _run_claim4_slice(
        ctx=ctx,
        stage=stage,
        policy=policy,
        point=point,
        repeat=repeat,
        resume=resume,
        verbose=verbose,
        show_progress=show_progress,
    )


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0

    return float(statistics.stdev(values))


def _load_or_init_noise_audit(path: Path) -> dict[str, Any]:
    if path.exists():
        return _read_json(path)

    return {
        "claim_id": CLAIM4,
        "stages": {},
        "updated_at": _now_iso(),
    }


def audit_claim4_stage(
    *,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    stage: str,
) -> dict[str, Any]:
    """Audit one Claim 4 stage for completeness and repeat-level noise.

    Args:
        run_id: Prepared Claim 4 run identifier.
        reports_root: Root directory that stores experiment outputs.
        stage: Stage name whose completed slices should be audited.

    Returns:
        A stage-level audit payload with pass/fail status per policy and point.

    Raises:
        FileNotFoundError: If required slice metrics are missing.
        ValueError: If the requested stage is unsupported.
    """

    manifest = _read_json(Path(reports_root) / run_id / "claim4" / "manifest.json")

    ctx = _build_claim4_context(
        freeze_run_id=str(manifest["freeze_run_id"]),
        source_claim1_run_id=str(manifest["source_claim1_run_id"]),
        run_id=run_id,
        reports_root=reports_root,
        input_path=str(manifest["input_path"]),
    )

    if stage not in CLAIM4_STAGES:
        raise ValueError(f"Unsupported Claim 4 stage: {stage}")

    fixed_points: list[dict[str, Any]] = []
    adaptive_points: list[dict[str, Any]] = []
    fixed_passed = True
    adaptive_passed = True

    for point_name in CLAIM4_FIXED_BUDGET_POINTS:
        repeats = _stage_repeats(stage)

        repeat_payloads = [
            _read_json(
                _slice_metrics_path(
                    ctx,
                    stage=stage,
                    policy=CLAIM4_FIXED_POLICY,
                    point=point_name,
                    repeat=repeat,
                )
            )
            for repeat in repeats
        ]

        method_rows: dict[str, Any] = {}

        for method_name in CLAIM4_FIXED_METHOD_NAMES:
            repeat_scores = [
                float(
                    payload["methods"][method_name]["primary_metric"]["point_estimate"]
                )
                for payload in repeat_payloads
            ]

            method_rows[method_name] = {
                "repeat_scores": repeat_scores,
                "mean_primary_score": float(sum(repeat_scores) / len(repeat_scores)),
                "std_primary_score": _std(repeat_scores),
            }

        passed = True
        failure_reason = ""
        min_gap = None
        max_std = max(row["std_primary_score"] for row in method_rows.values())

        if stage == "dev":
            pairwise_gaps = [
                abs(
                    method_rows[left]["mean_primary_score"]
                    - method_rows[right]["mean_primary_score"]
                )
                for idx, left in enumerate(CLAIM4_FIXED_METHOD_NAMES)
                for right in CLAIM4_FIXED_METHOD_NAMES[idx + 1 :]
            ]

            min_gap = min(pairwise_gaps) if pairwise_gaps else 0.0
            passed = bool(min_gap > 0.0 and max_std <= (min_gap / 3.0))

            if min_gap == 0.0:
                passed = False
                failure_reason = "zero_effect_gap"

            elif not passed:
                failure_reason = "noise_exceeds_one_third_gap"

        fixed_points.append(
            {
                "point": point_name,
                "budget": dict(ctx.budget_points[point_name]),
                "methods": method_rows,
                "min_gap": None if min_gap is None else float(min_gap),
                "max_std": float(max_std),
                "passed": passed,
                "failure_reason": failure_reason,
                "expected_repeats": list(repeats),
            }
        )

        fixed_passed = fixed_passed and passed

    for point_name in CLAIM4_ADAPTIVE_POINTS:
        repeats = _stage_repeats(stage)

        repeat_payloads = [
            _read_json(
                _slice_metrics_path(
                    ctx,
                    stage=stage,
                    policy=CLAIM4_ADAPTIVE_POLICY,
                    point=point_name,
                    repeat=repeat,
                )
            )
            for repeat in repeats
        ]

        runtime_payloads = [
            _read_json(
                _slice_runtime_path(
                    ctx,
                    stage=stage,
                    policy=CLAIM4_ADAPTIVE_POLICY,
                    point=point_name,
                    repeat=repeat,
                )
            )
            for repeat in repeats
        ]

        repeat_scores = [
            float(payload["methods"]["main_system"]["primary_metric"]["point_estimate"])
            for payload in repeat_payloads
        ]

        realized_turns = []

        for repeat in repeats:
            case_turn_counts = []

            for uid in ctx.stage_uids_by_name[stage]:
                result = _read_json(
                    _slice_raw_root(
                        ctx,
                        stage=stage,
                        policy=CLAIM4_ADAPTIVE_POLICY,
                        point=point_name,
                        repeat=repeat,
                    )
                    / "main_system"
                    / f"{uid}.json"
                )

                case_turn_counts.append(
                    float(len(result.get("trace", {}).get("turn_artifacts", []) or []))
                )

            realized_turns.append(_mean(case_turn_counts))

        adaptive_points.append(
            {
                "point": point_name,
                "budget": dict(ctx.budget_points[point_name]),
                "methods": {
                    "main_system": {
                        "repeat_scores": repeat_scores,
                        "mean_primary_score": _mean(repeat_scores),
                        "std_primary_score": _std(repeat_scores),
                        "repeat_mean_total_tokens_per_case": [
                            float(
                                payload["methods"]["main_system"][
                                    "mean_total_tokens_per_case"
                                ]
                            )
                            for payload in runtime_payloads
                        ],
                        "repeat_mean_realized_turns_per_case": realized_turns,
                    }
                },
                "passed": True,
                "failure_reason": "",
                "expected_repeats": list(repeats),
            }
        )

    stage_payload = {
        "claim_id": CLAIM4,
        "run_id": run_id,
        "stage": stage,
        "soft_fail_mode": stage == "dev",
        "fixed_policy": {
            "passed": fixed_passed,
            "points": fixed_points,
        },
        "adaptive_policy": {
            "passed": adaptive_passed,
            "points": adaptive_points,
        },
        "passed": fixed_passed and adaptive_passed,
        "updated_at": _now_iso(),
    }

    audit_payload = _load_or_init_noise_audit(ctx.noise_audit_path)
    audit_payload["claim_id"] = CLAIM4
    audit_payload["run_id"] = run_id
    audit_payload.setdefault("stages", {})
    audit_payload["stages"][stage] = stage_payload
    audit_payload["updated_at"] = _now_iso()
    _write_json(ctx.noise_audit_path, audit_payload)
    return stage_payload


def _status_map_for_result(
    result: dict[str, Any],
    *,
    claim_ids: list[str],
) -> dict[str, str]:
    lookup = {
        str(row.get("claim_id", "") or ""): normalize_status_eval(
            str(row.get("status_eval", "") or "")
        )
        for row in result.get("status", [])
        if str(row.get("claim_id", "") or "").strip()
    }

    return {claim_id: lookup.get(claim_id, "HYPOTHETICAL") for claim_id in claim_ids}


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _build_pareto_rows(
    *,
    stage: str,
    ctx: Claim4Context,
    noise_stage: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for point_row in noise_stage["fixed_policy"]["points"]:
        point_name = str(point_row["point"])
        passed = bool(point_row["passed"])
        repeats = _stage_repeats(stage)

        repeat_metrics = [
            _read_json(
                _slice_metrics_path(
                    ctx,
                    stage=stage,
                    policy=CLAIM4_FIXED_POLICY,
                    point=point_name,
                    repeat=repeat,
                )
            )
            for repeat in repeats
        ]

        repeat_runtime = [
            _read_json(
                _slice_runtime_path(
                    ctx,
                    stage=stage,
                    policy=CLAIM4_FIXED_POLICY,
                    point=point_name,
                    repeat=repeat,
                )
            )
            for repeat in repeats
        ]

        for method_name in CLAIM4_FIXED_METHOD_NAMES:
            rows.append(
                {
                    "stage": stage,
                    "policy": CLAIM4_FIXED_POLICY,
                    "method_name": method_name,
                    "point": point_name,
                    "budget_max_turns": int(ctx.budget_points[point_name]["max_turns"]),
                    "mean_primary_score": _mean(
                        [
                            float(
                                payload["methods"][method_name]["primary_metric"][
                                    "point_estimate"
                                ]
                            )
                            for payload in repeat_metrics
                        ]
                    ),
                    "mean_total_tokens_per_case": _mean(
                        [
                            float(
                                payload["methods"][method_name][
                                    "mean_total_tokens_per_case"
                                ]
                            )
                            for payload in repeat_runtime
                        ]
                    ),
                    "passed_noise_gate": passed,
                    "is_pareto_frontier": False,
                }
            )

    eligible = [row for row in rows if bool(row["passed_noise_gate"])]

    for row in eligible:
        dominated = False

        for other in eligible:
            if other is row:
                continue

            better_or_equal = float(other["mean_total_tokens_per_case"]) <= float(
                row["mean_total_tokens_per_case"]
            ) and float(other["mean_primary_score"]) >= float(row["mean_primary_score"])

            strictly_better = float(other["mean_total_tokens_per_case"]) < float(
                row["mean_total_tokens_per_case"]
            ) or float(other["mean_primary_score"]) > float(row["mean_primary_score"])

            if better_or_equal and strictly_better:
                dominated = True
                break

        row["is_pareto_frontier"] = not dominated

    return rows


def summarize_claim4_experiment(
    *,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
) -> dict[str, Any]:
    """Summarize Claim 4 diagnostics from completed slice outputs.

    Args:
        run_id: Prepared Claim 4 run identifier.
        reports_root: Root directory that stores experiment outputs.

    Returns:
        A dev-only diagnostic summary with delta tables and overlay artifacts.

    Raises:
        FileNotFoundError: If prepared Claim 4 artifacts are missing.
        ValueError: If required dev-stage audits have not been completed.
    """

    manifest = _read_json(Path(reports_root) / run_id / "claim4" / "manifest.json")

    ctx = _build_claim4_context(
        freeze_run_id=str(manifest["freeze_run_id"]),
        source_claim1_run_id=str(manifest["source_claim1_run_id"]),
        run_id=run_id,
        reports_root=reports_root,
        input_path=str(manifest["input_path"]),
    )

    noise_audit = _read_json(ctx.noise_audit_path)

    available_stage_names = [
        stage_name
        for stage_name in CLAIM4_OFFICIAL_STAGES
        if stage_name in noise_audit.get("stages", {})
    ]

    if not available_stage_names:
        raise ValueError("Missing Claim 4 audit for official dev stage.")

    gold_claim_ids_by_uid: dict[str, list[str]] = {}

    for row in ctx.gold_claim_rows:
        uid = str(row.get("uid", "") or "")
        claim_id = str(row.get("claim_id", "") or "")

        if uid and claim_id:
            gold_claim_ids_by_uid.setdefault(uid, []).append(claim_id)

    delta_root_flip_rows: list[dict[str, Any]] = []
    delta_err_rows: list[dict[str, Any]] = []
    delta_conflict_rows: list[dict[str, Any]] = []
    delta_phi_rows: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    adaptive_overlay_rows: list[dict[str, Any]] = []
    budget_result_cache: dict[tuple[str, str, str, int, str, str], dict[str, Any]] = {}

    def _get_budget_result(
        stage: str,
        policy: str,
        point: str,
        repeat: int,
        method_name: str,
        uid: str,
    ) -> dict[str, Any]:
        key = (stage, policy, point, repeat, method_name, uid)

        if key not in budget_result_cache:
            budget_result_cache[key] = _read_json(
                _slice_raw_root(
                    ctx,
                    stage=stage,
                    policy=policy,
                    point=point,
                    repeat=repeat,
                )
                / method_name
                / f"{uid}.json"
            )

        return budget_result_cache[key]

    for stage_name in available_stage_names:
        stage_uids = list(ctx.stage_uids_by_name[stage_name])
        repeats = _stage_repeats(stage_name)

        pareto_rows.extend(
            _build_pareto_rows(
                stage=stage_name,
                ctx=ctx,
                noise_stage=noise_audit["stages"][stage_name],
            )
        )

        full_metrics_by_repeat = {
            repeat: _read_json(
                _slice_metrics_path(
                    ctx,
                    stage=stage_name,
                    policy=CLAIM4_FIXED_POLICY,
                    point="full",
                    repeat=repeat,
                )
            )
            for repeat in repeats
        }

        full_runtime_by_repeat = {
            repeat: _read_json(
                _slice_runtime_path(
                    ctx,
                    stage=stage_name,
                    policy=CLAIM4_FIXED_POLICY,
                    point="full",
                    repeat=repeat,
                )
            )
            for repeat in repeats
        }

        for point_name in CLAIM4_FIXED_BUDGET_POINTS:
            for repeat in repeats:
                repeat_metrics = _read_json(
                    _slice_metrics_path(
                        ctx,
                        stage=stage_name,
                        policy=CLAIM4_FIXED_POLICY,
                        point=point_name,
                        repeat=repeat,
                    )
                )

                full_repeat_metrics = full_metrics_by_repeat[repeat]

                for method_name in CLAIM4_FIXED_METHOD_NAMES:
                    budget_case_scores = repeat_metrics["methods"][method_name][
                        "case_scores"
                    ]

                    full_case_scores = full_repeat_metrics["methods"][method_name][
                        "case_scores"
                    ]

                    for uid in stage_uids:
                        claim_ids = list(gold_claim_ids_by_uid.get(uid, []))

                        budget_result = _get_budget_result(
                            stage_name,
                            CLAIM4_FIXED_POLICY,
                            point_name,
                            repeat,
                            method_name,
                            uid,
                        )

                        full_result = _get_budget_result(
                            stage_name,
                            CLAIM4_FIXED_POLICY,
                            "full",
                            repeat,
                            method_name,
                            uid,
                        )

                        full_statuses = _status_map_for_result(
                            full_result,
                            claim_ids=claim_ids,
                        )

                        budget_statuses = _status_map_for_result(
                            budget_result,
                            claim_ids=claim_ids,
                        )

                        flips = [
                            1.0
                            if budget_statuses[claim_id] != full_statuses[claim_id]
                            else 0.0
                            for claim_id in claim_ids
                        ]

                        delta_root_flip_rows.append(
                            {
                                "stage": stage_name,
                                "policy": CLAIM4_FIXED_POLICY,
                                "method_name": method_name,
                                "point": point_name,
                                "anchor_point": "full",
                                "repeat": repeat,
                                "case_uid": uid,
                                "delta_root_flip": _mean(flips),
                            }
                        )

                        full_score = float(full_case_scores[uid])
                        budget_score = float(budget_case_scores[uid])

                        delta_err_rows.append(
                            {
                                "stage": stage_name,
                                "policy": CLAIM4_FIXED_POLICY,
                                "method_name": method_name,
                                "point": point_name,
                                "anchor_point": "full",
                                "repeat": repeat,
                                "case_uid": uid,
                                "full_fixed_score": full_score,
                                "budget_point_score": budget_score,
                                "delta_err": full_score - budget_score,
                            }
                        )

                        budget_parse = parse_method_result(budget_result)
                        full_parse = parse_method_result(full_result)
                        has_conflict_budget = 1 if budget_parse.conflict_edges else 0
                        has_conflict_full = 1 if full_parse.conflict_edges else 0

                        delta_conflict_rows.append(
                            {
                                "stage": stage_name,
                                "policy": CLAIM4_FIXED_POLICY,
                                "method_name": method_name,
                                "point": point_name,
                                "anchor_point": "full",
                                "repeat": repeat,
                                "case_uid": uid,
                                "has_conflict_budget": has_conflict_budget,
                                "has_conflict_full_fixed": has_conflict_full,
                                "delta_conflict": has_conflict_budget
                                - has_conflict_full,
                            }
                        )

                        turn_artifacts = list(
                            budget_result.get("trace", {}).get("turn_artifacts", [])
                            or []
                        )

                        for turn_idx, artifact in enumerate(turn_artifacts, start=1):
                            if turn_idx not in CLAIM4_ROUND_POINTS:
                                continue

                            convergence = dict(artifact.get("convergence", {}) or {})

                            delta_phi_rows.append(
                                {
                                    "stage": stage_name,
                                    "policy": CLAIM4_FIXED_POLICY,
                                    "method_name": method_name,
                                    "point": point_name,
                                    "repeat": repeat,
                                    "turn": turn_idx,
                                    "case_uid": uid,
                                    "delta_phi": float(
                                        convergence.get("delta_phi", 0.0)
                                    ),
                                    "sma": float(convergence.get("sma", 0.0)),
                                    "epsilon": float(convergence.get("epsilon", 0.0)),
                                }
                            )

        for repeat in repeats:
            adaptive_metrics = _read_json(
                _slice_metrics_path(
                    ctx,
                    stage=stage_name,
                    policy=CLAIM4_ADAPTIVE_POLICY,
                    point="full",
                    repeat=repeat,
                )
            )

            adaptive_runtime = _read_json(
                _slice_runtime_path(
                    ctx,
                    stage=stage_name,
                    policy=CLAIM4_ADAPTIVE_POLICY,
                    point="full",
                    repeat=repeat,
                )
            )

            fixed_metrics = full_metrics_by_repeat[repeat]
            fixed_runtime = full_runtime_by_repeat[repeat]
            adaptive_turn_counts = []
            fixed_turn_counts = []

            for uid in stage_uids:
                adaptive_result = _get_budget_result(
                    stage_name,
                    CLAIM4_ADAPTIVE_POLICY,
                    "full",
                    repeat,
                    "main_system",
                    uid,
                )

                fixed_result = _get_budget_result(
                    stage_name,
                    CLAIM4_FIXED_POLICY,
                    "full",
                    repeat,
                    "main_system",
                    uid,
                )

                adaptive_turn_counts.append(
                    float(
                        len(
                            adaptive_result.get("trace", {}).get("turn_artifacts", [])
                            or []
                        )
                    )
                )

                fixed_turn_counts.append(
                    float(
                        len(
                            fixed_result.get("trace", {}).get("turn_artifacts", [])
                            or []
                        )
                    )
                )

            adaptive_score = float(
                adaptive_metrics["methods"]["main_system"]["primary_metric"][
                    "point_estimate"
                ]
            )

            fixed_score = float(
                fixed_metrics["methods"]["main_system"]["primary_metric"][
                    "point_estimate"
                ]
            )

            adaptive_tokens = float(
                adaptive_runtime["methods"]["main_system"]["mean_total_tokens_per_case"]
            )

            fixed_tokens = float(
                fixed_runtime["methods"]["main_system"]["mean_total_tokens_per_case"]
            )

            adaptive_turns = _mean(adaptive_turn_counts)
            fixed_turns = _mean(fixed_turn_counts)

            adaptive_overlay_rows.append(
                {
                    "stage": stage_name,
                    "method_name": "main_system",
                    "point": "full",
                    "repeat": repeat,
                    "adaptive_primary_score": adaptive_score,
                    "fixed_primary_score": fixed_score,
                    "delta_primary_score_vs_fixed": adaptive_score - fixed_score,
                    "adaptive_mean_total_tokens_per_case": adaptive_tokens,
                    "fixed_mean_total_tokens_per_case": fixed_tokens,
                    "delta_tokens_vs_fixed": adaptive_tokens - fixed_tokens,
                    "adaptive_mean_realized_turns_per_case": adaptive_turns,
                    "fixed_mean_realized_turns_per_case": fixed_turns,
                    "delta_realized_turns_vs_fixed": adaptive_turns - fixed_turns,
                }
            )

    delta_phi_agg_rows: list[dict[str, Any]] = []
    phi_groups: dict[tuple[str, str, str, int, int], list[dict[str, Any]]] = {}

    for row in delta_phi_rows:
        key = (
            str(row["stage"]),
            str(row["policy"]),
            str(row["method_name"]),
            str(row["point"]),
            int(row["repeat"]),
            int(row["turn"]),
        )

        phi_groups.setdefault(key, []).append(row)

    for key in sorted(phi_groups):
        grouped = phi_groups[key]
        stage_name, policy, method_name, point_name, repeat, turn = key

        delta_phi_agg_rows.append(
            {
                "stage": stage_name,
                "policy": policy,
                "method_name": method_name,
                "point": point_name,
                "repeat": repeat,
                "turn": turn,
                "case_count": len(grouped),
                "mean_delta_phi": _mean([float(item["delta_phi"]) for item in grouped]),
                "mean_sma": _mean([float(item["sma"]) for item in grouped]),
                "mean_epsilon": _mean([float(item["epsilon"]) for item in grouped]),
            }
        )

    _write_csv(
        ctx.claim_root / "delta_root_flip.csv",
        [
            "stage",
            "policy",
            "method_name",
            "point",
            "anchor_point",
            "repeat",
            "case_uid",
            "delta_root_flip",
        ],
        delta_root_flip_rows,
    )

    _write_csv(
        ctx.claim_root / "delta_err.csv",
        [
            "stage",
            "policy",
            "method_name",
            "point",
            "anchor_point",
            "repeat",
            "case_uid",
            "full_fixed_score",
            "budget_point_score",
            "delta_err",
        ],
        delta_err_rows,
    )

    _write_csv(
        ctx.claim_root / "delta_conflict.csv",
        [
            "stage",
            "policy",
            "method_name",
            "point",
            "anchor_point",
            "repeat",
            "case_uid",
            "has_conflict_budget",
            "has_conflict_full_fixed",
            "delta_conflict",
        ],
        delta_conflict_rows,
    )

    _write_csv(
        ctx.claim_root / "delta_phi_curve.csv",
        [
            "stage",
            "policy",
            "method_name",
            "point",
            "repeat",
            "turn",
            "case_count",
            "mean_delta_phi",
            "mean_sma",
            "mean_epsilon",
        ],
        delta_phi_agg_rows,
    )

    _write_csv(
        ctx.claim_root / "pareto_points.csv",
        [
            "stage",
            "policy",
            "method_name",
            "point",
            "budget_max_turns",
            "mean_primary_score",
            "mean_total_tokens_per_case",
            "passed_noise_gate",
            "is_pareto_frontier",
        ],
        pareto_rows,
    )

    _write_csv(
        ctx.claim_root / "adaptive_overlay.csv",
        [
            "stage",
            "method_name",
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
        adaptive_overlay_rows,
    )

    summary = {
        "claim_id": CLAIM4,
        "run_id": run_id,
        "status": (
            "completed"
            if bool(noise_audit["stages"]["dev"]["fixed_policy"]["passed"])
            else "completed_with_warnings"
        ),
        "protocol_version": CLAIM4_PROTOCOL_VERSION,
        "task_focus": CLAIM4_TASK_FOCUS,
        "method_names": list(ctx.method_names),
        "official_stages": list(available_stage_names),
        "soft_fail_mode": True,
        "primary_metric_key": CLAIM4_PRIMARY_CASE_SCORE_KEY,
        "noise_audit": noise_audit,
        "artifacts": {
            "manifest": str(ctx.manifest_path),
            "budget_grid": str(ctx.budget_grid_path),
            "prereg_points": str(ctx.prereg_points_path),
            "reference_full_budget": str(ctx.reference_full_budget_path),
            "delta_root_flip_csv": str(ctx.claim_root / "delta_root_flip.csv"),
            "delta_phi_curve_csv": str(ctx.claim_root / "delta_phi_curve.csv"),
            "delta_err_csv": str(ctx.claim_root / "delta_err.csv"),
            "delta_conflict_csv": str(ctx.claim_root / "delta_conflict.csv"),
            "pareto_points_csv": str(ctx.claim_root / "pareto_points.csv"),
            "adaptive_overlay_csv": str(ctx.claim_root / "adaptive_overlay.csv"),
            "noise_audit_json": str(ctx.noise_audit_path),
        },
    }

    _write_json(ctx.claim_root / "run_summary.json", summary)
    return summary
