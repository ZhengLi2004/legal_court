"""Claim 3 experiment runner for Step 12."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
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
from benchmarks.experiments.eval.metrics_claim3 import summarize_claim3_curve
from benchmarks.experiments.methods.base import case_uid, validate_method_result
from benchmarks.experiments.methods.factory import build_default_registry
from mas.config import SystemConfig
from mas.infrastructure.evidence_pack_provider import (
    build_fixed_evidence_pack_map_for_contexts,
    extract_fixed_evidence_pack_for_uid,
    read_fixed_evidence_pack_map,
    write_fixed_evidence_pack_map,
)

CLAIM3 = 3
CLAIM3_PROTOCOL_VERSION = "step12_claim3_v1"
CLAIM3_TASK_FOCUS = "silent_warmup_frozen_test"
CLAIM3_MAIN_METHOD = "main_system"
CLAIM3_PRIMARY_CASE_SCORE_KEY = "e2e_status_acc"
CLAIM3_FIXED_PACK_REFERENCE_POINT = 0
CLAIM3_FIXED_PACK_POINTS = (0, 100)
CLAIM3_DEFAULT_WARMUP_POINTS = (0, 25, 50, 100)


class Claim3ExecutionError(RuntimeError):
    """Raised when Claim 3 execution cannot continue.

    Attributes:
        args: Standard runtime-error payload describing the failure.
    """


@dataclass(frozen=True)
class Claim3Context:
    """Frozen inputs, paths, and protocol state for one Claim 3 run.

    Attributes:
        freeze_run_id: Step 09A freeze bundle used to seed the run.
        source_claim1_run_id: Claim 1 run reused as the reference source.
        run_id: Destination Step 12 run identifier.
        reports_root: Root directory storing all experiment reports.
        input_path: Original case JSONL used for warmup and evaluation cases.
        claim_root: Root output directory for Claim 3 artifacts.
        manifest_path: Path to the warmup manifest written by prepare.
        job_plan_path: Path to the generated execution plan.
        fixed_pack_path: Path to the serialized fixed evidence pack.
        reference_anchor_path: Path to reference Claim 1 anchor metrics.
        warmup_points: Warmup prefixes evaluated by the experiment.
        fixed_pack_points: Warmup prefixes evaluated by the fixed-pack branch.
        eval_uids: Ordered evaluation case ids.
        warmup_pool_uids: Ordered pool of warmup case ids.
        gold_claim_rows: Gold claim rows restricted to the source task universe.
        gold_status_rows: Gold status rows restricted to the source task universe.
        matching_config: Frozen matching configuration reused from Claim 1.
        budget: Full-budget configuration reused by Claim 3.
        retrieval_config: Frozen retrieval config snapshot reused by the runner.
        seed: Shared deterministic seed for repeated execution.
    """

    freeze_run_id: str
    source_claim1_run_id: str
    run_id: str
    reports_root: Path
    input_path: Path
    claim_root: Path
    manifest_path: Path
    job_plan_path: Path
    fixed_pack_path: Path
    reference_anchor_path: Path
    warmup_points: tuple[int, ...]
    fixed_pack_points: tuple[int, ...]
    eval_uids: list[str]
    warmup_pool_uids: list[str]
    gold_claim_rows: list[dict[str, Any]]
    gold_status_rows: list[dict[str, Any]]
    matching_config: MatchingConfig
    budget: dict[str, Any]
    retrieval_config: dict[str, Any]
    seed: int


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


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("CSV rows must be non-empty.")

    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_ids(path: str | Path, key: str = "ids") -> list[str]:
    payload = _read_json(path)
    values = payload.get(key, [])

    if not isinstance(values, list) or not values:
        raise ValueError(f"{Path(path).name} must contain a non-empty `{key}` list.")

    return [str(value) for value in values]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _hash_directory(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()

    if not root.exists():
        digest.update(b"missing")
        return digest.hexdigest()

    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(root)).encode("utf-8"))
        digest.update(_sha256_file(file_path).encode("utf-8"))

    return digest.hexdigest()


def _clear_dir(path: str | Path) -> None:
    target = Path(path)

    if target.exists():
        shutil.rmtree(target)


def _copy_tree(src: str | Path, dst: str | Path) -> None:
    source = Path(src)

    if not source.exists():
        raise FileNotFoundError(f"Missing source directory: {source}")

    _clear_dir(dst)
    shutil.copytree(source, dst)


def _load_cases_by_uid(
    input_path: str | Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    ordered_cases = load_cases_from_jsonl(input_path)
    order = [case_uid(case) for case in ordered_cases]
    return {case_uid(case): case for case in ordered_cases}, order


def _load_gold_rows(path: str | Path) -> list[dict[str, Any]]:
    return [dict(row) for row in load_cases_from_jsonl(path)]


def _load_freeze_bundle(
    *,
    freeze_run_id: str,
    reports_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_root = Path(reports_root) / freeze_run_id
    manifest = _read_json(run_root / "protocol_freeze_manifest.json")

    if not manifest.get("preflight_passed", False):
        raise ValueError("Step 09A preflight did not pass; Step 12 cannot start.")

    if not manifest.get("live_dryrun_passed", False):
        raise ValueError("Step 09A live dry-run did not pass; Step 12 cannot start.")

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
        raise ValueError("Step 12 source run must be one Claim 1 run.")

    if str(summary.get("status", "") or "") != "completed":
        raise ValueError("Step 12 source Claim 1 run must be completed.")

    if str(summary.get("freeze_run_id", "") or "") != freeze_run_id:
        raise ValueError(
            "Step 12 freeze_run_id must match source Claim 1 freeze_run_id."
        )

    method_names = [str(name) for name in list(summary.get("method_names", []))]

    if CLAIM3_MAIN_METHOD not in method_names:
        raise ValueError("Step 12 source Claim 1 run must contain main_system.")

    return summary, claim_root


def _resolve_metric_root(
    *,
    claim_root: Path,
) -> Path:
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


def _extract_case_date(case: dict[str, Any]) -> str:
    meta = case.get("metaInfo") or {}
    raw = str(meta.get("裁判日期", "") or "").strip()
    return raw


def _extract_case_fact_finding(case: dict[str, Any]) -> str:
    content = case.get("content") or {}

    return str(
        case.get("fact_finding", "")
        or content.get("审理查明", "")
        or content.get("原告诉称", "")
        or ""
    ).strip()


def _order_warmup_pool(
    *,
    uids: list[str],
    case_map: dict[str, dict[str, Any]],
    input_order: list[str],
) -> list[str]:
    order_lookup = {uid: idx for idx, uid in enumerate(input_order)}

    def _sort_key(uid: str) -> tuple[int, str, int]:
        case = case_map[uid]
        raw_date = _extract_case_date(case)

        if raw_date:
            return (0, raw_date, order_lookup.get(uid, 10**9))

        return (1, "9999-99-99", order_lookup.get(uid, 10**9))

    return sorted(uids, key=_sort_key)


def _build_claim3_context(
    *,
    freeze_run_id: str,
    source_claim1_run_id: str,
    run_id: str,
    reports_root: str | Path,
    input_path: str | Path,
    warmup_points: tuple[int, ...],
    fixed_pack_points: tuple[int, ...] = CLAIM3_FIXED_PACK_POINTS,
) -> Claim3Context:
    manifest, split_refs, gold_refs, matching_protocol_snapshot = _load_freeze_bundle(
        freeze_run_id=freeze_run_id,
        reports_root=reports_root,
    )

    source_summary, source_claim_root = _load_claim1_source_summary(
        source_claim1_run_id=source_claim1_run_id,
        reports_root=reports_root,
        freeze_run_id=freeze_run_id,
    )

    case_map, input_order = _load_cases_by_uid(input_path)
    gold_claim_rows = _load_gold_rows(gold_refs["gold_claims_path"])
    gold_status_rows = _load_gold_rows(gold_refs["gold_status_path"])

    gold_uid_set = {
        str(row.get("uid", "") or "").strip()
        for row in gold_claim_rows
        if str(row.get("uid", "") or "").strip()
    }

    eval_scope_path = Path(str(source_summary["artifacts"]["dev_scope"]))
    eval_uids = [uid for uid in _read_ids(eval_scope_path) if uid in gold_uid_set]

    if not eval_uids:
        raise ValueError("Step 12 eval scope is empty after gold coverage filtering.")

    remaining_gold_uids = [
        uid for uid in gold_uid_set if uid not in set(eval_uids) and uid in case_map
    ]

    warmup_pool_uids = _order_warmup_pool(
        uids=remaining_gold_uids,
        case_map=case_map,
        input_order=input_order,
    )

    if max(warmup_points) > len(warmup_pool_uids):
        raise ValueError(
            f"Warmup pool too small for max point {max(warmup_points)}: only {len(warmup_pool_uids)}"
        )

    invalid_fixed_pack_points = [
        point for point in fixed_pack_points if point not in set(warmup_points)
    ]

    if invalid_fixed_pack_points:
        raise ValueError(
            "Claim 3 fixed-pack points must be a subset of warmup points: "
            f"{invalid_fixed_pack_points}"
        )

    claim_root = Path(reports_root) / run_id / "claim3"
    claim_root.mkdir(parents=True, exist_ok=True)

    return Claim3Context(
        freeze_run_id=freeze_run_id,
        source_claim1_run_id=source_claim1_run_id,
        run_id=run_id,
        reports_root=Path(reports_root),
        input_path=Path(input_path),
        claim_root=claim_root,
        manifest_path=claim_root / "warmup_manifest.json",
        job_plan_path=claim_root / "job_plan.json",
        fixed_pack_path=claim_root / "fixed_evidence_pack" / "warmup0_dev37.json",
        reference_anchor_path=claim_root / "reference_anchor_metrics.json",
        warmup_points=warmup_points,
        fixed_pack_points=tuple(sorted(set(int(point) for point in fixed_pack_points))),
        eval_uids=list(eval_uids),
        warmup_pool_uids=list(warmup_pool_uids),
        gold_claim_rows=gold_claim_rows,
        gold_status_rows=gold_status_rows,
        matching_config=MatchingConfig(
            **dict(matching_protocol_snapshot.get("base_config", {}) or {})
        ),
        budget=dict(
            _read_json(Path(reports_root) / freeze_run_id / "budget_grid.json").get(
                "full_budget", {}
            )
            or {}
        ),
        retrieval_config=dict(
            manifest.get("shared_constraints", {}).get("retrieval_config_snapshot", {})
            or {}
        ),
        seed=int(manifest.get("shared_constraints", {}).get("seed", 20260307)),
    )


def _snapshot_path(ctx: Claim3Context, point: int) -> Path:
    return ctx.claim_root / "memory_snapshots" / CLAIM3_MAIN_METHOD / f"warmup_{point}"


def _learning_runtime_dir(ctx: Claim3Context) -> Path:
    return ctx.claim_root / "runtime_learning" / CLAIM3_MAIN_METHOD / "cumulative"


def _learning_raw_dir(ctx: Claim3Context) -> Path:
    return ctx.claim_root / "warmup_runs" / CLAIM3_MAIN_METHOD / "raw_predictions"


def _branch_dir(ctx: Claim3Context, branch_name: str) -> Path:
    return ctx.claim_root / branch_name


def _branch_raw_dir(ctx: Claim3Context, branch_name: str) -> Path:
    return _branch_dir(ctx, branch_name) / "raw_predictions"


def _branch_metric_path(ctx: Claim3Context, branch_name: str, point: int) -> Path:
    return _branch_dir(ctx, branch_name) / "point_metrics" / f"warmup_{point}.json"


def _branch_runtime_dir(ctx: Claim3Context, branch_name: str, point: int) -> Path:
    return ctx.claim_root / "runtime_eval" / branch_name / f"warmup_{point}"


def _raw_result_path(root: Path, uid: str) -> Path:
    return root / f"{uid}.json"


def _try_load_valid_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        result = _read_json(path)
        validate_method_result(result)
        return result

    except Exception:
        return None


def _write_reference_anchor_metrics(ctx: Claim3Context) -> dict[str, Any]:
    metric_root = _resolve_metric_root(
        claim_root=ctx.reports_root / ctx.source_claim1_run_id / "claim1"
    )

    dev_metrics = _read_json(metric_root / "dev" / "method_metrics.json")

    anchors: dict[str, Any] = {
        "source_claim1_run_id": ctx.source_claim1_run_id,
        "metric_root": str(metric_root),
        "metric_key": CLAIM3_PRIMARY_CASE_SCORE_KEY,
        "eval_scope_case_count": len(ctx.eval_uids),
        "methods": {},
    }

    for method_name, metric in dev_metrics.items():
        point = metric["main_metrics"][CLAIM3_PRIMARY_CASE_SCORE_KEY]

        anchors["methods"][method_name] = {
            "point_estimate": float(point["point_estimate"]),
            "ci_low": float(point["ci_low"]),
            "ci_high": float(point["ci_high"]),
        }

    _write_json(ctx.reference_anchor_path, anchors)
    return anchors


def _write_manifest_and_job_plan(ctx: Claim3Context) -> dict[str, Any]:
    manifest = {
        "claim_id": CLAIM3,
        "protocol_version": CLAIM3_PROTOCOL_VERSION,
        "task_focus": CLAIM3_TASK_FOCUS,
        "run_id": ctx.run_id,
        "freeze_run_id": ctx.freeze_run_id,
        "source_claim1_run_id": ctx.source_claim1_run_id,
        "input_path": str(ctx.input_path),
        "method_name": CLAIM3_MAIN_METHOD,
        "warmup_points": list(ctx.warmup_points),
        "fixed_pack_reference_point": CLAIM3_FIXED_PACK_REFERENCE_POINT,
        "fixed_pack_points": list(ctx.fixed_pack_points),
        "eval_uids": list(ctx.eval_uids),
        "eval_case_count": len(ctx.eval_uids),
        "warmup_pool_uids": list(ctx.warmup_pool_uids),
        "warmup_pool_count": len(ctx.warmup_pool_uids),
        "warmup_pool_date_coverage": {
            "missing_date_count": sum(
                1
                for uid in ctx.warmup_pool_uids
                if not _extract_case_date(_load_cases_by_uid(ctx.input_path)[0][uid])
            ),
        },
        "budget": dict(ctx.budget),
        "retrieval_config": dict(ctx.retrieval_config),
        "seed": int(ctx.seed),
        "primary_metric_key": CLAIM3_PRIMARY_CASE_SCORE_KEY,
        "fixed_pack_path": str(ctx.fixed_pack_path),
    }

    _write_json(ctx.manifest_path, manifest)

    job_plan = {
        "claim_id": CLAIM3,
        "run_id": ctx.run_id,
        "prepare_completed_at": _now_iso(),
        "normal_branch_points": list(ctx.warmup_points),
        "fixed_pack_branch_points": list(ctx.fixed_pack_points),
        "expected_task_count": {
            "warmup_cases": max(ctx.warmup_points),
            "normal_eval_cases": len(ctx.eval_uids) * len(ctx.warmup_points),
            "fixed_pack_eval_cases": len(ctx.eval_uids) * len(ctx.fixed_pack_points),
        },
        "artifacts": {
            "manifest": str(ctx.manifest_path),
            "fixed_pack": str(ctx.fixed_pack_path),
            "reference_anchor_metrics": str(ctx.reference_anchor_path),
        },
    }

    _write_json(ctx.job_plan_path, job_plan)
    return job_plan


def prepare_claim3_experiment(
    *,
    freeze_run_id: str,
    source_claim1_run_id: str,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    warmup_points: tuple[int, ...] = CLAIM3_DEFAULT_WARMUP_POINTS,
    fixed_pack_points: tuple[int, ...] = CLAIM3_FIXED_PACK_POINTS,
) -> dict[str, Any]:
    """Prepare manifests and fixed evidence packs for Claim 3.

    Args:
        freeze_run_id: Step 09A freeze bundle that defines the shared protocol.
        source_claim1_run_id: Completed Claim 1 run that provides reference metrics.
        run_id: Destination Step 12 run identifier.
        reports_root: Root directory that stores experiment outputs.
        input_path: Original case JSONL used to build warmup and eval pools.
        warmup_points: Warmup prefixes to snapshot and later score.

    Returns:
        A preparation summary with manifest, job-plan, and anchor artifact paths.

    Raises:
        FileNotFoundError: If the referenced Claim 1 artifacts are missing.
        ValueError: If the warmup points or frozen inputs are inconsistent.
    """

    ctx = _build_claim3_context(
        freeze_run_id=freeze_run_id,
        source_claim1_run_id=source_claim1_run_id,
        run_id=run_id,
        reports_root=reports_root,
        input_path=input_path,
        warmup_points=tuple(sorted(set(int(point) for point in warmup_points))),
        fixed_pack_points=tuple(sorted(set(int(point) for point in fixed_pack_points))),
    )

    _write_reference_anchor_metrics(ctx)
    _snapshot_path(ctx, 0).mkdir(parents=True, exist_ok=True)
    case_map, _ = _load_cases_by_uid(ctx.input_path)

    fixed_pack_rows = build_fixed_evidence_pack_map_for_contexts(
        storage_root_dir=str(_snapshot_path(ctx, 0)),
        contexts_by_uid={
            uid: _extract_case_fact_finding(case_map[uid]) for uid in ctx.eval_uids
        },
    )

    write_fixed_evidence_pack_map(ctx.fixed_pack_path, fixed_pack_rows)
    job_plan = _write_manifest_and_job_plan(ctx)

    summary = {
        "claim_id": CLAIM3,
        "status": "prepared",
        "run_id": run_id,
        "freeze_run_id": freeze_run_id,
        "source_claim1_run_id": source_claim1_run_id,
        "method_name": CLAIM3_MAIN_METHOD,
        "warmup_points": list(ctx.warmup_points),
        "fixed_pack_points": list(ctx.fixed_pack_points),
        "eval_case_count": len(ctx.eval_uids),
        "warmup_pool_count": len(ctx.warmup_pool_uids),
        "artifacts": {
            "warmup_manifest": str(ctx.manifest_path),
            "job_plan": str(ctx.job_plan_path),
            "fixed_evidence_pack": str(ctx.fixed_pack_path),
            "reference_anchor_metrics": str(ctx.reference_anchor_path),
        },
        "job_plan": job_plan,
    }

    _write_json(ctx.claim_root / "run_summary.json", summary)
    return summary


def _run_main_system_once(
    *,
    case_payload: dict[str, Any],
    storage_root_dir: str,
    budget: dict[str, Any],
    seed: int,
    retrieval_config: dict[str, Any],
    test_mode_no_learning: bool,
) -> dict[str, Any]:
    runner = build_default_registry()[CLAIM3_MAIN_METHOD]

    return runner(
        case=case_payload,
        storage_root_dir=storage_root_dir,
        budget=budget,
        seed=seed,
        retrieval_config=retrieval_config,
        test_mode_no_learning=test_mode_no_learning,
    )


def _rebuild_runtime_to_completed_prefix(
    *,
    ctx: Claim3Context,
    case_map: dict[str, dict[str, Any]],
    completed_prefix: int,
) -> None:
    runtime_dir = _learning_runtime_dir(ctx)
    _clear_dir(runtime_dir)

    latest_snapshot_point = max(
        point
        for point in ctx.warmup_points
        if point <= completed_prefix and _snapshot_path(ctx, point).exists()
    )

    if latest_snapshot_point > 0:
        _copy_tree(_snapshot_path(ctx, latest_snapshot_point), runtime_dir)

    else:
        runtime_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(latest_snapshot_point, completed_prefix):
        uid = ctx.warmup_pool_uids[idx]

        _run_main_system_once(
            case_payload=dict(case_map[uid]),
            storage_root_dir=str(runtime_dir),
            budget=ctx.budget,
            seed=ctx.seed,
            retrieval_config=ctx.retrieval_config,
            test_mode_no_learning=False,
        )


def _ensure_warmup_snapshots(
    *,
    ctx: Claim3Context,
    show_progress: bool,
) -> dict[str, Any]:
    case_map, _ = _load_cases_by_uid(ctx.input_path)
    runtime_dir = _learning_runtime_dir(ctx)
    raw_dir = _learning_raw_dir(ctx)
    snapshot_zero = _snapshot_path(ctx, 0)
    snapshot_zero.mkdir(parents=True, exist_ok=True)
    completed_prefix = 0

    for uid in ctx.warmup_pool_uids[: max(ctx.warmup_points)]:
        if _try_load_valid_result(_raw_result_path(raw_dir, uid)) is None:
            break

        completed_prefix += 1

    if completed_prefix > 0 and (
        not runtime_dir.exists() or not any(runtime_dir.iterdir())
    ):
        _rebuild_runtime_to_completed_prefix(
            ctx=ctx,
            case_map=case_map,
            completed_prefix=completed_prefix,
        )

    runtime_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(
        total=max(ctx.warmup_points),
        desc="Claim3 warmup",
        unit="case",
        dynamic_ncols=True,
        disable=not show_progress,
        initial=completed_prefix,
    )

    try:
        for idx, uid in enumerate(
            ctx.warmup_pool_uids[: max(ctx.warmup_points)], start=1
        ):
            result_path = _raw_result_path(raw_dir, uid)

            if (
                idx <= completed_prefix
                and _try_load_valid_result(result_path) is not None
            ):
                if idx in ctx.warmup_points and not _snapshot_path(ctx, idx).exists():
                    _copy_tree(runtime_dir, _snapshot_path(ctx, idx))

                continue

            result = _run_main_system_once(
                case_payload=dict(case_map[uid]),
                storage_root_dir=str(runtime_dir),
                budget=ctx.budget,
                seed=ctx.seed,
                retrieval_config=ctx.retrieval_config,
                test_mode_no_learning=False,
            )

            _write_json(result_path, result)
            progress.update(1)

            if idx in ctx.warmup_points and not _snapshot_path(ctx, idx).exists():
                _copy_tree(runtime_dir, _snapshot_path(ctx, idx))

    finally:
        progress.close()

    summary = {
        "status": "completed",
        "completed_prefix": max(ctx.warmup_points),
        "warmup_points": list(ctx.warmup_points),
        "snapshot_paths": {
            f"warmup_{point}": str(_snapshot_path(ctx, point))
            for point in ctx.warmup_points
        },
        "raw_dir": str(raw_dir),
    }

    _write_json(
        ctx.claim_root / "warmup_runs" / CLAIM3_MAIN_METHOD / "summary.json", summary
    )

    return summary


def _evaluate_branch_point(
    *,
    ctx: Claim3Context,
    branch_name: str,
    point: int,
    case_map: dict[str, dict[str, Any]],
    fixed_pack_map: dict[str, dict[str, Any]] | None,
    resume: bool,
    show_progress: bool,
) -> dict[str, Any]:
    snapshot_dir = _snapshot_path(ctx, point)

    if not snapshot_dir.exists():
        raise FileNotFoundError(
            f"Missing warmup snapshot for point={point}: {snapshot_dir}"
        )

    runtime_dir = _branch_runtime_dir(ctx, branch_name, point)

    if not runtime_dir.exists():
        _copy_tree(snapshot_dir, runtime_dir)

    pre_hash = _hash_directory(runtime_dir)
    raw_dir = _branch_raw_dir(ctx, branch_name) / f"warmup_{point}"
    results: list[dict[str, Any]] = []
    flattened_predictions: list[dict[str, Any]] = []

    progress = tqdm(
        total=len(ctx.eval_uids),
        desc=f"Claim3 {branch_name} warmup_{point}",
        unit="case",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    try:
        for uid in ctx.eval_uids:
            result_path = _raw_result_path(raw_dir, uid)

            if resume:
                cached = _try_load_valid_result(result_path)

                if cached is not None:
                    results.append(cached)
                    flattened_predictions.extend(_flatten_prediction_rows(cached))
                    progress.update(1)
                    continue

            case_payload = dict(case_map[uid])

            if fixed_pack_map is not None:
                case_payload["experiment_fixed_evidence_pack"] = (
                    extract_fixed_evidence_pack_for_uid(fixed_pack_map, uid)
                )

            result = _run_main_system_once(
                case_payload=case_payload,
                storage_root_dir=str(runtime_dir),
                budget=ctx.budget,
                seed=ctx.seed,
                retrieval_config=ctx.retrieval_config,
                test_mode_no_learning=True,
            )

            _write_json(result_path, result)
            results.append(result)
            flattened_predictions.extend(_flatten_prediction_rows(result))
            progress.update(1)

    finally:
        progress.close()

    gold_claim_rows = [
        row
        for row in ctx.gold_claim_rows
        if str(row.get("uid", "") or "") in set(ctx.eval_uids)
    ]

    gold_status_rows = [
        row
        for row in ctx.gold_status_rows
        if str(row.get("uid", "") or "") in set(ctx.eval_uids)
    ]

    encoder = SentenceTransformerTextEncoder(SystemConfig().path.embedding_model_path)

    bundles = build_method_case_matches(
        case_uids=ctx.eval_uids,
        gold_rows=gold_claim_rows,
        prediction_rows=flattened_predictions,
        method_name=CLAIM3_MAIN_METHOD,
        encoder=encoder,
        config=ctx.matching_config,
    )

    metric = evaluate_claim1_metrics(bundles, gold_status_rows)
    post_hash = _hash_directory(runtime_dir)

    payload = {
        "branch_name": branch_name,
        "warmup_point": point,
        "metric_key": CLAIM3_PRIMARY_CASE_SCORE_KEY,
        "method_name": CLAIM3_MAIN_METHOD,
        "runtime_dir": str(runtime_dir),
        "snapshot_dir": str(snapshot_dir),
        "runtime_hash_before": pre_hash,
        "runtime_hash_after": post_hash,
        "runtime_hash_unchanged": pre_hash == post_hash,
        "used_fixed_evidence_pack": fixed_pack_map is not None,
        "fixed_pack_reference_point": (
            CLAIM3_FIXED_PACK_REFERENCE_POINT if fixed_pack_map is not None else None
        ),
        "metric": metric,
    }

    _write_json(_branch_metric_path(ctx, branch_name, point), payload)
    return payload


def run_claim3_branch(
    *,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    branch: str,
    resume: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run one Claim 3 branch across all configured warmup points.

    Args:
        run_id: Prepared Claim 3 run identifier.
        reports_root: Root directory that stores experiment outputs.
        branch: Branch name, either `normal` or `fixed-pack`.
        resume: Whether to reuse per-case raw predictions already on disk.
        show_progress: Whether to render progress bars for long-running work.

    Returns:
        A branch summary containing completed points and artifact paths.

    Raises:
        FileNotFoundError: If required manifests or snapshots are missing.
        ValueError: If the branch name is unsupported.
        Claim3ExecutionError: If execution cannot continue after preparation.
    """

    claim_root = Path(reports_root) / run_id / "claim3"
    manifest = _read_json(claim_root / "warmup_manifest.json")

    ctx = _build_claim3_context(
        freeze_run_id=str(manifest["freeze_run_id"]),
        source_claim1_run_id=str(manifest["source_claim1_run_id"]),
        run_id=run_id,
        reports_root=reports_root,
        input_path=str(manifest["input_path"]),
        warmup_points=tuple(int(point) for point in manifest["warmup_points"]),
        fixed_pack_points=tuple(
            int(point)
            for point in manifest.get("fixed_pack_points", CLAIM3_FIXED_PACK_POINTS)
        ),
    )

    case_map, _ = _load_cases_by_uid(ctx.input_path)
    warmup_summary = _ensure_warmup_snapshots(ctx=ctx, show_progress=show_progress)
    fixed_pack_map = None
    branch_name = str(branch).strip().lower()

    if branch_name not in {"normal", "fixed-pack"}:
        raise ValueError("Claim 3 branch must be one of: normal, fixed-pack")

    points = list(ctx.warmup_points)

    if branch_name == "fixed-pack":
        points = [
            point for point in ctx.fixed_pack_points if point in set(ctx.warmup_points)
        ]

        fixed_pack_map = read_fixed_evidence_pack_map(ctx.fixed_pack_path)

    branch_dir = _branch_dir(ctx, branch_name)
    branch_dir.mkdir(parents=True, exist_ok=True)
    stage_rows = []

    for point in points:
        stage_rows.append(
            _evaluate_branch_point(
                ctx=ctx,
                branch_name=branch_name,
                point=point,
                case_map=case_map,
                fixed_pack_map=fixed_pack_map,
                resume=resume,
                show_progress=show_progress,
            )
        )

    summary = {
        "claim_id": CLAIM3,
        "run_id": run_id,
        "branch": branch_name,
        "status": "completed",
        "completed_at": _now_iso(),
        "warmup_summary": warmup_summary,
        "points": points,
        "artifacts": {
            "branch_dir": str(branch_dir),
            "point_metrics": [
                str(_branch_metric_path(ctx, branch_name, point)) for point in points
            ],
        },
    }

    _write_json(branch_dir / "summary.json", summary)
    return summary


def _attribution_label(normal_gain: float, fixed_gain: float) -> str:
    if normal_gain > 0.0 and fixed_gain <= 0.0:
        return "memory_evidence_input_dominant"

    if normal_gain > 0.0 and fixed_gain > 0.0:
        return "frozen_pack_retains_gain"

    return "no_clear_learning_gain"


def summarize_claim3_experiment(
    *,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
) -> dict[str, Any]:
    """Aggregate finished Claim 3 branches into curves and attribution labels.

    Args:
        run_id: Prepared Claim 3 run identifier.
        reports_root: Root directory that stores experiment outputs.

    Returns:
        A summary payload with normal and fixed-pack curves plus attribution labels.

    Raises:
        FileNotFoundError: If prepared branch artifacts are missing.
        ValueError: If required branch summaries have not been produced yet.
    """

    claim_root = Path(reports_root) / run_id / "claim3"
    manifest = _read_json(claim_root / "warmup_manifest.json")

    ctx = _build_claim3_context(
        freeze_run_id=str(manifest["freeze_run_id"]),
        source_claim1_run_id=str(manifest["source_claim1_run_id"]),
        run_id=run_id,
        reports_root=reports_root,
        input_path=str(manifest["input_path"]),
        warmup_points=tuple(int(item) for item in manifest["warmup_points"]),
        fixed_pack_points=tuple(
            int(point)
            for point in manifest.get("fixed_pack_points", CLAIM3_FIXED_PACK_POINTS)
        ),
    )

    normal_summary = _read_json(claim_root / "normal" / "summary.json")
    fixed_summary = _read_json(claim_root / "fixed-pack" / "summary.json")

    curve_scores: dict[str, dict[str, dict[str, float]]] = {
        "normal": {},
        "fixed_pack": {},
    }

    for branch_name, summary in (
        ("normal", normal_summary),
        ("fixed_pack", fixed_summary),
    ):
        for point in summary["points"]:
            payload = _read_json(
                _branch_metric_path(
                    ctx,
                    "fixed-pack" if branch_name == "fixed_pack" else "normal",
                    int(point),
                )
            )

            curve_scores[branch_name][f"warmup_{point}"] = payload["metric"][
                "case_scores"
            ][CLAIM3_PRIMARY_CASE_SCORE_KEY]

    curve_summary = summarize_claim3_curve(curve_scores=curve_scores)
    normal_rows = []
    fixed_rows = []

    for point_name, summary in curve_summary["curve_summaries"]["normal"].items():
        normal_rows.append(
            {
                "branch": "normal",
                "warmup_point": point_name,
                **summary,
            }
        )

    for point_name, summary in curve_summary["curve_summaries"]["fixed_pack"].items():
        fixed_rows.append(
            {
                "branch": "fixed_pack",
                "warmup_point": point_name,
                **summary,
            }
        )

    _write_csv(claim_root / "status_curve_normal.csv", normal_rows)
    _write_csv(claim_root / "status_curve_fixed_pack.csv", fixed_rows)

    normal_point0 = curve_summary["curve_summaries"]["normal"]["warmup_0"][
        "point_estimate"
    ]

    normal_point_max = curve_summary["curve_summaries"]["normal"][
        f"warmup_{max(int(item) for item in manifest['warmup_points'])}"
    ]["point_estimate"]

    fixed_point0 = curve_summary["curve_summaries"]["fixed_pack"]["warmup_0"][
        "point_estimate"
    ]

    fixed_target_point = max(int(item) for item in fixed_summary["points"])

    fixed_point_target = curve_summary["curve_summaries"]["fixed_pack"][
        f"warmup_{fixed_target_point}"
    ]["point_estimate"]

    attribution = {
        "claim_id": CLAIM3,
        "metric_key": CLAIM3_PRIMARY_CASE_SCORE_KEY,
        "normal_curve_gain_0_to_max": float(normal_point_max) - float(normal_point0),
        "fixed_pack_target_point": int(fixed_target_point),
        "fixed_pack_gain_0_to_target": float(fixed_point_target) - float(fixed_point0),
    }

    attribution["attribution_label"] = _attribution_label(
        attribution["normal_curve_gain_0_to_max"],
        attribution["fixed_pack_gain_0_to_target"],
    )

    _write_json(claim_root / "attribution_summary.json", attribution)

    run_summary = {
        "claim_id": CLAIM3,
        "run_id": run_id,
        "status": "completed",
        "freeze_run_id": manifest["freeze_run_id"],
        "source_claim1_run_id": manifest["source_claim1_run_id"],
        "method_name": CLAIM3_MAIN_METHOD,
        "task_focus": CLAIM3_TASK_FOCUS,
        "warmup_points": list(manifest["warmup_points"]),
        "fixed_pack_points": list(manifest["fixed_pack_points"]),
        "primary_metric_key": CLAIM3_PRIMARY_CASE_SCORE_KEY,
        "artifacts": {
            "warmup_manifest": str(claim_root / "warmup_manifest.json"),
            "job_plan": str(claim_root / "job_plan.json"),
            "reference_anchor_metrics": str(
                claim_root / "reference_anchor_metrics.json"
            ),
            "status_curve_normal_csv": str(claim_root / "status_curve_normal.csv"),
            "status_curve_fixed_pack_csv": str(
                claim_root / "status_curve_fixed_pack.csv"
            ),
            "attribution_summary": str(claim_root / "attribution_summary.json"),
        },
        "curve_summary": curve_summary,
        "attribution_summary": attribution,
    }

    _write_json(claim_root / "run_summary.json", run_summary)
    return run_summary
