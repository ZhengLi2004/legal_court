"""Experiment orchestration entrypoints for Claim 1-4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from benchmarks.experiments.core.claim2_runner import (
    run_claim2_experiment as _run_claim2_experiment,
)
from benchmarks.experiments.core.claim3_runner import (
    prepare_claim3_experiment as _prepare_claim3_experiment,
)
from benchmarks.experiments.core.claim3_runner import (
    run_claim3_branch as _run_claim3_branch,
)
from benchmarks.experiments.core.claim3_runner import (
    summarize_claim3_experiment as _summarize_claim3_experiment,
)
from benchmarks.experiments.core.step09a import (
    DEFAULT_INPUT_PATH,
    DEFAULT_REPORTS_ROOT,
    METRIC_CONTRACT_VERSION,
)
from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.eval import (
    FROZEN_MATCHING_PROTOCOL_VERSION,
    MatchingConfig,
    MatchingRobustnessGateError,
    PrimaryMetricResult,
    SentenceTransformerTextEncoder,
    build_method_case_matches,
    evaluate_claim1_metrics,
    paired_case_bootstrap,
    run_frozen_matching_protocol,
)
from benchmarks.experiments.methods.base import case_uid, validate_method_result
from benchmarks.experiments.methods.factory import (
    INTERNAL_DEFAULT_PROFILE,
    build_default_registry,
    build_method_registry,
)
from mas.config import SystemConfig

PILOT_STAGE = "pilot"
FULL_STAGE = "full"
CLAIM1 = 1
ALLOWED_PILOT_VERDICTS = {"pass", "minor_issue", "critical_issue"}
TASK_MAX_ATTEMPTS = 2
CLAIM1_TASK_FOCUS = "status_eval_on_gold_claims"
CLAIM1_ROOT_CLAIM_SOURCE = "gold_package_seeded"
CLAIM1_MATCHING_MODE = "claim_id_direct_when_gold_seeded"
CLAIM1_ADJUDICATION_MODE = "direct_status_json"
CLAIM1_TEST_MODE_NO_LEARNING = True


@dataclass(frozen=True)
class FreezeBundle:
    freeze_run_id: str
    run_root: Path
    manifest: dict[str, Any]
    split_refs: dict[str, Any]
    gold_refs: dict[str, Any]
    budget_grid: dict[str, Any]
    matching_protocol_snapshot: dict[str, Any]
    metric_contract_snapshot: dict[str, Any]
    method_registry_snapshot: dict[str, Any]
    selected_dryrun_cases: dict[str, Any]
    preflight_health: dict[str, Any]
    live_dryrun_summary: dict[str, Any]


class Claim1ExecutionError(RuntimeError):
    """Raised when Claim 1 execution cannot continue."""


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


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _read_ids(path: str | Path, key: str) -> list[str]:
    payload = _read_json(path)
    values = payload.get(key, [])

    if not isinstance(values, list) or not values:
        raise ValueError(f"{Path(path).name} must contain a non-empty `{key}` list.")

    return [str(value) for value in values]


def _build_skeleton_result(
    *,
    claim_id: int,
    cases: list[dict[str, Any]] | None = None,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case_count = len(cases) if cases is not None else 0
    method_names = sorted(registry.keys()) if registry is not None else []

    return {
        "claim_id": claim_id,
        "status": "skeleton",
        "case_count": case_count,
        "methods": method_names,
        "metrics": {},
        "artifacts": [],
        "errors": [],
    }


def _load_freeze_bundle(
    *,
    freeze_run_id: str,
    reports_root: str | Path,
) -> FreezeBundle:
    run_root = Path(reports_root) / freeze_run_id
    manifest_path = run_root / "protocol_freeze_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Step 09A freeze manifest: {manifest_path}")

    manifest = _read_json(manifest_path)
    runtime_files = manifest.get("runtime_files", [])
    file_hashes = manifest.get("file_hashes", {})

    if not manifest.get("preflight_passed", False):
        raise ValueError("Step 09A preflight did not pass; Step 10 cannot start.")

    if not manifest.get("live_dryrun_passed", False):
        raise ValueError("Step 09A live dry-run did not pass; Step 10 cannot start.")

    if not isinstance(runtime_files, list) or not runtime_files:
        raise ValueError("Freeze manifest must contain `runtime_files`.")

    if not isinstance(file_hashes, dict) or not file_hashes:
        raise ValueError("Freeze manifest must contain `file_hashes`.")

    for rel_path in runtime_files:
        file_path = run_root / str(rel_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Missing frozen runtime file: {file_path}")

        actual_hash = _sha256_file(file_path)
        expected_hash = str(file_hashes.get(rel_path, "") or "")

        if actual_hash != expected_hash:
            raise ValueError(
                f"Frozen runtime file hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
            )

    split_refs = _read_json(run_root / "split_refs.json")
    gold_refs = _read_json(run_root / "gold_refs.json")
    budget_grid = _read_json(run_root / "budget_grid.json")

    matching_protocol_snapshot = _read_json(
        run_root / "matching_protocol_snapshot.json"
    )

    metric_contract_snapshot = _read_json(run_root / "metric_contract_snapshot.json")
    method_registry_snapshot = _read_json(run_root / "method_registry_snapshot.json")

    selected_dryrun_cases = _read_json(
        run_root / "preflight" / "selected_dryrun_cases.json"
    )

    preflight_health = _read_json(run_root / "preflight" / "preflight_health.json")

    live_dryrun_summary = _read_json(
        run_root / "preflight" / "live_dryrun" / "summary.json"
    )

    matching_protocol_version = str(
        matching_protocol_snapshot.get("protocol_version", "") or ""
    )

    metric_contract_version = str(
        metric_contract_snapshot.get("metric_contract_version", "") or ""
    )

    if matching_protocol_version != FROZEN_MATCHING_PROTOCOL_VERSION:
        raise ValueError(
            "Current matching protocol version does not match Step 09A freeze snapshot."
        )

    if metric_contract_version != METRIC_CONTRACT_VERSION:
        raise ValueError(
            "Current metric contract version does not match Step 09A freeze snapshot."
        )

    shared_constraints = manifest.get("shared_constraints", {})

    if (
        str(shared_constraints.get("matching_protocol_version", "") or "")
        != matching_protocol_version
    ):
        raise ValueError("Freeze manifest matching protocol version is inconsistent.")

    if (
        str(shared_constraints.get("metric_contract_version", "") or "")
        != metric_contract_version
    ):
        raise ValueError("Freeze manifest metric contract version is inconsistent.")

    if list(selected_dryrun_cases.get("selected_case_uids", [])) != list(
        preflight_health.get("selected_dryrun_case_uids", [])
    ):
        raise ValueError("Freeze dry-run case selection is internally inconsistent.")

    if list(selected_dryrun_cases.get("selected_case_uids", [])) != list(
        live_dryrun_summary.get("selected_case_uids", [])
    ):
        raise ValueError("Freeze dry-run summary does not match selected pilot cases.")

    return FreezeBundle(
        freeze_run_id=freeze_run_id,
        run_root=run_root,
        manifest=manifest,
        split_refs=split_refs,
        gold_refs=gold_refs,
        budget_grid=budget_grid,
        matching_protocol_snapshot=matching_protocol_snapshot,
        metric_contract_snapshot=metric_contract_snapshot,
        method_registry_snapshot=method_registry_snapshot,
        selected_dryrun_cases=selected_dryrun_cases,
        preflight_health=preflight_health,
        live_dryrun_summary=live_dryrun_summary,
    )


def _resolve_registry(
    *,
    freeze_bundle: FreezeBundle,
    registry: dict[str, Any] | None = None,
    methods: list[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    if registry is None:
        registry_profile = str(
            freeze_bundle.method_registry_snapshot.get(
                "registry_profile", INTERNAL_DEFAULT_PROFILE
            )
            or INTERNAL_DEFAULT_PROFILE
        )

        if registry_profile == INTERNAL_DEFAULT_PROFILE:
            full_registry = dict(build_default_registry())

        else:
            full_registry = dict(build_method_registry(profile=registry_profile))

    else:
        full_registry = dict(registry)

    frozen_method_names = list(
        freeze_bundle.method_registry_snapshot.get("method_names", [])
    )

    if sorted(full_registry.keys()) != sorted(frozen_method_names):
        raise ValueError(
            "Current Step 10 registry does not match the frozen Step 09A method registry."
        )

    active_method_names = list(frozen_method_names)

    if methods is not None:
        requested = [str(name).strip() for name in methods if str(name).strip()]
        unknown = sorted(set(requested) - set(frozen_method_names))

        if unknown:
            raise ValueError(f"Unknown Step 10 method names requested: {unknown}")

        active_method_names = [
            name for name in frozen_method_names if name in requested
        ]

    if not active_method_names:
        raise ValueError("Step 10 must run at least one method.")

    active_registry = {name: full_registry[name] for name in active_method_names}
    return active_method_names, active_registry


def _load_cases_by_uid(input_path: str | Path) -> dict[str, dict[str, Any]]:
    return {case_uid(case): case for case in load_cases_from_jsonl(input_path)}


def _select_cases(
    *,
    case_map: dict[str, dict[str, Any]],
    case_uids: list[str],
) -> list[dict[str, Any]]:
    selected = [case_map[uid] for uid in case_uids if uid in case_map]

    if len(selected) != len(case_uids):
        missing = sorted(set(case_uids) - set(case_map.keys()))
        raise ValueError(f"Missing Step 10 cases for uids: {missing}")

    return selected


def _load_gold_rows(
    *,
    gold_claims_path: str | Path,
    gold_status_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        load_cases_from_jsonl(gold_claims_path),
        load_cases_from_jsonl(gold_status_path),
    )


def _filter_rows_by_uids(
    rows: list[dict[str, Any]],
    case_uids: list[str],
) -> list[dict[str, Any]]:
    uid_set = set(case_uids)
    return [dict(row) for row in rows if str(row.get("uid", "") or "") in uid_set]


def _restrict_case_uids_to_gold_scope(
    *,
    case_uids: list[str],
    gold_rows: list[dict[str, Any]],
    split_name: str,
) -> list[str]:
    gold_uid_set = {
        str(row.get("uid", "") or "").strip()
        for row in gold_rows
        if str(row.get("uid", "") or "").strip()
    }

    scoped = [uid for uid in case_uids if uid in gold_uid_set]

    if not scoped:
        raise ValueError(
            f"Claim 1 {split_name} split has no cases covered by gold claims."
        )

    return scoped


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


def _summarize_result_row(method_name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method_name,
        "case_uid": str(result["case_uid"]),
        "claim_count": len(result["claims"]),
        "status_count": len(result["status"]),
        "warning_count": len(result["warnings"]),
    }


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
    )

    return any(marker in message for marker in retryable_markers)


def _try_load_existing_raw_result(
    *,
    path: Path,
    method_name: str,
    case_uid_value: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        result = _read_json(path)
        validate_method_result(result)

    except Exception:
        return None

    if str(result.get("method", "") or "") != method_name:
        return None

    if str(result.get("case_uid", "") or "") != case_uid_value:
        return None

    return result


def _run_method_matrix(
    *,
    cases: list[dict[str, Any]],
    case_uids: list[str],
    active_method_names: list[str],
    active_registry: dict[str, Any],
    storage_root_dir: str | None,
    test_mode_no_learning: bool,
    budget: dict[str, Any],
    seed: int,
    retrieval_config: dict[str, Any],
    output_dir: Path,
    verbose: bool,
    show_progress: bool,
    resume: bool = False,
) -> dict[str, Any]:
    raw_predictions_dir = output_dir / "raw_predictions"
    validation_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    retry_events: list[dict[str, Any]] = []

    results_by_method: dict[str, list[dict[str, Any]]] = {
        method_name: [] for method_name in active_method_names
    }

    flattened_predictions: dict[str, list[dict[str, Any]]] = {
        method_name: [] for method_name in active_method_names
    }

    summary: dict[str, Any] = {
        "status": "running",
        "started_at": _now_iso(),
        "case_uids": list(case_uids),
        "method_names": list(active_method_names),
        "total_task_count": len(cases) * len(active_method_names),
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

    summary_path = output_dir / "execution_summary.json"
    _write_json(summary_path, summary)

    progress = tqdm(
        total=summary["total_task_count"],
        desc=f"Claim1 {output_dir.name}",
        unit="task",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    try:
        for method_name in active_method_names:
            runner = active_registry[method_name]

            for case in cases:
                uid = case_uid(case)
                raw_result_path = raw_predictions_dir / method_name / f"{uid}.json"

                if resume:
                    cached_result = _try_load_existing_raw_result(
                        path=raw_result_path,
                        method_name=method_name,
                        case_uid_value=uid,
                    )

                    if cached_result is not None:
                        results_by_method[method_name].append(cached_result)

                        flattened_predictions[method_name].extend(
                            _flatten_prediction_rows(cached_result)
                        )

                        validation_rows.append(
                            _summarize_result_row(method_name, cached_result)
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
                        runner_kwargs = {
                            "case": case,
                            "budget": budget,
                            "seed": seed,
                            "retrieval_config": retrieval_config,
                            "verbose": verbose,
                        }

                        if storage_root_dir:
                            runner_kwargs["storage_root_dir"] = storage_root_dir

                        if test_mode_no_learning:
                            runner_kwargs["test_mode_no_learning"] = True

                        result = runner(**runner_kwargs)

                        validate_method_result(result)
                        results_by_method[method_name].append(result)

                        flattened_predictions[method_name].extend(
                            _flatten_prediction_rows(result)
                        )

                        validation_rows.append(
                            _summarize_result_row(method_name, result)
                        )

                        _write_json(
                            raw_result_path,
                            result,
                        )

                        break

                    except Exception as exc:
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
                                "error_detail": str(exc),
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
                                "error_detail": str(exc),
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
        raise Claim1ExecutionError(
            f"Claim 1 {output_dir.name} execution failed for {len(failures)} task(s)."
        )

    return {
        "summary": summary,
        "results_by_method": results_by_method,
        "flattened_predictions": flattened_predictions,
    }


def _make_matching_config(snapshot: dict[str, Any]) -> MatchingConfig:
    return MatchingConfig(**dict(snapshot.get("base_config", {}) or {}))


def _build_case_bundles_by_method(
    *,
    case_uids: list[str],
    gold_claim_rows: list[dict[str, Any]],
    flattened_predictions: dict[str, list[dict[str, Any]]],
    active_method_names: list[str],
    encoder: Any,
    config: MatchingConfig,
) -> dict[str, list[Any]]:
    bundles_by_method: dict[str, list[Any]] = {}

    for method_name in active_method_names:
        bundles_by_method[method_name] = build_method_case_matches(
            case_uids=case_uids,
            gold_rows=gold_claim_rows,
            prediction_rows=flattened_predictions[method_name],
            method_name=method_name,
            encoder=encoder,
            config=config,
        )

    return bundles_by_method


def _evaluate_method_metrics(
    *,
    bundles_by_method: dict[str, list[Any]],
    gold_status_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        method_name: evaluate_claim1_metrics(bundles, gold_status_rows)
        for method_name, bundles in bundles_by_method.items()
    }


def _summary_row(method_name: str, metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "method_name": method_name,
        "e2e_status_acc": dict(metric["main_metrics"]["e2e_status_acc"]),
        "status_acc_matched": dict(metric["main_metrics"]["status_acc_matched"]),
    }


def _appendix_stepa_row(method_name: str, metric: dict[str, Any]) -> dict[str, Any]:
    appendix = metric["appendix_metrics"]

    return {
        "method_name": method_name,
        "step_a_precision": dict(appendix["step_a_precision"]),
        "step_a_recall": dict(appendix["step_a_recall"]),
        "step_a_f1": dict(appendix["step_a_f1"]),
    }


def _appendix_fp_row(method_name: str, metric: dict[str, Any]) -> dict[str, Any]:
    appendix = metric["appendix_metrics"]

    return {
        "method_name": method_name,
        "e2e_f1_fp_sensitive": dict(appendix["e2e_f1_fp_sensitive"]),
        "over_generation_rate": dict(appendix["over_generation_rate"]),
        "macro_f1_3class_matched": dict(appendix["macro_f1_3class_matched"]),
        "balanced_accuracy_3class_matched": dict(
            appendix["balanced_accuracy_3class_matched"]
        ),
    }


def _sign(value: float, *, atol: float = 1e-12) -> int:
    if value > atol:
        return 1

    if value < -atol:
        return -1

    return 0


def _build_soft_consistency_payload(
    *,
    method_metrics: dict[str, dict[str, Any]],
    method_order: list[str],
) -> dict[str, Any]:
    rows = []

    for method_name in method_order:
        appendix = method_metrics[method_name]["appendix_metrics"]

        rows.append(
            {
                "method_name": method_name,
                "soft_e2e_f1": dict(appendix["soft_e2e_f1"]),
                "over_generation_rate": dict(appendix["over_generation_rate"]),
            }
        )

    checks: list[dict[str, Any]] = []
    reference_method = "main_system"

    if reference_method in method_metrics:
        ref_main = method_metrics[reference_method]["main_metrics"]["e2e_status_acc"]
        ref_soft = method_metrics[reference_method]["appendix_metrics"]["soft_e2e_f1"]

        for method_name in method_order:
            if method_name == reference_method:
                continue

            main_delta = float(ref_main["point_estimate"]) - float(
                method_metrics[method_name]["main_metrics"]["e2e_status_acc"][
                    "point_estimate"
                ]
            )

            soft_delta = float(ref_soft["point_estimate"]) - float(
                method_metrics[method_name]["appendix_metrics"]["soft_e2e_f1"][
                    "point_estimate"
                ]
            )

            checks.append(
                {
                    "reference_method": reference_method,
                    "other_method": method_name,
                    "e2e_status_acc_delta": main_delta,
                    "soft_e2e_f1_delta": soft_delta,
                    "direction_consistent": _sign(main_delta) == _sign(soft_delta),
                }
            )

    return {
        "rows": rows,
        "direction_checks_vs_main_system": checks,
        "all_direction_consistent": all(
            bool(item["direction_consistent"]) for item in checks
        )
        if checks
        else True,
    }


def _build_pairwise_delta_payload(
    *,
    method_metrics: dict[str, dict[str, Any]],
    method_order: list[str],
) -> dict[str, Any]:
    reference_method = "main_system"

    if reference_method not in method_metrics:
        return {"reference_method": reference_method, "rows": []}

    rows: list[dict[str, Any]] = []
    metric_names = ("e2e_status_acc", "soft_e2e_f1")

    for method_name in method_order:
        if method_name == reference_method:
            continue

        metrics_row: dict[str, Any] = {
            "reference_method": reference_method,
            "other_method": method_name,
        }

        for metric_name in metric_names:
            if metric_name in method_metrics[reference_method]["main_metrics"]:
                ref_case_scores = method_metrics[reference_method]["case_scores"][
                    metric_name
                ]

                other_case_scores = method_metrics[method_name]["case_scores"][
                    metric_name
                ]

                ref_point = method_metrics[reference_method]["main_metrics"][
                    metric_name
                ]["point_estimate"]

                other_point = method_metrics[method_name]["main_metrics"][metric_name][
                    "point_estimate"
                ]

            else:
                ref_case_scores = method_metrics[reference_method]["case_scores"][
                    metric_name
                ]

                other_case_scores = method_metrics[method_name]["case_scores"][
                    metric_name
                ]

                ref_point = method_metrics[reference_method]["appendix_metrics"][
                    metric_name
                ]["point_estimate"]

                other_point = method_metrics[method_name]["appendix_metrics"][
                    metric_name
                ]["point_estimate"]

            delta = paired_case_bootstrap(ref_case_scores, other_case_scores).to_dict()
            delta["point_estimate"] = float(ref_point) - float(other_point)
            metrics_row[metric_name] = delta

        rows.append(metrics_row)

    return {"reference_method": reference_method, "rows": rows}


def _write_main_table_csv(
    *,
    path: str | Path,
    method_order: list[str],
    method_metrics: dict[str, dict[str, Any]],
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_name",
                "e2e_status_acc",
                "e2e_status_acc_ci_low",
                "e2e_status_acc_ci_high",
                "status_acc_matched",
                "status_acc_matched_ci_low",
                "status_acc_matched_ci_high",
            ],
        )

        writer.writeheader()

        for method_name in method_order:
            main_metrics = method_metrics[method_name]["main_metrics"]

            writer.writerow(
                {
                    "method_name": method_name,
                    "e2e_status_acc": main_metrics["e2e_status_acc"]["point_estimate"],
                    "e2e_status_acc_ci_low": main_metrics["e2e_status_acc"]["ci_low"],
                    "e2e_status_acc_ci_high": main_metrics["e2e_status_acc"]["ci_high"],
                    "status_acc_matched": main_metrics["status_acc_matched"][
                        "point_estimate"
                    ],
                    "status_acc_matched_ci_low": main_metrics["status_acc_matched"][
                        "ci_low"
                    ],
                    "status_acc_matched_ci_high": main_metrics["status_acc_matched"][
                        "ci_high"
                    ],
                }
            )


def _write_metric_payloads(
    *,
    output_dir: Path,
    method_order: list[str],
    method_metrics: dict[str, dict[str, Any]],
    run_id: str,
    stage: str,
    split_name: str,
) -> dict[str, str]:
    main_table_payload = {
        "claim_id": CLAIM1,
        "run_id": run_id,
        "stage": stage,
        "split": split_name,
        "task_focus": CLAIM1_TASK_FOCUS,
        "root_claim_source": CLAIM1_ROOT_CLAIM_SOURCE,
        "matching_mode": CLAIM1_MATCHING_MODE,
        "adjudication_mode": CLAIM1_ADJUDICATION_MODE,
        "method_order": method_order,
        "rows": [_summary_row(name, method_metrics[name]) for name in method_order],
    }

    stepa_payload = {
        "claim_id": CLAIM1,
        "run_id": run_id,
        "stage": stage,
        "split": split_name,
        "task_focus": CLAIM1_TASK_FOCUS,
        "root_claim_source": CLAIM1_ROOT_CLAIM_SOURCE,
        "matching_mode": CLAIM1_MATCHING_MODE,
        "adjudication_mode": CLAIM1_ADJUDICATION_MODE,
        "method_order": method_order,
        "rows": [
            _appendix_stepa_row(name, method_metrics[name]) for name in method_order
        ],
    }

    fp_payload = {
        "claim_id": CLAIM1,
        "run_id": run_id,
        "stage": stage,
        "split": split_name,
        "task_focus": CLAIM1_TASK_FOCUS,
        "root_claim_source": CLAIM1_ROOT_CLAIM_SOURCE,
        "matching_mode": CLAIM1_MATCHING_MODE,
        "adjudication_mode": CLAIM1_ADJUDICATION_MODE,
        "method_order": method_order,
        "rows": [_appendix_fp_row(name, method_metrics[name]) for name in method_order],
        "pairwise_deltas_vs_main_system": _build_pairwise_delta_payload(
            method_metrics=method_metrics,
            method_order=method_order,
        ),
    }

    soft_payload = {
        "claim_id": CLAIM1,
        "run_id": run_id,
        "stage": stage,
        "split": split_name,
        "task_focus": CLAIM1_TASK_FOCUS,
        "root_claim_source": CLAIM1_ROOT_CLAIM_SOURCE,
        "matching_mode": CLAIM1_MATCHING_MODE,
        "adjudication_mode": CLAIM1_ADJUDICATION_MODE,
        **_build_soft_consistency_payload(
            method_metrics=method_metrics,
            method_order=method_order,
        ),
    }

    main_json_path = output_dir / "main_table.json"
    main_csv_path = output_dir / "main_table.csv"
    stepa_path = output_dir / "appendix_stepa_metrics.json"
    fp_path = output_dir / "appendix_e2e_fp_sensitive.json"
    soft_path = output_dir / "appendix_e2e_soft_consistency.json"
    _write_json(main_json_path, main_table_payload)

    _write_main_table_csv(
        path=main_csv_path,
        method_order=method_order,
        method_metrics=method_metrics,
    )

    _write_json(stepa_path, stepa_payload)
    _write_json(fp_path, fp_payload)
    _write_json(soft_path, soft_payload)

    return {
        "main_table_json": str(main_json_path),
        "main_table_csv": str(main_csv_path),
        "appendix_stepa_metrics": str(stepa_path),
        "appendix_e2e_fp_sensitive": str(fp_path),
        "appendix_e2e_soft_consistency": str(soft_path),
    }


def _gold_status_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("uid", "") or ""), str(row.get("claim_id", "") or "")): dict(row)
        for row in rows
    }


def _prediction_lookup(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["claim_id"]): row for row in _flatten_prediction_rows(result)}


def _serialize_review_packet(
    *,
    case: dict[str, Any],
    result: dict[str, Any],
    bundle: Any,
    gold_status_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    uid = str(bundle.uid)
    content = case.get("content") or {}
    extra = case.get("extraInfo") or {}
    meta = case.get("metaInfo") or {}
    prediction_lookup = _prediction_lookup(result)

    def _gold_with_status(row: dict[str, Any]) -> dict[str, Any]:
        claim_id = str(row.get("claim_id", "") or "")
        payload = dict(row)

        payload["status_eval"] = str(
            gold_status_lookup.get((uid, claim_id), {}).get("status_eval", "") or ""
        )

        return payload

    matched_pairs = []

    for pair in bundle.matched_pairs:
        gold_claim_id = str(pair.gold_row.get("claim_id", "") or "")
        pred_claim_id = str(pair.prediction_row.get("claim_id", "") or "")

        matched_pairs.append(
            {
                "gold_claim_id": gold_claim_id,
                "prediction_claim_id": pred_claim_id,
                "matching_mode": getattr(pair, "matching_mode", "semantic"),
                "direct_claim_id": getattr(pair, "direct_claim_id", ""),
                "gold_status_eval": str(
                    gold_status_lookup.get((uid, gold_claim_id), {}).get(
                        "status_eval", ""
                    )
                    or ""
                ),
                "prediction_status_eval": str(
                    prediction_lookup.get(pred_claim_id, {}).get("status_eval", "")
                    or ""
                ),
                "semantic_similarity": pair.semantic_similarity,
                "length_gap": pair.length_gap,
                "cost": pair.cost,
            }
        )

    return {
        "uid": uid,
        "method_name": str(result["method"]),
        "task_focus": CLAIM1_TASK_FOCUS,
        "root_claim_source": CLAIM1_ROOT_CLAIM_SOURCE,
        "matching_mode": str(getattr(bundle, "matching_mode", CLAIM1_MATCHING_MODE)),
        "adjudication_mode": CLAIM1_ADJUDICATION_MODE,
        "case_title": str(meta.get("caseName", "") or case.get("title", "") or ""),
        "stage": str(extra.get("stage", "") or ""),
        "cause": str(extra.get("sample_cause", "") or ""),
        "plaintiff_text": str(content.get("原告诉称", "") or ""),
        "judgment_result_text": str(content.get("裁判结果", "") or ""),
        "court_opinion_text": str(content.get("法院观点", "") or ""),
        "gold_claims": [_gold_with_status(dict(row)) for row in bundle.gold_rows],
        "prediction_claims": list(prediction_lookup.values()),
        "matched_pairs": matched_pairs,
        "unmatched_gold_claims": [
            _gold_with_status(dict(row)) for row in bundle.unmatched_gold_rows
        ],
        "unmatched_prediction_claims": [
            prediction_lookup[str(row.get("claim_id", "") or "")]
            for row in bundle.unmatched_prediction_rows
        ],
        "warnings": list(result["warnings"]),
        "trace": dict(result["trace"]),
    }


def _write_pilot_review_artifacts(
    *,
    output_dir: Path,
    pilot_cases: list[dict[str, Any]],
    method_order: list[str],
    results_by_method: dict[str, list[dict[str, Any]]],
    bundles_by_method: dict[str, list[Any]],
    gold_status_rows: list[dict[str, Any]],
) -> dict[str, str]:
    case_map = {case_uid(case): case for case in pilot_cases}
    gold_lookup = _gold_status_lookup(gold_status_rows)
    template_rows: list[dict[str, Any]] = []
    review_packets_dir = output_dir / "review_packets"

    for method_name in method_order:
        result_map = {
            str(result["case_uid"]): result for result in results_by_method[method_name]
        }

        for bundle in bundles_by_method[method_name]:
            uid = str(bundle.uid)

            packet = _serialize_review_packet(
                case=case_map[uid],
                result=result_map[uid],
                bundle=bundle,
                gold_status_lookup=gold_lookup,
            )

            _write_json(review_packets_dir / method_name / f"{uid}.json", packet)

            template_rows.append(
                {
                    "uid": uid,
                    "method": method_name,
                    "verdict": "",
                    "issue_tags": [],
                    "notes": "",
                    "reviewer": "",
                    "reviewed_at": "",
                }
            )

    template_path = output_dir / "pilot_review_template.jsonl"
    summary_path = output_dir / "pilot_review_summary.json"
    _write_jsonl(template_path, template_rows)

    _write_json(
        summary_path,
        {
            "row_count": len(template_rows),
            "method_count": len(method_order),
            "case_count": len(pilot_cases),
            "template_path": str(template_path),
        },
    )

    return {
        "review_packets_dir": str(review_packets_dir),
        "pilot_review_template": str(template_path),
        "pilot_review_summary": str(summary_path),
    }


def _load_and_validate_pilot_review(
    *,
    pilot_review_path: str | Path,
    expected_method_names: list[str],
    expected_case_uids: list[str],
) -> dict[str, Any]:
    rows = load_cases_from_jsonl(pilot_review_path)

    expected_pairs = {
        (uid, method_name)
        for uid in expected_case_uids
        for method_name in expected_method_names
    }

    actual_pairs: set[tuple[str, str]] = set()
    critical_rows: list[dict[str, Any]] = []

    for row in rows:
        uid = str(row.get("uid", "") or "")
        method_name = str(row.get("method", "") or "")
        verdict = str(row.get("verdict", "") or "")

        if not uid or not method_name:
            raise ValueError("Pilot review rows must contain non-empty uid and method.")

        if verdict not in ALLOWED_PILOT_VERDICTS:
            raise ValueError(
                f"Invalid pilot review verdict for uid={uid} method={method_name}: {verdict}"
            )

        key = (uid, method_name)

        if key in actual_pairs:
            raise ValueError(f"Duplicate pilot review row for {key}")

        actual_pairs.add(key)

        if verdict == "critical_issue":
            critical_rows.append(row)

    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extras = sorted(actual_pairs - expected_pairs)

        raise ValueError(
            f"Pilot review coverage mismatch. Missing={missing} Extras={extras}"
        )

    return {
        "row_count": len(rows),
        "critical_issue_count": len(critical_rows),
        "critical_rows": critical_rows,
        "rows": rows,
    }


def _claim1_primary_metric_factory(
    *,
    gold_status_rows: list[dict[str, Any]],
) -> Any:
    gold_lookup = _gold_status_lookup(gold_status_rows)

    def _primary_metric(case_bundles: list[Any]) -> PrimaryMetricResult:
        case_scores: dict[str, float] = {}

        for bundle in case_bundles:
            uid = str(bundle.uid)
            gold_count = len(bundle.gold_rows)

            if gold_count == 0:
                case_scores[uid] = 0.0
                continue

            correct = 0

            for pair in bundle.matched_pairs:
                gold_claim_id = str(pair.gold_row.get("claim_id", "") or "")

                gold_status = str(
                    gold_lookup.get((uid, gold_claim_id), {}).get("status_eval", "")
                    or ""
                )

                pred_status = str(pair.prediction_row.get("status_eval", "") or "")

                gold_2class = (
                    "DEFEATED" if gold_status == "HYPOTHETICAL" else gold_status
                )

                pred_2class = (
                    "DEFEATED" if pred_status == "HYPOTHETICAL" else pred_status
                )

                if gold_2class == pred_2class:
                    correct += 1

            case_scores[uid] = correct / gold_count

        mean_dev = sum(case_scores.values()) / len(case_scores) if case_scores else 0.0

        return PrimaryMetricResult(
            case_scores=case_scores,
            mean_dev=mean_dev,
            details={"metric_name": "e2e_status_acc"},
        )

    return _primary_metric


def _write_pilot_gate(
    *,
    output_dir: Path,
    method_order: list[str],
    case_uids: list[str],
    summary: dict[str, Any],
) -> str:
    gate_path = output_dir / "pilot_gate.json"

    payload = {
        "status": "ready_for_manual_review",
        "task_focus": CLAIM1_TASK_FOCUS,
        "root_claim_source": CLAIM1_ROOT_CLAIM_SOURCE,
        "matching_mode": CLAIM1_MATCHING_MODE,
        "adjudication_mode": CLAIM1_ADJUDICATION_MODE,
        "runtime_passed": bool(summary.get("failure_count", 0) == 0),
        "schema_passed": True,
        "matching_passed": True,
        "metrics_passed": True,
        "requires_manual_review": True,
        "selected_case_uids": list(case_uids),
        "method_names": list(method_order),
        "review_row_count_expected": len(case_uids) * len(method_order),
    }

    _write_json(gate_path, payload)
    return str(gate_path)


def _promote_split_outputs(
    *,
    split_dir: Path,
    claim_root: Path,
) -> dict[str, str]:
    claim_root.mkdir(parents=True, exist_ok=True)

    promoted = {
        "main_table_json": str(claim_root / "main_table.json"),
        "main_table_csv": str(claim_root / "main_table.csv"),
        "appendix_stepa_metrics": str(claim_root / "appendix_stepa_metrics.json"),
        "appendix_e2e_fp_sensitive": str(claim_root / "appendix_e2e_fp_sensitive.json"),
        "appendix_e2e_soft_consistency": str(
            claim_root / "appendix_e2e_soft_consistency.json"
        ),
    }

    for name, target in promoted.items():
        source = split_dir / Path(target).name
        Path(target).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    return promoted


def run_claim1_experiment(
    *,
    stage: str,
    freeze_run_id: str,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    storage_root_dir: str | Path | None = None,
    pilot_review_path: str | Path | None = None,
    methods: list[str] | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    resume: bool = False,
    encoder: Any | None = None,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in {PILOT_STAGE, FULL_STAGE}:
        raise ValueError(f"Unsupported Claim 1 stage: {stage}")

    freeze_bundle = _load_freeze_bundle(
        freeze_run_id=freeze_run_id,
        reports_root=reports_root,
    )

    method_order, active_registry = _resolve_registry(
        freeze_bundle=freeze_bundle,
        registry=registry,
        methods=methods,
    )

    claim_root = Path(reports_root) / run_id / "claim1"
    case_map = _load_cases_by_uid(input_path)

    gold_claim_rows, gold_status_rows = _load_gold_rows(
        gold_claims_path=freeze_bundle.gold_refs["gold_claims_path"],
        gold_status_path=freeze_bundle.gold_refs["gold_status_path"],
    )

    matching_config = _make_matching_config(freeze_bundle.matching_protocol_snapshot)
    budget = dict(freeze_bundle.budget_grid.get("full_budget", {}) or {})

    retrieval_config = dict(
        freeze_bundle.manifest.get("shared_constraints", {}).get(
            "retrieval_config_snapshot", {}
        )
        or {}
    )

    seed = int(
        freeze_bundle.manifest.get("shared_constraints", {}).get("seed", 20260307)
    )

    resolved_encoder = encoder

    if resolved_encoder is None:
        resolved_encoder = SentenceTransformerTextEncoder(
            SystemConfig().path.embedding_model_path
        )

    if stage == PILOT_STAGE:
        pilot_dir = claim_root / "pilot"

        pilot_case_uids = _read_ids(
            freeze_bundle.run_root / "preflight" / "selected_dryrun_cases.json",
            "selected_case_uids",
        )

        pilot_cases = _select_cases(case_map=case_map, case_uids=pilot_case_uids)
        pilot_gold_claim_rows = _filter_rows_by_uids(gold_claim_rows, pilot_case_uids)
        pilot_gold_status_rows = _filter_rows_by_uids(gold_status_rows, pilot_case_uids)

        execution = _run_method_matrix(
            cases=pilot_cases,
            case_uids=pilot_case_uids,
            active_method_names=method_order,
            active_registry=active_registry,
            storage_root_dir=str(storage_root_dir) if storage_root_dir else None,
            test_mode_no_learning=CLAIM1_TEST_MODE_NO_LEARNING,
            budget=budget,
            seed=seed,
            retrieval_config=retrieval_config,
            output_dir=pilot_dir,
            verbose=verbose,
            show_progress=show_progress,
            resume=resume,
        )

        bundles_by_method = _build_case_bundles_by_method(
            case_uids=pilot_case_uids,
            gold_claim_rows=pilot_gold_claim_rows,
            flattened_predictions=execution["flattened_predictions"],
            active_method_names=method_order,
            encoder=resolved_encoder,
            config=matching_config,
        )

        method_metrics = _evaluate_method_metrics(
            bundles_by_method=bundles_by_method,
            gold_status_rows=pilot_gold_status_rows,
        )

        metric_paths = _write_metric_payloads(
            output_dir=pilot_dir,
            method_order=method_order,
            method_metrics=method_metrics,
            run_id=run_id,
            stage=PILOT_STAGE,
            split_name="pilot",
        )

        review_paths = _write_pilot_review_artifacts(
            output_dir=pilot_dir,
            pilot_cases=pilot_cases,
            method_order=method_order,
            results_by_method=execution["results_by_method"],
            bundles_by_method=bundles_by_method,
            gold_status_rows=pilot_gold_status_rows,
        )

        gate_path = _write_pilot_gate(
            output_dir=pilot_dir,
            method_order=method_order,
            case_uids=pilot_case_uids,
            summary=execution["summary"],
        )

        return {
            "claim_id": CLAIM1,
            "stage": PILOT_STAGE,
            "status": "ready_for_manual_review",
            "registry_profile": str(
                freeze_bundle.method_registry_snapshot.get(
                    "registry_profile", INTERNAL_DEFAULT_PROFILE
                )
            ),
            "run_id": run_id,
            "freeze_run_id": freeze_run_id,
            "method_names": method_order,
            "case_uids": pilot_case_uids,
            "artifacts": {
                **metric_paths,
                **review_paths,
                "pilot_gate": gate_path,
                "execution_summary": str(pilot_dir / "execution_summary.json"),
            },
        }

    pilot_dir = claim_root / "pilot"
    pilot_gate_path = pilot_dir / "pilot_gate.json"

    if not pilot_gate_path.exists():
        raise FileNotFoundError(
            f"Missing pilot gate file; run Claim 1 pilot first: {pilot_gate_path}"
        )

    pilot_gate = _read_json(pilot_gate_path)

    for gate_field in (
        "runtime_passed",
        "schema_passed",
        "matching_passed",
        "metrics_passed",
    ):
        if not bool(pilot_gate.get(gate_field, False)):
            raise ValueError(
                f"Claim 1 pilot gate has not passed `{gate_field}`; cannot start full run."
            )

    if pilot_review_path is None:
        raise ValueError("Claim 1 full run requires `pilot_review_path`.")

    pilot_case_uids = list(pilot_gate.get("selected_case_uids", []))

    if list(pilot_gate.get("method_names", [])) != method_order:
        raise ValueError(
            "Claim 1 full run method set does not match the pilot gate method set."
        )

    review_summary = _load_and_validate_pilot_review(
        pilot_review_path=pilot_review_path,
        expected_method_names=method_order,
        expected_case_uids=pilot_case_uids,
    )

    if review_summary["critical_issue_count"] > 0:
        raise ValueError(
            "Claim 1 pilot review contains critical issues; abort full run."
        )

    _write_json(claim_root / "pilot" / "pilot_review_validation.json", review_summary)

    dev_case_uids = _read_ids(
        Path(freeze_bundle.split_refs["dev_ids_path"]),
        "ids",
    )

    test_case_uids = _read_ids(
        Path(freeze_bundle.split_refs["test_ids_path"]),
        "ids",
    )

    dev_case_uids = _restrict_case_uids_to_gold_scope(
        case_uids=dev_case_uids,
        gold_rows=gold_claim_rows,
        split_name="dev",
    )

    test_case_uids = _restrict_case_uids_to_gold_scope(
        case_uids=test_case_uids,
        gold_rows=gold_claim_rows,
        split_name="test",
    )

    dev_cases = _select_cases(case_map=case_map, case_uids=dev_case_uids)
    test_cases = _select_cases(case_map=case_map, case_uids=test_case_uids)
    dev_gold_claim_rows = _filter_rows_by_uids(gold_claim_rows, dev_case_uids)
    dev_gold_status_rows = _filter_rows_by_uids(gold_status_rows, dev_case_uids)
    test_gold_claim_rows = _filter_rows_by_uids(gold_claim_rows, test_case_uids)
    test_gold_status_rows = _filter_rows_by_uids(gold_status_rows, test_case_uids)
    dev_dir = claim_root / "dev"
    test_dir = claim_root / "test"
    dev_scope_path = claim_root / "dev_case_uids_claim1_scope.json"
    test_scope_path = claim_root / "test_case_uids_claim1_scope.json"

    _write_json(
        dev_scope_path,
        {
            "ids": list(dev_case_uids),
            "raw_case_count": len(
                _read_ids(Path(freeze_bundle.split_refs["dev_ids_path"]), "ids")
            ),
            "filtered_case_count": len(dev_case_uids),
        },
    )

    _write_json(
        test_scope_path,
        {
            "ids": list(test_case_uids),
            "raw_case_count": len(
                _read_ids(Path(freeze_bundle.split_refs["test_ids_path"]), "ids")
            ),
            "filtered_case_count": len(test_case_uids),
        },
    )

    dev_execution = _run_method_matrix(
        cases=dev_cases,
        case_uids=dev_case_uids,
        active_method_names=method_order,
        active_registry=active_registry,
        storage_root_dir=str(storage_root_dir) if storage_root_dir else None,
        test_mode_no_learning=CLAIM1_TEST_MODE_NO_LEARNING,
        budget=budget,
        seed=seed,
        retrieval_config=retrieval_config,
        output_dir=dev_dir,
        verbose=verbose,
        show_progress=show_progress,
        resume=resume,
    )

    try:
        dev_gate = run_frozen_matching_protocol(
            gold_rows=dev_gold_claim_rows,
            method_predictions=dev_execution["flattened_predictions"],
            primary_metric_fn=_claim1_primary_metric_factory(
                gold_status_rows=dev_gold_status_rows
            ),
            encoder=resolved_encoder,
            artifact_dir=dev_dir / "matching",
            dev_ids_path=dev_scope_path,
        )

    except MatchingRobustnessGateError as exc:
        _write_json(
            claim_root / "run_summary.json",
            {
                "claim_id": CLAIM1,
                "stage": FULL_STAGE,
                "status": "failed",
                "failed_phase": "dev_gate",
                "task_focus": CLAIM1_TASK_FOCUS,
                "root_claim_source": CLAIM1_ROOT_CLAIM_SOURCE,
                "matching_mode": CLAIM1_MATCHING_MODE,
                "adjudication_mode": CLAIM1_ADJUDICATION_MODE,
                "error_type": type(exc).__name__,
                "error_detail": str(exc),
                "freeze_run_id": freeze_run_id,
                "run_id": run_id,
            },
        )

        raise

    _write_json(dev_dir / "robustness_gate.json", dev_gate.gate)

    dev_method_metrics = _evaluate_method_metrics(
        bundles_by_method={
            method_name: list(
                dev_gate.scenario_results["base"][method_name].case_bundles
            )
            for method_name in method_order
        },
        gold_status_rows=dev_gold_status_rows,
    )

    _write_json(dev_dir / "method_metrics.json", dev_method_metrics)

    test_execution = _run_method_matrix(
        cases=test_cases,
        case_uids=test_case_uids,
        active_method_names=method_order,
        active_registry=active_registry,
        storage_root_dir=str(storage_root_dir) if storage_root_dir else None,
        test_mode_no_learning=CLAIM1_TEST_MODE_NO_LEARNING,
        budget=budget,
        seed=seed,
        retrieval_config=retrieval_config,
        output_dir=test_dir,
        verbose=verbose,
        show_progress=show_progress,
        resume=resume,
    )

    test_bundles_by_method = _build_case_bundles_by_method(
        case_uids=test_case_uids,
        gold_claim_rows=test_gold_claim_rows,
        flattened_predictions=test_execution["flattened_predictions"],
        active_method_names=method_order,
        encoder=resolved_encoder,
        config=matching_config,
    )

    test_method_metrics = _evaluate_method_metrics(
        bundles_by_method=test_bundles_by_method,
        gold_status_rows=test_gold_status_rows,
    )

    _write_json(test_dir / "method_metrics.json", test_method_metrics)

    _write_json(
        test_dir / "pairwise_deltas.json",
        _build_pairwise_delta_payload(
            method_metrics=test_method_metrics,
            method_order=method_order,
        ),
    )

    metric_paths = _write_metric_payloads(
        output_dir=test_dir,
        method_order=method_order,
        method_metrics=test_method_metrics,
        run_id=run_id,
        stage=FULL_STAGE,
        split_name="test",
    )

    promoted_paths = _promote_split_outputs(split_dir=test_dir, claim_root=claim_root)

    summary = {
        "claim_id": CLAIM1,
        "stage": FULL_STAGE,
        "status": "completed",
        "registry_profile": str(
            freeze_bundle.method_registry_snapshot.get(
                "registry_profile", INTERNAL_DEFAULT_PROFILE
            )
        ),
        "task_focus": CLAIM1_TASK_FOCUS,
        "root_claim_source": CLAIM1_ROOT_CLAIM_SOURCE,
        "matching_mode": CLAIM1_MATCHING_MODE,
        "adjudication_mode": CLAIM1_ADJUDICATION_MODE,
        "run_id": run_id,
        "freeze_run_id": freeze_run_id,
        "method_names": method_order,
        "pilot_review_path": str(pilot_review_path),
        "pilot_review_summary": review_summary,
        "dev_gate": dev_gate.gate,
        "artifacts": {
            **metric_paths,
            **promoted_paths,
            "dev_scope": str(dev_scope_path),
            "dev_execution_summary": str(dev_dir / "execution_summary.json"),
            "dev_method_metrics": str(dev_dir / "method_metrics.json"),
            "dev_matching_dir": str(dev_dir / "matching"),
            "test_scope": str(test_scope_path),
            "test_execution_summary": str(test_dir / "execution_summary.json"),
            "test_method_metrics": str(test_dir / "method_metrics.json"),
            "test_pairwise_deltas": str(test_dir / "pairwise_deltas.json"),
        },
    }

    _write_json(claim_root / "run_summary.json", summary)
    return summary


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
    top_k: int = 3,
    aggregation: str = "any",
    agreement_threshold: float = 0.70,
    alignment_threshold: float = 0.10,
) -> dict[str, Any]:
    """Run the finalized Step 11 Claim 2 orchestration flow."""
    return _run_claim2_experiment(
        freeze_run_id=freeze_run_id,
        source_claim1_run_id=source_claim1_run_id,
        run_id=run_id,
        reports_root=reports_root,
        input_path=input_path,
        evaluator_profile_path=evaluator_profile_path,
        methods=methods,
        resume=resume,
        verbose=verbose,
        show_progress=show_progress,
        top_k=top_k,
        aggregation=aggregation,
        agreement_threshold=agreement_threshold,
        alignment_threshold=alignment_threshold,
    )


def run_claim3_experiment(
    *,
    command: str,
    freeze_run_id: str | None = None,
    source_claim1_run_id: str | None = None,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    warmup_points: tuple[int, ...] = (0, 25, 50, 100),
    branch: str | None = None,
    resume: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run Claim 3 prepare/run/summarize orchestration."""
    action = str(command).strip().lower()

    if action == "prepare":
        if not freeze_run_id or not source_claim1_run_id:
            raise ValueError(
                "Claim 3 prepare requires freeze_run_id and source_claim1_run_id."
            )

        return _prepare_claim3_experiment(
            freeze_run_id=freeze_run_id,
            source_claim1_run_id=source_claim1_run_id,
            run_id=run_id,
            reports_root=reports_root,
            input_path=input_path,
            warmup_points=warmup_points,
        )

    if action == "run":
        if not branch:
            raise ValueError("Claim 3 run requires one branch name.")

        return _run_claim3_branch(
            run_id=run_id,
            reports_root=reports_root,
            branch=branch,
            resume=resume,
            show_progress=show_progress,
        )

    if action == "summarize":
        return _summarize_claim3_experiment(
            run_id=run_id,
            reports_root=reports_root,
        )

    raise ValueError(f"Unknown Claim 3 command: {command}")


def run_claim4_experiment(
    *,
    cases: list[dict[str, Any]] | None = None,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Claim 4 experiment skeleton and return structured result."""
    return _build_skeleton_result(claim_id=4, cases=cases, registry=registry)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot_parser = subparsers.add_parser("claim1-pilot")
    pilot_parser.add_argument("--freeze-run-id", required=True)
    pilot_parser.add_argument("--run-id", required=True)
    pilot_parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    pilot_parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    pilot_parser.add_argument("--memory-dir", default="")
    pilot_parser.add_argument("--methods", nargs="*", default=None)
    pilot_parser.add_argument("--verbose", action="store_true")
    pilot_parser.add_argument("--resume", action="store_true")
    pilot_parser.add_argument("--no-progress", action="store_true")
    full_parser = subparsers.add_parser("claim1-full")
    full_parser.add_argument("--freeze-run-id", required=True)
    full_parser.add_argument("--run-id", required=True)
    full_parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    full_parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    full_parser.add_argument("--memory-dir", default="")
    full_parser.add_argument("--pilot-review-path", required=True)
    full_parser.add_argument("--methods", nargs="*", default=None)
    full_parser.add_argument("--verbose", action="store_true")
    full_parser.add_argument("--resume", action="store_true")
    full_parser.add_argument("--no-progress", action="store_true")
    claim2_parser = subparsers.add_parser("claim2-run")
    claim2_parser.add_argument("--freeze-run-id", required=True)
    claim2_parser.add_argument("--source-claim1-run-id", required=True)
    claim2_parser.add_argument("--run-id", required=True)
    claim2_parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    claim2_parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    claim2_parser.add_argument("--evaluator-profile-path", required=True)
    claim2_parser.add_argument("--methods", nargs="*", default=None)
    claim2_parser.add_argument("--resume", action="store_true")
    claim2_parser.add_argument("--verbose", action="store_true")
    claim2_parser.add_argument("--no-progress", action="store_true")
    claim2_parser.add_argument("--top-k", type=int, default=3)
    claim2_parser.add_argument("--aggregation", default="any")
    claim2_parser.add_argument("--agreement-threshold", type=float, default=0.70)
    claim2_parser.add_argument("--alignment-threshold", type=float, default=0.10)
    claim3_prepare_parser = subparsers.add_parser("claim3-prepare")
    claim3_prepare_parser.add_argument("--freeze-run-id", required=True)
    claim3_prepare_parser.add_argument("--source-claim1-run-id", required=True)
    claim3_prepare_parser.add_argument("--run-id", required=True)

    claim3_prepare_parser.add_argument(
        "--reports-root", default=str(DEFAULT_REPORTS_ROOT)
    )

    claim3_prepare_parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    claim3_prepare_parser.add_argument("--warmup-points", default="0,25,50,100")
    claim3_run_parser = subparsers.add_parser("claim3-run")
    claim3_run_parser.add_argument("--run-id", required=True)
    claim3_run_parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))

    claim3_run_parser.add_argument(
        "--branch", required=True, choices=("normal", "fixed-pack")
    )

    claim3_run_parser.add_argument("--resume", action="store_true")
    claim3_run_parser.add_argument("--no-progress", action="store_true")
    claim3_summarize_parser = subparsers.add_parser("claim3-summarize")
    claim3_summarize_parser.add_argument("--run-id", required=True)

    claim3_summarize_parser.add_argument(
        "--reports-root", default=str(DEFAULT_REPORTS_ROOT)
    )

    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "claim1-pilot":
        payload = run_claim1_experiment(
            stage=PILOT_STAGE,
            freeze_run_id=args.freeze_run_id,
            run_id=args.run_id,
            reports_root=args.reports_root,
            input_path=args.input_path,
            storage_root_dir=args.memory_dir or None,
            methods=args.methods,
            verbose=bool(args.verbose),
            show_progress=not bool(args.no_progress),
            resume=bool(args.resume),
        )

    elif args.command == "claim1-full":
        payload = run_claim1_experiment(
            stage=FULL_STAGE,
            freeze_run_id=args.freeze_run_id,
            run_id=args.run_id,
            reports_root=args.reports_root,
            input_path=args.input_path,
            storage_root_dir=args.memory_dir or None,
            pilot_review_path=args.pilot_review_path,
            methods=args.methods,
            verbose=bool(args.verbose),
            show_progress=not bool(args.no_progress),
            resume=bool(args.resume),
        )

    elif args.command == "claim2-run":
        payload = run_claim2_experiment(
            freeze_run_id=args.freeze_run_id,
            source_claim1_run_id=args.source_claim1_run_id,
            run_id=args.run_id,
            reports_root=args.reports_root,
            input_path=args.input_path,
            evaluator_profile_path=args.evaluator_profile_path,
            methods=args.methods,
            resume=bool(args.resume),
            verbose=bool(args.verbose),
            show_progress=not bool(args.no_progress),
            top_k=int(args.top_k),
            aggregation=str(args.aggregation),
            agreement_threshold=float(args.agreement_threshold),
            alignment_threshold=float(args.alignment_threshold),
        )

    elif args.command == "claim3-prepare":
        warmup_points = tuple(
            int(item.strip())
            for item in str(args.warmup_points).split(",")
            if item.strip()
        )

        payload = run_claim3_experiment(
            command="prepare",
            freeze_run_id=args.freeze_run_id,
            source_claim1_run_id=args.source_claim1_run_id,
            run_id=args.run_id,
            reports_root=args.reports_root,
            input_path=args.input_path,
            warmup_points=warmup_points,
        )

    elif args.command == "claim3-run":
        payload = run_claim3_experiment(
            command="run",
            run_id=args.run_id,
            reports_root=args.reports_root,
            branch=args.branch,
            resume=bool(args.resume),
            show_progress=not bool(args.no_progress),
        )

    else:
        payload = run_claim3_experiment(
            command="summarize",
            run_id=args.run_id,
            reports_root=args.reports_root,
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
