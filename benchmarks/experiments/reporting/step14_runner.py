"""Step 14 appendix rebuild runner.

This module intentionally avoids any model execution. It rebuilds appendix-style
artifacts from existing raw prediction outputs after label-only status updates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.experiments.core import claim3_runner, claim4_runner
from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.eval import (
    FROZEN_MATCHING_PROTOCOL_VERSION,
    MatchingConfig,
    SentenceTransformerTextEncoder,
    build_method_case_matches,
    evaluate_claim1_metrics,
    holm_bonferroni,
    mcnemar_test,
    paired_case_bootstrap,
    stuart_maxwell_test,
    summarize_claim3_curve,
)
from benchmarks.experiments.eval._metric_utils import (
    STATUS_CLASSES_3,
    collapse_status_eval,
    normalize_status_eval,
)
from benchmarks.experiments.methods.base import validate_method_result
from mas.config import SystemConfig

DEFAULT_STEP14_INTERNAL_CLAIM1_RUN_ID = "20260323_step10_claim1_status_small"
DEFAULT_STEP14_EXTERNAL_CLAIM1_RUN_ID = "20260326_step10_claim1_external_v2"
DEFAULT_STEP14_INTERNAL_CLAIM2_RUN_ID = "20260327_step11_claim2_internal_assistant3"
DEFAULT_STEP14_EXTERNAL_CLAIM2_RUN_ID = "20260327_step11_claim2_external_assistant3"
DEFAULT_STEP14_CLAIM3_RUN_ID = "20260327_step12_claim3_internal_current"
DEFAULT_STEP14_CLAIM4_RUN_ID = "20260328_step13_claim4_internal_current"
STEP14_PROTOCOL_VERSION = "step14_appendix_rebuild_v1"
STEP14_STATUS_MODE = "label_only"


@dataclass(frozen=True)
class Step14Context:
    run_id: str
    reports_root: Path
    run_root: Path
    step14_root: Path
    manifest_path: Path
    dependency_state_path: Path
    rebuild_plan_path: Path
    appendix_index_json_path: Path
    appendix_index_md_path: Path
    rebuild_summary_path: Path
    gold_claims_path: Path
    prepared_status_path: Path
    source_runs: dict[str, str]


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


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        file_path.write_text("", encoding="utf-8")
        return

    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)

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


def _copy_file(src: str | Path, dst: str | Path) -> None:
    source = Path(src)

    if not source.exists():
        raise FileNotFoundError(f"Missing source file: {source}")

    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    return [dict(row) for row in load_cases_from_jsonl(path)]


def _make_matching_config(snapshot: dict[str, Any]) -> MatchingConfig:
    return MatchingConfig(**dict(snapshot.get("base_config", {}) or {}))


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

        payload = dict(claim_row)
        payload["status_eval"] = status_eval
        flattened.append(payload)

    return flattened


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
        "claim_id": 1,
        "run_id": run_id,
        "stage": stage,
        "split": split_name,
        "method_order": method_order,
        "rows": [_summary_row(name, method_metrics[name]) for name in method_order],
    }

    stepa_payload = {
        "claim_id": 1,
        "run_id": run_id,
        "stage": stage,
        "split": split_name,
        "method_order": method_order,
        "rows": [
            _appendix_stepa_row(name, method_metrics[name]) for name in method_order
        ],
    }

    fp_payload = {
        "claim_id": 1,
        "run_id": run_id,
        "stage": stage,
        "split": split_name,
        "method_order": method_order,
        "rows": [_appendix_fp_row(name, method_metrics[name]) for name in method_order],
        "pairwise_deltas_vs_main_system": _build_pairwise_delta_payload(
            method_metrics=method_metrics,
            method_order=method_order,
        ),
    }

    soft_payload = {
        "claim_id": 1,
        "run_id": run_id,
        "stage": stage,
        "split": split_name,
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


def _claim_identity_hash(rows: list[dict[str, Any]]) -> tuple[str, int]:
    items = sorted(
        (str(row.get("uid", "") or ""), str(row.get("claim_id", "") or ""))
        for row in rows
        if str(row.get("uid", "") or "").strip()
        and str(row.get("claim_id", "") or "").strip()
    )

    digest = hashlib.sha256(json.dumps(items, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest(), len(items)


def _resolve_metric_root(claim_root: Path) -> Path:
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


def _read_ids(path: str | Path, key: str = "ids") -> list[str]:
    payload = _read_json(path)
    values = payload.get(key, [])

    if not isinstance(values, list) or not values:
        raise ValueError(f"{Path(path).name} must contain a non-empty `{key}` list.")

    return [str(value) for value in values]


def _load_source_claim1_run(
    *,
    reports_root: Path,
    run_id: str,
) -> tuple[dict[str, Any], Path]:
    claim_root = reports_root / run_id / "claim1"
    summary = _read_json(claim_root / "run_summary.json")

    if int(summary.get("claim_id", 0) or 0) != 1:
        raise ValueError(f"{run_id} is not a Claim 1 run.")

    return summary, claim_root


def _load_matching_snapshot(
    *,
    reports_root: Path,
    freeze_run_id: str,
) -> dict[str, Any]:
    snapshot = _read_json(
        reports_root / freeze_run_id / "matching_protocol_snapshot.json"
    )

    if (
        str(snapshot.get("protocol_version", "") or "")
        != FROZEN_MATCHING_PROTOCOL_VERSION
    ):
        raise ValueError(
            "Matching protocol version is incompatible with current Step 14 rebuild."
        )

    return snapshot


def _load_gold_refs(
    *,
    reports_root: Path,
    freeze_run_id: str,
) -> dict[str, Any]:
    return _read_json(reports_root / freeze_run_id / "gold_refs.json")


def _build_ctx(
    *,
    run_id: str,
    reports_root: str | Path,
    status_path: str | Path,
    source_runs: dict[str, str],
) -> Step14Context:
    reports_root_path = Path(reports_root)
    run_root = reports_root_path / run_id
    step14_root = run_root / "step14"

    internal_summary, _ = _load_source_claim1_run(
        reports_root=reports_root_path,
        run_id=source_runs["claim1_internal"],
    )

    gold_refs = _load_gold_refs(
        reports_root=reports_root_path,
        freeze_run_id=str(internal_summary["freeze_run_id"]),
    )

    return Step14Context(
        run_id=run_id,
        reports_root=reports_root_path,
        run_root=run_root,
        step14_root=step14_root,
        manifest_path=step14_root / "manifest.json",
        dependency_state_path=step14_root / "dependency_state.json",
        rebuild_plan_path=step14_root / "rebuild_plan.json",
        appendix_index_json_path=step14_root / "appendix_index.json",
        appendix_index_md_path=step14_root / "appendix_index.md",
        rebuild_summary_path=step14_root / "rebuild_summary.json",
        gold_claims_path=Path(gold_refs["gold_claims_path"]),
        prepared_status_path=Path(status_path),
        source_runs=source_runs,
    )


def _source_dependency_state(
    *,
    reports_root: Path,
    source_runs: dict[str, str],
) -> dict[str, Any]:
    claim1_internal_summary, claim1_internal_root = _load_source_claim1_run(
        reports_root=reports_root,
        run_id=source_runs["claim1_internal"],
    )

    claim1_external_summary, claim1_external_root = _load_source_claim1_run(
        reports_root=reports_root,
        run_id=source_runs["claim1_external"],
    )

    claim2_internal_root = reports_root / source_runs["claim2_internal"] / "claim2"
    claim2_external_root = reports_root / source_runs["claim2_external"] / "claim2"
    claim3_root = reports_root / source_runs["claim3"] / "claim3"
    claim4_root = reports_root / source_runs["claim4"] / "claim4"

    return {
        "claim1_internal": {
            "run_id": source_runs["claim1_internal"],
            "run_summary_path": str(claim1_internal_root / "run_summary.json"),
            "run_summary_sha256": _sha256_file(
                claim1_internal_root / "run_summary.json"
            ),
            "freeze_run_id": str(claim1_internal_summary["freeze_run_id"]),
            "dev_scope_path": str(claim1_internal_summary["artifacts"]["dev_scope"]),
            "dev_scope_sha256": _sha256_file(
                claim1_internal_summary["artifacts"]["dev_scope"]
            ),
            "test_scope_path": str(claim1_internal_summary["artifacts"]["test_scope"]),
            "test_scope_sha256": _sha256_file(
                claim1_internal_summary["artifacts"]["test_scope"]
            ),
        },
        "claim1_external": {
            "run_id": source_runs["claim1_external"],
            "run_summary_path": str(claim1_external_root / "run_summary.json"),
            "run_summary_sha256": _sha256_file(
                claim1_external_root / "run_summary.json"
            ),
            "freeze_run_id": str(claim1_external_summary["freeze_run_id"]),
            "dev_scope_path": str(claim1_external_summary["artifacts"]["dev_scope"]),
            "dev_scope_sha256": _sha256_file(
                claim1_external_summary["artifacts"]["dev_scope"]
            ),
            "test_scope_path": str(claim1_external_summary["artifacts"]["test_scope"]),
            "test_scope_sha256": _sha256_file(
                claim1_external_summary["artifacts"]["test_scope"]
            ),
        },
        "claim2_internal": {
            "run_id": source_runs["claim2_internal"],
            "run_summary_path": str(claim2_internal_root / "run_summary.json"),
            "run_summary_sha256": _sha256_file(
                claim2_internal_root / "run_summary.json"
            ),
        },
        "claim2_external": {
            "run_id": source_runs["claim2_external"],
            "run_summary_path": str(claim2_external_root / "run_summary.json"),
            "run_summary_sha256": _sha256_file(
                claim2_external_root / "run_summary.json"
            ),
        },
        "claim3": {
            "run_id": source_runs["claim3"],
            "manifest_path": str(claim3_root / "warmup_manifest.json"),
            "manifest_sha256": _sha256_file(claim3_root / "warmup_manifest.json"),
        },
        "claim4": {
            "run_id": source_runs["claim4"],
            "manifest_path": str(claim4_root / "manifest.json"),
            "manifest_sha256": _sha256_file(claim4_root / "manifest.json"),
        },
    }


def prepare_step14_reporting(
    *,
    run_id: str,
    reports_root: str | Path,
    status_path: str | Path,
    claim1_internal_run_id: str = DEFAULT_STEP14_INTERNAL_CLAIM1_RUN_ID,
    claim1_external_run_id: str = DEFAULT_STEP14_EXTERNAL_CLAIM1_RUN_ID,
    claim2_internal_run_id: str = DEFAULT_STEP14_INTERNAL_CLAIM2_RUN_ID,
    claim2_external_run_id: str = DEFAULT_STEP14_EXTERNAL_CLAIM2_RUN_ID,
    claim3_run_id: str = DEFAULT_STEP14_CLAIM3_RUN_ID,
    claim4_run_id: str = DEFAULT_STEP14_CLAIM4_RUN_ID,
) -> dict[str, Any]:
    source_runs = {
        "claim1_internal": claim1_internal_run_id,
        "claim1_external": claim1_external_run_id,
        "claim2_internal": claim2_internal_run_id,
        "claim2_external": claim2_external_run_id,
        "claim3": claim3_run_id,
        "claim4": claim4_run_id,
    }

    ctx = _build_ctx(
        run_id=run_id,
        reports_root=reports_root,
        status_path=status_path,
        source_runs=source_runs,
    )

    ctx.step14_root.mkdir(parents=True, exist_ok=True)
    gold_claim_rows = _load_jsonl_rows(ctx.gold_claims_path)
    status_rows = _load_jsonl_rows(status_path)
    gold_claim_identity_hash, gold_claim_count = _claim_identity_hash(gold_claim_rows)
    status_identity_hash, status_count = _claim_identity_hash(status_rows)

    dependency_state = {
        "status_mode": STEP14_STATUS_MODE,
        "gold_claims_path": str(ctx.gold_claims_path),
        "gold_claims_sha256": _sha256_file(ctx.gold_claims_path),
        "gold_claim_identity_hash": gold_claim_identity_hash,
        "gold_claim_row_count": gold_claim_count,
        "prepared_status_path": str(status_path),
        "prepared_status_sha256": _sha256_file(status_path),
        "prepared_status_identity_hash": status_identity_hash,
        "prepared_status_row_count": status_count,
        "sources": _source_dependency_state(
            reports_root=ctx.reports_root,
            source_runs=source_runs,
        ),
    }

    rebuild_plan = {
        "claim1_internal": "rebuild",
        "claim1_external": "rebuild",
        "claim1_comparison": "index_only",
        "claim2_internal": "reference_only",
        "claim2_external": "reference_only",
        "claim3": "rebuild_from_raw_predictions",
        "claim4": "rebuild_dev_only_from_saved_slices",
    }

    appendix_index = {
        "claim1_internal": {"status": "planned", "mode": "rebuild"},
        "claim1_external": {"status": "planned", "mode": "rebuild"},
        "claim2_internal": {"status": "planned", "mode": "reference_only"},
        "claim2_external": {"status": "planned", "mode": "reference_only"},
        "claim3": {"status": "planned", "mode": "rebuild"},
        "claim4": {"status": "planned", "mode": "rebuild_dev_only"},
        "significance": {"status": "planned", "mode": "rebuild_claim1_only"},
    }

    manifest = {
        "step": 14,
        "protocol_version": STEP14_PROTOCOL_VERSION,
        "run_id": run_id,
        "status_mode": STEP14_STATUS_MODE,
        "created_at": _now_iso(),
        "source_runs": source_runs,
        "gold_claims_path": str(ctx.gold_claims_path),
        "prepared_status_path": str(status_path),
        "artifacts": {
            "dependency_state": str(ctx.dependency_state_path),
            "rebuild_plan": str(ctx.rebuild_plan_path),
            "appendix_index_json": str(ctx.appendix_index_json_path),
            "appendix_index_md": str(ctx.appendix_index_md_path),
        },
    }

    _write_json(ctx.manifest_path, manifest)
    _write_json(ctx.dependency_state_path, dependency_state)
    _write_json(ctx.rebuild_plan_path, rebuild_plan)
    _write_json(ctx.appendix_index_json_path, appendix_index)

    _write_text(
        ctx.appendix_index_md_path,
        "\n".join(
            [
                "# Step 14 Appendix Index",
                "",
                "- Claim 1 internal: planned rebuild",
                "- Claim 1 external: planned rebuild",
                "- Claim 2 internal/external: reference only",
                "- Claim 3: rebuild from raw predictions",
                "- Claim 4: dev-only rebuild from saved slices",
            ]
        )
        + "\n",
    )

    summary = {
        "step": 14,
        "run_id": run_id,
        "status": "prepared",
        "protocol_version": STEP14_PROTOCOL_VERSION,
        "status_mode": STEP14_STATUS_MODE,
        "source_runs": source_runs,
        "artifacts": manifest["artifacts"],
    }

    _write_json(ctx.rebuild_summary_path, summary)
    return summary


def _verify_dependency_state(
    ctx: Step14Context, *, status_path: str | Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dependency_state = _read_json(ctx.dependency_state_path)
    gold_claim_rows = _load_jsonl_rows(ctx.gold_claims_path)
    gold_claim_identity_hash, _ = _claim_identity_hash(gold_claim_rows)

    if gold_claim_identity_hash != str(dependency_state["gold_claim_identity_hash"]):
        raise ValueError(
            "Gold claims identity changed; Step 14 rebuild only supports label-only status updates."
        )

    if _sha256_file(ctx.gold_claims_path) != str(
        dependency_state["gold_claims_sha256"]
    ):
        raise ValueError(
            "Gold claims file changed; Step 14 rebuild only supports label-only status updates."
        )

    status_rows = _load_jsonl_rows(status_path)
    status_identity_hash, _ = _claim_identity_hash(status_rows)

    if status_identity_hash != str(dependency_state["prepared_status_identity_hash"]):
        raise ValueError(
            "Status relabel changed the (uid, claim_id) universe; label-only rebuild cannot continue."
        )

    for section_name, payload in dict(
        dependency_state.get("sources", {}) or {}
    ).items():
        for key, value in payload.items():
            if key.endswith("_path") and (
                "run_summary" in key or "scope" in key or "manifest" in key
            ):
                current_hash_key = key.replace("_path", "_sha256")

                if current_hash_key in payload and _sha256_file(value) != str(
                    payload[current_hash_key]
                ):
                    raise ValueError(
                        f"Source dependency changed for {section_name}: {key}"
                    )

    return gold_claim_rows, status_rows, dependency_state


def _filter_rows_by_uids(
    rows: list[dict[str, Any]], case_uids: list[str]
) -> list[dict[str, Any]]:
    uid_set = set(case_uids)
    return [dict(row) for row in rows if str(row.get("uid", "") or "") in uid_set]


def _copy_test_outputs_to_root(source_dir: Path, target_dir: Path) -> None:
    for file_name in (
        "main_table.json",
        "main_table.csv",
        "appendix_stepa_metrics.json",
        "appendix_e2e_fp_sensitive.json",
        "appendix_e2e_soft_consistency.json",
    ):
        _copy_file(source_dir / file_name, target_dir / file_name)


def _status_lookup_for_result(result: dict[str, Any]) -> dict[str, str]:
    return {
        str(row.get("claim_id", "") or ""): normalize_status_eval(
            str(row.get("status_eval", "") or "")
        )
        for row in result.get("status", [])
        if str(row.get("claim_id", "") or "").strip()
    }


def _build_claim1_significance_rows(
    *,
    source_name: str,
    method_order: list[str],
    result_maps: dict[str, dict[str, dict[str, Any]]],
    case_uids: list[str],
    gold_status_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gold_lookup = {
        (
            str(row.get("uid", "") or ""),
            str(row.get("claim_id", "") or ""),
        ): normalize_status_eval(str(row.get("status_eval", "") or ""))
        for row in gold_status_rows
    }

    claim_ids_by_uid: dict[str, list[str]] = {}

    for uid, claim_id in gold_lookup:
        claim_ids_by_uid.setdefault(uid, []).append(claim_id)

    mcnemar_rows: list[dict[str, Any]] = []
    stuart_rows: list[dict[str, Any]] = []
    mcnemar_p: dict[str, float] = {}
    stuart_p: dict[str, float] = {}

    for idx, left in enumerate(method_order):
        for right in method_order[idx + 1 :]:
            y_true_2: list[str] = []
            pred_a_2: list[str] = []
            pred_b_2: list[str] = []
            pred_a_3: list[str] = []
            pred_b_3: list[str] = []

            for uid in case_uids:
                left_lookup = _status_lookup_for_result(result_maps[left][uid])
                right_lookup = _status_lookup_for_result(result_maps[right][uid])

                for claim_id in sorted(claim_ids_by_uid.get(uid, [])):
                    gold_3 = gold_lookup[(uid, claim_id)]
                    a_3 = left_lookup.get(claim_id, "HYPOTHETICAL")
                    b_3 = right_lookup.get(claim_id, "HYPOTHETICAL")
                    y_true_2.append(collapse_status_eval(gold_3))
                    pred_a_2.append(collapse_status_eval(a_3))
                    pred_b_2.append(collapse_status_eval(b_3))
                    pred_a_3.append(a_3)
                    pred_b_3.append(b_3)

            label = f"{left}__vs__{right}"
            mcnemar_result = mcnemar_test(y_true_2, pred_a_2, pred_b_2)

            stuart_result = stuart_maxwell_test(
                pred_a_3,
                pred_b_3,
                classes=STATUS_CLASSES_3,
            )

            mcnemar_p[label] = float(mcnemar_result.p_value)
            stuart_p[label] = float(stuart_result.p_value)

            mcnemar_rows.append(
                {
                    "source_name": source_name,
                    "stage": "test",
                    "method_a": left,
                    "method_b": right,
                    **mcnemar_result.to_dict(),
                }
            )

            stuart_rows.append(
                {
                    "source_name": source_name,
                    "stage": "test",
                    "method_a": left,
                    "method_b": right,
                    **stuart_result.to_dict(),
                }
            )

    mcnemar_holm = holm_bonferroni(mcnemar_p).to_dict()["items"] if mcnemar_p else []
    stuart_holm = holm_bonferroni(stuart_p).to_dict()["items"] if stuart_p else []
    mcnemar_lookup = {item["label"]: item for item in mcnemar_holm}
    stuart_lookup = {item["label"]: item for item in stuart_holm}

    for row in mcnemar_rows:
        label = f"{row['method_a']}__vs__{row['method_b']}"
        holm = mcnemar_lookup.get(label, {})
        row["holm_threshold"] = holm.get("threshold")
        row["holm_rejected"] = holm.get("rejected")

    for row in stuart_rows:
        label = f"{row['method_a']}__vs__{row['method_b']}"
        holm = stuart_lookup.get(label, {})
        row["holm_threshold"] = holm.get("threshold")
        row["holm_rejected"] = holm.get("rejected")

    return mcnemar_rows, stuart_rows


def _rebuild_claim1_source(
    *,
    source_name: str,
    source_run_id: str,
    reports_root: Path,
    output_dir: Path,
    gold_claim_rows: list[dict[str, Any]],
    gold_status_rows: list[dict[str, Any]],
    encoder: Any,
) -> dict[str, Any]:
    summary, claim_root = _load_source_claim1_run(
        reports_root=reports_root,
        run_id=source_run_id,
    )

    matching_snapshot = _load_matching_snapshot(
        reports_root=reports_root,
        freeze_run_id=str(summary["freeze_run_id"]),
    )

    matching_config = _make_matching_config(matching_snapshot)
    method_order = [str(name) for name in list(summary.get("method_names", []))]
    result_maps_by_stage: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    stage_artifacts: dict[str, Any] = {}

    for stage_name, artifact_key in (("dev", "dev_scope"), ("test", "test_scope")):
        case_uids = _read_ids(summary["artifacts"][artifact_key])
        flattened_predictions: dict[str, list[dict[str, Any]]] = {}
        result_maps: dict[str, dict[str, dict[str, Any]]] = {}

        for method_name in method_order:
            method_results: dict[str, dict[str, Any]] = {}
            flattened_predictions[method_name] = []

            for uid in case_uids:
                result = _read_json(
                    claim_root
                    / stage_name
                    / "raw_predictions"
                    / method_name
                    / f"{uid}.json"
                )

                method_results[uid] = result

                flattened_predictions[method_name].extend(
                    _flatten_prediction_rows(result)
                )

            result_maps[method_name] = method_results

        stage_gold_claim_rows = _filter_rows_by_uids(gold_claim_rows, case_uids)
        stage_gold_status_rows = _filter_rows_by_uids(gold_status_rows, case_uids)

        bundles_by_method = _build_case_bundles_by_method(
            case_uids=case_uids,
            gold_claim_rows=stage_gold_claim_rows,
            flattened_predictions=flattened_predictions,
            active_method_names=method_order,
            encoder=encoder,
            config=matching_config,
        )

        method_metrics = _evaluate_method_metrics(
            bundles_by_method=bundles_by_method,
            gold_status_rows=stage_gold_status_rows,
        )

        stage_dir = output_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        _write_json(stage_dir / "method_metrics.json", method_metrics)

        _write_json(
            stage_dir / "pairwise_deltas.json",
            _build_pairwise_delta_payload(
                method_metrics=method_metrics,
                method_order=method_order,
            ),
        )

        metric_paths = _write_metric_payloads(
            output_dir=stage_dir,
            method_order=method_order,
            method_metrics=method_metrics,
            run_id=source_run_id,
            stage=stage_name,
            split_name=stage_name,
        )

        result_maps_by_stage[stage_name] = result_maps

        stage_artifacts[stage_name] = {
            "case_uids": case_uids,
            "metric_paths": metric_paths,
        }

    _copy_test_outputs_to_root(output_dir / "test", output_dir)

    mcnemar_rows, stuart_rows = _build_claim1_significance_rows(
        source_name=source_name,
        method_order=method_order,
        result_maps=result_maps_by_stage["test"],
        case_uids=stage_artifacts["test"]["case_uids"],
        gold_status_rows=_filter_rows_by_uids(
            gold_status_rows,
            stage_artifacts["test"]["case_uids"],
        ),
    )

    rebuild_summary = {
        "source_name": source_name,
        "source_run_id": source_run_id,
        "status": "rebuilt",
        "freeze_run_id": str(summary["freeze_run_id"]),
        "method_names": method_order,
        "artifacts": {
            "main_table_json": str(output_dir / "main_table.json"),
            "main_table_csv": str(output_dir / "main_table.csv"),
            "appendix_stepa_metrics": str(output_dir / "appendix_stepa_metrics.json"),
            "appendix_e2e_fp_sensitive": str(
                output_dir / "appendix_e2e_fp_sensitive.json"
            ),
            "appendix_e2e_soft_consistency": str(
                output_dir / "appendix_e2e_soft_consistency.json"
            ),
        },
    }

    _write_json(output_dir / "run_summary.json", rebuild_summary)

    return {
        "summary": rebuild_summary,
        "mcnemar_rows": mcnemar_rows,
        "stuart_rows": stuart_rows,
    }


def _copy_optional_file(src: Path, dst: Path) -> None:
    if src.exists():
        _copy_file(src, dst)


def _build_claim3_point_metric(
    *,
    raw_dir: Path,
    case_uids: list[str],
    gold_claim_rows: list[dict[str, Any]],
    gold_status_rows: list[dict[str, Any]],
    matching_config: Any,
    encoder: Any,
) -> dict[str, Any]:
    flattened_predictions: list[dict[str, Any]] = []

    for uid in case_uids:
        result = _read_json(raw_dir / f"{uid}.json")
        flattened_predictions.extend(_flatten_prediction_rows(result))

    bundles = claim3_runner.build_method_case_matches(
        case_uids=case_uids,
        gold_rows=gold_claim_rows,
        prediction_rows=flattened_predictions,
        method_name=claim3_runner.CLAIM3_MAIN_METHOD,
        encoder=encoder,
        config=matching_config,
    )

    return claim3_runner.evaluate_claim1_metrics(bundles, gold_status_rows)


def _claim3_attribution_label(normal_gain: float, fixed_gain: float) -> str:
    if normal_gain > 0.0 and fixed_gain <= 0.0:
        return "memory_evidence_input_dominant"

    if normal_gain > 0.0 and fixed_gain > 0.0:
        return "frozen_pack_retains_gain"

    return "no_clear_learning_gain"


def _rebuild_claim3(
    *,
    source_run_id: str,
    target_run_id: str,
    reports_root: Path,
    step14_root: Path,
    gold_claim_rows: list[dict[str, Any]],
    gold_status_rows: list[dict[str, Any]],
    encoder: Any,
) -> dict[str, Any]:
    source_claim_root = reports_root / source_run_id / "claim3"
    target_claim_root = reports_root / target_run_id / "claim3"
    _clear_dir(target_claim_root)
    target_claim_root.mkdir(parents=True, exist_ok=True)
    source_manifest = _read_json(source_claim_root / "warmup_manifest.json")

    _copy_file(
        source_claim_root / "warmup_manifest.json",
        target_claim_root / "warmup_manifest.json",
    )

    _copy_optional_file(
        source_claim_root / "job_plan.json", target_claim_root / "job_plan.json"
    )

    _copy_optional_file(
        source_claim_root / "reference_anchor_metrics.json",
        target_claim_root / "reference_anchor_metrics.json",
    )

    if (source_claim_root / "fixed_evidence_pack").exists():
        _copy_tree(
            source_claim_root / "fixed_evidence_pack",
            target_claim_root / "fixed_evidence_pack",
        )

    source_claim1_summary, _ = _load_source_claim1_run(
        reports_root=reports_root,
        run_id=str(source_manifest["source_claim1_run_id"]),
    )

    case_uids = _read_ids(source_claim1_summary["artifacts"]["dev_scope"])

    matching_config = _make_matching_config(
        _load_matching_snapshot(
            reports_root=reports_root,
            freeze_run_id=str(source_manifest["freeze_run_id"]),
        )
    )

    stage_gold_claim_rows = _filter_rows_by_uids(gold_claim_rows, case_uids)
    stage_gold_status_rows = _filter_rows_by_uids(gold_status_rows, case_uids)

    for branch_name in ("normal", "fixed-pack"):
        source_branch_dir = source_claim_root / branch_name
        target_branch_dir = target_claim_root / branch_name

        _copy_tree(
            source_branch_dir / "raw_predictions", target_branch_dir / "raw_predictions"
        )

        source_summary = _read_json(source_branch_dir / "summary.json")

        for point in source_summary["points"]:
            source_point_payload = _read_json(
                source_branch_dir / "point_metrics" / f"warmup_{point}.json"
            )

            metric = _build_claim3_point_metric(
                raw_dir=source_branch_dir / "raw_predictions" / f"warmup_{point}",
                case_uids=case_uids,
                gold_claim_rows=stage_gold_claim_rows,
                gold_status_rows=stage_gold_status_rows,
                matching_config=matching_config,
                encoder=encoder,
            )

            source_point_payload["metric"] = metric

            _write_json(
                target_branch_dir / "point_metrics" / f"warmup_{point}.json",
                source_point_payload,
            )

        _write_json(target_branch_dir / "summary.json", source_summary)

    curve_scores: dict[str, dict[str, dict[str, float]]] = {
        "normal": {},
        "fixed_pack": {},
    }

    normal_summary = _read_json(target_claim_root / "normal" / "summary.json")
    fixed_summary = _read_json(target_claim_root / "fixed-pack" / "summary.json")

    for branch_name, branch_summary in (
        ("normal", normal_summary),
        ("fixed_pack", fixed_summary),
    ):
        target_branch_name = "fixed-pack" if branch_name == "fixed_pack" else "normal"

        for point in branch_summary["points"]:
            payload = _read_json(
                target_claim_root
                / target_branch_name
                / "point_metrics"
                / f"warmup_{point}.json"
            )

            curve_scores[branch_name][f"warmup_{point}"] = payload["metric"][
                "case_scores"
            ][claim3_runner.CLAIM3_PRIMARY_CASE_SCORE_KEY]

    curve_summary = summarize_claim3_curve(curve_scores=curve_scores)
    normal_rows = []
    fixed_rows = []

    for point_name, payload in curve_summary["curve_summaries"]["normal"].items():
        normal_rows.append({"branch": "normal", "warmup_point": point_name, **payload})

    for point_name, payload in curve_summary["curve_summaries"]["fixed_pack"].items():
        fixed_rows.append(
            {"branch": "fixed_pack", "warmup_point": point_name, **payload}
        )

    _write_csv(target_claim_root / "status_curve_normal.csv", normal_rows)
    _write_csv(target_claim_root / "status_curve_fixed_pack.csv", fixed_rows)

    normal_point0 = curve_summary["curve_summaries"]["normal"]["warmup_0"][
        "point_estimate"
    ]

    normal_point_max = curve_summary["curve_summaries"]["normal"][
        f"warmup_{max(int(item) for item in source_manifest['warmup_points'])}"
    ]["point_estimate"]

    fixed_point0 = curve_summary["curve_summaries"]["fixed_pack"]["warmup_0"][
        "point_estimate"
    ]

    fixed_target_point = max(int(item) for item in fixed_summary["points"])

    fixed_point_target = curve_summary["curve_summaries"]["fixed_pack"][
        f"warmup_{fixed_target_point}"
    ]["point_estimate"]

    attribution = {
        "claim_id": claim3_runner.CLAIM3,
        "metric_key": claim3_runner.CLAIM3_PRIMARY_CASE_SCORE_KEY,
        "normal_curve_gain_0_to_max": float(normal_point_max) - float(normal_point0),
        "fixed_pack_target_point": int(fixed_target_point),
        "fixed_pack_gain_0_to_target": float(fixed_point_target) - float(fixed_point0),
    }

    attribution["attribution_label"] = _claim3_attribution_label(
        attribution["normal_curve_gain_0_to_max"],
        attribution["fixed_pack_gain_0_to_target"],
    )

    _write_json(target_claim_root / "attribution_summary.json", attribution)

    summary = {
        "claim_id": claim3_runner.CLAIM3,
        "run_id": target_run_id,
        "status": "completed",
        "freeze_run_id": source_manifest["freeze_run_id"],
        "source_claim1_run_id": source_manifest["source_claim1_run_id"],
        "method_name": claim3_runner.CLAIM3_MAIN_METHOD,
        "task_focus": claim3_runner.CLAIM3_TASK_FOCUS,
        "warmup_points": list(source_manifest["warmup_points"]),
        "fixed_pack_points": list(source_manifest["fixed_pack_points"]),
        "primary_metric_key": claim3_runner.CLAIM3_PRIMARY_CASE_SCORE_KEY,
        "artifacts": {
            "warmup_manifest": str(target_claim_root / "warmup_manifest.json"),
            "job_plan": str(target_claim_root / "job_plan.json"),
            "reference_anchor_metrics": str(
                target_claim_root / "reference_anchor_metrics.json"
            ),
            "status_curve_normal_csv": str(
                target_claim_root / "status_curve_normal.csv"
            ),
            "status_curve_fixed_pack_csv": str(
                target_claim_root / "status_curve_fixed_pack.csv"
            ),
            "attribution_summary": str(target_claim_root / "attribution_summary.json"),
        },
        "curve_summary": curve_summary,
        "attribution_summary": attribution,
    }

    _write_json(target_claim_root / "run_summary.json", summary)
    target_export_dir = step14_root / "claim3"
    target_export_dir.mkdir(parents=True, exist_ok=True)

    for file_name in (
        "status_curve_normal.csv",
        "status_curve_fixed_pack.csv",
        "attribution_summary.json",
        "run_summary.json",
    ):
        _copy_file(target_claim_root / file_name, target_export_dir / file_name)

    return {
        "summary": summary,
        "artifacts": {
            "status_curve_normal_csv": str(
                target_export_dir / "status_curve_normal.csv"
            ),
            "status_curve_fixed_pack_csv": str(
                target_export_dir / "status_curve_fixed_pack.csv"
            ),
            "attribution_summary": str(target_export_dir / "attribution_summary.json"),
            "run_summary": str(target_export_dir / "run_summary.json"),
        },
    }


def _build_claim4_reference_full_budget(
    *,
    source_claim1_root: Path,
    method_names: list[str],
    stage_uids_by_name: dict[str, list[str]],
    matching_config: Any,
    gold_claim_rows: list[dict[str, Any]],
    gold_status_rows: list[dict[str, Any]],
    encoder: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "claim_id": claim4_runner.CLAIM4,
        "primary_metric_key": claim4_runner.CLAIM4_PRIMARY_CASE_SCORE_KEY,
        "method_names": list(method_names),
        "stages": {},
    }

    for stage_name, case_uids in stage_uids_by_name.items():
        flattened_predictions: dict[str, list[dict[str, Any]]] = {}
        stage_methods: dict[str, Any] = {}
        stage_gold_claim_rows = _filter_rows_by_uids(gold_claim_rows, case_uids)
        stage_gold_status_rows = _filter_rows_by_uids(gold_status_rows, case_uids)

        for method_name in method_names:
            flattened_predictions[method_name] = []
            token_values: list[float] = []

            for uid in case_uids:
                result = _read_json(
                    source_claim1_root
                    / stage_name
                    / "raw_predictions"
                    / method_name
                    / f"{uid}.json"
                )

                flattened_predictions[method_name].extend(
                    _flatten_prediction_rows(result)
                )

                token_values.append(
                    float(result.get("token_usage", {}).get("total_tokens", 0) or 0)
                )

            bundles_by_method = _build_case_bundles_by_method(
                case_uids=case_uids,
                gold_claim_rows=stage_gold_claim_rows,
                flattened_predictions={method_name: flattened_predictions[method_name]},
                active_method_names=[method_name],
                encoder=encoder,
                config=matching_config,
            )

            method_metrics = _evaluate_method_metrics(
                bundles_by_method=bundles_by_method,
                gold_status_rows=stage_gold_status_rows,
            )[method_name]

            stage_methods[method_name] = {
                "primary_metric": dict(
                    method_metrics["main_metrics"][
                        claim4_runner.CLAIM4_PRIMARY_CASE_SCORE_KEY
                    ]
                ),
                "case_scores": dict(
                    method_metrics["case_scores"][
                        claim4_runner.CLAIM4_PRIMARY_CASE_SCORE_KEY
                    ]
                ),
                "raw_predictions_dir": str(
                    source_claim1_root / stage_name / "raw_predictions" / method_name
                ),
                "mean_total_tokens_per_case": (
                    float(sum(token_values) / len(token_values))
                    if token_values
                    else 0.0
                ),
            }

        payload["stages"][stage_name] = {
            "case_uids": list(case_uids),
            "methods": stage_methods,
        }

    return payload


def _rebuild_claim4_repeat_metric(
    *,
    raw_root: Path,
    case_uids: list[str],
    method_names: list[str],
    matching_config: Any,
    gold_claim_rows: list[dict[str, Any]],
    gold_status_rows: list[dict[str, Any]],
    encoder: Any,
    stage: str,
    policy: str,
    point: str,
    repeat: int,
) -> dict[str, Any]:
    flattened_predictions: dict[str, list[dict[str, Any]]] = {
        name: [] for name in method_names
    }

    for method_name in method_names:
        for uid in case_uids:
            result = _read_json(raw_root / method_name / f"{uid}.json")
            flattened_predictions[method_name].extend(_flatten_prediction_rows(result))

    bundles_by_method = _build_case_bundles_by_method(
        case_uids=case_uids,
        gold_claim_rows=gold_claim_rows,
        flattened_predictions=flattened_predictions,
        active_method_names=method_names,
        encoder=encoder,
        config=matching_config,
    )

    method_metrics = _evaluate_method_metrics(
        bundles_by_method=bundles_by_method,
        gold_status_rows=gold_status_rows,
    )

    return {
        "claim_id": claim4_runner.CLAIM4,
        "stage": stage,
        "policy": policy,
        "point": point,
        "repeat": repeat,
        "method_names": list(method_names),
        "primary_metric_key": claim4_runner.CLAIM4_PRIMARY_CASE_SCORE_KEY,
        "methods": {
            method_name: {
                "primary_metric": dict(
                    method_metrics[method_name]["main_metrics"][
                        claim4_runner.CLAIM4_PRIMARY_CASE_SCORE_KEY
                    ]
                ),
                "case_scores": dict(
                    method_metrics[method_name]["case_scores"][
                        claim4_runner.CLAIM4_PRIMARY_CASE_SCORE_KEY
                    ]
                ),
            }
            for method_name in method_names
        },
    }


def _rebuild_claim4(
    *,
    source_run_id: str,
    target_run_id: str,
    reports_root: Path,
    step14_root: Path,
    gold_claim_rows: list[dict[str, Any]],
    gold_status_rows: list[dict[str, Any]],
    encoder: Any,
) -> dict[str, Any]:
    source_claim_root = reports_root / source_run_id / "claim4"
    source_manifest = _read_json(source_claim_root / "manifest.json")

    claim4_runner.prepare_claim4_experiment(
        freeze_run_id=str(source_manifest["freeze_run_id"]),
        source_claim1_run_id=str(source_manifest["source_claim1_run_id"]),
        run_id=target_run_id,
        reports_root=reports_root,
        input_path=str(source_manifest["input_path"]),
    )

    target_claim_root = reports_root / target_run_id / "claim4"

    for relative in (
        Path("dev/raw_predictions"),
        Path("dev/repeat_runtime"),
        Path("dev/execution_summary"),
    ):
        _copy_tree(source_claim_root / relative, target_claim_root / relative)

    target_ctx = claim4_runner._build_claim4_context(
        freeze_run_id=str(source_manifest["freeze_run_id"]),
        source_claim1_run_id=str(source_manifest["source_claim1_run_id"]),
        run_id=target_run_id,
        reports_root=reports_root,
        input_path=str(source_manifest["input_path"]),
    )

    reference_payload = _build_claim4_reference_full_budget(
        source_claim1_root=target_ctx.source_claim_root,
        method_names=list(target_ctx.method_names),
        stage_uids_by_name=target_ctx.stage_uids_by_name,
        matching_config=target_ctx.matching_config,
        gold_claim_rows=gold_claim_rows,
        gold_status_rows=gold_status_rows,
        encoder=encoder,
    )

    _write_json(target_ctx.reference_full_budget_path, reference_payload)

    for policy in (
        claim4_runner.CLAIM4_FIXED_POLICY,
        claim4_runner.CLAIM4_ADAPTIVE_POLICY,
    ):
        for point in claim4_runner._policy_points(policy):
            for repeat in claim4_runner._stage_repeats("dev"):
                method_names = list(claim4_runner._policy_method_names(policy))
                case_uids = list(target_ctx.stage_uids_by_name["dev"])
                stage_gold_claim_rows = _filter_rows_by_uids(gold_claim_rows, case_uids)

                stage_gold_status_rows = _filter_rows_by_uids(
                    gold_status_rows, case_uids
                )

                repeat_metrics = _rebuild_claim4_repeat_metric(
                    raw_root=claim4_runner._slice_raw_root(
                        target_ctx,
                        stage="dev",
                        policy=policy,
                        point=point,
                        repeat=repeat,
                    ),
                    case_uids=case_uids,
                    method_names=method_names,
                    matching_config=target_ctx.matching_config,
                    gold_claim_rows=stage_gold_claim_rows,
                    gold_status_rows=stage_gold_status_rows,
                    encoder=encoder,
                    stage="dev",
                    policy=policy,
                    point=point,
                    repeat=repeat,
                )

                _write_json(
                    claim4_runner._slice_metrics_path(
                        target_ctx,
                        stage="dev",
                        policy=policy,
                        point=point,
                        repeat=repeat,
                    ),
                    repeat_metrics,
                )

    claim4_runner.audit_claim4_stage(
        run_id=target_run_id,
        reports_root=reports_root,
        stage="dev",
    )

    summary = claim4_runner.summarize_claim4_experiment(
        run_id=target_run_id,
        reports_root=reports_root,
    )

    target_export_dir = step14_root / "claim4"
    target_export_dir.mkdir(parents=True, exist_ok=True)

    for file_name in (
        "delta_root_flip.csv",
        "delta_phi_curve.csv",
        "delta_err.csv",
        "delta_conflict.csv",
        "pareto_points.csv",
        "adaptive_overlay.csv",
        "noise_audit.json",
        "run_summary.json",
        "reference_full_budget.json",
    ):
        _copy_file(target_claim_root / file_name, target_export_dir / file_name)

    return {
        "summary": summary,
        "artifacts": {
            "delta_root_flip_csv": str(target_export_dir / "delta_root_flip.csv"),
            "delta_phi_curve_csv": str(target_export_dir / "delta_phi_curve.csv"),
            "delta_err_csv": str(target_export_dir / "delta_err.csv"),
            "delta_conflict_csv": str(target_export_dir / "delta_conflict.csv"),
            "pareto_points_csv": str(target_export_dir / "pareto_points.csv"),
            "adaptive_overlay_csv": str(target_export_dir / "adaptive_overlay.csv"),
            "noise_audit_json": str(target_export_dir / "noise_audit.json"),
            "run_summary": str(target_export_dir / "run_summary.json"),
        },
    }


def _build_claim2_ref(
    *,
    source_name: str,
    source_run_id: str,
    reports_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    claim_root = reports_root / source_run_id / "claim2"
    summary = _read_json(claim_root / "run_summary.json")

    payload = {
        "source_name": source_name,
        "source_run_id": source_run_id,
        "status": "reused_reference",
        "reason": "status_relabel_unaffected",
        "run_summary": summary,
        "artifacts": {
            "run_summary": str(claim_root / "run_summary.json"),
            "main_table_json": str(claim_root / "main_table.json"),
            "main_table_csv": str(claim_root / "main_table.csv"),
        },
    }

    _write_json(output_dir / f"{source_name}_ref.json", payload)
    return payload


def rebuild_step14_reporting(
    *,
    run_id: str,
    reports_root: str | Path,
    status_path: str | Path,
) -> dict[str, Any]:
    manifest = _read_json(Path(reports_root) / run_id / "step14" / "manifest.json")

    ctx = _build_ctx(
        run_id=run_id,
        reports_root=reports_root,
        status_path=manifest["prepared_status_path"],
        source_runs=dict(manifest["source_runs"]),
    )

    gold_claim_rows, gold_status_rows, dependency_state = _verify_dependency_state(
        ctx,
        status_path=status_path,
    )

    encoder = SentenceTransformerTextEncoder(SystemConfig().path.embedding_model_path)
    claim1_internal_dir = ctx.step14_root / "claim1_internal"
    claim1_external_dir = ctx.step14_root / "claim1_external"
    claim1_internal_dir.mkdir(parents=True, exist_ok=True)
    claim1_external_dir.mkdir(parents=True, exist_ok=True)

    claim1_internal = _rebuild_claim1_source(
        source_name="claim1_internal",
        source_run_id=ctx.source_runs["claim1_internal"],
        reports_root=ctx.reports_root,
        output_dir=claim1_internal_dir,
        gold_claim_rows=gold_claim_rows,
        gold_status_rows=gold_status_rows,
        encoder=encoder,
    )

    claim1_external = _rebuild_claim1_source(
        source_name="claim1_external",
        source_run_id=ctx.source_runs["claim1_external"],
        reports_root=ctx.reports_root,
        output_dir=claim1_external_dir,
        gold_claim_rows=gold_claim_rows,
        gold_status_rows=gold_status_rows,
        encoder=encoder,
    )

    significance_dir = ctx.step14_root / "significance"
    significance_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        significance_dir / "claim1_mcnemar.csv",
        claim1_internal["mcnemar_rows"] + claim1_external["mcnemar_rows"],
    )

    _write_csv(
        significance_dir / "claim1_stuart_maxwell.csv",
        claim1_internal["stuart_rows"] + claim1_external["stuart_rows"],
    )

    _write_json(
        significance_dir / "claim4_status.json",
        {
            "status": "not_applicable_dev_only",
            "reason": "Step 13 current official protocol is dev-only diagnostic with soft-fail audit.",
        },
    )

    claim2_internal_ref = _build_claim2_ref(
        source_name="claim2_internal",
        source_run_id=ctx.source_runs["claim2_internal"],
        reports_root=ctx.reports_root,
        output_dir=ctx.step14_root,
    )

    claim2_external_ref = _build_claim2_ref(
        source_name="claim2_external",
        source_run_id=ctx.source_runs["claim2_external"],
        reports_root=ctx.reports_root,
        output_dir=ctx.step14_root,
    )

    claim3_summary = _rebuild_claim3(
        source_run_id=ctx.source_runs["claim3"],
        target_run_id=run_id,
        reports_root=ctx.reports_root,
        step14_root=ctx.step14_root,
        gold_claim_rows=gold_claim_rows,
        gold_status_rows=gold_status_rows,
        encoder=encoder,
    )

    claim4_summary = _rebuild_claim4(
        source_run_id=ctx.source_runs["claim4"],
        target_run_id=run_id,
        reports_root=ctx.reports_root,
        step14_root=ctx.step14_root,
        gold_claim_rows=gold_claim_rows,
        gold_status_rows=gold_status_rows,
        encoder=encoder,
    )

    appendix_index = {
        "claim1_internal": {
            "status": "rebuilt",
            "artifacts": claim1_internal["summary"]["artifacts"],
        },
        "claim1_external": {
            "status": "rebuilt",
            "artifacts": claim1_external["summary"]["artifacts"],
        },
        "claim2_internal": {
            "status": "reused_reference",
            "artifact": str(ctx.step14_root / "claim2_internal_ref.json"),
        },
        "claim2_external": {
            "status": "reused_reference",
            "artifact": str(ctx.step14_root / "claim2_external_ref.json"),
        },
        "claim3": {
            "status": "rebuilt",
            "artifacts": claim3_summary["artifacts"],
        },
        "claim4": {
            "status": "rebuilt_dev_only",
            "artifacts": claim4_summary["artifacts"],
        },
        "significance": {
            "status": "rebuilt",
            "artifacts": {
                "claim1_mcnemar_csv": str(significance_dir / "claim1_mcnemar.csv"),
                "claim1_stuart_maxwell_csv": str(
                    significance_dir / "claim1_stuart_maxwell.csv"
                ),
                "claim4_status_json": str(significance_dir / "claim4_status.json"),
            },
        },
    }

    _write_json(ctx.appendix_index_json_path, appendix_index)

    _write_text(
        ctx.appendix_index_md_path,
        "\n".join(
            [
                "# Step 14 Appendix Index",
                "",
                f"- Claim 1 internal: `{appendix_index['claim1_internal']['status']}`",
                f"- Claim 1 external: `{appendix_index['claim1_external']['status']}`",
                f"- Claim 2 internal: `{appendix_index['claim2_internal']['status']}`",
                f"- Claim 2 external: `{appendix_index['claim2_external']['status']}`",
                f"- Claim 3: `{appendix_index['claim3']['status']}`",
                f"- Claim 4: `{appendix_index['claim4']['status']}`",
            ]
        )
        + "\n",
    )

    summary = {
        "step": 14,
        "run_id": run_id,
        "status": "completed",
        "protocol_version": STEP14_PROTOCOL_VERSION,
        "status_mode": STEP14_STATUS_MODE,
        "rebuilt_at": _now_iso(),
        "prepared_status_path": str(ctx.prepared_status_path),
        "current_status_path": str(status_path),
        "prepared_status_sha256": str(dependency_state["prepared_status_sha256"]),
        "current_status_sha256": _sha256_file(status_path),
        "rebuilt_sections": [
            "claim1_internal",
            "claim1_external",
            "claim3",
            "claim4",
            "significance",
        ],
        "reused_sections": [
            "claim2_internal",
            "claim2_external",
        ],
        "artifacts": {
            "appendix_index_json": str(ctx.appendix_index_json_path),
            "appendix_index_md": str(ctx.appendix_index_md_path),
            "claim1_internal": str(claim1_internal_dir),
            "claim1_external": str(claim1_external_dir),
            "claim3": str(ctx.step14_root / "claim3"),
            "claim4": str(ctx.step14_root / "claim4"),
            "significance_dir": str(significance_dir),
        },
    }

    _write_json(ctx.rebuild_summary_path, summary)
    return summary
