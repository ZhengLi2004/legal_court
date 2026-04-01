"""Offline merge utility for Claim 1 internal and external comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from benchmarks.experiments.core import orchestrator


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _cohort_map(method_names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for method_name in method_names:
        mapping[method_name] = (
            "external" if method_name.startswith("external_") else "internal"
        )

    return mapping


def _read_scope_ids(path: str | Path) -> list[str]:
    payload = _read_json(path)
    ids = payload.get("ids", [])

    if not isinstance(ids, list):
        raise ValueError(f"{path} must contain an `ids` list.")

    return [str(item) for item in ids]


def _load_internal_bundle(internal_claim_root: str | Path) -> dict[str, Any]:
    root = Path(internal_claim_root)
    summary = _read_json(root / "recomputed_summary.json")
    main_table = _read_json(root / "main_table.json")
    source_claim_root = root.parent

    return {
        "root": root,
        "summary": summary,
        "main_table": main_table,
        "dev_method_metrics": _read_json(root / "dev" / "method_metrics.json"),
        "test_method_metrics": _read_json(root / "test" / "method_metrics.json"),
        "dev_scope_ids": _read_scope_ids(
            source_claim_root / "dev_case_uids_claim1_scope.json"
        ),
        "test_scope_ids": _read_scope_ids(
            source_claim_root / "test_case_uids_claim1_scope.json"
        ),
        "task_focus": str(main_table.get("task_focus", "") or ""),
        "root_claim_source": str(main_table.get("root_claim_source", "") or ""),
        "matching_mode": str(main_table.get("matching_mode", "") or ""),
        "adjudication_mode": str(main_table.get("adjudication_mode", "") or ""),
        "metric_contract_version": str(
            summary.get("metric_contract_version", "") or ""
        ),
    }


def _load_external_bundle(external_claim_root: str | Path) -> dict[str, Any]:
    root = Path(external_claim_root)
    summary = _read_json(root / "run_summary.json")
    main_table = _read_json(root / "main_table.json")
    freeze_run_id = str(summary.get("freeze_run_id", "") or "").strip()

    if not freeze_run_id:
        raise ValueError("External Claim 1 run_summary.json missing freeze_run_id.")

    freeze_root = root.parent.parent / freeze_run_id
    metric_contract_snapshot = _read_json(freeze_root / "metric_contract_snapshot.json")

    return {
        "root": root,
        "summary": summary,
        "main_table": main_table,
        "dev_method_metrics": _read_json(root / "dev" / "method_metrics.json"),
        "test_method_metrics": _read_json(root / "test" / "method_metrics.json"),
        "dev_scope_ids": _read_scope_ids(summary["artifacts"]["dev_scope"]),
        "test_scope_ids": _read_scope_ids(summary["artifacts"]["test_scope"]),
        "task_focus": str(
            summary.get("task_focus", "") or main_table.get("task_focus", "")
        ),
        "root_claim_source": str(
            summary.get("root_claim_source", "")
            or main_table.get("root_claim_source", "")
        ),
        "matching_mode": str(
            summary.get("matching_mode", "") or main_table.get("matching_mode", "")
        ),
        "adjudication_mode": str(
            summary.get("adjudication_mode", "")
            or main_table.get("adjudication_mode", "")
        ),
        "metric_contract_version": str(
            metric_contract_snapshot.get("metric_contract_version", "") or ""
        ),
    }


def _validate_same_protocol(internal: dict[str, Any], external: dict[str, Any]) -> None:
    for key in (
        "task_focus",
        "root_claim_source",
        "matching_mode",
        "adjudication_mode",
        "metric_contract_version",
    ):
        if internal[key] != external[key]:
            raise ValueError(
                f"Claim 1 merge mismatch on `{key}`: internal={internal[key]!r} external={external[key]!r}"
            )

    if internal["dev_scope_ids"] != external["dev_scope_ids"]:
        raise ValueError("Claim 1 merge requires identical dev scope ids.")

    if internal["test_scope_ids"] != external["test_scope_ids"]:
        raise ValueError("Claim 1 merge requires identical test scope ids.")


def _merge_method_metrics(*payloads: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for payload in payloads:
        overlap = sorted(set(merged.keys()) & set(payload.keys()))

        if overlap:
            raise ValueError(f"Duplicate method names across merged inputs: {overlap}")

        merged.update(payload)

    return merged


def _write_main_table_csv(
    *,
    path: str | Path,
    method_order: list[str],
    method_metrics: dict[str, dict[str, Any]],
    cohort_by_method: dict[str, str],
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_name",
                "cohort",
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
                    "cohort": cohort_by_method[method_name],
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


def merge_claim1_results(
    *,
    internal_claim_root: str | Path,
    external_claim_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Merge already-computed internal and external Claim 1 result bundles.

    Args:
        internal_claim_root: Internal Claim 1 result directory.
        external_claim_root: External Claim 1 result directory.
        output_dir: Directory for merged artifacts.

    Returns:
        Merge summary payload describing generated artifacts.
    """

    internal = _load_internal_bundle(internal_claim_root)
    external = _load_external_bundle(external_claim_root)
    _validate_same_protocol(internal, external)

    combined_dev_metrics = _merge_method_metrics(
        internal["dev_method_metrics"],
        external["dev_method_metrics"],
    )

    combined_test_metrics = _merge_method_metrics(
        internal["test_method_metrics"],
        external["test_method_metrics"],
    )

    internal_order = list(internal["main_table"].get("method_order", []))
    external_order = list(external["main_table"].get("method_order", []))

    method_order = internal_order + [
        name for name in external_order if name not in internal_order
    ]

    if not method_order:
        method_order = sorted(combined_test_metrics.keys())

    cohort_by_method = _cohort_map(method_order)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_table_rows = []
    stepa_rows = []
    fp_rows = []

    for method_name in method_order:
        summary_row = orchestrator._summary_row(
            method_name, combined_test_metrics[method_name]
        )

        summary_row["cohort"] = cohort_by_method[method_name]
        main_table_rows.append(summary_row)

        stepa_row = orchestrator._appendix_stepa_row(
            method_name, combined_test_metrics[method_name]
        )

        stepa_row["cohort"] = cohort_by_method[method_name]
        stepa_rows.append(stepa_row)

        fp_row = orchestrator._appendix_fp_row(
            method_name, combined_test_metrics[method_name]
        )

        fp_row["cohort"] = cohort_by_method[method_name]
        fp_rows.append(fp_row)

    soft_payload = orchestrator._build_soft_consistency_payload(
        method_metrics=combined_test_metrics,
        method_order=method_order,
    )

    pairwise_payload = orchestrator._build_pairwise_delta_payload(
        method_metrics=combined_test_metrics,
        method_order=method_order,
    )

    main_table_payload = {
        "claim_id": 1,
        "stage": "merged_comparison",
        "task_focus": internal["task_focus"],
        "root_claim_source": internal["root_claim_source"],
        "matching_mode": internal["matching_mode"],
        "adjudication_mode": internal["adjudication_mode"],
        "metric_contract_version": internal["metric_contract_version"],
        "method_order": method_order,
        "cohort_by_method": cohort_by_method,
        "rows": main_table_rows,
    }

    stepa_payload = {
        "claim_id": 1,
        "stage": "merged_comparison",
        "cohort_by_method": cohort_by_method,
        "rows": stepa_rows,
    }

    fp_payload = {
        "claim_id": 1,
        "stage": "merged_comparison",
        "cohort_by_method": cohort_by_method,
        "rows": fp_rows,
        "pairwise_deltas_vs_main_system": pairwise_payload,
    }

    merged_soft_payload = {
        "claim_id": 1,
        "stage": "merged_comparison",
        "cohort_by_method": cohort_by_method,
        **soft_payload,
    }

    _write_json(out_dir / "main_table.json", main_table_payload)

    _write_main_table_csv(
        path=out_dir / "main_table.csv",
        method_order=method_order,
        method_metrics=combined_test_metrics,
        cohort_by_method=cohort_by_method,
    )

    _write_json(out_dir / "appendix_stepa_metrics.json", stepa_payload)
    _write_json(out_dir / "appendix_e2e_fp_sensitive.json", fp_payload)
    _write_json(out_dir / "appendix_e2e_soft_consistency.json", merged_soft_payload)
    _write_json(out_dir / "dev" / "method_metrics.json", combined_dev_metrics)
    _write_json(out_dir / "test" / "method_metrics.json", combined_test_metrics)
    _write_json(out_dir / "test" / "pairwise_deltas.json", pairwise_payload)

    summary = {
        "status": "completed",
        "internal_claim_root": str(Path(internal_claim_root)),
        "external_claim_root": str(Path(external_claim_root)),
        "output_dir": str(out_dir),
        "task_focus": internal["task_focus"],
        "root_claim_source": internal["root_claim_source"],
        "matching_mode": internal["matching_mode"],
        "adjudication_mode": internal["adjudication_mode"],
        "method_order": method_order,
        "cohort_by_method": cohort_by_method,
        "artifacts": {
            "main_table_json": str(out_dir / "main_table.json"),
            "main_table_csv": str(out_dir / "main_table.csv"),
            "appendix_stepa_metrics": str(out_dir / "appendix_stepa_metrics.json"),
            "appendix_e2e_fp_sensitive": str(
                out_dir / "appendix_e2e_fp_sensitive.json"
            ),
            "appendix_e2e_soft_consistency": str(
                out_dir / "appendix_e2e_soft_consistency.json"
            ),
            "dev_method_metrics": str(out_dir / "dev" / "method_metrics.json"),
            "test_method_metrics": str(out_dir / "test" / "method_metrics.json"),
            "test_pairwise_deltas": str(out_dir / "test" / "pairwise_deltas.json"),
        },
    }

    _write_json(out_dir / "merge_summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge internal/external Claim 1 results"
    )

    parser.add_argument("--internal-claim-root", required=True)
    parser.add_argument("--external-claim-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint for offline Claim 1 table merging.

    Args:
        argv: Optional command-line argument override.

    Returns:
        Merge summary payload written by the CLI.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    payload = merge_claim1_results(
        internal_claim_root=args.internal_claim_root,
        external_claim_root=args.external_claim_root,
        output_dir=args.output_dir,
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
