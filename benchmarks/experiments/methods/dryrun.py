"""Dry-run utility for Step 09 method registry smoke execution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.methods.base import case_uid, validate_method_result
from benchmarks.experiments.methods.factory import build_default_registry


def _load_dev_case_uids(dev_ids_path: str | Path) -> list[str]:
    payload = json.loads(Path(dev_ids_path).read_text(encoding="utf-8"))
    ids = payload.get("ids", [])

    if not isinstance(ids, list) or not ids:
        raise ValueError("dev_ids.json must contain a non-empty `ids` list.")

    return [str(item) for item in ids]


def run_step09_dryrun(
    *,
    input_path: str | Path,
    dev_ids_path: str | Path,
    output_dir: str | Path,
    registry: Mapping[str, Any] | None = None,
    sample_size: int = 3,
    budget: Mapping[str, Any] | None = None,
    seed: int = 20260307,
    retrieval_config: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run all registered Step 09 methods on a small deterministic Dev slice."""
    cases = load_cases_from_jsonl(input_path)
    case_map = {case_uid(case): case for case in cases}
    selected_uids = _load_dev_case_uids(dev_ids_path)[: max(1, int(sample_size))]
    selected_cases = [case_map[uid] for uid in selected_uids if uid in case_map]

    if len(selected_cases) != len(selected_uids):
        missing = sorted(set(selected_uids) - {case_uid(case) for case in selected_cases})
        raise ValueError(f"Missing selected Dev cases: {missing}")

    active_registry = dict(registry or build_default_registry())
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_rows: list[dict[str, Any]] = []

    for method_name, runner in active_registry.items():
        method_dir = out_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        for case in selected_cases:
            result = runner(
                case=case,
                budget=budget,
                seed=seed,
                retrieval_config=retrieval_config,
                verbose=verbose,
            )
            validate_method_result(result)
            uid = str(result["case_uid"])
            (method_dir / f"{uid}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
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

    summary = {
        "sample_size": len(selected_cases),
        "selected_case_uids": selected_uids,
        "method_names": sorted(active_registry.keys()),
        "validation_row_count": len(validation_rows),
        "validation_rows": validation_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
