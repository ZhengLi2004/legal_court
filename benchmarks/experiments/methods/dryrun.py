"""Dry-run utility for Step 09 method registry smoke execution."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.methods.base import case_uid, validate_method_result
from benchmarks.experiments.methods.factory import (
    INTERNAL_DEFAULT_PROFILE,
    build_method_registry,
    normalize_registry_profile,
)


def _load_dev_case_uids(dev_ids_path: str | Path) -> list[str]:
    payload = json.loads(Path(dev_ids_path).read_text(encoding="utf-8"))
    ids = payload.get("ids", [])

    if not isinstance(ids, list) or not ids:
        raise ValueError("dev_ids.json must contain a non-empty `ids` list.")

    return [str(item) for item in ids]


def _load_selected_case_uids(selected_uids_path: str | Path) -> list[str]:
    payload = json.loads(Path(selected_uids_path).read_text(encoding="utf-8"))
    ids = payload.get("selected_case_uids", [])

    if not isinstance(ids, list) or not ids:
        raise ValueError(
            "selected_uids_path must contain a non-empty `selected_case_uids` list."
        )

    return [str(item) for item in ids]


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _write_summary(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _summarize_result_row(
    method_name: str, result: Mapping[str, Any]
) -> dict[str, Any]:
    uid = str(result["case_uid"])

    return {
        "method": method_name,
        "case_uid": uid,
        "claim_count": len(result["claims"]),
        "status_count": len(result["status"]),
        "warning_count": len(result["warnings"]),
    }


def _load_existing_validation_rows(
    *,
    output_dir: Path,
    selected_uids: list[str],
    method_names: list[str],
    skip_methods: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for method_name in method_names:
        if method_name in skip_methods:
            continue

        method_dir = output_dir / method_name

        if not method_dir.is_dir():
            continue

        for uid in selected_uids:
            result_path = method_dir / f"{uid}.json"

            if not result_path.exists():
                continue

            result = json.loads(result_path.read_text(encoding="utf-8"))
            validate_method_result(result)
            rows.append(_summarize_result_row(method_name, result))

    return rows


def _load_existing_validation_index(
    *,
    output_dir: Path,
    selected_uids: list[str],
    method_names: list[str],
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, str]] = set()

    for method_name in method_names:
        method_dir = output_dir / method_name

        if not method_dir.is_dir():
            continue

        for uid in selected_uids:
            result_path = method_dir / f"{uid}.json"

            if not result_path.exists():
                continue

            result = json.loads(result_path.read_text(encoding="utf-8"))
            validate_method_result(result)
            rows.append(_summarize_result_row(method_name, result))
            completed_keys.add((method_name, uid))

    return rows, completed_keys


def _clear_selected_method_outputs(
    *,
    output_dir: Path,
    selected_uids: list[str],
    method_names: list[str],
) -> None:
    for method_name in method_names:
        method_dir = output_dir / method_name

        if not method_dir.exists():
            continue

        for uid in selected_uids:
            result_path = method_dir / f"{uid}.json"

            if result_path.exists():
                result_path.unlink()


def run_step09_dryrun(
    *,
    input_path: str | Path,
    dev_ids_path: str | Path,
    output_dir: str | Path,
    storage_root_dir: str | Path | None = None,
    registry: Mapping[str, Any] | None = None,
    methods: list[str] | None = None,
    selected_uids_path: str | Path | None = None,
    sample_size: int = 3,
    budget: Mapping[str, Any] | None = None,
    seed: int = 20260307,
    retrieval_config: Mapping[str, Any] | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    resume: bool = False,
    registry_profile: str = INTERNAL_DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Run all registered Step 09 methods on a small deterministic Dev slice.

    Args:
        input_path: Case JSONL path.
        dev_ids_path: Dev-scope uid list path.
        output_dir: Directory for dry-run artifacts.
        storage_root_dir: Optional runtime storage root override.
        registry: Optional prebuilt method registry.
        methods: Optional subset of method names.
        selected_uids_path: Optional explicit case-selection file.
        sample_size: Number of dev cases to evaluate when no selection file exists.
        budget: Optional per-method budget overrides.
        seed: Deterministic random seed.
        retrieval_config: Optional retrieval overrides.
        verbose: Whether to enable verbose method logging.
        show_progress: Whether to show the tqdm progress bar.
        resume: Whether to reuse existing completed outputs.
        registry_profile: Registry profile identifier to activate.

    Returns:
        Dry-run summary payload written to ``summary.json``.

    Raises:
        ValueError: If requested methods or selected cases are invalid.
    """
    cases = load_cases_from_jsonl(input_path)
    case_map = {case_uid(case): case for case in cases}

    if selected_uids_path:
        selected_uids = _load_selected_case_uids(selected_uids_path)

    else:
        selected_uids = _load_dev_case_uids(dev_ids_path)[: max(1, int(sample_size))]

    selected_cases = [case_map[uid] for uid in selected_uids if uid in case_map]

    if len(selected_cases) != len(selected_uids):
        missing = sorted(
            set(selected_uids) - {case_uid(case) for case in selected_cases}
        )
        raise ValueError(f"Missing selected Dev cases: {missing}")

    resolved_registry_profile = normalize_registry_profile(registry_profile)

    full_registry = dict(
        registry or build_method_registry(profile=resolved_registry_profile)
    )

    if methods:
        requested_methods = [str(item).strip() for item in methods if str(item).strip()]
        unknown_methods = sorted(set(requested_methods) - set(full_registry.keys()))

        if unknown_methods:
            raise ValueError(f"Unknown dry-run methods: {unknown_methods}")

        active_registry = {
            name: full_registry[name]
            for name in full_registry
            if name in set(requested_methods)
        }

    else:
        active_registry = dict(full_registry)

    if not active_registry:
        raise ValueError("At least one method must be selected for Step 09 dry-run.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_method_names = sorted(
        {
            path.name
            for path in out_dir.iterdir()
            if path.is_dir() and path.name in set(full_registry.keys())
        }
    )

    summary_method_names = sorted(
        (set(existing_method_names) | set(active_registry.keys()))
    )

    existing_completed_keys: set[tuple[str, str]] = set()

    if resume:
        validation_rows, existing_completed_keys = _load_existing_validation_index(
            output_dir=out_dir,
            selected_uids=selected_uids,
            method_names=summary_method_names,
        )

    else:
        _clear_selected_method_outputs(
            output_dir=out_dir,
            selected_uids=selected_uids,
            method_names=list(active_registry.keys()),
        )

        validation_rows = _load_existing_validation_rows(
            output_dir=out_dir,
            selected_uids=selected_uids,
            method_names=summary_method_names,
            skip_methods=set(active_registry.keys()),
        )

    total_tasks = len(summary_method_names) * len(selected_cases)
    summary_path = out_dir / "summary.json"

    summary: dict[str, Any] = {
        "status": "running",
        "started_at": _now_iso(),
        "sample_size": len(selected_cases),
        "selected_case_uids": selected_uids,
        "registry_profile": resolved_registry_profile,
        "method_names": summary_method_names,
        "total_task_count": total_tasks,
        "completed_task_count": len(validation_rows),
        "validation_row_count": len(validation_rows),
        "validation_rows": validation_rows,
        "failure_count": 0,
        "failures": [],
        "resume_enabled": bool(resume),
        "resumed_task_count": len(validation_rows) if resume else 0,
        "current_task": None,
    }

    _write_summary(summary_path, summary)

    progress = tqdm(
        total=total_tasks,
        desc="Step09 dry-run",
        unit="task",
        initial=len(validation_rows),
        disable=not show_progress,
        dynamic_ncols=True,
    )

    try:
        for method_name, runner in active_registry.items():
            method_dir = out_dir / method_name
            method_dir.mkdir(parents=True, exist_ok=True)

            for case in selected_cases:
                selected_uid = case_uid(case)

                if resume and (method_name, selected_uid) in existing_completed_keys:
                    continue

                progress.set_postfix_str(f"{method_name}:{selected_uid[-8:]}")

                summary["current_task"] = {
                    "method": method_name,
                    "case_uid": selected_uid,
                    "started_at": _now_iso(),
                }

                _write_summary(summary_path, summary)

                try:
                    runner_kwargs = {
                        "case": case,
                        "budget": budget,
                        "seed": seed,
                        "retrieval_config": retrieval_config,
                        "verbose": verbose,
                    }

                    if storage_root_dir:
                        runner_kwargs["storage_root_dir"] = str(storage_root_dir)

                    result = runner(
                        **runner_kwargs,
                    )

                    validate_method_result(result)

                except Exception as exc:
                    failure = {
                        "method": method_name,
                        "case_uid": selected_uid,
                        "error_type": type(exc).__name__,
                        "error_detail": str(exc),
                        "failed_at": _now_iso(),
                    }

                    summary["status"] = "failed"
                    summary["failure_count"] = int(summary["failure_count"]) + 1
                    summary["failures"].append(failure)

                    summary["current_task"] = {
                        "method": method_name,
                        "case_uid": selected_uid,
                        "started_at": summary["current_task"]["started_at"],
                        "failed_at": failure["failed_at"],
                    }

                    _write_summary(summary_path, summary)
                    raise

                uid = str(result["case_uid"])

                (method_dir / f"{uid}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                validation_rows.append(_summarize_result_row(method_name, result))

                summary["completed_task_count"] = (
                    int(summary["completed_task_count"]) + 1
                )

                summary["validation_row_count"] = len(validation_rows)

                summary["last_completed_task"] = {
                    "method": method_name,
                    "case_uid": uid,
                    "completed_at": _now_iso(),
                }

                _write_summary(summary_path, summary)
                progress.update(1)

    finally:
        progress.close()

    summary["status"] = "completed"
    summary["completed_at"] = _now_iso()
    summary["current_task"] = None
    _write_summary(summary_path, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 09 dry-run on Dev cases")

    parser.add_argument(
        "--input-path",
        default="data/sampling/cleaned_samples.jsonl",
    )

    parser.add_argument(
        "--dev-ids-path",
        default="benchmarks/experiments/artifacts/splits/dev_ids.json",
    )

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-uids-path", default="")

    parser.add_argument(
        "--methods",
        nargs="*",
        default=[],
        help="Optional subset of method names to run.",
    )

    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260307)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--memory-dir", default="")

    parser.add_argument(
        "--registry-profile",
        default=INTERNAL_DEFAULT_PROFILE,
    )

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    """Run the Step 09 dry-run CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()

    summary = run_step09_dryrun(
        input_path=args.input_path,
        dev_ids_path=args.dev_ids_path,
        output_dir=args.output_dir,
        storage_root_dir=args.memory_dir or None,
        methods=list(args.methods or []),
        selected_uids_path=args.selected_uids_path or None,
        sample_size=args.sample_size,
        budget={"max_turns": int(args.max_turns)},
        seed=int(args.seed),
        verbose=bool(args.verbose),
        show_progress=True,
        resume=bool(args.resume),
        registry_profile=args.registry_profile,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
