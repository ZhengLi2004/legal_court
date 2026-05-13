"""Extend an existing Claim 3 run with additional warmup points.

The script only patches the run manifest and job plan. The actual warmup
continuation, branch evaluation, and summarization are still performed by the
standard Claim 3 runner.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORTS_ROOT = REPO_ROOT / "reports/experiments"
DEFAULT_WARMUP_POINTS = "0,25,50,75,100"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_points(value: str) -> tuple[int, ...]:
    points = tuple(
        sorted(set(int(item.strip()) for item in value.split(",") if item.strip()))
    )

    if not points:
        raise ValueError("Point list must be non-empty.")

    if points[0] != 0:
        raise ValueError("Claim 3 warmup points must include 0 as the baseline.")

    return points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--warmup-points", default=DEFAULT_WARMUP_POINTS)

    parser.add_argument(
        "--fixed-pack-points",
        default=None,
        help="Defaults to the same value as --warmup-points.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the patch without writing files.",
    )

    return parser.parse_args()


def _validate_existing_points_preserved(
    *,
    old_points: tuple[int, ...],
    requested_points: tuple[int, ...],
) -> None:
    missing_old_points = [
        point for point in old_points if point not in requested_points
    ]

    if missing_old_points:
        raise ValueError(
            "Requested warmup points must preserve existing points; "
            f"missing={missing_old_points}"
        )


def main() -> None:
    args = parse_args()
    claim_root = args.reports_root / args.run_id / "claim3"
    manifest_path = claim_root / "warmup_manifest.json"
    job_plan_path = claim_root / "job_plan.json"
    run_summary_path = claim_root / "run_summary.json"
    manifest = _read_json(manifest_path)
    job_plan = _read_json(job_plan_path)
    requested_warmup_points = _parse_points(str(args.warmup_points))

    requested_fixed_points = _parse_points(
        str(args.fixed_pack_points or args.warmup_points)
    )

    old_warmup_points = tuple(int(point) for point in manifest["warmup_points"])
    old_fixed_points = tuple(int(point) for point in manifest["fixed_pack_points"])
    warmup_pool_count = int(manifest["warmup_pool_count"])

    _validate_existing_points_preserved(
        old_points=old_warmup_points,
        requested_points=requested_warmup_points,
    )

    missing_fixed_points = [
        point
        for point in requested_fixed_points
        if point not in requested_warmup_points
    ]

    if missing_fixed_points:
        raise ValueError(
            "Fixed-pack points must be a subset of warmup points; "
            f"missing={missing_fixed_points}"
        )

    if max(requested_warmup_points) > warmup_pool_count:
        raise ValueError(
            f"Warmup pool too small for max point {max(requested_warmup_points)}: "
            f"only {warmup_pool_count}"
        )

    extension = {
        "extended_at": _now_iso(),
        "old_warmup_points": list(old_warmup_points),
        "new_warmup_points": list(requested_warmup_points),
        "old_fixed_pack_points": list(old_fixed_points),
        "new_fixed_pack_points": list(requested_fixed_points),
    }

    patched_manifest = dict(manifest)
    patched_manifest["warmup_points"] = list(requested_warmup_points)
    patched_manifest["fixed_pack_points"] = list(requested_fixed_points)
    patched_manifest.setdefault("warmup_point_extensions", []).append(extension)
    patched_job_plan = dict(job_plan)
    patched_job_plan["normal_branch_points"] = list(requested_warmup_points)
    patched_job_plan["fixed_pack_branch_points"] = list(requested_fixed_points)
    patched_job_plan.setdefault("warmup_point_extensions", []).append(extension)
    expected_task_count = dict(patched_job_plan.get("expected_task_count", {}) or {})
    expected_task_count["warmup_cases"] = max(requested_warmup_points)

    expected_task_count["normal_eval_cases"] = int(manifest["eval_case_count"]) * len(
        requested_warmup_points
    )

    expected_task_count["fixed_pack_eval_cases"] = int(
        manifest["eval_case_count"]
    ) * len(requested_fixed_points)

    patched_job_plan["expected_task_count"] = expected_task_count

    patch_summary = {
        "run_id": args.run_id,
        "status": "validated" if args.dry_run else "warmup_points_updated",
        "manifest_path": str(manifest_path),
        "job_plan_path": str(job_plan_path),
        **extension,
        "next_commands": [
            (
                "python benchmarks/experiments/core/orchestrator.py "
                f"claim3-run --run-id {args.run_id} --branch normal --resume"
            ),
            (
                "python benchmarks/experiments/core/orchestrator.py "
                f"claim3-run --run-id {args.run_id} --branch fixed-pack --resume"
            ),
            (
                "python benchmarks/experiments/core/orchestrator.py "
                f"claim3-summarize --run-id {args.run_id}"
            ),
        ],
    }

    if args.dry_run:
        print(json.dumps(patch_summary, ensure_ascii=False, indent=2))
        return

    _write_json(manifest_path, patched_manifest)
    _write_json(job_plan_path, patched_job_plan)

    if run_summary_path.exists():
        run_summary = _read_json(run_summary_path)
        run_summary["warmup_points"] = list(requested_warmup_points)
        run_summary["fixed_pack_points"] = list(requested_fixed_points)
        run_summary.setdefault("warmup_point_extensions", []).append(extension)
        _write_json(run_summary_path, run_summary)

    _write_json(claim_root / "warmup_point_extension_summary.json", patch_summary)
    print(json.dumps(patch_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
