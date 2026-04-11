"""Extend an existing Claim 3 run so fixed-pack can evaluate extra warmup points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.experiments.core.claim3_runner import CLAIM3_FIXED_PACK_POINTS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORTS_ROOT = REPO_ROOT / "reports/experiments"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--fixed-pack-points", default="0,25,50,100")
    return parser.parse_args()


def _parse_points(value: str) -> tuple[int, ...]:
    points = tuple(
        sorted(set(int(item.strip()) for item in value.split(",") if item.strip()))
    )

    if not points:
        raise ValueError("fixed-pack-points must be non-empty.")

    return points


def main() -> None:
    args = parse_args()
    claim_root = args.reports_root / args.run_id / "claim3"
    manifest_path = claim_root / "warmup_manifest.json"
    job_plan_path = claim_root / "job_plan.json"
    manifest = _read_json(manifest_path)
    job_plan = _read_json(job_plan_path)
    requested_points = _parse_points(str(args.fixed_pack_points))
    warmup_points = tuple(int(point) for point in manifest["warmup_points"])

    missing_from_warmup = [
        point for point in requested_points if point not in set(warmup_points)
    ]

    if missing_from_warmup:
        raise ValueError(
            f"Requested fixed-pack points are not in warmup_points: {missing_from_warmup}"
        )

    missing_snapshots = [
        point
        for point in requested_points
        if not (
            claim_root / "memory_snapshots" / "main_system" / f"warmup_{point}"
        ).exists()
    ]

    if missing_snapshots:
        raise FileNotFoundError(
            f"Missing warmup snapshots for fixed-pack extension: {missing_snapshots}"
        )

    manifest["fixed_pack_points"] = list(requested_points)
    job_plan["fixed_pack_branch_points"] = list(requested_points)

    if (
        "expected_task_count" in job_plan
        and "fixed_pack_eval_cases" in job_plan["expected_task_count"]
    ):
        job_plan["expected_task_count"]["fixed_pack_eval_cases"] = int(
            manifest["eval_case_count"]
        ) * len(requested_points)

    run_summary_path = claim_root / "run_summary.json"

    if run_summary_path.exists():
        run_summary = _read_json(run_summary_path)
        run_summary["fixed_pack_points"] = list(requested_points)
        _write_json(run_summary_path, run_summary)

    _write_json(manifest_path, manifest)
    _write_json(job_plan_path, job_plan)

    patch_summary = {
        "run_id": args.run_id,
        "status": "fixed_pack_points_updated",
        "manifest_path": str(manifest_path),
        "job_plan_path": str(job_plan_path),
        "old_default_fixed_pack_points": list(CLAIM3_FIXED_PACK_POINTS),
        "updated_fixed_pack_points": list(requested_points),
    }

    _write_json(claim_root / "fixed_pack_extension_summary.json", patch_summary)
    print(json.dumps(patch_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
