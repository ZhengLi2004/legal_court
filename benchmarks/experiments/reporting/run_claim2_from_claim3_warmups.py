"""Run Claim 2 dev-only evaluation on Claim 3 warmup outputs.

This bridge script reuses the existing Claim 2 stage logic and consumes
Claim 3 `raw_predictions/warmup_*` outputs directly, without pretending those
outputs are a full Claim 1 run.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Any

from benchmarks.experiments.core.claim2_runner import (
    CLAIM2,
    CLAIM2_AGREEMENT_THRESHOLD,
    CLAIM2_ALIGNMENT_THRESHOLD,
    CLAIM2_DEFAULT_AGGREGATION,
    CLAIM2_PROTOCOL_VERSION,
    CLAIM2_TASK_FOCUS,
    CLAIM2_TOP_K,
    Claim2ExecutionError,
    _build_aligner,
    _load_cases_by_uid,
    _load_step09a_manifest,
    _now_iso,
    _run_claim2_stage,
    _write_json,
    _write_text,
)
from benchmarks.experiments.core.claim3_runner import (
    DEFAULT_REPORTS_ROOT,
)
from benchmarks.experiments.eval.claim2_evaluators import (
    build_claim2_prompt_snapshot,
    build_primary_labelers,
    build_supplementary_labelers,
    load_evaluator_profile,
    probe_evaluator_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLAIM3_RUN_ID = "20260327_step12_claim3_internal_current"

DEFAULT_EVALUATOR_PROFILE_PATH = (
    REPO_ROOT
    / "benchmarks/experiments/artifacts/evaluator_profiles/claim2_primary_qwen35_mdeberta_current.template.json"
)

BRANCH_CHOICES = ("normal", "fixed-pack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim3-run-id", default=DEFAULT_CLAIM3_RUN_ID)
    parser.add_argument("--branch", choices=BRANCH_CHOICES, default="normal")
    parser.add_argument("--warmup-points", type=int, nargs="+", default=[0, 100])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--input-path", default="")

    parser.add_argument(
        "--evaluator-profile-path",
        default=str(DEFAULT_EVALUATOR_PROFILE_PATH),
    )

    parser.add_argument("--method-name", default="main_system")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Proceed even if one or more primary evaluators fail the preflight health check.",
    )

    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--top-k", type=int, default=CLAIM2_TOP_K)
    parser.add_argument("--aggregation", default=CLAIM2_DEFAULT_AGGREGATION)

    parser.add_argument(
        "--agreement-threshold", type=float, default=CLAIM2_AGREEMENT_THRESHOLD
    )

    parser.add_argument(
        "--alignment-threshold", type=float, default=CLAIM2_ALIGNMENT_THRESHOLD
    )

    return parser.parse_args()


def _read_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _read_claim3_manifest(claim3_root: Path) -> dict[str, Any]:
    manifest_path = claim3_root / "warmup_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Claim 3 warmup manifest: {manifest_path}")

    return _read_json(manifest_path)


def _ensure_prediction_view(
    *, source_dir: Path, target_method_dir: Path, resume: bool
) -> None:
    target_method_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_method_dir.is_symlink():
        if target_method_dir.resolve() == source_dir.resolve():
            return

        target_method_dir.unlink()

    if target_method_dir.exists():
        if resume:
            return

        if target_method_dir.is_dir():
            shutil.rmtree(target_method_dir)

        else:
            target_method_dir.unlink()

    try:
        os.symlink(source_dir, target_method_dir, target_is_directory=True)

    except OSError:
        shutil.copytree(source_dir, target_method_dir)


def _write_bridge_reproducibility_block(
    *,
    path: Path,
    freeze_run_id: str,
    source_claim1_run_id: str,
    source_claim3_run_id: str,
    branch: str,
    warmup_point: int,
    evaluator_profile_snapshot_path: str,
    prompt_snapshot_path: str,
    top_k: int,
    aggregation: str,
    agreement_threshold: float,
    alignment_threshold: float,
) -> None:
    content = "\n".join(
        [
            "# Claim 2 Dev-Only Bridge From Claim 3",
            "",
            f"- freeze_run_id: `{freeze_run_id}`",
            f"- source_claim1_run_id: `{source_claim1_run_id}`",
            f"- source_claim3_run_id: `{source_claim3_run_id}`",
            f"- branch: `{branch}`",
            f"- warmup_point: `{warmup_point}`",
            f"- top_k: `{top_k}`",
            f"- aggregation: `{aggregation}`",
            f"- agreement_threshold: `{agreement_threshold}`",
            f"- alignment_threshold: `{alignment_threshold}`",
            f"- evaluator_profile_snapshot: `{evaluator_profile_snapshot_path}`",
            f"- prompt_snapshot: `{prompt_snapshot_path}`",
        ]
    )

    _write_text(path, content)


def _write_warmup_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_bridge(args: argparse.Namespace) -> dict[str, Any]:
    reports_root = Path(args.reports_root).resolve()
    claim3_root = reports_root / args.claim3_run_id / "claim3"
    claim3_manifest = _read_claim3_manifest(claim3_root)
    freeze_run_id = str(claim3_manifest["freeze_run_id"])
    source_claim1_run_id = str(claim3_manifest["source_claim1_run_id"])
    input_path = Path(args.input_path or claim3_manifest["input_path"]).resolve()
    eval_uids = [str(uid) for uid in list(claim3_manifest["eval_uids"])]
    _load_step09a_manifest(freeze_run_id=freeze_run_id, reports_root=reports_root)
    case_map = _load_cases_by_uid(input_path)
    missing_case_uids = sorted(set(eval_uids) - set(case_map.keys()))

    if missing_case_uids:
        raise ValueError(f"Missing case payloads for eval uids: {missing_case_uids}")

    run_root = reports_root / args.run_id / "claim2_from_claim3" / args.branch
    run_root.mkdir(parents=True, exist_ok=True)
    profile = load_evaluator_profile(args.evaluator_profile_path)
    prompt_snapshot = build_claim2_prompt_snapshot()
    profile_snapshot_path = run_root / "evaluator_profile_snapshot.json"
    prompt_snapshot_path = run_root / "prompt_snapshot.json"
    health_path = run_root / "evaluator_health.json"
    protocol_snapshot_path = run_root / "bridge_protocol_snapshot.json"
    _write_json(profile_snapshot_path, profile.to_snapshot())
    _write_json(prompt_snapshot_path, prompt_snapshot)
    health_rows = probe_evaluator_profile(profile)
    _write_json(health_path, health_rows)

    failed_primary = [
        row
        for row in health_rows
        if row.get("cohort") == "primary" and not bool(row.get("ok", False))
    ]

    if failed_primary and not args.skip_health_check:
        raise Claim2ExecutionError(
            "One or more primary evaluators failed health check."
        )

    _write_json(
        protocol_snapshot_path,
        {
            "claim_id": CLAIM2,
            "protocol_version": CLAIM2_PROTOCOL_VERSION,
            "task_focus": f"{CLAIM2_TASK_FOCUS}_dev_only_from_claim3",
            "run_id": args.run_id,
            "freeze_run_id": freeze_run_id,
            "source_claim1_run_id": source_claim1_run_id,
            "source_claim3_run_id": args.claim3_run_id,
            "branch": args.branch,
            "warmup_points": [int(point) for point in args.warmup_points],
            "input_path": str(input_path),
            "eval_case_count": len(eval_uids),
            "method_names": [args.method_name],
            "top_k": int(args.top_k),
            "aggregation": str(args.aggregation),
            "agreement_threshold": float(args.agreement_threshold),
            "alignment_threshold": float(args.alignment_threshold),
            "evaluator_profile_name": profile.profile_name,
            "skip_health_check": bool(args.skip_health_check),
            "failed_primary_health_checks": failed_primary,
        },
    )

    labeler_cache_root = run_root / "cache" / "votes"

    primary_labelers = build_primary_labelers(
        profile, cache_dir=labeler_cache_root / "primary"
    )

    supplementary_labelers = build_supplementary_labelers(
        profile, cache_dir=labeler_cache_root / "supplementary"
    )

    aligner = _build_aligner(alignment_threshold=args.alignment_threshold)
    comparison_rows: list[dict[str, Any]] = []
    point_summaries: list[dict[str, Any]] = []

    for warmup_point in args.warmup_points:
        source_prediction_dir = (
            claim3_root / args.branch / "raw_predictions" / f"warmup_{warmup_point}"
        )

        if not source_prediction_dir.exists():
            raise FileNotFoundError(
                f"Missing Claim 3 raw predictions: {source_prediction_dir}"
            )

        point_root = run_root / f"warmup_{warmup_point}"
        source_view_root = point_root / "source_raw_predictions"
        target_method_dir = source_view_root / args.method_name

        _ensure_prediction_view(
            source_dir=source_prediction_dir,
            target_method_dir=target_method_dir,
            resume=args.resume,
        )

        dev_dir = point_root / "dev"
        source_claim1_tag = f"{args.claim3_run_id}:{args.branch}:warmup_{warmup_point}"

        dev_execution = _run_claim2_stage(
            stage_name="dev",
            case_uids=eval_uids,
            case_map=case_map,
            raw_predictions_dir=source_view_root,
            method_order=[args.method_name],
            stage_dir=dev_dir,
            primary_labelers=primary_labelers,
            supplementary_labelers=supplementary_labelers,
            aligner=aligner,
            top_k=args.top_k,
            aggregation=args.aggregation,
            agreement_threshold=args.agreement_threshold,
            alignment_threshold=args.alignment_threshold,
            run_id=args.run_id,
            source_claim1_run_id=source_claim1_tag,
            evaluator_profile_name=profile.profile_name,
            show_progress=not args.no_progress,
            resume=args.resume,
        )

        method_metric = dev_execution["method_metrics"][args.method_name]
        main_metrics = method_metric["main_metrics"]
        appendix = method_metric["appendix_metrics"]
        agreement = method_metric["agreement_matrix"]
        gate = method_metric["agreement_gate"]
        _write_json(dev_dir / "agreement_gate.json", gate)

        _write_bridge_reproducibility_block(
            path=dev_dir / "reproducibility_block.md",
            freeze_run_id=freeze_run_id,
            source_claim1_run_id=source_claim1_run_id,
            source_claim3_run_id=args.claim3_run_id,
            branch=args.branch,
            warmup_point=int(warmup_point),
            evaluator_profile_snapshot_path=str(profile_snapshot_path),
            prompt_snapshot_path=str(prompt_snapshot_path),
            top_k=args.top_k,
            aggregation=args.aggregation,
            agreement_threshold=args.agreement_threshold,
            alignment_threshold=args.alignment_threshold,
        )

        point_summary = {
            "claim_id": CLAIM2,
            "stage": "dev_only_from_claim3",
            "status": "completed",
            "task_focus": f"{CLAIM2_TASK_FOCUS}_dev_only_from_claim3",
            "run_id": args.run_id,
            "freeze_run_id": freeze_run_id,
            "source_claim1_run_id": source_claim1_run_id,
            "source_claim3_run_id": args.claim3_run_id,
            "branch": args.branch,
            "warmup_point": int(warmup_point),
            "eval_case_count": len(eval_uids),
            "method_names": [args.method_name],
            "evaluator_profile_name": profile.profile_name,
            "agreement_gate": gate,
            "artifacts": {
                "source_prediction_dir": str(source_prediction_dir),
                "source_view_dir": str(source_view_root),
                "dev_execution_summary": str(dev_dir / "execution_summary.json"),
                "dev_method_metrics": str(dev_dir / "method_metrics.json"),
                "dev_main_table_csv": str(dev_dir / "main_table.csv"),
                "dev_main_table_json": str(dev_dir / "main_table.json"),
                "dev_evaluator_agreement_matrix": str(
                    dev_dir / "evaluator_agreement_matrix.csv"
                ),
                "dev_parse_failure_audit": str(dev_dir / "parse_failure_audit.json"),
                "dev_faithfulness_decomposition": str(
                    dev_dir / "faithfulness_decomposition.json"
                ),
                "dev_reproducibility_block": str(dev_dir / "reproducibility_block.md"),
            },
        }

        _write_json(point_root / "run_summary.json", point_summary)
        point_summaries.append(point_summary)

        comparison_rows.append(
            {
                "branch": args.branch,
                "warmup_point": int(warmup_point),
                "method_name": args.method_name,
                "overall_faithfulness": main_metrics["overall_faithfulness"][
                    "point_estimate"
                ],
                "overall_faithfulness_ci_low": main_metrics["overall_faithfulness"][
                    "ci_low"
                ],
                "overall_faithfulness_ci_high": main_metrics["overall_faithfulness"][
                    "ci_high"
                ],
                "alignment_success_rate": appendix["alignment_success_rate"][
                    "point_estimate"
                ],
                "entailment_rate_on_aligned": appendix["entailment_rate_on_aligned"][
                    "point_estimate"
                ],
                "agreement_overall_majority": agreement["overall_majority_agreement"],
                "agreement_uncertain_rate": agreement["uncertain_rate"],
                "agreement_gate_passed": bool(gate.get("passed", False)),
            }
        )

    comparison_path = run_root / "warmup_comparison.csv"
    _write_warmup_comparison(comparison_path, comparison_rows)

    summary = {
        "claim_id": CLAIM2,
        "stage": "dev_only_from_claim3",
        "status": "completed",
        "completed_at": _now_iso(),
        "run_id": args.run_id,
        "freeze_run_id": freeze_run_id,
        "source_claim1_run_id": source_claim1_run_id,
        "source_claim3_run_id": args.claim3_run_id,
        "branch": args.branch,
        "warmup_points": [int(point) for point in args.warmup_points],
        "eval_case_count": len(eval_uids),
        "method_names": [args.method_name],
        "evaluator_profile_name": profile.profile_name,
        "artifacts": {
            "protocol_snapshot": str(protocol_snapshot_path),
            "evaluator_profile_snapshot": str(profile_snapshot_path),
            "prompt_snapshot": str(prompt_snapshot_path),
            "evaluator_health": str(health_path),
            "warmup_comparison": str(comparison_path),
        },
        "point_runs": point_summaries,
    }

    _write_json(run_root / "run_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    run_bridge(args)


if __name__ == "__main__":
    main()
