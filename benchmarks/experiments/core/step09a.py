"""Step 09A preflight and protocol freeze helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.eval.matching_robustness import (
    FROZEN_BASE_CONFIG,
    FROZEN_GATE_FLIP_THRESHOLD,
    FROZEN_GATE_RHO_THRESHOLD,
    FROZEN_MATCHING_PROTOCOL_VERSION,
    default_matching_scenarios,
)
from benchmarks.experiments.methods.base import case_uid
from benchmarks.experiments.methods.factory import build_default_registry
from mas.config import SystemConfig
from mas.infrastructure.embedding import EmbeddingFunc

DEFAULT_INPUT_PATH = Path("data/sampling/cleaned_samples.jsonl")
DEFAULT_DEV_IDS_PATH = Path("benchmarks/experiments/artifacts/splits/dev_ids.json")

DEFAULT_GOLD_CLAIMS_PATH = Path(
    "benchmarks/experiments/artifacts/gold/gold_claims_final.jsonl"
)

DEFAULT_GOLD_STATUS_PATH = Path(
    "benchmarks/experiments/artifacts/gold/gold_status_final.jsonl"
)

DEFAULT_SPLIT_MANIFEST_PATH = Path(
    "benchmarks/experiments/artifacts/splits/split_manifest.json"
)

DEFAULT_REPORTS_ROOT = Path("reports/experiments")
DEFAULT_DRYRUN_SAMPLE_SIZE = 3
DEFAULT_SEED = 20260307
DEFAULT_FULL_MAX_TURNS = 10
METRIC_CONTRACT_VERSION = "step08_claim1_v1"


def _utc_now_iso() -> str:
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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _secret_fingerprint(value: str) -> dict[str, Any]:
    text = str(value or "")

    if not text:
        return {"set": False, "sha256_prefix": ""}

    return {
        "set": True,
        "sha256_prefix": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    }


def _probe_result(
    *,
    check_name: str,
    ok: bool,
    start_ts: float,
    endpoint: str = "",
    error: Exception | None = None,
    response_excerpt: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "check_name": check_name,
        "ok": bool(ok),
        "latency_ms": round((time.perf_counter() - start_ts) * 1000.0, 3),
        "endpoint": endpoint,
        "error_type": "",
        "error_detail": "",
        "response_excerpt": response_excerpt,
    }

    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_detail"] = str(error)

    if extra:
        payload.update(extra)

    return payload


def _probe_embedding(model_path: str) -> dict[str, Any]:
    start_ts = time.perf_counter()

    try:
        embedding = EmbeddingFunc(model_path)
        vector = embedding.embed_query("测试")

        return _probe_result(
            check_name="embedding",
            ok=len(vector) > 0,
            start_ts=start_ts,
            endpoint=model_path,
            extra={"vector_dim": len(vector)},
        )

    except Exception as exc:
        return _probe_result(
            check_name="embedding",
            ok=False,
            start_ts=start_ts,
            endpoint=model_path,
            error=exc,
        )


def _probe_es_json(url: str, *, check_name: str) -> dict[str, Any]:
    start_ts = time.perf_counter()

    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

        extra: dict[str, Any] = {}

        if "status" in payload:
            extra["cluster_status"] = payload.get("status")

        if "count" in payload:
            extra["count"] = payload.get("count")

        return _probe_result(
            check_name=check_name,
            ok=True,
            start_ts=start_ts,
            endpoint=url,
            extra=extra,
            response_excerpt=json.dumps(payload, ensure_ascii=False)[:200],
        )

    except Exception as exc:
        return _probe_result(
            check_name=check_name,
            ok=False,
            start_ts=start_ts,
            endpoint=url,
            error=exc,
        )


def _probe_openai_chat(
    *,
    check_name: str,
    base_url: str,
    api_key: str,
    model_name: str,
) -> dict[str, Any]:
    start_ts = time.perf_counter()

    try:
        client = OpenAI(base_url=base_url, api_key=api_key)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个测试助手。只输出OK。"},
                {"role": "user", "content": "请只输出OK"},
            ],
            max_tokens=4,
            temperature=0.0,
        )

        answer = ""

        if response.choices:
            answer = str(response.choices[0].message.content or "").strip()

        if not answer:
            raise RuntimeError("LLM probe returned empty content")

        return _probe_result(
            check_name=check_name,
            ok=True,
            start_ts=start_ts,
            endpoint=base_url,
            response_excerpt=answer[:80],
            extra={"model_name": model_name},
        )

    except Exception as exc:
        return _probe_result(
            check_name=check_name,
            ok=False,
            start_ts=start_ts,
            endpoint=base_url,
            error=exc,
            extra={"model_name": model_name},
        )


def _load_dev_uids(dev_ids_path: str | Path) -> list[str]:
    payload = _read_json(dev_ids_path)
    ids = payload.get("ids", [])

    if not isinstance(ids, list) or not ids:
        raise ValueError("dev_ids.json must contain a non-empty `ids` list.")

    return [str(item) for item in ids]


def select_step09a_dryrun_case_uids(
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    dev_ids_path: str | Path = DEFAULT_DEV_IDS_PATH,
    claims_path: str | Path = DEFAULT_GOLD_CLAIMS_PATH,
    sample_size: int = DEFAULT_DRYRUN_SAMPLE_SIZE,
) -> list[str]:
    cases = load_cases_from_jsonl(input_path)
    case_uid_set = {case_uid(case) for case in cases}
    dev_uids = _load_dev_uids(dev_ids_path)
    gold_rows = load_cases_from_jsonl(claims_path)
    gold_uid_set = {str(row.get("uid", "") or "") for row in gold_rows}

    selected = [uid for uid in dev_uids if uid in case_uid_set and uid in gold_uid_set][
        : max(1, int(sample_size))
    ]

    if len(selected) < max(1, int(sample_size)):
        raise ValueError(
            "Unable to select enough Dev cases with Gold Claims for Step 09A dry-run."
        )

    return selected


def _build_budget_grid(full_max_turns: int = DEFAULT_FULL_MAX_TURNS) -> dict[str, Any]:
    q25 = max(1, int((full_max_turns * 0.25) + 0.999999))
    q50 = max(1, int((full_max_turns * 0.50) + 0.999999))
    q75 = max(1, int((full_max_turns * 0.75) + 0.999999))

    return {
        "budget_axis": "max_turns",
        "full_budget": {"max_turns": int(full_max_turns)},
        "budget_points": {
            "q25": {"max_turns": q25},
            "q50": {"max_turns": q50},
            "q75": {"max_turns": q75},
        },
    }


def _build_prereg_points() -> dict[str, Any]:
    return {
        "comparisons": {
            "main_system": [
                "baseline_b1_structured_rag",
                "baseline_b2_vanilla_mad",
                "baseline_b3_stateful_no_axioms",
            ]
        },
        "budget_points": ["q25", "q50", "q75"],
        "round_points": [1, 3, 5],
        "temperature": 0,
        "repeats_per_budget_point": 3,
    }


def _build_runtime_config_snapshot(cfg: SystemConfig) -> dict[str, Any]:
    snapshot = asdict(cfg)
    snapshot["llm"]["api_key"] = _secret_fingerprint(cfg.llm.api_key)
    snapshot["judge"]["api_key"] = _secret_fingerprint(cfg.judge.api_key)
    return snapshot


def _build_method_registry_snapshot() -> dict[str, Any]:
    registry = build_default_registry()

    return {
        "method_names": sorted(registry.keys()),
        "method_semantics": {
            "main_system": {"mode": "full_debate"},
            "baseline_b1_structured_rag": {
                "mode": "structured_single_pass",
                "direct_adjudication": True,
                "creates_debate_engine": False,
            },
            "baseline_b2_vanilla_mad": {
                "mode": "debate_without_recall_or_initial_insights",
                "disable_recall_worker": True,
                "disable_initial_insights": True,
            },
            "baseline_b3_stateful_no_axioms": {
                "mode": "debate_without_validate_step",
                "skip_validate_step": True,
            },
        },
    }


def _build_matching_protocol_snapshot() -> dict[str, Any]:
    return {
        "protocol_version": FROZEN_MATCHING_PROTOCOL_VERSION,
        "base_config": asdict(FROZEN_BASE_CONFIG),
        "gate_rho_threshold": FROZEN_GATE_RHO_THRESHOLD,
        "gate_flip_threshold": FROZEN_GATE_FLIP_THRESHOLD,
        "scenarios": [
            {
                "name": scenario.name,
                "description": scenario.description,
                "config_overrides": scenario.config_overrides,
            }
            for scenario in default_matching_scenarios(FROZEN_BASE_CONFIG)
        ],
    }


def _build_metric_contract_snapshot() -> dict[str, Any]:
    return {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "claim1": {
            "main_metrics": ["e2e_status_acc", "status_acc_matched"],
            "appendix_metrics": [
                "step_a_precision",
                "step_a_recall",
                "step_a_f1",
                "e2e_f1_fp_sensitive",
                "soft_e2e_f1",
                "over_generation_rate",
                "macro_f1_3class_matched",
                "balanced_accuracy_3class_matched",
            ],
            "status_collapse_rule": {
                "from": "HYPOTHETICAL",
                "to": "DEFEATED",
                "scope": "claim1_main_metrics_only",
            },
        },
    }


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )

    return completed.stdout.strip()


def _git_is_clean() -> bool:
    return _git_output("status", "--short") == ""


def _git_head_short() -> str:
    return _git_output("rev-parse", "--short", "HEAD")


def _ensure_git_tag(tag_name: str) -> None:
    existing = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag_name}"],
        check=False,
        capture_output=True,
        text=True,
    )

    if existing.returncode == 0:
        tag_commit = existing.stdout.strip()
        head_commit = _git_output("rev-parse", "HEAD")

        if tag_commit != head_commit:
            raise ValueError(
                f"Git tag `{tag_name}` already exists on a different commit."
            )

        return

    subprocess.run(["git", "tag", tag_name], check=True)


def run_step09a_preflight(
    *,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    dev_ids_path: str | Path = DEFAULT_DEV_IDS_PATH,
    claims_path: str | Path = DEFAULT_GOLD_CLAIMS_PATH,
    gold_status_path: str | Path = DEFAULT_GOLD_STATUS_PATH,
    split_manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    sample_size: int = DEFAULT_DRYRUN_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    cfg = SystemConfig()
    run_root = Path(reports_root) / run_id
    preflight_dir = run_root / "preflight"

    selected_uids = select_step09a_dryrun_case_uids(
        input_path=input_path,
        dev_ids_path=dev_ids_path,
        claims_path=claims_path,
        sample_size=sample_size,
    )

    checks = {
        "embedding": _probe_embedding(cfg.path.embedding_model_path),
        "primary_llm": _probe_openai_chat(
            check_name="primary_llm",
            base_url=cfg.llm.base_url,
            api_key=cfg.llm.api_key,
            model_name=cfg.llm.model_name,
        ),
        "judge_llm": _probe_openai_chat(
            check_name="judge_llm",
            base_url=cfg.judge.base_url,
            api_key=cfg.judge.api_key,
            model_name=cfg.judge.model_name,
        ),
        "es_cluster": _probe_es_json(
            f"{cfg.es.host.rstrip('/')}/_cluster/health",
            check_name="es_cluster",
        ),
        "es_cases_index": _probe_es_json(
            f"{cfg.es.host.rstrip('/')}/rag_legal_cases/_count",
            check_name="es_cases_index",
        ),
        "es_laws_index": _probe_es_json(
            f"{cfg.es.host.rstrip('/')}/rag_legal_laws/_count",
            check_name="es_laws_index",
        ),
    }

    passed = all(bool(check["ok"]) for check in checks.values())

    preflight_summary = {
        "run_id": run_id,
        "checked_at": _utc_now_iso(),
        "passed": passed,
        "required_checks": list(checks.keys()),
        "checks": checks,
        "selected_dryrun_case_uids": selected_uids,
        "seed": int(seed),
    }

    _write_json(preflight_dir / "preflight_health.json", preflight_summary)

    _write_json(
        preflight_dir / "selected_dryrun_cases.json",
        {
            "run_id": run_id,
            "selected_case_uids": selected_uids,
            "sample_size": len(selected_uids),
            "seed": int(seed),
        },
    )

    _write_json(
        run_root / "runtime_config_snapshot.json", _build_runtime_config_snapshot(cfg)
    )

    _write_json(
        run_root / "method_registry_snapshot.json", _build_method_registry_snapshot()
    )

    _write_json(
        run_root / "matching_protocol_snapshot.json",
        _build_matching_protocol_snapshot(),
    )

    _write_json(
        run_root / "metric_contract_snapshot.json",
        _build_metric_contract_snapshot(),
    )

    _write_json(
        run_root / "split_refs.json",
        {
            "dev_ids_path": str(Path(dev_ids_path)),
            "test_ids_path": "benchmarks/experiments/artifacts/splits/test_ids.json",
            "split_manifest_path": str(Path(split_manifest_path)),
        },
    )

    _write_json(
        run_root / "gold_refs.json",
        {
            "gold_claims_path": str(Path(claims_path)),
            "gold_status_path": str(Path(gold_status_path)),
        },
    )

    _write_json(run_root / "budget_grid.json", _build_budget_grid())
    _write_json(run_root / "prereg_points.json", _build_prereg_points())
    return preflight_summary


def finalize_step09a_freeze(
    *,
    run_id: str,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    dryrun_summary_path: str | Path | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    run_root = Path(reports_root) / run_id
    preflight_path = run_root / "preflight" / "preflight_health.json"

    dryrun_path = Path(
        dryrun_summary_path or (run_root / "preflight" / "live_dryrun" / "summary.json")
    )

    preflight_summary = _read_json(preflight_path)

    if not bool(preflight_summary.get("passed", False)):
        raise ValueError("Step 09A preflight has not passed.")

    dryrun_summary = _read_json(dryrun_path)
    expected_methods = sorted(build_default_registry().keys())
    selected_case_uids = list(preflight_summary.get("selected_dryrun_case_uids", []))

    if sorted(dryrun_summary.get("method_names", [])) != expected_methods:
        raise ValueError("Live dry-run method set does not match the default registry.")

    if list(dryrun_summary.get("selected_case_uids", [])) != selected_case_uids:
        raise ValueError("Live dry-run cases do not match preflight-selected cases.")

    if int(dryrun_summary.get("validation_row_count", 0)) != len(
        expected_methods
    ) * len(selected_case_uids):
        raise ValueError("Live dry-run validation row count is incomplete.")

    if not _git_is_clean():
        raise ValueError("Step 09A freeze requires a clean git worktree.")

    tag_name = f"exp-protocol-freeze-{datetime.now().strftime('%Y%m%d')}"
    _ensure_git_tag(tag_name)

    file_paths = [
        run_root / "runtime_config_snapshot.json",
        run_root / "method_registry_snapshot.json",
        run_root / "matching_protocol_snapshot.json",
        run_root / "metric_contract_snapshot.json",
        run_root / "split_refs.json",
        run_root / "gold_refs.json",
        run_root / "budget_grid.json",
        run_root / "prereg_points.json",
        preflight_path,
        run_root / "preflight" / "selected_dryrun_cases.json",
        dryrun_path,
    ]

    for file_path in file_paths:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing Step 09A freeze input: {file_path}")

    retrieval_config_snapshot = asdict(SystemConfig().retrieval)
    budget_grid = _read_json(run_root / "budget_grid.json")

    manifest = {
        "run_id": run_id,
        "frozen_at": _utc_now_iso(),
        "git_commit": _git_head_short(),
        "git_tag": tag_name,
        "worktree_clean": True,
        "preflight_passed": True,
        "live_dryrun_passed": True,
        "runtime_files": [str(path.relative_to(run_root)) for path in file_paths],
        "file_hashes": {
            str(path.relative_to(run_root)): _sha256_file(path) for path in file_paths
        },
        "shared_constraints": {
            "case_uids_ref": str(
                (run_root / "preflight" / "selected_dryrun_cases.json").relative_to(
                    run_root
                )
            ),
            "seed": int(seed),
            "budget_axis": str(budget_grid["budget_axis"]),
            "budget_grid": dict(budget_grid["budget_points"]),
            "retrieval_config_snapshot": retrieval_config_snapshot,
            "matching_protocol_version": FROZEN_MATCHING_PROTOCOL_VERSION,
            "metric_contract_version": METRIC_CONTRACT_VERSION,
        },
        "notes": [
            "Step 09A freeze uses runtime snapshots, not legacy configs/aligner directories.",
            "Dry-run must pass before any Step 10-13 execution is allowed.",
        ],
    }

    _write_json(run_root / "protocol_freeze_manifest.json", manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 09A preflight and freeze helpers"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--run-id", required=True)
    preflight_parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    preflight_parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    preflight_parser.add_argument("--dev-ids-path", default=str(DEFAULT_DEV_IDS_PATH))

    preflight_parser.add_argument(
        "--claims-path", default=str(DEFAULT_GOLD_CLAIMS_PATH)
    )

    preflight_parser.add_argument(
        "--gold-status-path", default=str(DEFAULT_GOLD_STATUS_PATH)
    )

    preflight_parser.add_argument(
        "--split-manifest-path", default=str(DEFAULT_SPLIT_MANIFEST_PATH)
    )

    preflight_parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_DRYRUN_SAMPLE_SIZE
    )

    preflight_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--run-id", required=True)
    finalize_parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    finalize_parser.add_argument("--dryrun-summary-path", default="")
    finalize_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "preflight":
        payload = run_step09a_preflight(
            run_id=args.run_id,
            reports_root=args.reports_root,
            input_path=args.input_path,
            dev_ids_path=args.dev_ids_path,
            claims_path=args.claims_path,
            gold_status_path=args.gold_status_path,
            split_manifest_path=args.split_manifest_path,
            sample_size=args.sample_size,
            seed=args.seed,
        )

    else:
        payload = finalize_step09a_freeze(
            run_id=args.run_id,
            reports_root=args.reports_root,
            dryrun_summary_path=args.dryrun_summary_path or None,
            seed=args.seed,
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
