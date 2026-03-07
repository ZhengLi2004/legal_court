"""Build Step 06A Gold Status artifacts from frozen Gold Claims."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.data.status_inference import (
    CrossEncoderReranker,
    DenseRetriever,
    ZeroShotStatusClassifier,
    action_family,
    build_action_aliases,
    build_liability_aliases,
    build_status_summary,
    infer_case_statuses,
    load_status_rule_bundle,
)
from benchmarks.experiments.data.step06_claims import load_claim_ontology

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))

AUTO_STATUS_NAME = "gold_status.jsonl"
UNCERTAIN_STATUS_NAME = "gold_status_uncertain.jsonl"
SUMMARY_NAME = "status_classification_summary.json"
SMOKE_CLAIMS_NAME = "status_smoke_claims.jsonl"
SMOKE_PREDICTIONS_NAME = "status_smoke_predictions.jsonl"
SMOKE_SUMMARY_NAME = "status_smoke_summary.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _allocate(
    group_sizes: dict[tuple[Any, ...], int], sample_size: int
) -> dict[tuple[Any, ...], int]:
    total = sum(group_sizes.values())

    if sample_size > total:
        raise ValueError(f"sample_size={sample_size} exceeds available claims={total}")

    raw = {key: sample_size * size / total for key, size in group_sizes.items()}
    alloc = {key: min(group_sizes[key], int(value)) for key, value in raw.items()}
    used = sum(alloc.values())

    remainders = sorted(
        ((raw[key] - alloc[key], key) for key in group_sizes),
        reverse=True,
    )

    idx = 0

    while used < sample_size and remainders:
        _, key = remainders[idx]

        if alloc[key] < group_sizes[key]:
            alloc[key] += 1
            used += 1

        idx += 1

        if idx == len(remainders) and used < sample_size:
            idx = 0

    return alloc


def _sample_smoke_claims(
    rows: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    for row in rows:
        key = (
            str(row.get("stage", "")),
            action_family(str(row.get("action", ""))),
            int(row.get("hard_case", 0)),
            str(row.get("claim_source", "")),
        )

        grouped.setdefault(key, []).append(row)

    alloc = _allocate({key: len(items) for key, items in grouped.items()}, sample_size)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []

    for key in sorted(grouped):
        items = list(grouped[key])
        rng.shuffle(items)
        selected.extend(items[: alloc[key]])

    selected.sort(
        key=lambda row: (
            str(row.get("split", "")),
            str(row.get("stage", "")),
            -int(row.get("hard_case", 0)),
            str(row.get("uid", "")),
            str(row.get("claim_id", "")),
        )
    )

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--claims-path", required=True)
    parser.add_argument("--rules-path", required=True)
    parser.add_argument("--ontology-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--sample-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260307)
    parser.add_argument("--embed-model")
    parser.add_argument("--reranker-model")
    parser.add_argument("--zeroshot-model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    claims_path = Path(args.claims_path)
    rules_path = Path(args.rules_path)
    ontology_path = Path(args.ontology_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embed_model = args.embed_model or os.getenv("STATUS_EMBED_MODEL", "")
    reranker_model = args.reranker_model or os.getenv("STATUS_RERANKER_MODEL", "")
    zeroshot_model = args.zeroshot_model or os.getenv("STATUS_ZEROSHOT_MODEL", "")

    if not embed_model or not reranker_model or not zeroshot_model:
        raise ValueError("STATUS model paths are required via args or environment")

    cases = load_cases_from_jsonl(input_path)

    case_map = {
        str((case.get("metaInfo") or {}).get("uid", "") or ""): case for case in cases
    }

    claim_rows = _load_jsonl(claims_path)

    if args.mode == "smoke":
        claim_rows = _sample_smoke_claims(claim_rows, args.sample_size, args.seed)

    rules = load_status_rule_bundle(rules_path)
    ontology = load_claim_ontology(ontology_path)
    action_alias_map = build_action_aliases(ontology)
    liability_alias_map = build_liability_aliases(ontology)
    embedder = DenseRetriever(embed_model)
    reranker = CrossEncoderReranker(reranker_model)

    zeroshot = ZeroShotStatusClassifier(
        zeroshot_model, rules.zeroshot_hypothesis_template
    )

    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in claim_rows:
        grouped.setdefault(str(row["uid"]), []).append(row)

    auto_rows: list[dict[str, Any]] = []
    uncertain_rows: list[dict[str, Any]] = []

    for uid, rows in sorted(grouped.items()):
        case = case_map.get(uid)

        if case is None:
            raise ValueError(f"uid={uid} missing from input cases")

        case_auto, case_uncertain = infer_case_statuses(
            case,
            rows,
            rules=rules,
            action_alias_map=action_alias_map,
            liability_alias_map=liability_alias_map,
            embedder=embedder,
            reranker=reranker,
            zeroshot=zeroshot,
        )

        auto_rows.extend(case_auto)
        uncertain_rows.extend(case_uncertain)

    auto_rows.sort(key=lambda row: (row["uid"], row["claim_id"]))
    uncertain_rows.sort(key=lambda row: (row["uid"], row["claim_id"]))

    summary = build_status_summary(
        auto_rows=auto_rows,
        uncertain_rows=uncertain_rows,
        total_claims=len(claim_rows),
        mode=args.mode,
    )

    summary.update(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(input_path.resolve()),
            "claims_path": str(claims_path.resolve()),
            "rules_path": str(rules_path.resolve()),
            "ontology_path": str(ontology_path.resolve()),
            "embed_model": str(embed_model),
            "reranker_model": str(reranker_model),
            "zeroshot_model": str(zeroshot_model),
        }
    )

    if args.mode == "smoke":
        predictions = list(auto_rows) + [
            {
                "uid": row["uid"],
                "claim_id": row["claim_id"],
                "status_canonical": "",
                "status_eval": "",
                "status_source": "uncertain",
                "decision_mode": "uncertain",
                "confidence": None,
                "review_reason": ",".join(row.get("review_reasons", [])),
                "claim_text_raw": row["claim_text_raw"],
                "candidate_evidence": row.get("candidate_evidence", []),
            }
            for row in uncertain_rows
        ]

        predictions.sort(key=lambda row: (row["uid"], row["claim_id"]))
        _write_jsonl(output_dir / SMOKE_CLAIMS_NAME, claim_rows)
        _write_jsonl(output_dir / SMOKE_PREDICTIONS_NAME, predictions)
        _write_json(output_dir / SMOKE_SUMMARY_NAME, summary)
        return 0

    _write_jsonl(output_dir / AUTO_STATUS_NAME, auto_rows)
    _write_jsonl(output_dir / UNCERTAIN_STATUS_NAME, uncertain_rows)
    _write_json(output_dir / SUMMARY_NAME, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
