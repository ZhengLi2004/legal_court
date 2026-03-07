"""Extract Step 06 Gold Claims with rule-first and expert adjudication queue."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.data.step06_claims import (
    HanLPTokenizerSplitter,
    build_reason_count_rows,
    build_summary,
    build_uncertain_rows,
    configure_hanlp_env,
    extract_rule_claims,
    load_claim_ontology,
    load_rule_bundle,
    merge_claim_result,
    serialize_claim_rows,
)

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))

load_dotenv(override=True)
GOLD_CLAIMS_NAME = "gold_claims.jsonl"
UNCERTAIN_NAME = "gold_claims_uncertain.jsonl"
SUMMARY_NAME = "claim_extraction_summary.json"
REASON_COUNTS_NAME = "claim_extraction_reason_counts.csv"


def _load_id_set(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("ids")

    if not isinstance(values, list):
        raise ValueError(f"Missing `ids` in {path}")

    return {str(item) for item in values}


def _load_hard_case_map(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("labels")

    if not isinstance(rows, list):
        raise ValueError(f"Missing `labels` in {path}")

    return {str(row["uid"]): int(row["hard_case"]) for row in rows}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["reason", "count"]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def _build_split_meta(
    *,
    dev_ids: set[str],
    test_ids: set[str],
    hard_case_map: dict[str, int],
) -> dict[str, dict[str, Any]]:
    overlap = dev_ids & test_ids

    if overlap:
        raise ValueError(f"Dev/Test overlap detected: {sorted(overlap)[:5]}")

    split_meta: dict[str, dict[str, Any]] = {}

    for uid in sorted(dev_ids):
        if uid not in hard_case_map:
            raise ValueError(f"uid={uid} missing from hard_case labels")

        split_meta[uid] = {"split": "dev", "hard_case": hard_case_map[uid]}

    for uid in sorted(test_ids):
        if uid not in hard_case_map:
            raise ValueError(f"uid={uid} missing from hard_case labels")

        split_meta[uid] = {"split": "test", "hard_case": hard_case_map[uid]}

    return split_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--dev-ids", required=True)
    parser.add_argument("--test-ids", required=True)
    parser.add_argument("--hard-case-labels", required=True)
    parser.add_argument("--rules-path", required=True)
    parser.add_argument("--ontology-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--export-uncertain-only", action="store_true")

    parser.add_argument(
        "--disable-kimi",
        action="store_true",
        help="Deprecated. Step 06 now uses expert adjudication only.",
    )

    parser.add_argument("--hanlp-home")
    parser.add_argument("--hf-home")
    parser.add_argument("--hanlp-verbose")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    configure_hanlp_env(
        hanlp_home=args.hanlp_home,
        hf_home=args.hf_home,
        hanlp_verbose=args.hanlp_verbose,
    )

    input_path = Path(args.input)
    dev_ids_path = Path(args.dev_ids)
    test_ids_path = Path(args.test_ids)
    hard_case_path = Path(args.hard_case_labels)
    rules_path = Path(args.rules_path)
    ontology_path = Path(args.ontology_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases_from_jsonl(input_path)

    case_map = {
        str((case.get("metaInfo") or {}).get("uid", "") or ""): case for case in cases
    }

    rules = load_rule_bundle(rules_path)
    ontology = load_claim_ontology(ontology_path)
    dev_ids = _load_id_set(dev_ids_path)
    test_ids = _load_id_set(test_ids_path)
    hard_case_map = _load_hard_case_map(hard_case_path)

    split_meta = _build_split_meta(
        dev_ids=dev_ids, test_ids=test_ids, hard_case_map=hard_case_map
    )

    syntax_splitter = HanLPTokenizerSplitter()
    results = []

    for case in cases:
        uid = str((case.get("metaInfo") or {}).get("uid", "") or "")

        if uid not in split_meta:
            raise ValueError(f"uid={uid} missing from Step 05 split metadata")

        rule_result = extract_rule_claims(
            case,
            rules=rules,
            ontology=ontology,
            syntax_splitter=syntax_splitter,
        )

        results.append(
            merge_claim_result(
                case,
                split_meta=split_meta[uid],
                rule_result=rule_result,
            )
        )

    claim_rows = serialize_claim_rows(results)
    uncertain_rows = build_uncertain_rows(results, case_map)

    summary = build_summary(
        results,
        claim_rows,
        uncertain_rows,
        expert_audit_rows=[],
        total_cases=len(cases),
        rules=rules,
    )

    summary.update(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(input_path.resolve()),
            "dev_ids_path": str(dev_ids_path.resolve()),
            "test_ids_path": str(test_ids_path.resolve()),
            "hard_case_labels_path": str(hard_case_path.resolve()),
            "rules_path": str(rules_path.resolve()),
            "ontology_path": str(ontology_path.resolve()),
            "expert_protocol": "manual_adjudication_required",
            "export_uncertain_only": bool(args.export_uncertain_only),
            "hanlp_home": os.getenv("HANLP_HOME", ""),
            "hf_home": os.getenv("HF_HOME", ""),
        }
    )

    if not args.export_uncertain_only:
        _write_jsonl(output_dir / GOLD_CLAIMS_NAME, claim_rows)

    _write_jsonl(output_dir / UNCERTAIN_NAME, uncertain_rows)
    _write_json(output_dir / SUMMARY_NAME, summary)
    _write_csv(output_dir / REASON_COUNTS_NAME, build_reason_count_rows(uncertain_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
