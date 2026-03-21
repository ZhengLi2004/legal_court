"""Build disagreement reports and an adjudicated joint anchor review file."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from benchmarks.experiments.eval.compute_kappa import (
    _claim_entries,
    _normalize,
    _row_uid,
)


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


def _claim_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}

    for claim in _claim_entries(row):
        text = _normalize(str(claim.get("claim_text_raw", "") or ""))

        if text:
            mapping[text] = claim

    return mapping


def _build_disagreement_report(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]
) -> dict[str, Any]:
    map_a = {_row_uid(row): row for row in rows_a}
    map_b = {_row_uid(row): row for row in rows_b}
    details: list[dict[str, Any]] = []
    claim_only_in_a_count = 0
    claim_only_in_b_count = 0
    status_mismatch_count = 0

    for uid in sorted(set(map_a) | set(map_b)):
        row_a = map_a.get(uid, {})
        row_b = map_b.get(uid, {})
        claims_a = _claim_map(row_a)
        claims_b = _claim_map(row_b)
        only_in_a = sorted(set(claims_a) - set(claims_b))
        only_in_b = sorted(set(claims_b) - set(claims_a))
        matched = sorted(set(claims_a) & set(claims_b))
        status_mismatches = []

        for claim_key in matched:
            status_a = str(claims_a[claim_key].get("status", "") or "")
            status_b = str(claims_b[claim_key].get("status", "") or "")

            if status_a != status_b:
                status_mismatches.append(
                    {
                        "claim_text_raw": str(
                            claims_a[claim_key].get("claim_text_raw", "") or ""
                        ),
                        "status_a": status_a,
                        "status_b": status_b,
                        "reason_type": "status_conflict",
                    }
                )

        if not only_in_a and not only_in_b and not status_mismatches:
            continue

        claim_only_in_a_count += len(only_in_a)
        claim_only_in_b_count += len(only_in_b)
        status_mismatch_count += len(status_mismatches)
        only_in_a_items = []

        for claim_key in only_in_a:
            raw_text = str(claims_a[claim_key].get("claim_text_raw", "") or "")

            reason_type = (
                "meta_appeal_in_a"
                if raw_text.startswith(("撤销", "发回重审"))
                else "span_or_copy_diff"
            )

            only_in_a_items.append(
                {
                    "claim_text_raw": raw_text,
                    "status_a": str(claims_a[claim_key].get("status", "") or ""),
                    "reason_type": reason_type,
                }
            )

        only_in_b_items = [
            {
                "claim_text_raw": str(
                    claims_b[claim_key].get("claim_text_raw", "") or ""
                ),
                "status_b": str(claims_b[claim_key].get("status", "") or ""),
                "reason_type": "span_or_copy_diff",
            }
            for claim_key in only_in_b
        ]

        details.append(
            {
                "uid": uid,
                "anchor_case_id": row_a.get("anchor_case_id")
                or row_b.get("anchor_case_id"),
                "claim_only_in_a": only_in_a_items,
                "claim_only_in_b": only_in_b_items,
                "status_mismatch_on_matched_claims": status_mismatches,
            }
        )

    return {
        "case_disagreement_count": len(details),
        "claim_only_in_a_count": claim_only_in_a_count,
        "claim_only_in_b_count": claim_only_in_b_count,
        "status_mismatch_count": status_mismatch_count,
        "details": details,
    }


def _apply_decisions(
    rows_a: list[dict[str, Any]], decisions: dict[str, Any]
) -> list[dict[str, Any]]:
    overrides = decisions.get("overrides") or {}
    adjudicated: list[dict[str, Any]] = []

    for row in rows_a:
        uid = _row_uid(row)
        new_row = copy.deepcopy(row)

        if uid in overrides:
            override = overrides[uid]
            new_row["claims_review"] = list(override["claims_review"])
            new_row["adjudication_note"] = str(override.get("note", "") or "")
            new_row["adjudication_source"] = "joint_anchor_adjudication"

        adjudicated.append(new_row)

    return adjudicated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-a", required=True)
    parser.add_argument("--blind-b", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-adjudicated", required=True)
    parser.add_argument("--output-summary", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows_a = _load_jsonl(Path(args.blind_a))
    rows_b = _load_jsonl(Path(args.blind_b))
    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    report = _build_disagreement_report(rows_a, rows_b)
    adjudicated = _apply_decisions(rows_a, decisions)

    summary = {
        "input_row_count": len(rows_a),
        "adjudicated_row_count": len(adjudicated),
        "overridden_case_count": len((decisions.get("overrides") or {}).keys()),
        "case_disagreement_count": report["case_disagreement_count"],
        "claim_only_in_a_count": report["claim_only_in_a_count"],
        "claim_only_in_b_count": report["claim_only_in_b_count"],
        "status_mismatch_count": report["status_mismatch_count"],
    }

    Path(args.output_report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_jsonl(Path(args.output_adjudicated), adjudicated)

    Path(args.output_summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
