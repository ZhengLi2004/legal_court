"""Compute simple Cohen's kappa metrics for blind Step 06 review packages."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalize(text: str) -> str:
    return "".join(str(text).split())


def _extract_claim_presence(rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    mapping: dict[tuple[str, str], int] = {}

    for row in rows:
        uid = str(row.get("uid", "") or row.get("anchor_case_id", ""))
        claims = row.get("claims_review") or row.get("claims") or []

        for claim in claims:
            claim_text = _normalize(str(claim.get("claim_text_raw", "") or ""))

            if claim_text:
                mapping[(uid, claim_text)] = 1

    return mapping


def _extract_claim_status(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}

    for row in rows:
        uid = str(row.get("uid", "") or row.get("anchor_case_id", ""))
        claims = row.get("claims_review") or row.get("claims") or []

        for claim in claims:
            claim_text = _normalize(str(claim.get("claim_text_raw", "") or ""))

            if claim_text:
                mapping[(uid, claim_text)] = str(
                    claim.get("status", "ABSENT") or "ABSENT"
                )

    return mapping


def _case_exact_match_rate(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]
) -> float:
    case_map_a = {
        str(row.get("uid", "") or row.get("anchor_case_id", "")): sorted(
            _normalize(str(claim.get("claim_text_raw", "") or ""))
            for claim in (row.get("claims_review") or row.get("claims") or [])
            if _normalize(str(claim.get("claim_text_raw", "") or ""))
        )
        for row in rows_a
    }

    case_map_b = {
        str(row.get("uid", "") or row.get("anchor_case_id", "")): sorted(
            _normalize(str(claim.get("claim_text_raw", "") or ""))
            for claim in (row.get("claims_review") or row.get("claims") or [])
            if _normalize(str(claim.get("claim_text_raw", "") or ""))
        )
        for row in rows_b
    }

    uids = sorted(set(case_map_a) | set(case_map_b))

    if not uids:
        return 0.0

    exact = sum(case_map_a.get(uid, []) == case_map_b.get(uid, []) for uid in uids)
    return exact / len(uids)


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("label lengths must match")

    if not labels_a:
        return 0.0

    total = len(labels_a)
    agree = sum(a == b for a, b in zip(labels_a, labels_b)) / total
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    categories = set(counts_a) | set(counts_b)

    chance = sum(
        (counts_a[cat] / total) * (counts_b[cat] / total) for cat in categories
    )

    if chance == 1.0:
        return 1.0

    return (agree - chance) / (1 - chance)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-a", required=True)
    parser.add_argument("--blind-b", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows_a = _load_jsonl(Path(args.blind_a))
    rows_b = _load_jsonl(Path(args.blind_b))
    presence_a = _extract_claim_presence(rows_a)
    presence_b = _extract_claim_presence(rows_b)
    claim_units = sorted(set(presence_a) | set(presence_b))

    claim_labels_a = [
        "PRESENT" if presence_a.get(unit) else "ABSENT" for unit in claim_units
    ]

    claim_labels_b = [
        "PRESENT" if presence_b.get(unit) else "ABSENT" for unit in claim_units
    ]

    status_a = _extract_claim_status(rows_a)
    status_b = _extract_claim_status(rows_b)
    status_units = sorted(set(status_a) | set(status_b))
    status_labels_a = [status_a.get(unit, "ABSENT") for unit in status_units]
    status_labels_b = [status_b.get(unit, "ABSENT") for unit in status_units]

    payload = {
        "claim_extraction_kappa": _cohen_kappa(claim_labels_a, claim_labels_b),
        "claim_extraction_exact_match_rate": _case_exact_match_rate(rows_a, rows_b),
        "status_label_kappa": _cohen_kappa(status_labels_a, status_labels_b),
        "claim_unit_count": len(claim_units),
        "status_unit_count": len(status_units),
    }

    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
