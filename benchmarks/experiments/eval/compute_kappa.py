"""Compute agreement metrics for blind Step 06 / Step 06A review packages."""

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


def _row_uid(row: dict[str, Any]) -> str:
    return str(row.get("uid", "") or row.get("anchor_case_id", ""))


def _claim_entries(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("claims_review") or row.get("claims") or [])


def _extract_case_claims(
    rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}

    for row in rows:
        uid = _row_uid(row)

        claims = sorted(
            _normalize(str(claim.get("claim_text_raw", "") or ""))
            for claim in _claim_entries(row)
            if _normalize(str(claim.get("claim_text_raw", "") or ""))
        )

        mapping[uid] = claims

    return mapping


def _extract_claim_presence(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    mapping: set[tuple[str, str]] = set()

    for row in rows:
        uid = _row_uid(row)
        claims = _claim_entries(row)

        for claim in claims:
            claim_text = _normalize(str(claim.get("claim_text_raw", "") or ""))

            if claim_text:
                mapping.add((uid, claim_text))

    return mapping


def _extract_claim_status(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}

    for row in rows:
        uid = _row_uid(row)
        claims = _claim_entries(row)

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
    case_map_a = _extract_case_claims(rows_a)
    case_map_b = _extract_case_claims(rows_b)
    uids = sorted(set(case_map_a) | set(case_map_b))

    if not uids:
        return 0.0

    exact = sum(case_map_a.get(uid, []) == case_map_b.get(uid, []) for uid in uids)
    return exact / len(uids)


def _case_claim_set_kappa(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]
) -> float:
    case_map_a = _extract_case_claims(rows_a)
    case_map_b = _extract_case_claims(rows_b)
    uids = sorted(set(case_map_a) | set(case_map_b))
    labels_a = ["||".join(case_map_a.get(uid, [])) for uid in uids]
    labels_b = ["||".join(case_map_b.get(uid, [])) for uid in uids]
    return _cohen_kappa(labels_a, labels_b)


def _claim_prf(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]
) -> tuple[float, float, float, int, int, int, int]:
    claims_a = _extract_claim_presence(rows_a)
    claims_b = _extract_claim_presence(rows_b)
    matched = claims_a & claims_b
    union = claims_a | claims_b
    precision = len(matched) / len(claims_a) if claims_a else 0.0
    recall = len(matched) / len(claims_b) if claims_b else 0.0

    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    return (
        precision,
        recall,
        f1,
        len(claims_a),
        len(claims_b),
        len(matched),
        len(union),
    )


def _status_metrics(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]
) -> tuple[float, float, int]:
    status_a = _extract_claim_status(rows_a)
    status_b = _extract_claim_status(rows_b)
    status_units = sorted(set(status_a) & set(status_b))

    if not status_units:
        return 0.0, 0.0, 0

    status_labels_a = [status_a[unit] for unit in status_units]
    status_labels_b = [status_b[unit] for unit in status_units]

    agreement = sum(a == b for a, b in zip(status_labels_a, status_labels_b)) / len(
        status_units
    )

    mismatch_count = sum(a != b for a, b in zip(status_labels_a, status_labels_b))
    return _cohen_kappa(status_labels_a, status_labels_b), agreement, mismatch_count


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

    (
        claim_precision,
        claim_recall,
        claim_f1,
        claim_unit_count_a,
        claim_unit_count_b,
        matched_claim_unit_count,
        claim_unit_count,
    ) = _claim_prf(rows_a, rows_b)

    status_label_kappa, status_label_agreement_rate, status_mismatch_count = (
        _status_metrics(rows_a, rows_b)
    )

    payload = {
        "claim_extraction_kappa": _case_claim_set_kappa(rows_a, rows_b),
        "claim_extraction_exact_match_rate": _case_exact_match_rate(rows_a, rows_b),
        "claim_extraction_precision": claim_precision,
        "claim_extraction_recall": claim_recall,
        "claim_extraction_f1": claim_f1,
        "status_label_kappa": status_label_kappa,
        "status_label_agreement_rate": status_label_agreement_rate,
        "claim_unit_count": claim_unit_count,
        "claim_unit_count_a": claim_unit_count_a,
        "claim_unit_count_b": claim_unit_count_b,
        "matched_claim_unit_count": matched_claim_unit_count,
        "status_unit_count": matched_claim_unit_count,
        "status_mismatch_count": status_mismatch_count,
    }

    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
