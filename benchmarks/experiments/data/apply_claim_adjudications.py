"""Apply expert claim adjudications and build final Step 06 Gold Claim artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.data.step06_claims import (
    CaseClaimExtractionResult,
    build_claim_from_expert_span,
    build_summary,
    claim_item_from_row,
    extract_request_block,
    load_claim_ontology,
    load_rule_bundle,
    normalize_space,
    serialize_claim_rows,
)

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))

FINAL_CLAIMS_NAME = "gold_claims_final.jsonl"
FINAL_SUMMARY_NAME = "claim_extraction_summary.final.json"
AUDIT_NAME = "expert_claim_adjudication_audit.jsonl"

ALLOWED_DECISIONS = {
    "claims_confirmed",
    "no_substantive_claim",
    "procedural_only",
    "meta_appeal_only",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _case_uid(case: dict[str, Any]) -> str:
    return str((case.get("metaInfo") or {}).get("uid", "") or "")


def _build_rule_result(
    case: dict[str, Any], rows: list[dict[str, Any]], stage: str, rules
) -> CaseClaimExtractionResult:
    extra = case.get("extraInfo") or {}
    plaintiff_text = str((case.get("content") or {}).get("原告诉称", "") or "")
    request_block = extract_request_block(plaintiff_text, stage=stage, rules=rules)
    claims = [
        claim_item_from_row(row)
        for row in sorted(rows, key=lambda item: int(item["claim_index"]))
    ]
    first = rows[0]

    return CaseClaimExtractionResult(
        uid=_case_uid(case),
        split=str(first["split"]),
        hard_case=int(first["hard_case"]),
        sample_cause=str(first["sample_cause"]),
        stage=str(extra.get("stage", "") or ""),
        claims=claims,
        request_block=request_block,
        auto_candidates=[],
        review_reasons=[],
        needs_manual_review=False,
        claim_source="rule",
    )


def _build_expert_result(
    case: dict[str, Any], uncertain_row: dict[str, Any], claims, rules
) -> CaseClaimExtractionResult:
    extra = case.get("extraInfo") or {}
    plaintiff_text = str((case.get("content") or {}).get("原告诉称", "") or "")
    request_block = extract_request_block(
        plaintiff_text, stage=str(extra.get("stage", "") or ""), rules=rules
    )

    return CaseClaimExtractionResult(
        uid=_case_uid(case),
        split=str(uncertain_row["split"]),
        hard_case=int(uncertain_row["hard_case"]),
        sample_cause=str(uncertain_row["sample_cause"]),
        stage=str(extra.get("stage", "") or ""),
        claims=list(claims),
        request_block=request_block,
        auto_candidates=list(uncertain_row.get("auto_candidates", [])),
        review_reasons=[],
        needs_manual_review=False,
        claim_source="expert_adjudication",
    )


def _normalize_review_row(
    row: dict[str, Any],
    case: dict[str, Any],
    uncertain_row: dict[str, Any],
    ontology,
    rules,
) -> tuple[list[Any], dict[str, Any]]:
    uid = _case_uid(case)
    decision = str(row.get("decision", "") or "")

    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"uid={uid} has invalid decision: {decision}")

    claims_payload = row.get("claims", [])

    if decision == "claims_confirmed":
        if not isinstance(claims_payload, list) or not claims_payload:
            raise ValueError(
                f"uid={uid} decision=claims_confirmed requires non-empty claims"
            )

    else:
        if claims_payload not in ([], None):
            raise ValueError(f"uid={uid} decision={decision} must not contain claims")

        claims_payload = []

    claims = []
    plaintiff_text = str((case.get("content") or {}).get("原告诉称", "") or "")

    for claim_row in claims_payload:
        if not isinstance(claim_row, dict):
            raise ValueError(f"uid={uid} contains non-object claim row")

        span = claim_row.get("source_span") or {}

        try:
            start = int(span["start"])
            end = int(span["end"])

        except Exception as exc:
            raise ValueError(f"uid={uid} has invalid source_span") from exc

        raw_text = normalize_space(plaintiff_text[start:end])
        provided_text = normalize_space(str(claim_row.get("claim_text_raw", "") or ""))

        if not raw_text or raw_text != provided_text:
            raise ValueError(f"uid={uid} claim_text_raw does not match source_span")

        claims.append(
            build_claim_from_expert_span(
                case,
                start=start,
                end=end,
                rules=rules,
                ontology=ontology,
                adjudication_reason=str(claim_row.get("adjudication_reason", "") or ""),
            )
        )

    audit_row = {
        "uid": uid,
        "split": str(uncertain_row["split"]),
        "hard_case": int(uncertain_row["hard_case"]),
        "stage": str(uncertain_row["stage"]),
        "decision": decision,
        "accepted_claim_count": len(claims),
        "reviewer": str(row.get("reviewer", "expert_agent") or "expert_agent"),
        "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        "review_reasons": list(uncertain_row.get("review_reasons", [])),
        "claims": [
            {
                "claim_text_raw": str(claim_row.get("claim_text_raw", "") or ""),
                "source_span": {
                    "start": int((claim_row.get("source_span") or {}).get("start", 0)),
                    "end": int((claim_row.get("source_span") or {}).get("end", 0)),
                },
                "adjudication_reason": str(
                    claim_row.get("adjudication_reason", "") or ""
                ),
            }
            for claim_row in claims_payload
        ],
    }

    return claims, audit_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--rule-claims", required=True)
    parser.add_argument("--uncertain-path", required=True)
    parser.add_argument("--reviews-dir", required=True)
    parser.add_argument("--rules-path", required=True)
    parser.add_argument("--ontology-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _collect_reviews(reviews_dir: Path) -> dict[str, dict[str, Any]]:
    review_map: dict[str, dict[str, Any]] = {}

    for path in sorted(reviews_dir.glob("*.jsonl")):
        for row in _load_jsonl(path):
            uid = str(row.get("uid", "") or "")

            if not uid:
                raise ValueError(f"Missing uid in {path}")

            if uid in review_map:
                raise ValueError(f"Duplicate review uid detected: {uid}")

            review_map[uid] = row

    return review_map


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    rule_claims_path = Path(args.rule_claims)
    uncertain_path = Path(args.uncertain_path)
    reviews_dir = Path(args.reviews_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rules = load_rule_bundle(Path(args.rules_path))
    ontology = load_claim_ontology(Path(args.ontology_path))
    cases = load_cases_from_jsonl(input_path)
    rule_claim_rows = _load_jsonl(rule_claims_path)
    uncertain_rows = _load_jsonl(uncertain_path)
    uncertain_map = {str(row["uid"]): row for row in uncertain_rows}
    review_map = _collect_reviews(reviews_dir)
    missing_reviews = sorted(set(uncertain_map) - set(review_map))
    unexpected_reviews = sorted(set(review_map) - set(uncertain_map))

    if missing_reviews:
        raise ValueError(
            f"Missing expert reviews for {len(missing_reviews)} uncertain cases, e.g. {missing_reviews[:5]}"
        )

    if unexpected_reviews:
        raise ValueError(
            f"Unexpected expert reviews for non-uncertain cases: {unexpected_reviews[:5]}"
        )

    grouped_rule_rows: dict[str, list[dict[str, Any]]] = {}

    for row in rule_claim_rows:
        grouped_rule_rows.setdefault(str(row["uid"]), []).append(row)

    results: list[CaseClaimExtractionResult] = []
    audit_rows: list[dict[str, Any]] = []

    for case in cases:
        uid = _case_uid(case)

        if uid in uncertain_map:
            claims, audit_row = _normalize_review_row(
                review_map[uid], case, uncertain_map[uid], ontology, rules
            )
            results.append(
                _build_expert_result(case, uncertain_map[uid], claims, rules)
            )
            audit_rows.append(audit_row)
            continue

        rows = grouped_rule_rows.get(uid, [])

        if not rows:
            raise ValueError(
                f"uid={uid} missing from both rule claims and uncertain reviews"
            )

        stage = str((case.get("extraInfo") or {}).get("stage", "") or "")
        results.append(_build_rule_result(case, rows, stage, rules))

    final_claim_rows = serialize_claim_rows(results)

    summary = build_summary(
        results,
        final_claim_rows,
        uncertain_rows=[],
        expert_audit_rows=audit_rows,
        total_cases=len(cases),
        rules=rules,
    )

    summary.update(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(input_path.resolve()),
            "rule_claims_path": str(rule_claims_path.resolve()),
            "uncertain_path": str(uncertain_path.resolve()),
            "reviews_dir": str(reviews_dir.resolve()),
            "rules_path": str(Path(args.rules_path).resolve()),
            "ontology_path": str(Path(args.ontology_path).resolve()),
            "expert_protocol": "manual_adjudication_applied",
        }
    )

    _write_jsonl(output_dir / FINAL_CLAIMS_NAME, final_claim_rows)
    _write_jsonl(output_dir / AUDIT_NAME, audit_rows)
    _write_json(output_dir / FINAL_SUMMARY_NAME, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
