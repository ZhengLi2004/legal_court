"""Apply expert status adjudications and build final Step 06A artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.data.status_inference import (
    build_status_summary,
    canonical_to_eval,
)

FINAL_STATUS_NAME = "gold_status_final.jsonl"
FINAL_SUMMARY_NAME = "status_classification_summary.final.json"
AUDIT_NAME = "expert_status_adjudication_audit.jsonl"
ALLOWED_DECISIONS = {"ACCEPTED", "REJECTED", "UNMENTIONED"}


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


def _case_uid(case: dict[str, Any]) -> str:
    return str((case.get("metaInfo") or {}).get("uid", "") or "")


def _collect_reviews(reviews_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    review_map: dict[tuple[str, str], dict[str, Any]] = {}

    for path in sorted(reviews_dir.glob("*.jsonl")):
        for row in _load_jsonl(path):
            uid = str(row.get("uid", "") or "")
            claim_id = str(row.get("claim_id", "") or "")

            if not uid or not claim_id:
                raise ValueError(f"Missing uid/claim_id in {path}")

            key = (uid, claim_id)

            if key in review_map:
                raise ValueError(f"Duplicate review row detected: {uid}/{claim_id}")

            review_map[key] = row

    return review_map


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for applying manual status adjudications.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--review-queue-path", required=False)
    parser.add_argument("--uncertain-path", required=False)
    parser.add_argument("--reviews-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    """Merge expert status adjudications into the finalized gold status file.

    Returns:
        Process exit code.

    Raises:
        ValueError: If required review queues are missing or inconsistent.
    """
    args = parse_args()
    input_path = Path(args.input)
    status_path = Path(args.status_path)
    review_queue_path = Path(args.review_queue_path) if args.review_queue_path else None
    uncertain_path = Path(args.uncertain_path) if args.uncertain_path else None
    reviews_dir = Path(args.reviews_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if review_queue_path is None:
        if uncertain_path is None:
            raise ValueError("review-queue-path or uncertain-path is required")

        review_queue_path = uncertain_path

    cases = load_cases_from_jsonl(input_path)
    case_map = {_case_uid(case): case for case in cases}
    auto_rows = _load_jsonl(status_path)
    review_queue_rows = _load_jsonl(review_queue_path)
    review_map = _collect_reviews(reviews_dir)

    review_queue_map = {
        (str(row["uid"]), str(row["claim_id"])): row for row in review_queue_rows
    }

    missing = sorted(set(review_queue_map) - set(review_map))
    extra = sorted(set(review_map) - set(review_queue_map))

    if missing:
        raise ValueError(f"Missing expert status reviews, e.g. {missing[:5]}")

    if extra:
        raise ValueError(f"Unexpected status reviews, e.g. {extra[:5]}")

    audit_rows: list[dict[str, Any]] = []
    auto_map = {(str(row["uid"]), str(row["claim_id"])): row for row in auto_rows}
    final_map = dict(auto_map)

    for key, queue_row in review_queue_map.items():
        review = review_map[key]
        decision = str(review.get("decision", "") or "")

        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"Invalid decision for {key}: {decision}")

        evidence_spans = review.get("evidence_spans", [])

        if not isinstance(evidence_spans, list):
            raise ValueError(f"Invalid evidence_spans for {key}")

        case = case_map.get(key[0])

        if case is None:
            raise ValueError(f"Case missing for review key={key}")

        content = case.get("content") or {}
        evidence_texts: list[str] = []
        normalized_spans: list[dict[str, Any]] = []

        for span in evidence_spans:
            section = str(span.get("section", "") or "")
            start = int(span.get("start", -1))
            end = int(span.get("end", -1))
            section_text = str(content.get(section, "") or "")

            if section not in {"裁判结果", "法院观点"}:
                raise ValueError(f"Invalid evidence section for {key}: {section}")

            if start < 0 or end <= start or end > len(section_text):
                raise ValueError(f"Invalid evidence span for {key}: {span}")

            evidence_texts.append(section_text[start:end])
            normalized_spans.append({"section": section, "start": start, "end": end})

        base_row = dict(queue_row)
        base_row.pop("candidate_evidence", None)
        base_row.pop("needs_manual_review", None)
        base_row.pop("review_reasons", None)
        base_row.pop("review_origin", None)
        base_row.pop("auto_status_candidate", None)
        base_row.pop("auto_confidence", None)
        base_row.pop("auto_decision_mode", None)

        final_map[key] = {
            **base_row,
            "status_canonical": decision,
            "status_eval": canonical_to_eval(decision),
            "status_source": "expert_adjudication",
            "decision_mode": "expert_adjudication",
            "evidence_section": normalized_spans[0]["section"]
            if normalized_spans
            else "",
            "evidence_spans": normalized_spans,
            "evidence_text": " ".join(
                text.strip() for text in evidence_texts if text.strip()
            ),
            "confidence": None,
            "review_reason": str(review.get("adjudication_reason", "") or ""),
        }

        audit_rows.append(
            {
                "uid": key[0],
                "claim_id": key[1],
                "decision": decision,
                "reviewer": str(
                    review.get("reviewer", "expert_agent") or "expert_agent"
                ),
                "reviewed_at": datetime.now().isoformat(timespec="seconds"),
                "review_reasons": list(queue_row.get("review_reasons", [])),
                "review_origin": str(queue_row.get("review_origin", "uncertain")),
                "replaced_status_source": str(
                    auto_map.get(key, {}).get("status_source", "")
                ),
                "evidence_spans": normalized_spans,
                "adjudication_reason": str(review.get("adjudication_reason", "") or ""),
            }
        )

    final_rows = sorted(
        final_map.values(), key=lambda row: (str(row["uid"]), str(row["claim_id"]))
    )

    summary = build_status_summary(
        auto_rows=final_rows,
        uncertain_rows=[],
        total_claims=len(final_rows),
        mode="full_final",
    )

    summary.update(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(input_path.resolve()),
            "status_path": str(status_path.resolve()),
            "review_queue_path": str(review_queue_path.resolve()),
            "reviews_dir": str(reviews_dir.resolve()),
            "expert_protocol": "manual_status_adjudication_applied",
            "expert_adjudicated_count": len(audit_rows),
            "status_source_counts": dict(
                sorted(summary.get("status_source_counts", {}).items())
            ),
        }
    )

    _write_jsonl(output_dir / FINAL_STATUS_NAME, final_rows)
    _write_jsonl(output_dir / AUDIT_NAME, audit_rows)
    _write_json(output_dir / FINAL_SUMMARY_NAME, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
