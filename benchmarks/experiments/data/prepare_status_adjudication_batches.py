"""Prepare Step 06A expert adjudication batches from uncertain and zeroshot status rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

BATCH_PREFIX = "batch_"

REASON_PRIORITY = {
    "manual_audit_zeroshot": 0,
    "no_evidence_candidate": 1,
    "low_confidence": 2,
    "zeroshot_empty_result": 3,
    "zeroshot_unknown_label": 4,
}

QUEUE_NAME = "status_review_queue.jsonl"


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


def _primary_reason(row: dict[str, Any]) -> str:
    reasons = [str(item) for item in row.get("review_reasons", [])]

    if not reasons:
        return "unknown"

    return min(reasons, key=lambda item: (REASON_PRIORITY.get(item, 999), item))


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_priority = {"dev": 0, "test": 1}
    stage_priority = {"一审": 0, "二审": 1}

    return sorted(
        rows,
        key=lambda row: (
            REASON_PRIORITY.get(_primary_reason(row), 999),
            split_priority.get(str(row.get("split", "")), 9),
            stage_priority.get(str(row.get("stage", "")), 9),
            -int(row.get("hard_case", 0)),
            str(row.get("uid", "")),
            str(row.get("claim_id", "")),
        ),
    )


def _chunk(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[idx : idx + batch_size] for idx in range(0, len(rows), batch_size)]


def _render_batch_markdown(batch_id: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Step 06A Expert Status Adjudication {batch_id}",
        "",
        "输出协议：每条 claim 返回 `uid`、`claim_id`、`decision`、`evidence_spans`。",
        "允许 decision：`ACCEPTED` / `REJECTED` / `UNMENTIONED`。",
        "",
    ]

    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## Claim {index}: {row['uid']} / {row['claim_id']}",
                f"- split: {row.get('split', '')}",
                f"- hard_case: {row.get('hard_case', '')}",
                f"- stage: {row.get('stage', '')}",
                f"- stage_role: {row.get('stage_role', '')}",
                f"- claim_text_raw: {row.get('claim_text_raw', '')}",
                f"- review_reasons: {', '.join(row.get('review_reasons', []))}",
            ]
        )

        if row.get("review_origin") == "zeroshot":
            lines.extend(
                [
                    f"- auto_status_candidate: {row.get('auto_status_candidate', '')}",
                    f"- auto_confidence: {row.get('auto_confidence', '')}",
                    f"- auto_decision_mode: {row.get('auto_decision_mode', '')}",
                ]
            )

        lines.extend(
            [
                "- candidate_evidence:",
                json.dumps(
                    row.get("candidate_evidence", []), ensure_ascii=False, indent=2
                ),
                "",
            ]
        )

    return "\n".join(lines)


def _normalize_uncertain_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []

    for row in rows:
        normalized.append(
            {
                **row,
                "review_origin": "uncertain",
            }
        )

    return normalized


def _collect_zeroshot_rows(status_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []

    for row in status_rows:
        if str(row.get("status_source", "")) != "zeroshot":
            continue

        rows.append(
            {
                "uid": str(row["uid"]),
                "claim_id": str(row["claim_id"]),
                "split": str(row.get("split", "")),
                "hard_case": int(row.get("hard_case", 0)),
                "stage": str(row.get("stage", "")),
                "sample_cause": str(row.get("sample_cause", "")),
                "claim_text_raw": str(row.get("claim_text_raw", "")),
                "claim_text_norm": str(row.get("claim_text_norm", "")),
                "action": str(row.get("action", "")),
                "target": str(row.get("target", "")),
                "amount": str(row.get("amount", "")),
                "liability": str(row.get("liability", "")),
                "stage_role": str(row.get("stage_role", "")),
                "claim_source": str(row.get("claim_source", "")),
                "review_reasons": ["manual_audit_zeroshot"],
                "review_origin": "zeroshot",
                "auto_status_candidate": str(row.get("status_canonical", "")),
                "auto_confidence": row.get("confidence"),
                "auto_decision_mode": str(row.get("decision_mode", "")),
                "candidate_evidence": [
                    {
                        "section": str(row.get("evidence_section", "")),
                        "start": int(
                            (row.get("evidence_spans") or [{}])[0].get("start", 0)
                        )
                        if row.get("evidence_spans")
                        else 0,
                        "end": int((row.get("evidence_spans") or [{}])[0].get("end", 0))
                        if row.get("evidence_spans")
                        else 0,
                        "text": str(row.get("evidence_text", "")),
                        "bm25_score": None,
                        "dense_score": None,
                        "combined_score": None,
                        "rerank_score": None,
                    }
                ]
                if row.get("evidence_text")
                else [],
                "needs_manual_review": True,
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for status adjudication batch generation.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncertain-path", required=True)
    parser.add_argument("--status-path", required=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    """Build batched expert review queues for uncertain status judgments.

    Returns:
        Process exit code.

    Raises:
        ValueError: If batch size is invalid or review rows are duplicated.
    """
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    uncertain_path = Path(args.uncertain_path)
    status_path = Path(args.status_path) if args.status_path else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _normalize_uncertain_rows(_load_jsonl(uncertain_path))

    if status_path is not None:
        rows.extend(_collect_zeroshot_rows(_load_jsonl(status_path)))

    keys = [(str(row.get("uid", "")), str(row.get("claim_id", ""))) for row in rows]

    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate review queue rows detected")

    rows = _sorted_rows(rows)
    queue_path = output_dir / QUEUE_NAME
    _write_jsonl(queue_path, rows)
    batches = _chunk(rows, args.batch_size)
    manifest_batches: list[dict[str, Any]] = []

    for idx, batch_rows in enumerate(batches, start=1):
        batch_id = f"{BATCH_PREFIX}{idx:03d}"
        jsonl_path = output_dir / f"{batch_id}.jsonl"
        md_path = output_dir / f"{batch_id}.md"
        _write_jsonl(jsonl_path, batch_rows)

        md_path.write_text(
            _render_batch_markdown(batch_id, batch_rows), encoding="utf-8"
        )

        reason_counts = Counter(_primary_reason(row) for row in batch_rows)

        origin_counts = Counter(
            str(row.get("review_origin", "unknown")) for row in batch_rows
        )

        manifest_batches.append(
            {
                "batch_id": batch_id,
                "size": len(batch_rows),
                "claim_keys": [
                    f"{row.get('uid', '')}:{row.get('claim_id', '')}"
                    for row in batch_rows
                ],
                "primary_reason_counts": dict(sorted(reason_counts.items())),
                "review_origin_counts": dict(sorted(origin_counts.items())),
                "jsonl_path": str(jsonl_path.resolve()),
                "markdown_path": str(md_path.resolve()),
            }
        )

    _write_json(
        output_dir / "batch_manifest.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "uncertain_path": str(uncertain_path.resolve()),
            "status_path": "" if status_path is None else str(status_path.resolve()),
            "review_queue_path": str(queue_path.resolve()),
            "total_claims": len(rows),
            "batch_size": args.batch_size,
            "batch_count": len(batches),
            "protocol": {
                "reviewer": "expert_agent",
                "allowed_decisions": ["ACCEPTED", "REJECTED", "UNMENTIONED"],
                "zeroshot_policy": "manual_override_required",
            },
            "batches": manifest_batches,
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
