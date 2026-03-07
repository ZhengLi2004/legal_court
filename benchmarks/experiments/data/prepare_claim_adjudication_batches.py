"""Prepare Step 06 expert adjudication batches from uncertain claim cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

BATCH_PREFIX = "batch_"

REASON_PRIORITY = {
    "request_block_missing": 0,
    "number_tree_conflict": 1,
    "syntax_split_ambiguous": 2,
    "meta_appeal_only": 3,
    "no_substantive_claim": 4,
    "procedural_only": 5,
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _primary_reason(row: dict[str, Any]) -> str:
    reasons = [str(reason) for reason in row.get("review_reasons", [])]

    if not reasons:
        return "unknown"

    return min(reasons, key=lambda value: (REASON_PRIORITY.get(value, 999), value))


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage_priority = {"一审": 0, "二审": 1}
    split_priority = {"dev": 0, "test": 1}

    return sorted(
        rows,
        key=lambda row: (
            REASON_PRIORITY.get(_primary_reason(row), 999),
            stage_priority.get(str(row.get("stage", "")), 9),
            split_priority.get(str(row.get("split", "")), 9),
            -int(row.get("hard_case", 0)),
            str(row.get("uid", "")),
        ),
    )


def _chunk(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[idx : idx + batch_size] for idx in range(0, len(rows), batch_size)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_batch_markdown(batch_id: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Step 06 Expert Claim Adjudication {batch_id}",
        "",
        "输出协议：每案返回 `uid`、`decision`、`claims[]`；span 统一相对于 `plaintiff_text` 全文。",
        "允许 decision：`claims_confirmed` / `no_substantive_claim` / `procedural_only` / `meta_appeal_only`。",
        "",
    ]

    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## Case {index}: {row['uid']}",
                f"- split: {row.get('split', '')}",
                f"- hard_case: {row.get('hard_case', '')}",
                f"- stage: {row.get('stage', '')}",
                f"- sample_cause: {row.get('sample_cause', '')}",
                f"- review_reasons: {', '.join(row.get('review_reasons', []))}",
                f"- request_block_source: {row.get('request_block_source', '')}",
                "- auto_candidates:",
                json.dumps(
                    row.get("auto_candidates", []), ensure_ascii=False, indent=2
                ),
                "- request_block_text:",
                "```text",
                str(row.get("request_block_text", "")),
                "```",
                "- plaintiff_text:",
                "```text",
                str(row.get("plaintiff_text", "")),
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncertain-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    uncertain_path = Path(args.uncertain_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _sorted_rows(_load_jsonl(uncertain_path))
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

        manifest_batches.append(
            {
                "batch_id": batch_id,
                "size": len(batch_rows),
                "uids": [str(row.get("uid", "")) for row in batch_rows],
                "primary_reason_counts": dict(sorted(reason_counts.items())),
                "jsonl_path": str(jsonl_path.resolve()),
                "markdown_path": str(md_path.resolve()),
            }
        )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "uncertain_path": str(uncertain_path.resolve()),
        "total_cases": len(rows),
        "batch_size": args.batch_size,
        "batch_count": len(batches),
        "protocol": {
            "reviewer": "expert_agent",
            "span_reference": "plaintiff_text",
            "allowed_decisions": [
                "claims_confirmed",
                "no_substantive_claim",
                "procedural_only",
                "meta_appeal_only",
            ],
        },
        "batches": manifest_batches,
    }

    _write_json(output_dir / "batch_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
