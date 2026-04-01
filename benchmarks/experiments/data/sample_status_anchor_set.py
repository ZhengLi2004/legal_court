"""Sample blind status anchor rows from final Gold Status artifacts."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from benchmarks.experiments.data.loader import load_cases_from_jsonl

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))


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


def _allocate(
    group_sizes: dict[tuple[Any, ...], int], sample_size: int
) -> dict[tuple[Any, ...], int]:
    total = sum(group_sizes.values())

    if sample_size > total:
        raise ValueError(f"sample_size={sample_size} exceeds available rows={total}")

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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for status-anchor sampling.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260307)
    return parser.parse_args()


def main() -> int:
    """Sample status anchors and export paired blind-review files.

    Returns:
        Process exit code.
    """
    args = parse_args()
    status_rows = _load_jsonl(Path(args.status_path))
    case_map: dict[str, dict[str, Any]] = {}

    if args.input:
        case_map = {
            str((case.get("metaInfo") or {}).get("uid", "") or ""): case
            for case in load_cases_from_jsonl(Path(args.input))
        }

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    for row in status_rows:
        key = (
            str(row.get("split", "")),
            int(row.get("hard_case", 0)),
            str(row.get("stage", "")),
            str(row.get("status_source", "")),
            str(row.get("status_canonical", "")),
        )

        grouped.setdefault(key, []).append(row)

    alloc = _allocate(
        {key: len(rows) for key, rows in grouped.items()}, args.sample_size
    )

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []

    for key in sorted(grouped):
        rows = list(grouped[key])
        rng.shuffle(rows)
        selected.extend(rows[: alloc[key]])

    selected.sort(
        key=lambda row: (
            str(row.get("split", "")),
            str(row.get("stage", "")),
            -int(row.get("hard_case", 0)),
            str(row.get("uid", "")),
            str(row.get("claim_id", "")),
        )
    )

    blind_rows = []

    for idx, row in enumerate(selected, start=1):
        case = case_map.get(str(row.get("uid", "")), {})
        content = case.get("content") or {}

        blind_rows.append(
            {
                "anchor_case_id": f"STATUS_ANCHOR_{idx:03d}",
                "uid": row["uid"],
                "split": row["split"],
                "hard_case": row["hard_case"],
                "stage": row["stage"],
                "plaintiff_text": str(content.get("原告诉称", "") or ""),
                "judgment_result_text": str(content.get("裁判结果", "") or ""),
                "court_opinion_text": str(content.get("法院观点", "") or ""),
                "claims_review": [
                    {
                        "claim_text_raw": row["claim_text_raw"],
                        "status": "",
                    }
                ],
                "status_review": [],
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "status_anchor_set_blind_a.jsonl", blind_rows)
    _write_jsonl(output_dir / "status_anchor_set_blind_b.jsonl", blind_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
