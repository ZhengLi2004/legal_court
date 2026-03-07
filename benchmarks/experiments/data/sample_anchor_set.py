"""Sample blind anchor cases for Step 06 claim extraction / status review."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
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
        raise ValueError(f"sample_size={sample_size} exceeds available cases={total}")

    raw = {key: sample_size * size / total for key, size in group_sizes.items()}
    alloc = {key: min(size, int(value)) for key, value in raw.items()}
    used = sum(alloc.values())

    remainders = sorted(
        ((raw[key] - alloc[key], key) for key in group_sizes),
        reverse=True,
    )

    idx = 0

    while used < sample_size and idx < len(remainders):
        _, key = remainders[idx]

        if alloc[key] < group_sizes[key]:
            alloc[key] += 1
            used += 1

        idx += 1

        if idx == len(remainders) and used < sample_size:
            idx = 0

    return alloc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--claims-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260307)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    case_map = {
        str((case.get("metaInfo") or {}).get("uid", "") or ""): case
        for case in load_cases_from_jsonl(Path(args.input))
    }

    claim_rows = _load_jsonl(Path(args.claims_path))
    grouped: dict[str, dict[str, Any]] = {}

    for row in claim_rows:
        uid = str(row["uid"])

        packet = grouped.setdefault(
            uid,
            {
                "uid": uid,
                "split": row["split"],
                "hard_case": row["hard_case"],
                "stage": row["stage"],
                "claim_source": row["claim_source"],
                "claims": [],
            },
        )

        packet["claims"].append(
            {
                "claim_text_raw": row["claim_text_raw"],
                "source_span": row["source_span"],
            }
        )

    strata: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for uid, packet in grouped.items():
        stratum = (
            packet["split"],
            packet["hard_case"],
            packet["stage"],
            packet["claim_source"],
        )

        strata[stratum].append(packet)

    alloc = _allocate(
        {key: len(rows) for key, rows in strata.items()}, args.sample_size
    )

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []

    for key in sorted(strata):
        rows = list(strata[key])
        rng.shuffle(rows)
        selected.extend(rows[: alloc[key]])

    selected.sort(
        key=lambda row: (row["split"], row["stage"], -int(row["hard_case"]), row["uid"])
    )

    blind_rows = []

    for idx, row in enumerate(selected, start=1):
        case = case_map[row["uid"]]

        blind_rows.append(
            {
                "anchor_case_id": f"ANCHOR_{idx:03d}",
                "uid": row["uid"],
                "split": row["split"],
                "hard_case": row["hard_case"],
                "stage": row["stage"],
                "plaintiff_text": str(
                    (case.get("content") or {}).get("原告诉称", "") or ""
                ),
                "claims_review": [],
                "status_review": [],
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "anchor_set_blind_a.jsonl", blind_rows)
    _write_jsonl(output_dir / "anchor_set_blind_b.jsonl", blind_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
