"""Sample blind anchor cases for Step 06 / Step 06A review."""

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
    alloc = {key: min(group_sizes[key], int(value)) for key, value in raw.items()}
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
    """Parse CLI arguments for claim or joint-anchor sampling.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--claims-path", required=True)
    parser.add_argument("--status-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260307)
    return parser.parse_args()


def _load_case_map(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str((case.get("metaInfo") or {}).get("uid", "") or ""): case
        for case in load_cases_from_jsonl(path)
    }


def _group_claim_rows(claim_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
                "has_expert_claim": False,
                "has_expert_status": False,
                "claims": [],
            },
        )

        if row["claim_source"] == "expert_adjudication":
            packet["has_expert_claim"] = True

        packet["claims"].append(
            {
                "claim_text_raw": row["claim_text_raw"],
                "source_span": row["source_span"],
            }
        )

    return grouped


def _merge_status_flags(
    grouped: dict[str, dict[str, Any]], status_rows: list[dict[str, Any]]
) -> None:
    for row in status_rows:
        uid = str(row["uid"])
        packet = grouped.get(uid)

        if packet and row.get("status_source") == "expert_adjudication":
            packet["has_expert_status"] = True


def _select_packets(
    grouped: dict[str, dict[str, Any]], sample_size: int, seed: int, joint: bool
) -> list[dict[str, Any]]:
    strata: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for uid, packet in grouped.items():
        if joint:
            stratum = (
                packet["split"],
                packet["hard_case"],
                packet["stage"],
                int(packet["has_expert_claim"]),
                int(packet["has_expert_status"]),
            )

        else:
            stratum = (
                packet["split"],
                packet["hard_case"],
                packet["stage"],
                packet["claim_source"],
            )

        strata[stratum].append(packet)

    alloc = _allocate({key: len(rows) for key, rows in strata.items()}, sample_size)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []

    for key in sorted(strata):
        rows = list(strata[key])
        rng.shuffle(rows)
        selected.extend(rows[: alloc[key]])

    selected.sort(
        key=lambda row: (row["split"], row["stage"], -int(row["hard_case"]), row["uid"])
    )

    return selected


def _build_blind_rows(
    selected: list[dict[str, Any]],
    case_map: dict[str, dict[str, Any]],
    joint: bool,
) -> list[dict[str, Any]]:
    blind_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(selected, start=1):
        case = case_map[row["uid"]]
        content = case.get("content") or {}

        blind_row = {
            "anchor_case_id": (
                f"JOINT_ANCHOR_{idx:03d}" if joint else f"ANCHOR_{idx:03d}"
            ),
            "uid": row["uid"],
            "split": row["split"],
            "hard_case": row["hard_case"],
            "stage": row["stage"],
            "plaintiff_text": str(content.get("原告诉称", "") or ""),
            "claims_review": [],
            "status_review": [],
        }

        if joint:
            blind_row["judgment_result_text"] = str(content.get("裁判结果", "") or "")
            blind_row["court_opinion_text"] = str(content.get("法院观点", "") or "")

        blind_rows.append(blind_row)

    return blind_rows


def main() -> int:
    """Sample anchor review packets and write blind-review exports.

    Returns:
        Process exit code.
    """
    args = parse_args()
    joint = bool(args.status_path)
    case_map = _load_case_map(Path(args.input))
    claim_rows = _load_jsonl(Path(args.claims_path))
    grouped = _group_claim_rows(claim_rows)

    if joint:
        _merge_status_flags(grouped, _load_jsonl(Path(args.status_path)))

    selected = _select_packets(
        grouped=grouped,
        sample_size=args.sample_size,
        seed=args.seed,
        joint=joint,
    )

    blind_rows = _build_blind_rows(selected=selected, case_map=case_map, joint=joint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if joint:
        blind_a = output_dir / "joint_anchor_set_blind_a.jsonl"
        blind_b = output_dir / "joint_anchor_set_blind_b.jsonl"

    else:
        blind_a = output_dir / "anchor_set_blind_a.jsonl"
        blind_b = output_dir / "anchor_set_blind_b.jsonl"

    _write_jsonl(blind_a, blind_rows)
    _write_jsonl(blind_b, blind_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
