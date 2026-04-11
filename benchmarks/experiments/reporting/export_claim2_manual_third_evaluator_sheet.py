"""Export a manual worksheet for Claim 2 third-evaluator annotation.

This script prepares one deduplicated assertion-window worksheet from an
existing Claim 2 dev-only bridge run. It is designed for the case where the
first two evaluators have already been scored and a third evaluator (for
example, manual GPT-5.4 review) must be added offline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "reports/experiments/20260405_claim2_from_claim3_normal/claim2_from_claim3/normal"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT),
        help="Claim 2 bridge run root, e.g. .../claim2_from_claim3/normal",
    )

    parser.add_argument(
        "--output-csv",
        default="",
        help="Output worksheet CSV path. Defaults to <run-root>/manual_third_evaluator_sheet.csv",
    )

    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_alignment_items(run_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}

    for point in ("warmup_0", "warmup_100"):
        diag_root = run_root / point / "dev" / "case_diagnostics" / "main_system"

        for path in sorted(diag_root.glob("*.json")):
            case_diag = _read_json(path)

            for alignment in case_diag.get("alignment_results", []):
                assertion = dict(alignment.get("assertion", {}) or {})

                for hit in alignment.get("aligned_hits", []):
                    window = dict(dict(hit).get("window", {}) or {})

                    key = (
                        str(assertion.get("assertion_id", "") or ""),
                        str(window.get("window_id", "") or ""),
                    )

                    if not key[0] or not key[1]:
                        continue

                    row = rows.setdefault(
                        key,
                        {
                            "assertion_id": key[0],
                            "case_uid": str(assertion.get("case_uid", "") or ""),
                            "claim_id": str(assertion.get("claim_id", "") or ""),
                            "status_eval": str(assertion.get("status_eval", "") or ""),
                            "assertion_text": str(
                                assertion.get("assertion_text", "") or ""
                            ),
                            "window_id": key[1],
                            "section": str(window.get("section", "") or ""),
                            "window_text": str(window.get("text", "") or ""),
                            "used_in_warmup_0": False,
                            "used_in_warmup_100": False,
                            "qwen35_35b_a3b_primary": "",
                            "mdeberta_xnli_local_primary": "",
                        },
                    )

                    row[f"used_in_{point}"] = True

    return rows


def _inject_cached_votes(
    *,
    run_root: Path,
    rows: dict[tuple[str, str], dict[str, Any]],
) -> None:
    cache_root = run_root / "cache" / "votes" / "primary"

    for path in sorted(cache_root.glob("*.json")):
        payload = _read_json(path)

        key = (
            str(payload.get("assertion_id", "") or ""),
            str(payload.get("window_id", "") or ""),
        )

        if key not in rows:
            continue

        evaluator_name = str(payload.get("evaluator_name", "") or "")

        if evaluator_name in {
            "qwen35_35b_a3b_primary",
            "mdeberta_xnli_local_primary",
        }:
            rows[key][evaluator_name] = str(payload.get("label", "") or "")


def _build_output_rows(
    rows: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for row in rows.values():
        qwen = row["qwen35_35b_a3b_primary"]
        mdeberta = row["mdeberta_xnli_local_primary"]
        disagreement = bool(qwen and mdeberta and qwen != mdeberta)

        output.append(
            {
                "case_uid": row["case_uid"],
                "claim_id": row["claim_id"],
                "assertion_id": row["assertion_id"],
                "status_eval": row["status_eval"],
                "window_id": row["window_id"],
                "section": row["section"],
                "used_in_warmup_0": row["used_in_warmup_0"],
                "used_in_warmup_100": row["used_in_warmup_100"],
                "qwen35_35b_a3b_primary": qwen,
                "mdeberta_xnli_local_primary": mdeberta,
                "qwen_mdeberta_disagree": disagreement,
                "gpt54_manual_label": "",
                "assertion_text": row["assertion_text"],
                "window_text": row["window_text"],
            }
        )

    output.sort(
        key=lambda item: (
            not bool(item["qwen_mdeberta_disagree"]),
            item["case_uid"],
            item["assertion_id"],
            item["window_id"],
        )
    )

    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_uid",
        "claim_id",
        "assertion_id",
        "status_eval",
        "window_id",
        "section",
        "used_in_warmup_0",
        "used_in_warmup_100",
        "qwen35_35b_a3b_primary",
        "mdeberta_xnli_local_primary",
        "qwen_mdeberta_disagree",
        "gpt54_manual_label",
        "assertion_text",
        "window_text",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()

    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else run_root / "manual_third_evaluator_sheet.csv"
    )

    rows = _collect_alignment_items(run_root)
    _inject_cached_votes(run_root=run_root, rows=rows)
    output_rows = _build_output_rows(rows)
    _write_csv(output_csv, output_rows)
    disagree_count = sum(1 for row in output_rows if row["qwen_mdeberta_disagree"])

    payload = {
        "run_root": str(run_root),
        "output_csv": str(output_csv),
        "unique_assertion_window_items": len(output_rows),
        "disagreement_items": disagree_count,
        "agreement_items": len(output_rows) - disagree_count,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
