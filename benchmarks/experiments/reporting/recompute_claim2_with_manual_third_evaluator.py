"""Recompute Claim 2 dev summaries from a manually annotated third evaluator.

Input:
- one Claim 2 bridge run root
- one worksheet CSV exported by `export_claim2_manual_third_evaluator_sheet.py`

Output:
- per-warmup JSON summaries for:
  - GPT-5.4 single-evaluator estimates
  - final three-evaluator aggregates
- one comparison CSV across warmup points
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.experiments.eval.faithfulness_pipeline import (
    AlignmentHit,
    AlignmentResult,
    ClaimAssertion,
    EvidenceWindow,
    score_faithfulness,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "reports/experiments/20260405_claim2_from_claim3_normal/claim2_from_claim3/normal"
)


@dataclass(frozen=True)
class MappingLabeler:
    name: str
    mapping: dict[tuple[str, str], str]

    def label(self, assertion: ClaimAssertion, window: EvidenceWindow) -> str:
        key = (assertion.assertion_id, window.window_id)

        if key not in self.mapping:
            raise KeyError(f"Missing manual label for {key}")

        return self.mapping[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT),
        help="Claim 2 bridge run root, e.g. .../claim2_from_claim3/normal",
    )

    parser.add_argument(
        "--worksheet-csv",
        default="",
        help="Manual worksheet CSV. Defaults to <run-root>/manual_third_evaluator_sheet.csv",
    )

    parser.add_argument(
        "--fill-agreed-from-consensus",
        action="store_true",
        help="If gpt54_manual_label is blank and qwen/mdeberta agree, fill the agreed label automatically.",
    )

    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to <run-root>/manual_third_evaluator_results",
    )

    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_worksheet(
    path: Path,
    *,
    fill_agreed_from_consensus: bool,
) -> tuple[dict[tuple[str, str], dict[str, str]], list[dict[str, Any]]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    missing: list[dict[str, Any]] = []

    for row in rows:
        key = (str(row["assertion_id"]), str(row["window_id"]))
        qwen = str(row["qwen35_35b_a3b_primary"]).strip()
        mdeberta = str(row["mdeberta_xnli_local_primary"]).strip()
        gpt54 = str(row["gpt54_manual_label"]).strip().upper()

        if not gpt54 and fill_agreed_from_consensus and qwen and qwen == mdeberta:
            gpt54 = qwen

        mapping[key] = {
            "qwen35_35b_a3b_primary": qwen,
            "mdeberta_xnli_local_primary": mdeberta,
            "gpt54_primary": gpt54,
        }

        if not gpt54:
            missing.append(
                {
                    "assertion_id": row["assertion_id"],
                    "window_id": row["window_id"],
                    "section": row["section"],
                    "qwen35_35b_a3b_primary": qwen,
                    "mdeberta_xnli_local_primary": mdeberta,
                    "qwen_mdeberta_disagree": row["qwen_mdeberta_disagree"],
                }
            )

    return mapping, missing


def _load_alignment_results(case_diag_root: Path) -> list[AlignmentResult]:
    results: list[AlignmentResult] = []

    for path in sorted(case_diag_root.glob("*.json")):
        payload = _read_json(path)

        for item in payload.get("alignment_results", []):
            assertion_payload = dict(item.get("assertion", {}) or {})

            assertion = ClaimAssertion(
                assertion_id=str(assertion_payload["assertion_id"]),
                case_uid=str(assertion_payload["case_uid"]),
                claim_id=str(assertion_payload["claim_id"]),
                claim_text=str(assertion_payload["claim_text"]),
                status_eval=str(assertion_payload["status_eval"]),
                assertion_text=str(assertion_payload["assertion_text"]),
            )

            aligned_hits = []
            candidate_hits = []

            for hit in item.get("aligned_hits", []):
                window_payload = dict(hit.get("window", {}) or {})

                window = EvidenceWindow(
                    window_id=str(window_payload["window_id"]),
                    section=str(window_payload["section"]),
                    text=str(window_payload["text"]),
                    start=int(window_payload["start"]),
                    end=int(window_payload["end"]),
                )

                aligned_hits.append(
                    AlignmentHit(
                        window=window, score=float(hit.get("score", 0.0) or 0.0)
                    )
                )

            for hit in item.get("candidate_hits", []):
                window_payload = dict(hit.get("window", {}) or {})

                window = EvidenceWindow(
                    window_id=str(window_payload["window_id"]),
                    section=str(window_payload["section"]),
                    text=str(window_payload["text"]),
                    start=int(window_payload["start"]),
                    end=int(window_payload["end"]),
                )

                candidate_hits.append(
                    AlignmentHit(
                        window=window, score=float(hit.get("score", 0.0) or 0.0)
                    )
                )

            results.append(
                AlignmentResult(
                    assertion=assertion,
                    aligned_hits=tuple(aligned_hits),
                    success=bool(item.get("success", False)),
                    candidate_hits=tuple(candidate_hits),
                )
            )

    return results


def _metric_row(summary: Any) -> dict[str, float]:
    return {
        "overall_faithfulness": float(summary.overall_faithfulness),
        "alignment_success_rate": float(summary.alignment_success_rate),
        "entailment_rate_on_aligned": float(summary.entailment_rate_on_aligned),
        "uncertain_rate": float(summary.uncertain_rate),
        "agreement_overall_majority": float(
            summary.agreement_matrix.overall_majority_agreement
        ),
        "agreement_uncertain_rate": float(summary.agreement_matrix.uncertain_rate),
        "agreement_item_count": int(summary.agreement_matrix.item_count),
    }


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()

    worksheet_csv = (
        Path(args.worksheet_csv).resolve()
        if args.worksheet_csv
        else run_root / "manual_third_evaluator_sheet.csv"
    )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_root / "manual_third_evaluator_results"
    )

    label_mapping, missing = _load_worksheet(
        worksheet_csv,
        fill_agreed_from_consensus=bool(args.fill_agreed_from_consensus),
    )

    _write_json(output_dir / "missing_labels.json", missing)

    if missing:
        raise ValueError(
            f"Worksheet still has {len(missing)} missing gpt54_manual_label entries. "
            f"See {output_dir / 'missing_labels.json'}."
        )

    qwen_map = {
        key: values["qwen35_35b_a3b_primary"] for key, values in label_mapping.items()
    }

    mdeberta_map = {
        key: values["mdeberta_xnli_local_primary"]
        for key, values in label_mapping.items()
    }

    gpt54_map = {key: values["gpt54_primary"] for key, values in label_mapping.items()}
    comparison_rows: list[dict[str, Any]] = []

    for point in ("warmup_0", "warmup_100"):
        alignment_results = _load_alignment_results(
            run_root / point / "dev" / "case_diagnostics" / "main_system"
        )

        qwen_labeler = MappingLabeler("qwen35_35b_a3b_primary", qwen_map)
        mdeberta_labeler = MappingLabeler("mdeberta_xnli_local_primary", mdeberta_map)
        gpt54_labeler = MappingLabeler("gpt54_primary", gpt54_map)

        gpt54_single = score_faithfulness(
            alignment_results, [gpt54_labeler], aggregation="any"
        )

        final_three = score_faithfulness(
            alignment_results,
            [qwen_labeler, mdeberta_labeler, gpt54_labeler],
            aggregation="any",
        )

        point_payload = {
            "warmup_point": point,
            "gpt54_single": {
                "summary": _metric_row(gpt54_single),
                "raw": gpt54_single.to_dict(),
            },
            "final_three_evaluators": {
                "summary": _metric_row(final_three),
                "raw": final_three.to_dict(),
            },
        }

        _write_json(output_dir / f"{point}.json", point_payload)

        comparison_rows.append(
            {
                "warmup_point": point.replace("warmup_", ""),
                "gpt54_single_overall_faithfulness": gpt54_single.overall_faithfulness,
                "gpt54_single_alignment_success_rate": gpt54_single.alignment_success_rate,
                "gpt54_single_entailment_rate_on_aligned": gpt54_single.entailment_rate_on_aligned,
                "gpt54_single_uncertain_rate": gpt54_single.uncertain_rate,
                "final_three_overall_faithfulness": final_three.overall_faithfulness,
                "final_three_alignment_success_rate": final_three.alignment_success_rate,
                "final_three_entailment_rate_on_aligned": final_three.entailment_rate_on_aligned,
                "final_three_uncertain_rate": final_three.uncertain_rate,
                "final_three_agreement_overall_majority": final_three.agreement_matrix.overall_majority_agreement,
                "final_three_agreement_uncertain_rate": final_three.agreement_matrix.uncertain_rate,
                "final_three_agreement_item_count": final_three.agreement_matrix.item_count,
            }
        )

    _write_csv(output_dir / "comparison.csv", comparison_rows)

    print(
        json.dumps(
            {"output_dir": str(output_dir), "comparison_rows": comparison_rows},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
