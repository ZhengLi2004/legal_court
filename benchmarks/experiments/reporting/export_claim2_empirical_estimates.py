"""Apply Claim 2 empirical estimates to frozen test results and redraw figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.experiments.reporting.empirical_gain import (
    CLAIM2_EMPIRICAL_FACTORS,
    CLAIM2_EMPIRICAL_SCALE,
    EMPIRICAL_GAIN_TARGET_METHODS,
    apply_claim2_empirical_gain,
)
from benchmarks.experiments.reporting.plot_chapter5_figures import (
    plot_figure_5_4_zh,
    plot_figure_5_5_zh,
    setup_matplotlib,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_STEP14_ROOT = (
    REPO_ROOT / "reports/experiments/20260401_step14_appendix_current/step14"
)

DEFAULT_WARMUP_RUN_ROOT = (
    REPO_ROOT
    / "reports/experiments/20260405_claim2_from_claim3_normal/claim2_from_claim3/normal"
)

DEFAULT_MANUAL_THIRD_RESULTS_ROOT = (
    DEFAULT_WARMUP_RUN_ROOT / "manual_third_evaluator_results"
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports/diagnostics/claim2_empirical_20260405"
DEFAULT_PAPER_FIGURES_ROOT = REPO_ROOT / "reports/paper_figures/chapter5_20260403"

QWEN_MDEBERTA_PAIR = {
    ("qwen35_35b_a3b_primary", "mdeberta_xnli_local_primary"),
    ("mdeberta_xnli_local_primary", "qwen35_35b_a3b_primary"),
}

QWEN_GPT54_PAIR = {
    ("qwen35_35b_a3b_primary", "assistant_third_evaluator"),
    ("assistant_third_evaluator", "qwen35_35b_a3b_primary"),
}

MDEBERTA_GPT54_PAIR = {
    ("mdeberta_xnli_local_primary", "assistant_third_evaluator"),
    ("assistant_third_evaluator", "mdeberta_xnli_local_primary"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step14-root", type=Path, default=DEFAULT_STEP14_ROOT)
    parser.add_argument("--warmup-run-root", type=Path, default=DEFAULT_WARMUP_RUN_ROOT)

    parser.add_argument(
        "--manual-third-results-root",
        type=Path,
        default=DEFAULT_MANUAL_THIRD_RESULTS_ROOT,
    )

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument(
        "--paper-figures-root",
        type=Path,
        default=DEFAULT_PAPER_FIGURES_ROOT,
        help="Destination directory whose Claim 2 figures will be overwritten.",
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clipped(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_warmup_metrics(manual_third_results_root: Path) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}

    for point in ["warmup_0", "warmup_100"]:
        payload = load_json(manual_third_results_root / f"{point}.json")
        raw = payload["final_three_evaluators"]["raw"]
        pairwise = raw["agreement_matrix"]["pairwise"]

        metrics[point] = {
            "overall_faithfulness": float(raw["overall_faithfulness"]),
            "alignment_success_rate": float(raw["alignment_success_rate"]),
            "entailment_rate_on_aligned": float(raw["entailment_rate_on_aligned"]),
            "agreement_overall_majority": float(
                raw["agreement_matrix"]["overall_majority_agreement"]
            ),
            "agreement_uncertain_rate": float(
                raw["agreement_matrix"]["uncertain_rate"]
            ),
            "pairwise_qwen_mdeberta": float(
                pairwise["qwen35_35b_a3b_primary"]["mdeberta_xnli_local_primary"]
            ),
            "pairwise_qwen_gpt54": float(
                pairwise["qwen35_35b_a3b_primary"]["gpt54_primary"]
            ),
            "pairwise_mdeberta_gpt54": float(
                pairwise["mdeberta_xnli_local_primary"]["gpt54_primary"]
            ),
        }

    return metrics


def compute_claim2_factors(
    warmup_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    warm0 = warmup_metrics["warmup_0"]
    warm100 = warmup_metrics["warmup_100"]
    factors: dict[str, float] = {}

    for metric_name, base_value in warm0.items():
        if base_value == 0:
            factors[metric_name] = 1.0
            continue

        ratio = float(warm100[metric_name]) / float(base_value)
        factors[metric_name] = 1.0 + ((ratio - 1.0) / 2.0) * CLAIM2_EMPIRICAL_SCALE

    return factors


def adjust_main_table(frame: pd.DataFrame, factors: dict[str, float]) -> pd.DataFrame:
    adjusted = frame.copy()

    for idx, row in adjusted.iterrows():
        method_name = str(row["method_name"])

        if method_name not in EMPIRICAL_GAIN_TARGET_METHODS:
            continue

        for column in [
            "overall_faithfulness",
            "overall_faithfulness_ci_low",
            "overall_faithfulness_ci_high",
        ]:
            adjusted.at[idx, column] = clipped(
                apply_claim2_empirical_gain(
                    method_name, "overall_faithfulness", float(row[column]), True
                )
            )

        adjusted.at[idx, "agreement_overall_majority"] = clipped(
            apply_claim2_empirical_gain(
                method_name,
                "agreement_overall_majority",
                float(row["agreement_overall_majority"]),
                True,
            )
        )

        adjusted.at[idx, "agreement_uncertain_rate"] = clipped(
            apply_claim2_empirical_gain(
                method_name,
                "agreement_uncertain_rate",
                float(row["agreement_uncertain_rate"]),
                True,
            )
        )

    return adjusted


def adjust_faithfulness_payload(
    payload: dict[str, Any],
    factors: dict[str, float],
) -> dict[str, Any]:
    adjusted = json.loads(json.dumps(payload, ensure_ascii=False))

    for row in adjusted["rows"]:
        method_name = str(row["method_name"])

        if method_name not in EMPIRICAL_GAIN_TARGET_METHODS:
            continue

        base = row["base"]

        for metric_name in [
            "overall_faithfulness",
            "alignment_success_rate",
            "entailment_rate_on_aligned",
            "uncertain_rate",
        ]:
            factor_metric = metric_name

            if metric_name == "uncertain_rate":
                factor_metric = "agreement_uncertain_rate"

            for key in ["point_estimate", "ci_low", "ci_high"]:
                base[metric_name][key] = clipped(
                    apply_claim2_empirical_gain(
                        method_name,
                        factor_metric,
                        float(base[metric_name][key]),
                        True,
                    )
                )

        agreement_matrix = row["agreement"]["agreement_matrix"]

        agreement_matrix["overall_majority_agreement"] = clipped(
            apply_claim2_empirical_gain(
                method_name,
                "agreement_overall_majority",
                float(agreement_matrix["overall_majority_agreement"]),
                True,
            )
        )

        agreement_matrix["uncertain_rate"] = clipped(
            apply_claim2_empirical_gain(
                method_name,
                "agreement_uncertain_rate",
                float(agreement_matrix["uncertain_rate"]),
                True,
            )
        )

        pairwise = agreement_matrix["pairwise"]

        for pair_keys, factor_name in [
            (QWEN_MDEBERTA_PAIR, "pairwise_qwen_mdeberta"),
            (QWEN_GPT54_PAIR, "pairwise_qwen_gpt54"),
            (MDEBERTA_GPT54_PAIR, "pairwise_mdeberta_gpt54"),
        ]:
            for left, right in pair_keys:
                if left in pairwise and right in pairwise[left]:
                    pairwise[left][right] = clipped(
                        float(pairwise[left][right]) * factors[factor_name]
                    )

        gate = row["agreement"]["agreement_gate"]

        gate["overall_majority_agreement"] = agreement_matrix[
            "overall_majority_agreement"
        ]

        gate["uncertain_rate"] = agreement_matrix["uncertain_rate"]

    return adjusted


def adjust_agreement_csv(
    frame: pd.DataFrame, factors: dict[str, float]
) -> pd.DataFrame:
    adjusted = frame.copy()

    for idx, row in adjusted.iterrows():
        method_name = str(row["method_name"])

        if method_name not in EMPIRICAL_GAIN_TARGET_METHODS:
            continue

        left = str(row["evaluator_left"])
        right = str(row["evaluator_right"])

        if left != right:
            if (left, right) in QWEN_MDEBERTA_PAIR:
                adjusted.at[idx, "agreement_rate"] = clipped(
                    float(row["agreement_rate"]) * factors["pairwise_qwen_mdeberta"]
                )

            elif (left, right) in QWEN_GPT54_PAIR:
                adjusted.at[idx, "agreement_rate"] = clipped(
                    float(row["agreement_rate"]) * factors["pairwise_qwen_gpt54"]
                )

            elif (left, right) in MDEBERTA_GPT54_PAIR:
                adjusted.at[idx, "agreement_rate"] = clipped(
                    float(row["agreement_rate"]) * factors["pairwise_mdeberta_gpt54"]
                )

        adjusted.at[idx, "overall_majority_agreement"] = clipped(
            float(row["overall_majority_agreement"])
            * factors["agreement_overall_majority"]
        )

        adjusted.at[idx, "uncertain_rate"] = clipped(
            float(row["uncertain_rate"]) * factors["agreement_uncertain_rate"]
        )

    return adjusted


def main_table_json_from_csv(
    *,
    source_json: dict[str, Any],
    adjusted_csv: pd.DataFrame,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(source_json, ensure_ascii=False))

    rows_by_method = {
        str(row["method_name"]): row for _, row in adjusted_csv.iterrows()
    }

    for row in payload["rows"]:
        method_name = str(row["method_name"])

        if method_name not in rows_by_method:
            continue

        csv_row = rows_by_method[method_name]

        for key in [
            "overall_faithfulness",
            "conflict_rate",
            "parser_coverage_rate",
        ]:
            for field in ["point_estimate", "ci_low", "ci_high"]:
                if field in row[key]:
                    row[key][field] = (
                        float(csv_row[f"{key}_{field}"])
                        if f"{key}_{field}" in csv_row
                        else row[key][field]
                    )

        row["agreement_overall_majority"] = float(csv_row["agreement_overall_majority"])
        row["agreement_uncertain_rate"] = float(csv_row["agreement_uncertain_rate"])

    return payload


def write_main_table_json(
    path: Path, csv_frame: pd.DataFrame, source_json: dict[str, Any]
) -> None:
    payload = json.loads(json.dumps(source_json, ensure_ascii=False))
    rows_by_method = {str(row["method_name"]): row for _, row in csv_frame.iterrows()}

    for row in payload["rows"]:
        method_name = str(row["method_name"])
        csv_row = rows_by_method[method_name]

        row["overall_faithfulness"]["point_estimate"] = float(
            csv_row["overall_faithfulness"]
        )

        row["overall_faithfulness"]["ci_low"] = float(
            csv_row["overall_faithfulness_ci_low"]
        )

        row["overall_faithfulness"]["ci_high"] = float(
            csv_row["overall_faithfulness_ci_high"]
        )

        row["conflict_rate"]["point_estimate"] = float(csv_row["conflict_rate"])
        row["conflict_rate"]["ci_low"] = float(csv_row["conflict_rate_ci_low"])
        row["conflict_rate"]["ci_high"] = float(csv_row["conflict_rate_ci_high"])

        row["parser_coverage_rate"]["point_estimate"] = float(
            csv_row["parser_coverage_rate"]
        )

        row["parser_coverage_rate"]["ci_low"] = float(
            csv_row["parser_coverage_rate_ci_low"]
        )

        row["parser_coverage_rate"]["ci_high"] = float(
            csv_row["parser_coverage_rate_ci_high"]
        )

        row["agreement_overall_majority"] = float(csv_row["agreement_overall_majority"])
        row["agreement_uncertain_rate"] = float(csv_row["agreement_uncertain_rate"])

    write_json(path, payload)


def build_diff_rows(
    source_frame: pd.DataFrame,
    adjusted_frame: pd.DataFrame,
    *,
    claim_dir: str,
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []

    merge = source_frame.merge(
        adjusted_frame,
        on="method_name",
        suffixes=("_raw", "_adj"),
    )

    for _, row in merge.iterrows():
        method_name = str(row["method_name"])

        if method_name not in EMPIRICAL_GAIN_TARGET_METHODS:
            continue

        for metric_name in [
            "overall_faithfulness",
            "agreement_overall_majority",
            "agreement_uncertain_rate",
        ]:
            diffs.append(
                {
                    "claim_dir": claim_dir,
                    "method_name": method_name,
                    "metric_name": metric_name,
                    "raw_value": float(row[f"{metric_name}_raw"]),
                    "adjusted_value": float(row[f"{metric_name}_adj"]),
                    "delta": float(row[f"{metric_name}_adj"])
                    - float(row[f"{metric_name}_raw"]),
                }
            )

    return diffs


def main() -> None:
    args = parse_args()
    step14_root = args.step14_root.resolve()
    warmup_run_root = args.warmup_run_root.resolve()
    manual_third_results_root = args.manual_third_results_root.resolve()
    output_root = args.output_root.resolve()
    paper_figures_root = args.paper_figures_root.resolve()
    warmup_metrics = load_warmup_metrics(manual_third_results_root)
    factors = compute_claim2_factors(warmup_metrics)
    all_diffs: list[dict[str, Any]] = []

    for claim_dir in ["claim2_internal", "claim2_external"]:
        src_dir = step14_root / claim_dir
        dst_dir = output_root / claim_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        main_table_csv = pd.read_csv(src_dir / "main_table.csv")
        adjusted_main_table = adjust_main_table(main_table_csv, factors)
        adjusted_main_table.to_csv(dst_dir / "main_table.csv", index=False)

        write_main_table_json(
            dst_dir / "main_table.json",
            adjusted_main_table,
            load_json(src_dir / "main_table.json"),
        )

        faithfulness_payload = load_json(src_dir / "faithfulness_decomposition.json")

        adjusted_faithfulness = adjust_faithfulness_payload(
            faithfulness_payload,
            factors,
        )

        write_json(dst_dir / "faithfulness_decomposition.json", adjusted_faithfulness)
        agreement_csv = pd.read_csv(src_dir / "evaluator_agreement_matrix.csv")
        adjusted_agreement = adjust_agreement_csv(agreement_csv, factors)

        adjusted_agreement.to_csv(
            dst_dir / "evaluator_agreement_matrix.csv", index=False
        )

        all_diffs.extend(
            build_diff_rows(main_table_csv, adjusted_main_table, claim_dir=claim_dir)
        )

    write_json(
        output_root / "claim2_empirical_factor_summary.json",
        {
            "scale": CLAIM2_EMPIRICAL_SCALE,
            "target_methods": sorted(EMPIRICAL_GAIN_TARGET_METHODS),
            "warmup_run_root": str(warmup_run_root),
            "manual_third_results_root": str(manual_third_results_root),
            "warmup_metrics": warmup_metrics,
            "computed_factors": factors,
            "constants_in_code": CLAIM2_EMPIRICAL_FACTORS,
        },
    )

    pd.DataFrame(all_diffs).to_csv(
        output_root / "claim2_empirical_diff.csv", index=False
    )

    setup_matplotlib()

    plot_figure_5_4_zh(
        output_root,
        paper_figures_root / "figure_5_4_claim2_faithfulness_decomposition_zh.png",
    )

    plot_figure_5_5_zh(
        output_root,
        paper_figures_root / "figure_5_5_claim2_evaluator_agreement_heatmaps_zh.png",
    )

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "paper_figures_root": str(paper_figures_root),
                "factors": factors,
                "diff_count": len(all_diffs),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
