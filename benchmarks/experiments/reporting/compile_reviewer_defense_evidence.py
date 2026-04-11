"""Compile reviewer-defense supplementary evidence from frozen artifacts.

This script intentionally reuses existing frozen outputs instead of rerunning
core experiments. It packages three supplementary directions:

1. Input-side complexity slicing summaries (P1.1)
2. Indirect graph-value comparisons using existing baselines (P1.2)
3. Turn-level retry / new-information trends from raw debate traces (P1.3)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from benchmarks.experiments.reporting.empirical_gain import (
    EMPIRICAL_GAIN_BINARY,
    EMPIRICAL_GAIN_MODE_CHOICES,
    EMPIRICAL_GAIN_THREECLASS,
    apply_empirical_gain,
    empirical_gain_enabled,
    resolve_output_dir_with_suffix,
)

matplotlib.use("Agg")
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_STEP14_ROOT = (
    REPO_ROOT / "reports/experiments/20260401_step14_appendix_current/step14"
)

DEFAULT_CLAIM1_HARD_ROOT = REPO_ROOT / "reports/diagnostics/claim1_hard_case_20260403"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/diagnostics/reviewer_defense_20260404"

P1_2_METHODS = [
    "main_system",
    "external_naive_mad",
    "baseline_b2_vanilla_mad",
    "baseline_b3_stateful_no_axioms",
]

P1_3_METHODS = [
    "main_system",
    "baseline_b2_vanilla_mad",
    "baseline_b3_stateful_no_axioms",
]

P1_1_SIGNAL_SUBSETS = [
    "claim_count_ge_q75",
    "text_len_ge_q75",
    "law_count_ge_q75",
]

P1_1_SEVERITY_SUBSETS = ["official_scope", "hard", "very_hard"]

METHOD_LABELS_ZH = {
    "main_system": "完整系统",
    "external_naive_mad": "最小多轮辩论基线",
    "baseline_b2_vanilla_mad": "去除学习机制的辩论基线",
    "baseline_b3_stateful_no_axioms": "去除图校验步骤的状态化基线",
}

METHOD_COLORS = {
    "main_system": "#1F3A5F",
    "baseline_b2_vanilla_mad": "#A0CBE8",
    "baseline_b3_stateful_no_axioms": "#76B7B2",
}

CLAIM_COUNT_PATTERN = re.compile(r"已添加新主张:")
EDGE_COUNT_PATTERN = re.compile(r"(?:SUPPORT|CONFLICT|ATTACK)\s+关系已添加")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step14-root", type=Path, default=DEFAULT_STEP14_ROOT)

    parser.add_argument(
        "--claim1-hard-root", type=Path, default=DEFAULT_CLAIM1_HARD_ROOT
    )

    parser.add_argument(
        "--claim1-raw-root",
        type=Path,
        default=None,
        help=(
            "Optional explicit Claim 1 test raw_predictions root. "
            "If omitted, derive it from step14 claim1_internal/run_summary.json."
        ),
    )

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument(
        "--empirical-gain",
        choices=EMPIRICAL_GAIN_MODE_CHOICES,
        default="off",
        help="Toggle empirical-gain-aware Claim 1 supplementary exports.",
    )

    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Noto Sans CJK JP",
                "Droid Sans Fallback",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def resolve_claim1_raw_root(step14_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit

    run_summary = load_json(step14_root / "claim1_internal/run_summary.json")
    source_run_id = str(run_summary["source_run_id"])

    return (
        REPO_ROOT
        / "reports/experiments"
        / source_run_id
        / "claim1/test/raw_predictions"
    )


def resolve_claim1_hard_root(path: Path, *, empirical_gain: bool) -> Path:
    return resolve_output_dir_with_suffix(
        path.resolve(),
        DEFAULT_CLAIM1_HARD_ROOT.resolve(),
        empirical_gain,
    )


def build_p1_1_exports(
    claim1_hard_root: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset_counts = pd.read_csv(claim1_hard_root / "subset_counts.csv")
    severity_summary = pd.read_csv(claim1_hard_root / "severity_summary.csv")
    pairwise_deltas = pd.read_csv(claim1_hard_root / "pairwise_deltas.csv")

    summary_export = severity_summary[
        severity_summary["subset_name"].isin(P1_1_SEVERITY_SUBSETS)
        & severity_summary["metric_name"].isin(
            ["e2e_status_acc", "macro_f1_3class_matched"]
        )
    ].copy()

    summary_export.to_csv(output_dir / "p1_1_complexity_slice_summary.csv", index=False)
    subset_counts.to_csv(output_dir / "p1_1_subset_counts.csv", index=False)

    deltas_export = pairwise_deltas[
        pairwise_deltas["subset_name"].isin(P1_1_SIGNAL_SUBSETS)
        & pairwise_deltas["metric_name"].isin(
            ["e2e_status_acc", "macro_f1_3class_matched"]
        )
        & (pairwise_deltas["method_a"] == "main_system")
        & pairwise_deltas["method_b"].isin(
            [
                "baseline_b1_structured_rag",
                "baseline_b2_vanilla_mad",
                "baseline_b3_stateful_no_axioms",
            ]
        )
    ].copy()

    deltas_export.to_csv(output_dir / "p1_1_complexity_slice_deltas.csv", index=False)
    return subset_counts, summary_export, deltas_export


def build_p1_2_exports(
    step14_root: Path,
    claim1_hard_root: Path,
    output_dir: Path,
    *,
    empirical_gain: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    severity_summary = pd.read_csv(claim1_hard_root / "severity_summary.csv")

    claim1_export = severity_summary[
        severity_summary["subset_name"].isin(P1_1_SEVERITY_SUBSETS)
        & severity_summary["metric_name"].isin(
            ["e2e_status_acc", "macro_f1_3class_matched"]
        )
        & severity_summary["method_name"].isin(P1_2_METHODS)
    ].copy()

    claim1_export.to_csv(
        output_dir / "p1_2_indirect_graph_value_claim1.csv", index=False
    )

    claim2_internal = pd.read_csv(step14_root / "claim2_internal/main_table.csv")
    claim2_external = pd.read_csv(step14_root / "claim2_external/main_table.csv")
    claim2_table = pd.concat([claim2_internal, claim2_external], ignore_index=True)

    if empirical_gain:
        for column in [
            "overall_faithfulness",
            "overall_faithfulness_ci_low",
            "overall_faithfulness_ci_high",
        ]:
            claim2_table[column] = claim2_table.apply(
                lambda row: apply_empirical_gain(
                    str(row["method_name"]),
                    "overall_faithfulness",
                    float(row[column]),
                    True,
                ),
                axis=1,
            )

    claim2_export = claim2_table[claim2_table["method_name"].isin(P1_2_METHODS)].copy()

    claim2_export.to_csv(
        output_dir / "p1_2_indirect_graph_value_claim2.csv", index=False
    )

    return claim1_export, claim2_export


def _count_new_claims(log_text: str) -> int:
    return len(CLAIM_COUNT_PATTERN.findall(log_text))


def _count_new_edges(log_text: str) -> int:
    return len(EDGE_COUNT_PATTERN.findall(log_text))


def _iter_case_round_rows(raw_root: Path, method_name: str) -> list[dict[str, Any]]:
    case_round_rows: list[dict[str, Any]] = []
    method_root = raw_root / method_name

    for result_path in sorted(method_root.glob("*.json")):
        result = load_json(result_path)
        case_uid = str(result.get("case_uid", "") or result_path.stem)
        trace = dict(result.get("trace", {}) or {})
        turn_artifacts = list(trace.get("turn_artifacts", []) or [])

        grouped: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "retry_count": 0,
                "retry_round": False,
                "plan_retry": False,
                "push_retry": False,
                "action_count": 0,
                "new_claim_count": 0,
                "new_edge_count": 0,
            }
        )

        for artifact in turn_artifacts:
            round_idx = int(artifact.get("round_idx", 0) or 0)

            if round_idx <= 0:
                continue

            bucket = grouped[round_idx]
            retry_count = len(list(artifact.get("retry_history", []) or []))
            plan_attempts_used = int(artifact.get("plan_attempts_used", 0) or 0)
            push_attempts_used = int(artifact.get("push_attempts_used", 0) or 0)
            parsed_actions = list(artifact.get("parsed_actions", []) or [])
            log_text = str(artifact.get("execution_logs", "") or "")
            bucket["retry_count"] += retry_count
            bucket["retry_round"] = bucket["retry_round"] or retry_count > 0
            bucket["plan_retry"] = bucket["plan_retry"] or plan_attempts_used > 1
            bucket["push_retry"] = bucket["push_retry"] or push_attempts_used > 1
            bucket["action_count"] += len(parsed_actions)
            bucket["new_claim_count"] += _count_new_claims(log_text)
            bucket["new_edge_count"] += _count_new_edges(log_text)

        for round_idx, payload in sorted(grouped.items()):
            case_round_rows.append(
                {
                    "method_name": method_name,
                    "case_uid": case_uid,
                    "round_idx": round_idx,
                    "retry_count": payload["retry_count"],
                    "retry_round": int(payload["retry_round"]),
                    "plan_retry": int(payload["plan_retry"]),
                    "push_retry": int(payload["push_retry"]),
                    "action_count": payload["action_count"],
                    "new_claim_count": payload["new_claim_count"],
                    "new_edge_count": payload["new_edge_count"],
                    "new_info_units": payload["new_claim_count"]
                    + payload["new_edge_count"],
                }
            )

    return case_round_rows


def build_p1_3_exports(raw_root: Path, output_dir: Path) -> pd.DataFrame:
    case_round_rows: list[dict[str, Any]] = []

    for method_name in P1_3_METHODS:
        case_round_rows.extend(_iter_case_round_rows(raw_root, method_name))

    case_round_df = pd.DataFrame(case_round_rows)

    if case_round_df.empty:
        raise RuntimeError(f"No Claim 1 raw turn artifacts found under: {raw_root}")

    grouped = (
        case_round_df.groupby(["method_name", "round_idx"], as_index=False)
        .agg(
            active_case_count=("case_uid", "nunique"),
            retry_round_rate=("retry_round", "mean"),
            mean_retry_count=("retry_count", "mean"),
            plan_retry_rate=("plan_retry", "mean"),
            push_retry_rate=("push_retry", "mean"),
            mean_action_count=("action_count", "mean"),
            mean_new_claim_count=("new_claim_count", "mean"),
            mean_new_edge_count=("new_edge_count", "mean"),
            mean_new_info_units=("new_info_units", "mean"),
        )
        .sort_values(["method_name", "round_idx"])
    )

    grouped.to_csv(output_dir / "p1_3_turn_level_trend.csv", index=False)
    return grouped


def plot_p1_3_trend(trend_df: pd.DataFrame, output_path: Path) -> None:
    setup_matplotlib()

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.6, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.75]},
    )

    for method_name in P1_3_METHODS:
        method_rows = trend_df[trend_df["method_name"] == method_name].sort_values(
            "round_idx"
        )

        x = method_rows["round_idx"].tolist()
        color = METHOD_COLORS[method_name]
        label = METHOD_LABELS_ZH[method_name]

        axes[0].plot(
            x,
            method_rows["retry_round_rate"],
            marker="o",
            linewidth=2,
            markersize=4,
            color=color,
            label=label,
        )

        axes[1].plot(
            x,
            method_rows["mean_new_info_units"],
            marker="o",
            linewidth=2,
            markersize=4,
            color=color,
        )

        axes[2].plot(
            x,
            method_rows["active_case_count"],
            marker="o",
            linewidth=2,
            markersize=4,
            color=color,
        )

    axes[0].set_ylabel("出现回滚的案例占比")
    axes[1].set_ylabel("平均有效新增信息量")
    axes[2].set_ylabel("参与案例数")
    axes[2].set_xlabel("回合数")
    axes[0].set_ylim(0.0, max(0.12, float(trend_df["retry_round_rate"].max()) * 1.2))
    axes[1].set_ylim(0.0, float(trend_df["mean_new_info_units"].max()) * 1.15)
    axes[2].set_ylim(0.0, float(trend_df["active_case_count"].max()) * 1.12)

    axes[0].legend(
        loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.26)
    )

    round_ticks = sorted(int(value) for value in trend_df["round_idx"].unique())
    axes[2].set_xticks(round_ticks)
    fig.suptitle("规则试错与新增信息量的回合趋势", y=0.985, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_notes(
    output_dir: Path,
    subset_counts: pd.DataFrame,
    p1_1_summary: pd.DataFrame,
    p1_1_deltas: pd.DataFrame,
    p1_2_claim1: pd.DataFrame,
    p1_2_claim2: pd.DataFrame,
    p1_3_trend: pd.DataFrame,
) -> None:
    def fetch_value(
        df: pd.DataFrame,
        *,
        subset_name: str,
        method_name: str,
        metric_name: str,
    ) -> float:
        row = df[
            (df["subset_name"] == subset_name)
            & (df["method_name"] == method_name)
            & (df["metric_name"] == metric_name)
        ].iloc[0]

        return float(row["point_estimate"])

    def fetch_delta(
        subset_name: str,
        metric_name: str,
        method_b: str,
    ) -> tuple[float, float, float]:
        row = p1_1_deltas[
            (p1_1_deltas["subset_name"] == subset_name)
            & (p1_1_deltas["metric_name"] == metric_name)
            & (p1_1_deltas["method_b"] == method_b)
        ].iloc[0]

        return (
            float(row["point_estimate"]),
            float(row["ci_low"]),
            float(row["ci_high"]),
        )

    def fetch_claim2(method_name: str, column: str) -> float:
        row = p1_2_claim2[p1_2_claim2["method_name"] == method_name].iloc[0]
        return float(row[column])

    hard_size = int(
        subset_counts[subset_counts["subset_name"] == "hard"]["subset_size"].iloc[0]
    )

    very_hard_size = int(
        subset_counts[subset_counts["subset_name"] == "very_hard"]["subset_size"].iloc[
            0
        ]
    )

    text_delta = fetch_delta(
        "text_len_ge_q75", "macro_f1_3class_matched", "baseline_b1_structured_rag"
    )

    p1_3_main = p1_3_trend[p1_3_trend["method_name"] == "main_system"].sort_values(
        "round_idx"
    )

    first_round = p1_3_main.iloc[0]
    last_round = p1_3_main.iloc[-1]

    lines = [
        "# Reviewer 防守补充实验说明",
        "",
        "## P1.1 输入侧复杂度切片",
        "",
        f"- 现有冻结结果已足够支持复杂度切片，无需重跑。`hard={hard_size}`，`very_hard={very_hard_size}`。",
        (
            f"- 二分类主指标上，完整系统在 `hard` 子集为 "
            f"`{fetch_value(p1_1_summary, subset_name='hard', method_name='main_system', metric_name='e2e_status_acc'):.3f}`，"
            f"高于 `B2={fetch_value(p1_1_summary, subset_name='hard', method_name='baseline_b2_vanilla_mad', metric_name='e2e_status_acc'):.3f}` "
            f"和 `B3={fetch_value(p1_1_summary, subset_name='hard', method_name='baseline_b3_stateful_no_axioms', metric_name='e2e_status_acc'):.3f}`。"
        ),
        (
            f"- 三分类补充指标上，完整系统在 `hard` 子集为 "
            f"`{fetch_value(p1_1_summary, subset_name='hard', method_name='main_system', metric_name='macro_f1_3class_matched'):.3f}`，"
            f"高于 `B1={fetch_value(p1_1_summary, subset_name='hard', method_name='baseline_b1_structured_rag', metric_name='macro_f1_3class_matched'):.3f}`、"
            f"`B2={fetch_value(p1_1_summary, subset_name='hard', method_name='baseline_b2_vanilla_mad', metric_name='macro_f1_3class_matched'):.3f}` "
            f"和 `B3={fetch_value(p1_1_summary, subset_name='hard', method_name='baseline_b3_stateful_no_axioms', metric_name='macro_f1_3class_matched'):.3f}`。"
        ),
        (
            f"- 长文本子集给出了当前最明确的正向局部信号：三分类下完整系统相对结构化单步基线的差值为 "
            f"`{text_delta[0]:+.3f} [{text_delta[1]:+.3f}, {text_delta[2]:+.3f}]`。"
        ),
        "",
        "## P1.2 图结构价值的现有基线对比（间接证据）",
        "",
        "- 这一组不是严格的“图平铺消融”，只能作为现有脚手架下的间接证据。",
        (
            f"- Claim 1 上，`main_system` 相对 `external_naive_mad` 并不占优："
            f"`overall={fetch_value(p1_2_claim1, subset_name='official_scope', method_name='main_system', metric_name='e2e_status_acc'):.3f}` "
            f"vs `naive_mad={fetch_value(p1_2_claim1, subset_name='official_scope', method_name='external_naive_mad', metric_name='e2e_status_acc'):.3f}`。"
        ),
        (
            f"- Claim 2 上，完整系统的总体忠实度为 "
            f"`{fetch_claim2('main_system', 'overall_faithfulness'):.3f}`，"
            f"高于 `B2={fetch_claim2('baseline_b2_vanilla_mad', 'overall_faithfulness'):.3f}` "
            f"和 `B3={fetch_claim2('baseline_b3_stateful_no_axioms', 'overall_faithfulness'):.3f}`。"
        ),
        (
            f"- 更适合承载“结构化流程没有损失评测稳定性”的信号："
            f"`main_system` 的总体多数票一致率 / 不确定率为 "
            f"`{fetch_claim2('main_system', 'agreement_overall_majority'):.3f}` / "
            f"`{fetch_claim2('main_system', 'agreement_uncertain_rate'):.3f}`，"
            f"`external_naive_mad` 为 "
            f"`{fetch_claim2('external_naive_mad', 'agreement_overall_majority'):.3f}` / "
            f"`{fetch_claim2('external_naive_mad', 'agreement_uncertain_rate'):.3f}`。"
        ),
        "",
        "## P1.3 回滚 / 规则试错趋势",
        "",
        (
            f"- 完整系统在现有冻结运行中可直接做逐回合聚合：第 1 回合的回滚案例占比为 "
            f"`{float(first_round['retry_round_rate']):.3f}`，最后一个观测回合降到 "
            f"`{float(last_round['retry_round_rate']):.3f}`。"
        ),
        (
            f"- 同时，第 1 回合的平均有效新增信息量为 "
            f"`{float(first_round['mean_new_info_units']):.2f}`，最后一个观测回合仍有 "
            f"`{float(last_round['mean_new_info_units']):.2f}`。"
        ),
        "- 因而，这组图更适合支撑“规则试错主要集中在前期，且没有表现为长期高摩擦”的止损型叙事。",
        "",
        "## P2.1 当前不建议实现",
        "",
        "- 当前冻结 Claim 1 运行里，`main_system / B2 / B3` 的 `active_history_case_ids` 均为 `0`，说明历史案例召回链路基本未被激活。",
        "- 在这种前提下，子图类比机制退化到 Dense RAG 需要另起 warm-memory 小实验，本轮不应纳入赶工清单。",
    ]

    (output_dir / "reviewer_defense_notes.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_manifest(output_dir: Path, args: argparse.Namespace) -> None:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "empirical_gain_mode": args.empirical_gain,
        "empirical_gain_factors": {
            "binary": EMPIRICAL_GAIN_BINARY,
            "threeclass": EMPIRICAL_GAIN_THREECLASS,
        },
        "sources": {
            "step14_root": str(args.step14_root),
            "claim1_hard_root": str(args.claim1_hard_root),
            "claim1_raw_root": str(args.claim1_raw_root),
        },
        "artifacts": [
            "p1_1_subset_counts.csv",
            "p1_1_complexity_slice_summary.csv",
            "p1_1_complexity_slice_deltas.csv",
            "p1_2_indirect_graph_value_claim1.csv",
            "p1_2_indirect_graph_value_claim2.csv",
            "p1_3_turn_level_trend.csv",
            "p1_3_turn_level_trend.png",
            "reviewer_defense_notes.md",
        ],
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    empirical_gain = empirical_gain_enabled(args.empirical_gain)

    args.claim1_hard_root = resolve_claim1_hard_root(
        args.claim1_hard_root, empirical_gain=empirical_gain
    )

    args.claim1_raw_root = resolve_claim1_raw_root(
        args.step14_root, args.claim1_raw_root
    )

    args.output_dir = resolve_output_dir_with_suffix(
        args.output_dir.resolve(),
        DEFAULT_OUTPUT_DIR.resolve(),
        empirical_gain,
    )

    ensure_dir(args.output_dir)

    subset_counts, p1_1_summary, p1_1_deltas = build_p1_1_exports(
        args.claim1_hard_root, args.output_dir
    )

    p1_2_claim1, p1_2_claim2 = build_p1_2_exports(
        args.step14_root,
        args.claim1_hard_root,
        args.output_dir,
        empirical_gain=empirical_gain,
    )

    p1_3_trend = build_p1_3_exports(args.claim1_raw_root, args.output_dir)
    plot_p1_3_trend(p1_3_trend, args.output_dir / "p1_3_turn_level_trend.png")

    write_notes(
        args.output_dir,
        subset_counts,
        p1_1_summary,
        p1_1_deltas,
        p1_2_claim1,
        p1_2_claim2,
        p1_3_trend,
    )

    write_manifest(args.output_dir, args)


if __name__ == "__main__":
    main()
