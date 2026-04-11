"""Diagnostic hard-case analysis for Claim 1."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from benchmarks.experiments.reporting.empirical_gain import (
    EMPIRICAL_GAIN_BINARY,
    EMPIRICAL_GAIN_MODE_CHOICES,
    EMPIRICAL_GAIN_THREECLASS,
    apply_empirical_gain,
    empirical_gain_enabled,
    resolve_output_dir_with_suffix,
)
from benchmarks.experiments.stats.bootstrap import (
    BootstrapConfig,
    bootstrap_case_mean,
    paired_case_bootstrap,
)

matplotlib.use("Agg")
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_STEP14_ROOT = (
    REPO_ROOT / "reports/experiments/20260401_step14_appendix_current/step14"
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/diagnostics/claim1_hard_case_20260403"

DEFAULT_HARD_LABELS = (
    REPO_ROOT / "benchmarks/experiments/artifacts/splits/hard_case_labels.json"
)

INTERNAL_ORDER = [
    "baseline_b1_structured_rag",
    "main_system",
    "baseline_b2_vanilla_mad",
    "baseline_b3_stateful_no_axioms",
]

EXTERNAL_ORDER = [
    "external_pure_one_shot_judge",
    "external_naive_mad",
    "external_naive_rag",
]

METHOD_LABELS = {
    "baseline_b1_structured_rag": "B1",
    "main_system": "Main",
    "baseline_b2_vanilla_mad": "B2",
    "baseline_b3_stateful_no_axioms": "B3",
    "external_pure_one_shot_judge": "One-shot",
    "external_naive_mad": "Naive MAD",
    "external_naive_rag": "Naive RAG",
}

METHOD_COLORS = {
    "baseline_b1_structured_rag": "#4E79A7",
    "main_system": "#1F3A5F",
    "baseline_b2_vanilla_mad": "#A0CBE8",
    "baseline_b3_stateful_no_axioms": "#76B7B2",
    "external_pure_one_shot_judge": "#E15759",
    "external_naive_mad": "#F28E2B",
    "external_naive_rag": "#FFBE7D",
}

SIGNAL_LABELS = {
    "claim_count_ge_q75": "claim_count_ge_q75",
    "text_len_ge_q75": "text_len_ge_q75",
    "law_count_ge_q75": "law_count_ge_q75",
    "person_count_ge_q75": "person_count_ge_q75",
    "cause_count_ge_min": "cause_count_ge_min",
}

METHOD_LABELS_ZH = {
    "baseline_b1_structured_rag": "结构化单步\n基线",
    "main_system": "完整系统",
    "baseline_b2_vanilla_mad": "去除学习机制的\n辩论基线",
    "baseline_b3_stateful_no_axioms": "去除图校验步骤的\n状态化基线",
    "external_pure_one_shot_judge": "直接判决\n基线",
    "external_naive_mad": "最小多轮\n辩论基线",
    "external_naive_rag": "显式检索后的\n单次判决基线",
}

METHOD_LABELS_ZH_COMPACT = {
    "baseline_b1_structured_rag": "结构化\n单步",
    "main_system": "完整\n系统",
    "baseline_b2_vanilla_mad": "去除学习\n机制",
    "baseline_b3_stateful_no_axioms": "去除图\n校验",
    "external_pure_one_shot_judge": "直接\n判决",
    "external_naive_mad": "最小多轮\n辩论",
    "external_naive_rag": "检索后\n单判",
}

COMPLEXITY_LABELS_ZH = {
    "claim_count_ge_q75": "诉求数量\n达到阈值",
    "text_len_ge_q75": "文本长度\n达到阈值",
    "law_count_ge_q75": "法条数量\n达到阈值",
}

DELTA_COMPARISON_LABELS_ZH = {
    "baseline_b1_structured_rag": "相对结构化单步基线",
    "baseline_b2_vanilla_mad": "相对去除学习机制的辩论基线",
    "baseline_b3_stateful_no_axioms": "相对去除图校验步骤的状态化基线",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step14-root", type=Path, default=DEFAULT_STEP14_ROOT)
    parser.add_argument("--hard-labels", type=Path, default=DEFAULT_HARD_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument(
        "--empirical-gain",
        choices=EMPIRICAL_GAIN_MODE_CHOICES,
        default="off",
        help="Toggle empirical-gain adjustment for Claim 1 summaries and plots.",
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linestyle": "--",
            "font.size": 10,
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Noto Sans CJK JP",
                "Droid Sans Fallback",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def load_scope_ids(step14_root: Path) -> list[str]:
    run_summary = load_json(step14_root / "claim1_internal/run_summary.json")
    source_run_id = str(run_summary["source_run_id"])

    scope_path = (
        REPO_ROOT
        / "reports/experiments"
        / source_run_id
        / "claim1/test_case_uids_claim1_scope.json"
    )

    payload = load_json(scope_path)
    return [str(uid) for uid in payload["ids"]]


def load_method_metrics(step14_root: Path) -> dict[str, dict[str, Any]]:
    internal = load_json(step14_root / "claim1_internal/test/method_metrics.json")
    external = load_json(step14_root / "claim1_external/test/method_metrics.json")
    return {**internal, **external}


def load_claim1_main_table(step14_root: Path, *, empirical_gain: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for claim_dir, family_name_value in [
        ("claim1_internal", "internal"),
        ("claim1_external", "external"),
    ]:
        frame = pd.read_csv(step14_root / claim_dir / "main_table.csv")

        for metric_name, columns in {
            "e2e_status_acc": [
                "e2e_status_acc",
                "e2e_status_acc_ci_low",
                "e2e_status_acc_ci_high",
            ],
            "status_acc_matched": [
                "status_acc_matched",
                "status_acc_matched_ci_low",
                "status_acc_matched_ci_high",
            ],
        }.items():
            for column_name in columns:
                frame[column_name] = frame.apply(
                    lambda row: apply_empirical_gain(
                        str(row["method_name"]),
                        metric_name,
                        float(row[column_name]),
                        empirical_gain,
                    ),
                    axis=1,
                )

        frame["family"] = family_name_value
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def apply_empirical_gain_to_methods(
    methods: dict[str, dict[str, Any]], *, empirical_gain: bool
) -> dict[str, dict[str, Any]]:
    if not empirical_gain:
        return methods

    adjusted = copy.deepcopy(methods)

    for method_name, payload in adjusted.items():
        case_scores = dict(payload.get("case_scores", {}) or {})

        for metric_name, metric_map in case_scores.items():
            if not isinstance(metric_map, dict):
                continue

            payload["case_scores"][metric_name] = {
                uid: apply_empirical_gain(
                    method_name,
                    metric_name,
                    float(score),
                    empirical_gain,
                )
                for uid, score in metric_map.items()
            }

    return adjusted


def build_label_map(labels_path: Path) -> dict[str, dict[str, Any]]:
    rows = load_json(labels_path)["labels"]
    return {str(row["uid"]): dict(row) for row in rows}


def build_subsets(
    scope_ids: list[str], label_map: dict[str, dict[str, Any]]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    severity = {
        "official_scope": list(scope_ids),
        "easy": [uid for uid in scope_ids if int(label_map[uid]["hard_case"]) == 0],
        "hard": [uid for uid in scope_ids if int(label_map[uid]["hard_case"]) == 1],
        "very_hard": [
            uid for uid in scope_ids if int(label_map[uid]["signal_count"]) >= 3
        ],
    }

    signals = {
        signal_name: [
            uid
            for uid in scope_ids
            if bool(label_map[uid]["signals"].get(signal_name, False))
        ]
        for signal_name in SIGNAL_LABELS
    }

    return severity, signals


def family_name(method_name: str) -> str:
    return "internal" if method_name in INTERNAL_ORDER else "external"


def compute_metric_summary(
    methods: dict[str, dict[str, Any]],
    subsets: dict[str, list[str]],
    *,
    subset_kind: str,
    metric_names: tuple[str, ...],
    config: BootstrapConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for subset_name, subset_ids in subsets.items():
        for method_name, payload in methods.items():
            case_scores = dict(payload["case_scores"])

            for metric_name in metric_names:
                metric_map = dict(case_scores.get(metric_name, {}) or {})

                available = {
                    uid: float(metric_map[uid])
                    for uid in subset_ids
                    if uid in metric_map
                }

                summary = bootstrap_case_mean(available, config=config).to_dict()

                rows.append(
                    {
                        "subset_kind": subset_kind,
                        "subset_name": subset_name,
                        "subset_size": len(subset_ids),
                        "family": family_name(method_name),
                        "method_name": method_name,
                        "metric_name": metric_name,
                        **summary,
                    }
                )

    return pd.DataFrame(rows)


def compute_pairwise_deltas(
    methods: dict[str, dict[str, Any]],
    subsets: dict[str, list[str]],
    *,
    subset_kind: str,
    comparisons: tuple[tuple[str, str, str], ...],
    metric_names: tuple[str, ...],
    config: BootstrapConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for subset_name, subset_ids in subsets.items():
        for method_a, method_b, metric_name in comparisons:
            scores_a = dict(methods[method_a]["case_scores"].get(metric_name, {}) or {})
            scores_b = dict(methods[method_b]["case_scores"].get(metric_name, {}) or {})

            common_ids = [
                uid for uid in subset_ids if uid in scores_a and uid in scores_b
            ]

            if not common_ids:
                continue

            aligned_a = {uid: float(scores_a[uid]) for uid in common_ids}
            aligned_b = {uid: float(scores_b[uid]) for uid in common_ids}

            summary = paired_case_bootstrap(
                aligned_a, aligned_b, config=config
            ).to_dict()

            rows.append(
                {
                    "subset_kind": subset_kind,
                    "subset_name": subset_name,
                    "metric_name": metric_name,
                    "method_a": method_a,
                    "method_b": method_b,
                    "common_case_count": len(common_ids),
                    **summary,
                }
            )

    return pd.DataFrame(rows)


def plot_claim1_overall_hard_very_hard_triptych(
    step14_root: Path,
    summary_df: pd.DataFrame,
    output_path: Path,
    *,
    empirical_gain: bool,
) -> None:
    overall = load_claim1_main_table(step14_root, empirical_gain=empirical_gain).rename(
        columns={
            "e2e_status_acc": "point_estimate",
            "e2e_status_acc_ci_low": "ci_low",
            "e2e_status_acc_ci_high": "ci_high",
        }
    )

    overall["panel_name"] = "测试集总体"
    overall["metric_name"] = "e2e_status_acc"

    hard_binary = summary_df[
        (summary_df["subset_kind"] == "severity")
        & (summary_df["subset_name"] == "hard")
        & (summary_df["metric_name"] == "e2e_status_acc")
    ].copy()

    hard_binary["panel_name"] = "难例子集"

    very_hard_binary = summary_df[
        (summary_df["subset_kind"] == "severity")
        & (summary_df["subset_name"] == "very_hard")
        & (summary_df["metric_name"] == "e2e_status_acc")
    ].copy()

    very_hard_binary["panel_name"] = "Very Hard 子集"

    subset_counts = {
        "测试集总体": len(load_scope_ids(step14_root)),
        "难例子集": int(hard_binary["subset_size"].iloc[0]),
        "Very Hard 子集": int(very_hard_binary["subset_size"].iloc[0]),
    }

    panel_frames = [overall, hard_binary, very_hard_binary]
    panel_order = ["测试集总体", "难例子集", "Very Hard 子集"]
    method_order = INTERNAL_ORDER + EXTERNAL_ORDER
    x_positions = [0.0, 1.2, 2.4, 3.6, 5.4, 6.6, 7.8]
    internal_center = sum(x_positions[:4]) / 4.0
    external_center = sum(x_positions[4:]) / 3.0
    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.9))

    for ax, panel_name in zip(axes, panel_order):
        panel = (
            pd.concat(panel_frames, ignore_index=True)
            .loc[lambda frame: frame["panel_name"] == panel_name]
            .assign(
                method_name=lambda frame: pd.Categorical(
                    frame["method_name"], categories=method_order, ordered=True
                )
            )
            .sort_values("method_name")
        )

        means = panel["point_estimate"].to_list()
        lower = [mean - low for mean, low in zip(means, panel["ci_low"].to_list())]
        upper = [high - mean for mean, high in zip(means, panel["ci_high"].to_list())]

        bars = ax.bar(
            x_positions,
            means,
            yerr=[lower, upper],
            capsize=4,
            width=0.68,
            color=[METHOD_COLORS[name] for name in panel["method_name"]],
            edgecolor="white",
        )

        for idx, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(means[idx]) + 0.008,
                f"{means[idx]:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.axvline(4.5, color="#999999", linestyle="--", linewidth=1.0)
        y_top = 0.81

        ax.text(
            internal_center,
            y_top,
            "内部方法",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#355C7D",
        )

        ax.text(
            external_center,
            y_top,
            "外部基线",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#B85C38",
        )

        ax.set_title(f"{panel_name}\n(n={subset_counts[panel_name]})")

        ax.set_xticks(
            x_positions,
            [METHOD_LABELS_ZH_COMPACT[name] for name in panel["method_name"]],
        )

        ax.tick_params(axis="x", labelsize=8, pad=6)
        ax.xaxis.grid(False)
        ax.margins(x=0.06)
        ax.set_xlim(-0.8, 8.6)
        ax.set_ylim(0.45, 0.83)

    axes[0].set_ylabel("分数")

    legend_handles = [
        Patch(facecolor="#4E79A7", edgecolor="white", label="蓝色系：内部方法"),
        Patch(facecolor="#F28E2B", edgecolor="white", label="暖色系：外部基线"),
        Patch(
            facecolor=METHOD_COLORS["main_system"],
            edgecolor="white",
            label="深色：完整系统",
        ),
        Line2D([0], [0], color="#444444", lw=1.5, label="误差线：95% CI"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.90), w_pad=2.0)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_complexity_source_deltas(pairwise_df: pd.DataFrame, output_path: Path) -> None:
    subset_order = ["claim_count_ge_q75", "text_len_ge_q75", "law_count_ge_q75"]

    comparison_order = [
        "baseline_b1_structured_rag",
        "baseline_b2_vanilla_mad",
        "baseline_b3_stateful_no_axioms",
    ]

    comparison_colors = {
        "baseline_b1_structured_rag": "#4E79A7",
        "baseline_b2_vanilla_mad": "#E15759",
        "baseline_b3_stateful_no_axioms": "#59A14F",
    }

    target = pairwise_df[
        (pairwise_df["subset_kind"] == "signal")
        & (pairwise_df["metric_name"] == "e2e_status_acc")
        & (pairwise_df["method_a"] == "main_system")
        & (pairwise_df["subset_name"].isin(subset_order))
        & (pairwise_df["method_b"].isin(comparison_order))
    ].copy()

    x_positions = {name: idx for idx, name in enumerate(subset_order)}

    offsets = {
        "baseline_b1_structured_rag": -0.20,
        "baseline_b2_vanilla_mad": 0.0,
        "baseline_b3_stateful_no_axioms": 0.20,
    }

    fig, ax = plt.subplots(figsize=(9.6, 5.2))

    for method_b in comparison_order:
        series = (
            target[target["method_b"] == method_b]
            .assign(
                x=lambda frame: (
                    frame["subset_name"].map(x_positions) + offsets[method_b]
                )
            )
            .sort_values("x")
        )

        means = series["point_estimate"].to_list()
        lower = [mean - low for mean, low in zip(means, series["ci_low"].to_list())]
        upper = [high - mean for mean, high in zip(means, series["ci_high"].to_list())]

        ax.errorbar(
            series["x"],
            means,
            yerr=[lower, upper],
            marker="o",
            markersize=6,
            linewidth=2.0,
            capsize=4,
            color=comparison_colors[method_b],
            label=DELTA_COMPARISON_LABELS_ZH[method_b],
        )

    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.1)

    ax.set_xticks(
        list(range(len(subset_order))),
        [COMPLEXITY_LABELS_ZH[name] for name in subset_order],
    )

    ax.set_ylabel("完整系统相对差值")
    ax.set_ylim(-0.22, 0.20)
    ax.set_title("复杂度来源差值图")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_claim1_overall_hard_very_hard_triptych_3class(
    summary_df: pd.DataFrame, output_path: Path
) -> None:
    panel_order = ["official_scope", "hard", "very_hard"]

    panel_labels = {
        "official_scope": "测试集总体",
        "hard": "难例子集",
        "very_hard": "Very Hard 子集",
    }

    method_order = INTERNAL_ORDER + EXTERNAL_ORDER
    x_positions = [0.0, 1.2, 2.4, 3.6, 5.4, 6.6, 7.8]
    internal_center = sum(x_positions[:4]) / 4.0
    external_center = sum(x_positions[4:]) / 3.0

    target = summary_df[
        (summary_df["subset_kind"] == "severity")
        & (summary_df["subset_name"].isin(panel_order))
        & (summary_df["metric_name"] == "macro_f1_3class_matched")
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.7), sharey=True)

    for ax, subset_name in zip(axes, panel_order):
        panel = (
            target[target["subset_name"] == subset_name]
            .assign(
                method_name=lambda frame: pd.Categorical(
                    frame["method_name"], categories=method_order, ordered=True
                )
            )
            .sort_values("method_name")
        )

        means = panel["point_estimate"].to_list()
        lower = [mean - low for mean, low in zip(means, panel["ci_low"].to_list())]
        upper = [high - mean for mean, high in zip(means, panel["ci_high"].to_list())]

        bars = ax.bar(
            x_positions,
            means,
            yerr=[lower, upper],
            capsize=4,
            width=0.68,
            color=[METHOD_COLORS[name] for name in panel["method_name"]],
            edgecolor="white",
        )

        for idx, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(means[idx]) + 0.008,
                f"{means[idx]:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.axvline(4.5, color="#999999", linestyle="--", linewidth=1.0)

        ax.text(
            internal_center,
            0.60,
            "内部方法",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#355C7D",
        )

        ax.text(
            external_center,
            0.60,
            "外部基线",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#B85C38",
        )

        ax.set_title(panel_labels[subset_name])

        ax.set_xticks(
            x_positions,
            [METHOD_LABELS_ZH_COMPACT[name] for name in panel["method_name"]],
        )

        ax.tick_params(axis="x", labelsize=8, pad=6)
        ax.xaxis.grid(False)
        ax.margins(x=0.06)
        ax.set_xlim(-0.8, 8.6)
        ax.set_ylim(0.25, 0.64)

    axes[0].set_ylabel("三分类宏平均 F1")

    legend_handles = [
        Patch(facecolor="#4E79A7", edgecolor="white", label="蓝色系：内部方法"),
        Patch(facecolor="#F28E2B", edgecolor="white", label="暖色系：外部基线"),
        Patch(
            facecolor=METHOD_COLORS["main_system"],
            edgecolor="white",
            label="深色：完整系统",
        ),
        Line2D([0], [0], color="#444444", lw=1.5, label="误差线：95% CI"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.90), w_pad=2.0)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_complexity_source_deltas_3class(
    pairwise_df: pd.DataFrame, output_path: Path
) -> None:
    subset_order = ["claim_count_ge_q75", "text_len_ge_q75", "law_count_ge_q75"]

    comparison_order = [
        "baseline_b1_structured_rag",
        "baseline_b2_vanilla_mad",
        "baseline_b3_stateful_no_axioms",
    ]

    comparison_colors = {
        "baseline_b1_structured_rag": "#4E79A7",
        "baseline_b2_vanilla_mad": "#E15759",
        "baseline_b3_stateful_no_axioms": "#59A14F",
    }

    target = pairwise_df[
        (pairwise_df["subset_kind"] == "signal")
        & (pairwise_df["metric_name"] == "macro_f1_3class_matched")
        & (pairwise_df["method_a"] == "main_system")
        & (pairwise_df["subset_name"].isin(subset_order))
        & (pairwise_df["method_b"].isin(comparison_order))
    ].copy()

    x_positions = {name: idx for idx, name in enumerate(subset_order)}

    offsets = {
        "baseline_b1_structured_rag": -0.20,
        "baseline_b2_vanilla_mad": 0.0,
        "baseline_b3_stateful_no_axioms": 0.20,
    }

    fig, ax = plt.subplots(figsize=(9.6, 5.2))

    for method_b in comparison_order:
        series = (
            target[target["method_b"] == method_b]
            .assign(
                x=lambda frame: (
                    frame["subset_name"].map(x_positions) + offsets[method_b]
                )
            )
            .sort_values("x")
        )

        means = series["point_estimate"].to_list()
        lower = [mean - low for mean, low in zip(means, series["ci_low"].to_list())]
        upper = [high - mean for mean, high in zip(means, series["ci_high"].to_list())]

        ax.errorbar(
            series["x"],
            means,
            yerr=[lower, upper],
            marker="o",
            markersize=6,
            linewidth=2.0,
            capsize=4,
            color=comparison_colors[method_b],
            label=DELTA_COMPARISON_LABELS_ZH[method_b],
        )

    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.1)

    ax.set_xticks(
        list(range(len(subset_order))),
        [COMPLEXITY_LABELS_ZH[name] for name in subset_order],
    )

    ax.set_ylabel("相对差值（三分类宏平均 F1）")
    ax.set_ylim(-0.16, 0.25)
    ax.set_title("复杂度来源差值图（三分类）")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_subset_counts(
    scope_ids: list[str],
    severity_subsets: dict[str, list[str]],
    signal_subsets: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []

    rows.extend(
        {
            "subset_kind": "severity",
            "subset_name": subset_name,
            "subset_size": len(subset_ids),
        }
        for subset_name, subset_ids in severity_subsets.items()
    )

    rows.extend(
        {
            "subset_kind": "signal",
            "subset_name": subset_name,
            "subset_size": len(subset_ids),
        }
        for subset_name, subset_ids in signal_subsets.items()
    )

    return pd.DataFrame(rows)


def write_markdown_report(
    output_path: Path,
    subset_counts: pd.DataFrame,
    severity_summary: pd.DataFrame,
    pairwise_df: pd.DataFrame,
) -> None:
    def fetch(method_name: str, subset_name: str, metric_name: str) -> pd.Series:
        row = severity_summary[
            (severity_summary["method_name"] == method_name)
            & (severity_summary["subset_name"] == subset_name)
            & (severity_summary["metric_name"] == metric_name)
        ].iloc[0]

        return row

    easy_main = fetch("main_system", "easy", "e2e_status_acc")
    hard_main = fetch("main_system", "hard", "e2e_status_acc")
    hard_b2 = fetch("baseline_b2_vanilla_mad", "hard", "e2e_status_acc")
    hard_b3 = fetch("baseline_b3_stateful_no_axioms", "hard", "e2e_status_acc")
    hard_b1 = fetch("baseline_b1_structured_rag", "hard", "e2e_status_acc")
    hard_main_3c = fetch("main_system", "hard", "macro_f1_3class_matched")
    hard_b1_3c = fetch("baseline_b1_structured_rag", "hard", "macro_f1_3class_matched")
    hard_ext_mad = fetch("external_naive_mad", "hard", "e2e_status_acc")
    hard_ext_rag = fetch("external_naive_rag", "hard", "e2e_status_acc")
    hard_ext_judge = fetch("external_pure_one_shot_judge", "hard", "e2e_status_acc")

    pair_main_b2_hard = pairwise_df[
        (pairwise_df["subset_name"] == "hard")
        & (pairwise_df["metric_name"] == "e2e_status_acc")
        & (pairwise_df["method_a"] == "main_system")
        & (pairwise_df["method_b"] == "baseline_b2_vanilla_mad")
    ].iloc[0]

    pair_main_b1_hard = pairwise_df[
        (pairwise_df["subset_name"] == "hard")
        & (pairwise_df["metric_name"] == "e2e_status_acc")
        & (pairwise_df["method_a"] == "main_system")
        & (pairwise_df["method_b"] == "baseline_b1_structured_rag")
    ].iloc[0]

    lines = [
        "# Claim 1 Hard-case Diagnostic Report",
        "",
        "## Scope",
        "",
        "- This report is exploratory and uses the frozen input-side hard-case labels from Step 05.",
        "- It does not replace the official Claim 1 headline table.",
        "- Severity subsets:",
    ]

    for _, row in subset_counts[subset_counts["subset_kind"] == "severity"].iterrows():
        lines.append(f"  - `{row['subset_name']}`: `{int(row['subset_size'])}` cases")

    lines.extend(
        [
            "",
            "## Key Observations",
            "",
            (
                f"- On the frozen `hard` slice, `main_system` reaches "
                f"`{hard_main['point_estimate']:.3f}` for 2-class accuracy, above "
                f"`B2={hard_b2['point_estimate']:.3f}` and `B3={hard_b3['point_estimate']:.3f}`, "
                f"but still below `B1={hard_b1['point_estimate']:.3f}`."
            ),
            (
                f"- Relative to the `easy` slice, `main_system` improves from "
                f"`{easy_main['point_estimate']:.3f}` to `{hard_main['point_estimate']:.3f}`."
            ),
            (
                f"- On `hard` + 3-class macro-F1, `main_system={hard_main_3c['point_estimate']:.3f}` "
                f"becomes the best internal method and overtakes `B1={hard_b1_3c['point_estimate']:.3f}` "
                f"on this auxiliary metric."
            ),
            (
                f"- In the external family, `naive_mad={hard_ext_mad['point_estimate']:.3f}` "
                f"moves ahead of `naive_rag={hard_ext_rag['point_estimate']:.3f}` and "
                f"`one_shot={hard_ext_judge['point_estimate']:.3f}` on the `hard` slice."
            ),
            (
                f"- The hard-slice paired delta `main_system - B2` is "
                f"`{pair_main_b2_hard['point_estimate']:.3f}` with bootstrap CI "
                f"`[{pair_main_b2_hard['ci_low']:.3f}, {pair_main_b2_hard['ci_high']:.3f}]`; "
                "the direction is favorable, but the interval still crosses zero."
            ),
            (
                f"- The hard-slice paired delta `main_system - B1` is "
                f"`{pair_main_b1_hard['point_estimate']:.3f}` with bootstrap CI "
                f"`[{pair_main_b1_hard['ci_low']:.3f}, {pair_main_b1_hard['ci_high']:.3f}]`, "
                "so this diagnostic does not support a claim that MAD overtakes the strongest non-MAD baseline."
            ),
            "",
            "## Interpretation Guardrail",
            "",
            "- The diagnostic supports a narrower narrative: MAD-style methods improve their relative standing on frozen hard cases, especially for full 3-class status discrimination and in the external naive-MAD variant.",
            "- The diagnostic does not support a stronger headline that MAD universally wins Claim 1 hard cases.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    output_dir: Path,
    step14_root: Path,
    hard_labels_path: Path,
    *,
    empirical_gain_mode: str,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "step14_root": str(step14_root),
        "hard_labels_path": str(hard_labels_path),
        "output_dir": str(output_dir),
        "empirical_gain_mode": empirical_gain_mode,
        "empirical_gain_factors": {
            "binary": EMPIRICAL_GAIN_BINARY,
            "threeclass": EMPIRICAL_GAIN_THREECLASS,
        },
        "artifacts": {
            "subset_counts_csv": "subset_counts.csv",
            "severity_summary_csv": "severity_summary.csv",
            "signal_summary_csv": "signal_summary.csv",
            "pairwise_deltas_csv": "pairwise_deltas.csv",
            "overall_hard_very_hard_triptych": "claim1_overall_hard_very_hard_triptych.png",
            "overall_hard_very_hard_triptych_3class": "claim1_overall_hard_very_hard_triptych_3class.png",
            "complexity_delta_plot": "claim1_complexity_source_deltas.png",
            "complexity_delta_plot_3class": "claim1_complexity_source_deltas_3class.png",
            "report_md": "claim1_hard_case_report.md",
        },
    }


def main() -> None:
    args = parse_args()
    step14_root = args.step14_root.resolve()
    empirical_gain = empirical_gain_enabled(args.empirical_gain)

    output_dir = resolve_output_dir_with_suffix(
        args.output_dir.resolve(),
        DEFAULT_OUTPUT_DIR.resolve(),
        empirical_gain,
    )

    hard_labels_path = args.hard_labels.resolve()
    ensure_dir(output_dir)
    setup_plot_style()
    config = BootstrapConfig(n_resamples=2000, seed=20260307, confidence_level=0.95)
    scope_ids = load_scope_ids(step14_root)

    methods = apply_empirical_gain_to_methods(
        load_method_metrics(step14_root),
        empirical_gain=empirical_gain,
    )

    label_map = build_label_map(hard_labels_path)
    severity_subsets, signal_subsets = build_subsets(scope_ids, label_map)

    metric_names = (
        "e2e_status_acc",
        "macro_f1_3class_matched",
        "soft_e2e_f1",
        "step_a_f1",
    )

    severity_summary = compute_metric_summary(
        methods,
        severity_subsets,
        subset_kind="severity",
        metric_names=metric_names,
        config=config,
    )

    signal_summary = compute_metric_summary(
        methods,
        signal_subsets,
        subset_kind="signal",
        metric_names=("e2e_status_acc", "macro_f1_3class_matched"),
        config=config,
    )

    comparisons = (
        ("main_system", "baseline_b2_vanilla_mad", "e2e_status_acc"),
        ("main_system", "baseline_b3_stateful_no_axioms", "e2e_status_acc"),
        ("main_system", "baseline_b1_structured_rag", "e2e_status_acc"),
        ("main_system", "baseline_b2_vanilla_mad", "macro_f1_3class_matched"),
        ("main_system", "baseline_b3_stateful_no_axioms", "macro_f1_3class_matched"),
        ("main_system", "baseline_b1_structured_rag", "macro_f1_3class_matched"),
        ("external_naive_mad", "external_naive_rag", "e2e_status_acc"),
        ("external_naive_mad", "external_pure_one_shot_judge", "e2e_status_acc"),
        ("external_naive_mad", "external_naive_rag", "macro_f1_3class_matched"),
        (
            "external_naive_mad",
            "external_pure_one_shot_judge",
            "macro_f1_3class_matched",
        ),
    )

    pairwise_deltas = pd.concat(
        [
            compute_pairwise_deltas(
                methods,
                severity_subsets,
                subset_kind="severity",
                comparisons=comparisons,
                metric_names=("e2e_status_acc", "macro_f1_3class_matched"),
                config=config,
            ),
            compute_pairwise_deltas(
                methods,
                {
                    key: value
                    for key, value in signal_subsets.items()
                    if key
                    in ("text_len_ge_q75", "law_count_ge_q75", "claim_count_ge_q75")
                },
                subset_kind="signal",
                comparisons=comparisons,
                metric_names=("e2e_status_acc", "macro_f1_3class_matched"),
                config=config,
            ),
        ],
        ignore_index=True,
    )

    subset_counts = build_subset_counts(scope_ids, severity_subsets, signal_subsets)
    severity_summary.to_csv(output_dir / "severity_summary.csv", index=False)
    signal_summary.to_csv(output_dir / "signal_summary.csv", index=False)
    pairwise_deltas.to_csv(output_dir / "pairwise_deltas.csv", index=False)
    subset_counts.to_csv(output_dir / "subset_counts.csv", index=False)

    plot_claim1_overall_hard_very_hard_triptych(
        step14_root,
        severity_summary,
        output_dir / "claim1_overall_hard_very_hard_triptych.png",
        empirical_gain=empirical_gain,
    )

    plot_claim1_overall_hard_very_hard_triptych_3class(
        severity_summary,
        output_dir / "claim1_overall_hard_very_hard_triptych_3class.png",
    )

    plot_complexity_source_deltas(
        pairwise_deltas, output_dir / "claim1_complexity_source_deltas.png"
    )

    plot_complexity_source_deltas_3class(
        pairwise_deltas, output_dir / "claim1_complexity_source_deltas_3class.png"
    )

    write_markdown_report(
        output_dir / "claim1_hard_case_report.md",
        subset_counts,
        severity_summary,
        pairwise_deltas,
    )

    (output_dir / "manifest.json").write_text(
        json.dumps(
            build_manifest(
                output_dir,
                step14_root,
                hard_labels_path,
                empirical_gain_mode=args.empirical_gain,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
