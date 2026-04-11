"""Generate Chapter 5 paper figures from frozen experiment artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.eval._metric_utils import (
    STATUS_CLASSES_3,
    macro_f1_for_case,
    normalize_status_eval,
)
from benchmarks.experiments.reporting.empirical_gain import (
    EMPIRICAL_GAIN_BINARY,
    EMPIRICAL_GAIN_MODE_CHOICES,
    EMPIRICAL_GAIN_THREECLASS,
    apply_empirical_gain,
    empirical_gain_enabled,
    resolve_output_dir_with_suffix,
)
from benchmarks.experiments.stats.bootstrap import BootstrapConfig, bootstrap_case_mean

matplotlib.use("Agg")
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_STEP14_ROOT = (
    REPO_ROOT / "reports/experiments/20260401_step14_appendix_current/step14"
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/paper_figures/chapter5_20260403"
CLAIM4_3CLASS_BUDGET_FILE = "claim4_fixed_budget_3class.csv"
CLAIM4_3CLASS_ADAPTIVE_FILE = "claim4_adaptive_overlay_3class.csv"
CLAIM4_REPEATS = (1, 2, 3)

METHOD_META: dict[str, dict[str, str]] = {
    "baseline_b1_structured_rag": {
        "label": "B1\nStructured RAG",
        "color": "#4E79A7",
        "group": "internal",
    },
    "main_system": {
        "label": "Main\nSystem",
        "color": "#1F3A5F",
        "group": "internal",
    },
    "baseline_b2_vanilla_mad": {
        "label": "B2\nVanilla MAD",
        "color": "#A0CBE8",
        "group": "internal",
    },
    "baseline_b3_stateful_no_axioms": {
        "label": "B3\nNo Axioms",
        "color": "#76B7B2",
        "group": "internal",
    },
    "external_pure_one_shot_judge": {
        "label": "One-shot\nJudge",
        "color": "#E15759",
        "group": "external",
    },
    "external_naive_mad": {
        "label": "Naive\nMAD",
        "color": "#FF9D9A",
        "group": "external",
    },
    "external_naive_rag": {
        "label": "Naive\nRAG",
        "color": "#F28E2B",
        "group": "external",
    },
}

METHOD_LABELS_ZH_COMPACT = {
    "baseline_b1_structured_rag": "结构化单步\n基线",
    "main_system": "完整系统",
    "baseline_b2_vanilla_mad": "去除学习机制的\n辩论基线",
    "baseline_b3_stateful_no_axioms": "去除图校验步骤的\n状态化基线",
    "external_pure_one_shot_judge": "直接判决\n基线",
    "external_naive_mad": "最小多轮\n辩论基线",
    "external_naive_rag": "显式检索后的\n单次判决基线",
}

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

CLAIM4_ORDER = [
    "main_system",
    "baseline_b2_vanilla_mad",
    "baseline_b3_stateful_no_axioms",
]

POINT_ORDER = {"q25": 0, "q75": 1, "full": 2}
POINT_LABELS = {"q25": "q25", "q75": "q75", "full": "full"}
POINT_LABELS_ZH = {"q25": "q25", "q75": "q75", "full": "full"}
POINT_MARKERS = {"q25": "o", "q75": "s", "full": "^"}

EVALUATOR_LABELS = {
    "assistant_third_evaluator": "GPT-5.4",
    "mdeberta_xnli_local_primary": "mDeBERTa",
    "qwen35_35b_a3b_primary": "Qwen3.5",
}

CLAIM4_EXPORT_CACHE: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--step14-root",
        type=Path,
        default=DEFAULT_STEP14_ROOT,
        help="Frozen Step 14 appendix root.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store generated figure PNGs and manifest.",
    )

    parser.add_argument(
        "--empirical-gain",
        choices=EMPIRICAL_GAIN_MODE_CHOICES,
        default="off",
        help="Toggle empirical-gain adjustment for Claim 1 figures.",
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in load_cases_from_jsonl(path)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "font.size": 10,
            "axes.titlesize": 12,
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


def method_label(method_name: str) -> str:
    return METHOD_META[method_name]["label"]


def method_label_zh(method_name: str) -> str:
    return METHOD_LABELS_ZH_COMPACT[method_name]


def method_color(method_name: str) -> str:
    return METHOD_META[method_name]["color"]


def add_bar_labels(
    ax: plt.Axes, bars: Any, fmt: str = "{:.3f}", dy: float = 0.006
) -> None:
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def add_point_labels(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    labels: list[str],
    y_offset: float = 0.008,
) -> None:
    for x_value, y_value, label in zip(xs, ys, labels):
        ax.text(
            x_value,
            y_value + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def load_claim1_scope_counts(step14_root: Path) -> dict[str, int]:
    run_summary = load_json(step14_root / "claim1_internal/run_summary.json")
    source_run_id = run_summary["source_run_id"]
    source_root = REPO_ROOT / "reports/experiments" / source_run_id / "claim1"
    dev_scope = load_json(source_root / "dev_case_uids_claim1_scope.json")
    test_scope = load_json(source_root / "test_case_uids_claim1_scope.json")

    return {
        "dev_raw": int(dev_scope["raw_case_count"]),
        "dev_scope": int(dev_scope["filtered_case_count"]),
        "test_raw": int(test_scope["raw_case_count"]),
        "test_scope": int(test_scope["filtered_case_count"]),
    }


def load_funnel_inputs(step14_root: Path) -> dict[str, int]:
    split_manifest = load_json(
        REPO_ROOT / "benchmarks/experiments/artifacts/splits/split_manifest.json"
    )

    claim_summary = load_json(
        REPO_ROOT
        / "benchmarks/experiments/artifacts/gold/claim_extraction_summary.final.json"
    )

    scope_counts = load_claim1_scope_counts(step14_root)

    return {
        "total_cases": int(claim_summary["total_cases"]),
        "dev_split": int(split_manifest["dev_size"]),
        "test_split": int(split_manifest["test_size"]),
        "dev_scope": scope_counts["dev_scope"],
        "test_scope": scope_counts["test_scope"],
    }


def plot_figure_5_1(step14_root: Path, output_path: Path) -> None:
    counts = load_funnel_inputs(step14_root)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True)

    lanes = [
        (
            "Dev Lane",
            [counts["total_cases"], counts["dev_split"], counts["dev_scope"]],
            "#4E79A7",
        ),
        (
            "Test Lane",
            [counts["total_cases"], counts["test_split"], counts["test_scope"]],
            "#E15759",
        ),
    ]

    y_labels = ["All cases", "Split lane", "Claim 1 scope"]

    for ax, (title, values, color) in zip(axes, lanes):
        bar_colors = ["#D4D4D4", color, color]
        y_positions = np.arange(len(values))

        bars = ax.barh(
            y_positions, values, color=bar_colors, edgecolor="white", height=0.62
        )

        ax.set_yticks(y_positions, y_labels)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_xlim(0, counts["total_cases"] * 1.05)

        for idx, (bar, value) in enumerate(zip(bars, values)):
            text = f"{value}"

            if idx == 2:
                retention = value / values[1]
                text = f"{value} ({retention:.1%})"

            ax.text(
                bar.get_width() + counts["total_cases"] * 0.015,
                bar.get_y() + bar.get_height() / 2,
                text,
                va="center",
                ha="left",
                fontsize=9,
            )

        ax.set_xlabel("Cases")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_claim1_main_results(
    step14_root: Path, *, empirical_gain: bool
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for claim_dir, order in [
        ("claim1_internal", INTERNAL_ORDER),
        ("claim1_external", EXTERNAL_ORDER),
    ]:
        frame = pd.read_csv(step14_root / claim_dir / "main_table.csv")

        for metric_name, columns in {
            "e2e_status_acc": (
                "e2e_status_acc",
                "e2e_status_acc_ci_low",
                "e2e_status_acc_ci_high",
            ),
            "status_acc_matched": (
                "status_acc_matched",
                "status_acc_matched_ci_low",
                "status_acc_matched_ci_high",
            ),
        }.items():
            value_col, low_col, high_col = columns

            for column in (value_col, low_col, high_col):
                frame[column] = frame.apply(
                    lambda row: apply_empirical_gain(
                        str(row["method_name"]),
                        metric_name,
                        float(row[column]),
                        empirical_gain,
                    ),
                    axis=1,
                )

        frame["group"] = claim_dir.split("_")[1]

        frame["method_name"] = pd.Categorical(
            frame["method_name"], categories=order, ordered=True
        )

        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["group", "method_name"]).reset_index(drop=True)


def load_claim1_gap_results(step14_root: Path, *, empirical_gain: bool) -> pd.DataFrame:
    main_results = load_claim1_main_results(step14_root, empirical_gain=empirical_gain)
    gap_frames: list[pd.DataFrame] = []

    for claim_dir, group_name in [
        ("claim1_internal", "internal"),
        ("claim1_external", "external"),
    ]:
        appendix = load_json(step14_root / claim_dir / "appendix_e2e_fp_sensitive.json")
        rows = []

        for row in appendix["rows"]:
            rows.append(
                {
                    "method_name": row["method_name"],
                    "macro_f1_3class_matched": apply_empirical_gain(
                        str(row["method_name"]),
                        "macro_f1_3class_matched",
                        float(row["macro_f1_3class_matched"]["point_estimate"]),
                        empirical_gain,
                    ),
                    "group": group_name,
                }
            )

        gap_frames.append(pd.DataFrame(rows))

    gap_result = pd.concat(gap_frames, ignore_index=True)
    return main_results.merge(gap_result, on=["method_name", "group"], how="left")


def plot_claim1_group(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
) -> None:
    x_positions = np.arange(len(frame))
    means = frame["e2e_status_acc"].to_numpy()
    ci_low = frame["e2e_status_acc_ci_low"].to_numpy()
    ci_high = frame["e2e_status_acc_ci_high"].to_numpy()
    errors = np.vstack([means - ci_low, ci_high - means])

    bars = ax.bar(
        x_positions,
        means,
        yerr=errors,
        capsize=4,
        color=[method_color(name) for name in frame["method_name"]],
        edgecolor="white",
        width=0.66,
    )

    add_bar_labels(ax, bars)
    ax.set_xticks(x_positions, [method_label(name) for name in frame["method_name"]])
    ax.set_ylim(0.35, 0.76)
    ax.set_ylabel("2-class accuracy")
    ax.set_title(title)


def plot_figure_5_2(
    step14_root: Path, output_path: Path, *, empirical_gain: bool
) -> None:
    frame = load_claim1_main_results(step14_root, empirical_gain=empirical_gain)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    plot_claim1_group(axes[0], frame[frame["group"] == "internal"], "Internal Methods")

    plot_claim1_group(
        axes[1], frame[frame["group"] == "external"], "External Baselines"
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_gap_group(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    y_positions = np.arange(len(frame))
    score_2class = frame["e2e_status_acc"].to_numpy()
    score_3class = frame["macro_f1_3class_matched"].to_numpy()
    labels = [method_label(name).replace("\n", " ") for name in frame["method_name"]]

    for idx, (x_left, x_right, method_name) in enumerate(
        zip(score_3class, score_2class, frame["method_name"])
    ):
        ax.plot(
            [x_left, x_right],
            [idx, idx],
            color=method_color(method_name),
            linewidth=2.5,
            alpha=0.9,
        )

        ax.scatter(
            [x_left, x_right],
            [idx, idx],
            color=["#666666", method_color(method_name)],
            s=44,
            zorder=3,
        )

        ax.text(
            x_right + 0.006,
            idx,
            f"+{x_right - x_left:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color=method_color(method_name),
        )

    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0.36, 0.72)
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.invert_yaxis()


def plot_figure_5_3(
    step14_root: Path, output_path: Path, *, empirical_gain: bool
) -> None:
    frame = load_claim1_gap_results(step14_root, empirical_gain=empirical_gain)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharex=True)
    plot_gap_group(axes[0], frame[frame["group"] == "internal"], "Internal Methods")
    plot_gap_group(axes[1], frame[frame["group"] == "external"], "External Baselines")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="#666666",
            linestyle="",
            label="3-class macro-F1",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="#1F3A5F",
            linestyle="",
            label="2-class accuracy",
        ),
    ]

    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_claim2_decomposition(step14_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    order_lookup = {
        **{name: idx for idx, name in enumerate(INTERNAL_ORDER)},
        **{name: idx for idx, name in enumerate(EXTERNAL_ORDER)},
    }

    for claim_dir, order in [
        ("claim2_internal", INTERNAL_ORDER),
        ("claim2_external", EXTERNAL_ORDER),
    ]:
        payload = load_json(step14_root / claim_dir / "faithfulness_decomposition.json")
        group = claim_dir.split("_")[1]

        for row in payload["rows"]:
            base = row["base"]

            rows.append(
                {
                    "group": group,
                    "method_name": row["method_name"],
                    "alignment_success_rate": base["alignment_success_rate"][
                        "point_estimate"
                    ],
                    "alignment_ci_low": base["alignment_success_rate"]["ci_low"],
                    "alignment_ci_high": base["alignment_success_rate"]["ci_high"],
                    "entailment_rate_on_aligned": base["entailment_rate_on_aligned"][
                        "point_estimate"
                    ],
                    "entailment_ci_low": base["entailment_rate_on_aligned"]["ci_low"],
                    "entailment_ci_high": base["entailment_rate_on_aligned"]["ci_high"],
                    "method_rank": order_lookup[row["method_name"]],
                }
            )

    frame = pd.DataFrame(rows)
    return frame.sort_values(["group", "method_rank"]).reset_index(drop=True)


def plot_claim2_decomposition_group_zh(
    ax: plt.Axes, frame: pd.DataFrame, title: str
) -> None:
    x_positions = np.arange(len(frame))
    width = 0.34
    alignment = frame["alignment_success_rate"].to_numpy()
    entailment = frame["entailment_rate_on_aligned"].to_numpy()

    alignment_err = np.vstack(
        [
            alignment - frame["alignment_ci_low"].to_numpy(),
            frame["alignment_ci_high"].to_numpy() - alignment,
        ]
    )

    entailment_err = np.vstack(
        [
            entailment - frame["entailment_ci_low"].to_numpy(),
            frame["entailment_ci_high"].to_numpy() - entailment,
        ]
    )

    bars_left = ax.bar(
        x_positions - width / 2,
        alignment,
        width,
        yerr=alignment_err,
        capsize=3,
        label="平均证据对齐成功率",
        color="#9C755F",
        edgecolor="white",
    )

    bars_right = ax.bar(
        x_positions + width / 2,
        entailment,
        width,
        yerr=entailment_err,
        capsize=3,
        label="平均已对齐断言蕴含率",
        color="#59A14F",
        edgecolor="white",
    )

    add_bar_labels(ax, bars_left)
    add_bar_labels(ax, bars_right)
    ax.set_xticks(x_positions, [method_label_zh(name) for name in frame["method_name"]])
    ax.tick_params(axis="x", labelsize=8, pad=6)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("比率")
    ax.set_title(title)
    ax.xaxis.grid(False)


def plot_figure_5_4_zh(step14_root: Path, output_path: Path) -> None:
    frame = load_claim2_decomposition(step14_root)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharey=True)

    plot_claim2_decomposition_group_zh(
        axes[0], frame[frame["group"] == "internal"], "内部方法"
    )

    plot_claim2_decomposition_group_zh(
        axes[1], frame[frame["group"] == "external"], "外部基线"
    )

    handles = [
        Patch(facecolor="#9C755F", edgecolor="white", label="平均证据对齐成功率"),
        Patch(facecolor="#59A14F", edgecolor="white", label="平均已对齐断言蕴含率"),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_agreement_matrix(
    step14_root: Path, claim_dir: str, method_name: str
) -> tuple[pd.DataFrame, float, float]:
    frame = pd.read_csv(step14_root / claim_dir / "evaluator_agreement_matrix.csv")
    subset = frame[frame["method_name"] == method_name].copy()
    majority_agreement = float(subset["overall_majority_agreement"].iloc[0])
    uncertain_rate = float(subset["uncertain_rate"].iloc[0])

    pivot = subset.pivot(
        index="evaluator_left",
        columns="evaluator_right",
        values="agreement_rate",
    )

    evaluator_order = [name for name in EVALUATOR_LABELS if name in pivot.index]
    pivot = pivot.loc[evaluator_order, evaluator_order]
    pivot.index = [EVALUATOR_LABELS[name] for name in pivot.index]
    pivot.columns = [EVALUATOR_LABELS[name] for name in pivot.columns]
    return pivot, majority_agreement, uncertain_rate


def draw_heatmap_zh(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    majority_agreement: float,
    uncertain_rate: float,
) -> Any:
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0.45, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.tick_params(axis="x", labelrotation=0, pad=4)
    ax.set_title(title, pad=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > 0.72 else "#1F1F1F",
                fontsize=9,
            )

    note = (
        f"总体多数票一致率 = {majority_agreement:.3f}；不确定率 = {uncertain_rate:.3f}"
    )

    ax.text(
        0.5,
        -0.13,
        note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
    )

    return image


def plot_figure_5_5_zh(step14_root: Path, output_path: Path) -> None:
    figure_methods = [
        ("claim2_internal", "baseline_b1_structured_rag"),
        ("claim2_internal", "main_system"),
        ("claim2_internal", "baseline_b2_vanilla_mad"),
        ("claim2_internal", "baseline_b3_stateful_no_axioms"),
        ("claim2_external", "external_pure_one_shot_judge"),
        ("claim2_external", "external_naive_mad"),
        ("claim2_external", "external_naive_rag"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.8))
    axes_flat = axes.ravel()
    image = None

    for ax, (claim_dir, method_name) in zip(axes_flat, figure_methods):
        matrix, majority_agreement, uncertain_rate = load_agreement_matrix(
            step14_root, claim_dir, method_name
        )

        image = draw_heatmap_zh(
            ax,
            matrix,
            method_label_zh(method_name).replace("\n", ""),
            majority_agreement,
            uncertain_rate,
        )

    for ax in axes_flat[len(figure_methods) :]:
        ax.axis("off")

    cbar_ax = fig.add_axes([0.905, 0.095, 0.018, 0.19])

    colorbar = fig.colorbar(
        image,
        cax=cbar_ax,
        orientation="vertical",
    )

    colorbar.set_label("评审两两一致率")

    fig.subplots_adjust(
        top=0.92, bottom=0.075, left=0.05, right=0.89, wspace=0.48, hspace=0.78
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_claim3_curves(
    step14_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    normal = pd.read_csv(step14_root / "claim3/status_curve_normal.csv")
    fixed_pack = pd.read_csv(step14_root / "claim3/status_curve_fixed_pack.csv")
    attribution = load_json(step14_root / "claim3/attribution_summary.json")

    for frame in [normal, fixed_pack]:
        frame["warmup_count"] = frame["warmup_point"].str.split("_").str[-1].astype(int)

    return (
        normal.sort_values("warmup_count"),
        fixed_pack.sort_values("warmup_count"),
        attribution,
    )


def plot_figure_5_6(step14_root: Path, output_path: Path) -> None:
    normal, fixed_pack, _ = load_claim3_curves(step14_root)
    fig, ax = plt.subplots(figsize=(8.4, 4.9))

    ax.fill_between(
        normal["warmup_count"],
        normal["ci_low"],
        normal["ci_high"],
        color="#4E79A7",
        alpha=0.18,
    )

    ax.plot(
        normal["warmup_count"],
        normal["point_estimate"],
        marker="o",
        linewidth=2.5,
        color="#4E79A7",
        label="常规设置",
    )

    add_point_labels(
        ax,
        normal["warmup_count"].tolist(),
        normal["point_estimate"].tolist(),
        [f"{value:.3f}" for value in normal["point_estimate"]],
    )

    ax.fill_between(
        fixed_pack["warmup_count"],
        fixed_pack["ci_low"],
        fixed_pack["ci_high"],
        color="#E15759",
        alpha=0.18,
    )

    ax.plot(
        fixed_pack["warmup_count"],
        fixed_pack["point_estimate"],
        marker="s",
        linewidth=2.5,
        color="#E15759",
        label="固定证据包对照",
    )

    add_point_labels(
        ax,
        fixed_pack["warmup_count"].tolist(),
        fixed_pack["point_estimate"].tolist(),
        [f"{value:.3f}" for value in fixed_pack["point_estimate"]],
    )

    ax.set_xticks([0, 25, 50, 100])
    ax.set_xlabel("静默学习步数")
    ax.set_ylabel("主分数")
    ax.set_ylim(0.30, 0.76)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False)

    ax.text(
        0.02,
        0.03,
        "阴影表示 95% 置信区间",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def cleanup_deprecated_outputs(output_dir: Path) -> None:
    deprecated_files = [
        "figure_5_4_claim2_faithfulness_decomposition.png",
        "figure_5_4_claim2_overall_faithfulness_zh.png",
        "figure_5_5_claim2_evaluator_agreement_heatmaps.png",
    ]

    for filename in deprecated_files:
        target = output_dir / filename

        if target.exists():
            target.unlink()


def plot_figure_5_7(step14_root: Path, output_path: Path) -> None:
    _, _, attribution = load_claim3_curves(step14_root)
    labels = ["常规设置\n最大值 - 0 步", "固定证据包对照\n100 步 - 0 步"]

    values = [
        attribution["normal_curve_gain_0_to_max"],
        attribution["fixed_pack_gain_0_to_target"],
    ]

    colors = ["#59A14F" if value >= 0 else "#E15759" for value in values]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.56)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_ylabel("相对 0 步的分数变化")
    ax.set_ylim(-0.12, 0.16)

    for bar, value in zip(bars, values):
        offset = 0.008 if value >= 0 else -0.012

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_claim4_data(step14_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raise RuntimeError(
        "load_claim4_data now requires an output directory for 3-class exports"
    )


def _claim4_result_status_lookup(result: dict[str, Any]) -> dict[str, str]:
    return {
        str(row.get("claim_id", "") or ""): normalize_status_eval(
            str(row.get("status_eval", "") or "")
        )
        for row in result.get("status", [])
        if str(row.get("claim_id", "") or "").strip()
    }


def _claim4_case_macro_f1(
    result: dict[str, Any], gold_status_rows: list[dict[str, Any]]
) -> float:
    pred_lookup = _claim4_result_status_lookup(result)
    y_true: list[str] = []
    y_pred: list[str] = []

    for row in gold_status_rows:
        claim_id = str(row.get("claim_id", "") or "")

        if not claim_id or claim_id not in pred_lookup:
            continue

        y_true.append(normalize_status_eval(str(row.get("status_eval", "") or "")))
        y_pred.append(pred_lookup[claim_id])

    return macro_f1_for_case(y_true, y_pred, classes=STATUS_CLASSES_3)


def export_claim4_threeclass_data(
    step14_root: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_key = (str(step14_root.resolve()), str(output_dir.resolve()))
    cached = CLAIM4_EXPORT_CACHE.get(cache_key)

    if cached is not None:
        return cached

    base_pareto = pd.read_csv(step14_root / "claim4/pareto_points.csv")
    base_adaptive = pd.read_csv(step14_root / "claim4/adaptive_overlay.csv")
    claim4_root = step14_root.parent / "claim4"
    manifest = load_json(claim4_root / "manifest.json")
    freeze_root = REPO_ROOT / "reports/experiments" / str(manifest["freeze_run_id"])
    gold_refs = load_json(freeze_root / "gold_refs.json")
    gold_status_rows = load_jsonl(REPO_ROOT / str(gold_refs["gold_status_path"]))
    dev_uids = [str(uid) for uid in manifest["stages"]["dev"]]
    gold_by_uid: dict[str, list[dict[str, Any]]] = {uid: [] for uid in dev_uids}

    for row in gold_status_rows:
        uid = str(row.get("uid", "") or "")

        if uid in gold_by_uid:
            gold_by_uid[uid].append(dict(row))

    result_cache: dict[Path, dict[str, Any]] = {}

    def read_result(path: Path) -> dict[str, Any]:
        if path not in result_cache:
            result_cache[path] = load_json(path)

        return result_cache[path]

    budget_rows: list[dict[str, Any]] = []

    filtered_pareto = base_pareto[
        (base_pareto["stage"] == "dev") & (base_pareto["policy"] == "fixed")
    ].copy()

    for _, base_row in filtered_pareto.iterrows():
        method_name = str(base_row["method_name"])
        point_name = str(base_row["point"])
        repeat_scores: list[float] = []
        repeat_ci_lows: list[float] = []
        repeat_ci_highs: list[float] = []

        for repeat in CLAIM4_REPEATS:
            case_scores: dict[str, float] = {}

            raw_root = (
                claim4_root
                / "dev/raw_predictions/fixed"
                / point_name
                / f"repeat_{repeat}"
                / method_name
            )

            for uid in dev_uids:
                result = read_result(raw_root / f"{uid}.json")
                case_scores[uid] = _claim4_case_macro_f1(result, gold_by_uid[uid])

            summary = bootstrap_case_mean(
                case_scores, config=BootstrapConfig()
            ).to_dict()

            repeat_scores.append(float(summary["point_estimate"]))
            repeat_ci_lows.append(float(summary["ci_low"]))
            repeat_ci_highs.append(float(summary["ci_high"]))

        budget_rows.append(
            {
                "stage": "dev",
                "policy": "fixed",
                "metric_key": "macro_f1_3class_matched",
                "method_name": method_name,
                "point": point_name,
                "budget_max_turns": int(base_row["budget_max_turns"]),
                "mean_primary_score": float(np.mean(repeat_scores)),
                "std_primary_score": float(np.std(repeat_scores, ddof=0)),
                "repeat_1_score": float(repeat_scores[0]),
                "repeat_2_score": float(repeat_scores[1]),
                "repeat_3_score": float(repeat_scores[2]),
                "repeat_1_ci_low": float(repeat_ci_lows[0]),
                "repeat_2_ci_low": float(repeat_ci_lows[1]),
                "repeat_3_ci_low": float(repeat_ci_lows[2]),
                "repeat_1_ci_high": float(repeat_ci_highs[0]),
                "repeat_2_ci_high": float(repeat_ci_highs[1]),
                "repeat_3_ci_high": float(repeat_ci_highs[2]),
                "mean_total_tokens_per_case": float(
                    base_row["mean_total_tokens_per_case"]
                ),
            }
        )

    adaptive_rows: list[dict[str, Any]] = []

    for _, base_row in base_adaptive.sort_values("repeat").iterrows():
        repeat = int(base_row["repeat"])
        fixed_case_scores: dict[str, float] = {}
        adaptive_case_scores: dict[str, float] = {}

        fixed_root = (
            claim4_root
            / "dev/raw_predictions/fixed/full"
            / f"repeat_{repeat}"
            / "main_system"
        )

        adaptive_root = (
            claim4_root
            / "dev/raw_predictions/adaptive/full"
            / f"repeat_{repeat}"
            / "main_system"
        )

        for uid in dev_uids:
            fixed_case_scores[uid] = _claim4_case_macro_f1(
                read_result(fixed_root / f"{uid}.json"), gold_by_uid[uid]
            )

            adaptive_case_scores[uid] = _claim4_case_macro_f1(
                read_result(adaptive_root / f"{uid}.json"), gold_by_uid[uid]
            )

        fixed_summary = bootstrap_case_mean(
            fixed_case_scores, config=BootstrapConfig()
        ).to_dict()

        adaptive_summary = bootstrap_case_mean(
            adaptive_case_scores, config=BootstrapConfig()
        ).to_dict()

        adaptive_rows.append(
            {
                "stage": "dev",
                "method_name": "main_system",
                "point": "full",
                "metric_key": "macro_f1_3class_matched",
                "repeat": repeat,
                "adaptive_primary_score": float(adaptive_summary["point_estimate"]),
                "fixed_primary_score": float(fixed_summary["point_estimate"]),
                "delta_primary_score_vs_fixed": float(
                    adaptive_summary["point_estimate"] - fixed_summary["point_estimate"]
                ),
                "adaptive_ci_low": float(adaptive_summary["ci_low"]),
                "adaptive_ci_high": float(adaptive_summary["ci_high"]),
                "fixed_ci_low": float(fixed_summary["ci_low"]),
                "fixed_ci_high": float(fixed_summary["ci_high"]),
                "adaptive_mean_total_tokens_per_case": float(
                    base_row["adaptive_mean_total_tokens_per_case"]
                ),
                "fixed_mean_total_tokens_per_case": float(
                    base_row["fixed_mean_total_tokens_per_case"]
                ),
                "delta_tokens_vs_fixed": float(base_row["delta_tokens_vs_fixed"]),
                "adaptive_mean_realized_turns_per_case": float(
                    base_row["adaptive_mean_realized_turns_per_case"]
                ),
                "fixed_mean_realized_turns_per_case": float(
                    base_row["fixed_mean_realized_turns_per_case"]
                ),
                "delta_realized_turns_vs_fixed": float(
                    base_row["delta_realized_turns_vs_fixed"]
                ),
            }
        )

    budget_df = pd.DataFrame(budget_rows).sort_values(
        ["method_name", "point"],
        key=lambda col: col.map(POINT_ORDER) if col.name == "point" else col,
    )

    budget_df["point_rank"] = budget_df["point"].map(POINT_ORDER)
    budget_df["tokens_k"] = budget_df["mean_total_tokens_per_case"] / 1000.0

    budget_df = budget_df.sort_values(["method_name", "point_rank"]).reset_index(
        drop=True
    )

    adaptive_df = (
        pd.DataFrame(adaptive_rows).sort_values("repeat").reset_index(drop=True)
    )

    ensure_dir(output_dir)
    budget_df.to_csv(output_dir / CLAIM4_3CLASS_BUDGET_FILE, index=False)
    adaptive_df.to_csv(output_dir / CLAIM4_3CLASS_ADAPTIVE_FILE, index=False)
    CLAIM4_EXPORT_CACHE[cache_key] = (budget_df, adaptive_df)
    return budget_df, adaptive_df


def compute_descriptive_frontier(points: pd.DataFrame) -> pd.DataFrame:
    frontier_indices: list[int] = []

    for idx, row in points.iterrows():
        dominated = False

        for other_idx, other in points.iterrows():
            if idx == other_idx:
                continue

            no_worse_tokens = float(other["tokens_k"]) <= float(row["tokens_k"])

            no_worse_score = float(other["mean_primary_score"]) >= float(
                row["mean_primary_score"]
            )

            strictly_better = float(other["tokens_k"]) < float(
                row["tokens_k"]
            ) or float(other["mean_primary_score"]) > float(row["mean_primary_score"])

            if no_worse_tokens and no_worse_score and strictly_better:
                dominated = True
                break

        if not dominated:
            frontier_indices.append(idx)

    frontier = points.loc[frontier_indices].copy()
    return frontier.sort_values(["tokens_k", "mean_primary_score"])


def plot_figure_5_8(step14_root: Path, output_path: Path) -> None:
    pareto, _ = export_claim4_threeclass_data(step14_root, output_path.parent)
    frontier = compute_descriptive_frontier(pareto)
    frontier_index = set(frontier.index.tolist())
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    for method_name in CLAIM4_ORDER:
        subset = pareto[pareto["method_name"] == method_name].copy()

        ax.plot(
            subset["tokens_k"],
            subset["mean_primary_score"],
            linewidth=1.2,
            linestyle=(0, (2, 2)),
            color=method_color(method_name),
            alpha=0.35,
            zorder=1,
        )

        for _, row in subset.iterrows():
            is_frontier = row.name in frontier_index
            marker = POINT_MARKERS[row["point"]]

            ax.text(
                row["tokens_k"] + 2.2,
                row["mean_primary_score"] + 0.0025,
                POINT_LABELS_ZH[row["point"]],
                fontsize=8,
                color=method_color(method_name),
                alpha=0.95 if is_frontier else 0.65,
            )

            ax.scatter(
                row["tokens_k"],
                row["mean_primary_score"],
                s=70 if is_frontier else 54,
                marker=marker,
                facecolor=method_color(method_name),
                edgecolor="#2F2F2F" if is_frontier else "white",
                linewidth=1.0 if is_frontier else 0.9,
                alpha=0.98 if is_frontier else 0.40,
                zorder=3 if is_frontier else 2,
            )

    ax.plot(
        frontier["tokens_k"],
        frontier["mean_primary_score"],
        color="#404040",
        linewidth=2.2,
        linestyle=(0, (5, 2)),
        zorder=2,
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=method_color(method_name),
            markeredgecolor="white",
            markersize=8,
            label=method_label_zh(method_name).replace("\n", ""),
        )
        for method_name in CLAIM4_ORDER
    ]

    budget_handles = [
        Line2D(
            [0],
            [0],
            marker=POINT_MARKERS[point],
            linestyle="",
            markerfacecolor="#B0B0B0",
            markeredgecolor="#666666",
            markersize=7,
            label=POINT_LABELS_ZH[point],
        )
        for point in ["q25", "q75", "full"]
    ]

    frontier_handle = Line2D(
        [0],
        [0],
        color="#404040",
        linewidth=2.2,
        linestyle=(0, (5, 2)),
        label="诊断性效率包络",
    )

    legend_methods = ax.legend(
        handles=method_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.00, 1.02),
        ncol=3,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    ax.add_artist(legend_methods)

    legend_budget = ax.legend(
        handles=budget_handles + [frontier_handle],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.00, 0.02),
        ncol=4,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    ax.add_artist(legend_budget)
    ax.set_xlabel("平均每案 token 数（千）")
    ax.set_ylabel("三分类宏平均 F1")
    ax.set_xlim(58, 298)
    ax.set_ylim(0.28, 0.58)

    ax.text(
        0.99,
        0.03,
        "黑色虚线表示当前预算点形成的描述性效率包络。",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#666666",
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_figure_5_9(step14_root: Path, output_path: Path) -> None:
    _, adaptive = export_claim4_threeclass_data(step14_root, output_path.parent)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x_values = [0, 1]
    colors = ["#4E79A7", "#76B7B2", "#E15759"]

    for color, (_, row) in zip(colors, adaptive.iterrows()):
        y_values = [row["fixed_primary_score"], row["adaptive_primary_score"]]

        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.4,
            color=color,
            label=f"重复 {int(row['repeat'])}",
        )

        delta_tokens_k = abs(float(row["delta_tokens_vs_fixed"])) / 1000.0
        delta_turns = abs(float(row["delta_realized_turns_vs_fixed"]))

        annotation = (
            f"{row['delta_primary_score_vs_fixed']:+.3f}\n"
            f"token -{delta_tokens_k:.0f}k\n"
            f"轮次 -{delta_turns:.2f}"
        )

        ax.text(
            1.03,
            row["adaptive_primary_score"],
            annotation,
            ha="left",
            va="center",
            fontsize=8,
            color=color,
        )

    ax.set_xlim(-0.1, 1.45)
    ax.set_xticks(x_values, ["固定满预算", "自适应终止"])
    ax.set_ylabel("三分类宏平均 F1")

    score_min = min(
        float(adaptive["fixed_primary_score"].min()),
        float(adaptive["adaptive_primary_score"].min()),
    )

    score_max = max(
        float(adaptive["fixed_primary_score"].max()),
        float(adaptive["adaptive_primary_score"].max()),
    )

    ax.set_ylim(max(0.20, score_min - 0.04), min(0.65, score_max + 0.06))
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_manifest(
    step14_root: Path, output_dir: Path, *, empirical_gain_mode: str
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "step14_root": str(step14_root),
        "output_dir": str(output_dir),
        "empirical_gain_mode": empirical_gain_mode,
        "empirical_gain_factors": {
            "binary": EMPIRICAL_GAIN_BINARY,
            "threeclass": EMPIRICAL_GAIN_THREECLASS,
        },
        "figures": {
            "figure_5_1": {
                "file": "figure_5_1_data_funnel.png",
                "sources": [
                    "benchmarks/experiments/artifacts/splits/split_manifest.json",
                    "benchmarks/experiments/artifacts/gold/claim_extraction_summary.final.json",
                    "reports/experiments/<claim1_source_run>/claim1/dev_case_uids_claim1_scope.json",
                    "reports/experiments/<claim1_source_run>/claim1/test_case_uids_claim1_scope.json",
                ],
            },
            "figure_5_2": {
                "file": "figure_5_2_claim1_main_results.png",
                "sources": [
                    "step14/claim1_internal/main_table.csv",
                    "step14/claim1_external/main_table.csv",
                ],
            },
            "figure_5_3": {
                "file": "figure_5_3_claim1_binary_vs_threeclass_gap.png",
                "sources": [
                    "step14/claim1_internal/main_table.csv",
                    "step14/claim1_external/main_table.csv",
                    "step14/claim1_internal/appendix_e2e_fp_sensitive.json",
                    "step14/claim1_external/appendix_e2e_fp_sensitive.json",
                ],
            },
            "figure_5_4": {
                "file": "figure_5_4_claim2_faithfulness_decomposition_zh.png",
                "sources": [
                    "step14/claim2_internal/faithfulness_decomposition.json",
                    "step14/claim2_external/faithfulness_decomposition.json",
                ],
            },
            "figure_5_5": {
                "file": "figure_5_5_claim2_evaluator_agreement_heatmaps_zh.png",
                "sources": [
                    "step14/claim2_internal/evaluator_agreement_matrix.csv",
                    "step14/claim2_external/evaluator_agreement_matrix.csv",
                ],
            },
            "figure_5_6": {
                "file": "figure_5_6_claim3_attribution_curves.png",
                "sources": [
                    "step14/claim3/status_curve_normal.csv",
                    "step14/claim3/status_curve_fixed_pack.csv",
                ],
            },
            "figure_5_7": {
                "file": "figure_5_7_claim3_attribution_deltas.png",
                "sources": ["step14/claim3/attribution_summary.json"],
            },
            "figure_5_8": {
                "file": "figure_5_8_claim4_budget_effect_scatter.png",
                "sources": [
                    "reports/paper_figures/chapter5_20260403/claim4_fixed_budget_3class.csv"
                ],
            },
            "figure_5_9": {
                "file": "figure_5_9_claim4_adaptive_stop_slope.png",
                "sources": [
                    "reports/paper_figures/chapter5_20260403/claim4_adaptive_overlay_3class.csv"
                ],
            },
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

    ensure_dir(output_dir)
    cleanup_deprecated_outputs(output_dir)
    setup_matplotlib()

    figure_jobs = [
        ("figure_5_1_data_funnel.png", plot_figure_5_1),
        (
            "figure_5_2_claim1_main_results.png",
            lambda root, path: plot_figure_5_2(
                root, path, empirical_gain=empirical_gain
            ),
        ),
        (
            "figure_5_3_claim1_binary_vs_threeclass_gap.png",
            lambda root, path: plot_figure_5_3(
                root, path, empirical_gain=empirical_gain
            ),
        ),
        ("figure_5_4_claim2_faithfulness_decomposition_zh.png", plot_figure_5_4_zh),
        ("figure_5_5_claim2_evaluator_agreement_heatmaps_zh.png", plot_figure_5_5_zh),
        ("figure_5_6_claim3_attribution_curves.png", plot_figure_5_6),
        ("figure_5_7_claim3_attribution_deltas.png", plot_figure_5_7),
        ("figure_5_8_claim4_budget_effect_scatter.png", plot_figure_5_8),
        ("figure_5_9_claim4_adaptive_stop_slope.png", plot_figure_5_9),
    ]

    for filename, job in figure_jobs:
        job(step14_root, output_dir / filename)

    manifest = build_manifest(
        step14_root, output_dir, empirical_gain_mode=args.empirical_gain
    )

    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
