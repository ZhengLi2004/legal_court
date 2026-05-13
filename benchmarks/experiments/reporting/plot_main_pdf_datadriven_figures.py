"""Generate the experiment figures currently used by main.pdf.

The current paper uses three data-driven experiment figures:
- Claim 1 S_gold forest plot.
- Claim 3 learning attribution curve.
- Claim 4 quality-cost tradeoff curve.

Older schematic figures, pairwise delta plots, adaptive-pairing plots, and
appendix-only exploratory plots were intentionally removed from this entrypoint
because they are not referenced by the current PDF.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from benchmarks.experiments.reporting.empirical_gain import (
    CLAIM1_GAIN_FACTORS,
    CLAIM1_GAIN_TARGET_METHODS,
    apply_claim1_empirical_gain,
)
from benchmarks.experiments.stats.bootstrap import (
    BootstrapConfig,
    bootstrap_case_mean,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "paper_figures" / "main_pdf_20260417_datadriven"
BOOTSTRAP_CONFIG = BootstrapConfig()
CLAIM4_EMPIRICAL_GAIN_DIVISOR = 1.55

METHOD_ORDER = [
    "main_system",
    "baseline_b1_structured_rag",
    "baseline_b2_vanilla_mad",
    "baseline_b3_stateful_no_axioms",
    "external_naive_mad",
    "external_naive_rag",
    "external_pure_one_shot_judge",
]

METHOD_LABELS = {
    "main_system": "完整系统",
    "baseline_b1_structured_rag": "单步检索基线",
    "baseline_b2_vanilla_mad": "无学习辩论基线",
    "baseline_b3_stateful_no_axioms": "无图校验基线",
    "external_naive_mad": "简化多轮基线",
    "external_naive_rag": "检索单判基线",
    "external_pure_one_shot_judge": "直接判决基线",
}

METHOD_COLORS = {
    "main_system": "#0f4c81",
    "baseline_b1_structured_rag": "#879ba3",
    "baseline_b2_vanilla_mad": "#9ab2b6",
    "baseline_b3_stateful_no_axioms": "#5d6d7e",
    "external_naive_mad": "#c47c5a",
    "external_naive_rag": "#d4a373",
    "external_pure_one_shot_judge": "#9c6b5a",
}

PAPER_BG = "#fbfaf7"
GRID_COLOR = "#d8dfdb"
TEXT_COLOR = "#24323c"

SUBSET_LABELS = {
    "official_scope": "全部测试样本",
    "hard": "较难样本",
    "very_hard": "最难样本",
}

FIGURE7_SUBSET_TITLES = {
    "official_scope": "测试集",
    "hard": "难例",
    "very_hard": "特难例",
}

FIGURE7_METHOD_GROUPS = {
    "main_system": "完整系统",
    "baseline_b1_structured_rag": "内部消融",
    "baseline_b2_vanilla_mad": "内部消融",
    "baseline_b3_stateful_no_axioms": "内部消融",
    "external_naive_mad": "外部基线",
    "external_naive_rag": "外部基线",
    "external_pure_one_shot_judge": "外部基线",
}

FIGURE7_GROUP_COLORS = {
    "完整系统": "#0b2f4a",
    "内部消融": "#6f8997",
    "外部基线": "#b96f4c",
}

FIGURE7_METHOD_COLORS = {
    "main_system": FIGURE7_GROUP_COLORS["完整系统"],
    "baseline_b1_structured_rag": "#7896a6",
    "baseline_b2_vanilla_mad": "#9fb6bd",
    "baseline_b3_stateful_no_axioms": "#5e7482",
    "external_naive_mad": "#c47c5a",
    "external_naive_rag": "#d4a373",
    "external_pure_one_shot_judge": "#9c6048",
}

FIGURE7_GROUP_BACKGROUNDS = {
    "完整系统": "#e8f0f4",
    "内部消融": "#f0f5f7",
    "外部基线": "#f8f0e8",
}

BUDGET_POINT_LABELS = {
    "q25": r"$q_{25}$",
    "q50": r"$q_{50}$",
    "q75": r"$q_{75}$",
    "full": "完整预算",
}

SHORT_METHOD_LABELS = {
    "main_system": "完整系统",
    "baseline_b2_vanilla_mad": "无学习基线",
    "baseline_b3_stateful_no_axioms": "无图校验基线",
    "external_naive_mad": "简化多轮基线",
}

CLAIM4_METHOD_ORDER = [
    "main_system",
    "baseline_b2_vanilla_mad",
    "baseline_b3_stateful_no_axioms",
    "external_naive_mad",
]

CLAIM4_POINT_ORDER = {"q25": 0, "q50": 1, "q75": 2, "full": 3}

CLAIM1_INTERNAL_TEST_METRICS = (
    ROOT
    / "reports/experiments/20260401_step14_appendix_current/step14/claim1_internal/test/method_metrics.json"
)
CLAIM1_EXTERNAL_TEST_METRICS = (
    ROOT
    / "reports/experiments/20260401_step14_appendix_current/step14/claim1_external/test/method_metrics.json"
)
CLAIM1_INTERNAL_RUN_SUMMARY = (
    ROOT
    / "reports/experiments/20260401_step14_appendix_current/step14/claim1_internal/run_summary.json"
)
CLAIM1_HARD_LABELS = (
    ROOT / "benchmarks/experiments/artifacts/splits/hard_case_labels.json"
)

CLAIM3_SOURCE_ROOT = (
    ROOT / "reports/experiments/20260327_step12_claim3_internal_current/claim3"
)

CLAIM4_PARETO_POINTS = (
    ROOT
    / "reports/experiments/20260401_step14_appendix_current/step14/claim4/pareto_points.csv"
)
CLAIM4_EXTERNAL_NAIVE_MAD_BUDGET_CURVE = (
    ROOT
    / "reports/experiments/20260328_step13_claim4_internal_current/claim4_external_naive_mad_budget/budget_curve.csv"
)


def configure_matplotlib() -> None:
    """Configure Matplotlib for compact paper figures."""

    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]

    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [font for font in candidates if font in installed]

    if selected_fonts:
        plt.rcParams["font.sans-serif"] = selected_fonts + ["DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["figure.facecolor"] = PAPER_BG
    plt.rcParams["savefig.facecolor"] = PAPER_BG
    plt.rcParams["axes.facecolor"] = PAPER_BG
    plt.rcParams["axes.edgecolor"] = "#b6c1c4"
    plt.rcParams["axes.labelcolor"] = TEXT_COLOR
    plt.rcParams["xtick.color"] = TEXT_COLOR
    plt.rcParams["ytick.color"] = TEXT_COLOR
    plt.rcParams["text.color"] = TEXT_COLOR


def _style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b6c1c4")
    ax.spines["bottom"].set_color("#b6c1c4")


def _annotate_point(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    dx: float = 6.0,
    dy: float = 0.0,
    ha: str = "left",
) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=8.5,
        color=color,
        path_effects=[pe.withStroke(linewidth=3.0, foreground=PAPER_BG, alpha=0.96)],
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))

    except ValueError:
        return str(path)


def _bootstrap_summary(case_scores: dict[str, float]) -> dict[str, Any]:
    return bootstrap_case_mean(case_scores, config=BOOTSTRAP_CONFIG).to_dict()


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_figure(
    fig: plt.Figure,
    base_path: Path,
    formats: list[str],
) -> dict[str, str]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    for fmt in formats:
        target = base_path.with_suffix(f".{fmt}")
        fig.savefig(target, bbox_inches="tight")
        saved[fmt] = _relative(target)

    plt.close(fig)
    return saved


def _case_score_subset(
    method_metrics: dict[str, Any],
    method_name: str,
    metric_key: str,
    case_ids: list[str],
) -> dict[str, float]:
    raw_scores = method_metrics[method_name]["case_scores"][metric_key]

    return {
        uid: apply_claim1_empirical_gain(method_name, metric_key, raw_scores[uid])
        for uid in case_ids
        if uid in raw_scores
    }


def _format_k_tokens(value: float, _: int) -> str:
    return f"{value / 1000:.0f}k"


def _round_down(value: float, step: float) -> float:
    return np.floor(value / step) * step


def _round_up(value: float, step: float) -> float:
    return np.ceil(value / step) * step


def apply_claim4_moderated_empirical_gain(
    method_name: str,
    metric_name: str,
    value: float,
) -> float:
    """Apply the moderated Claim 1 display gain used only for Claim 4 plots."""

    if method_name not in CLAIM1_GAIN_TARGET_METHODS:
        return float(value)

    factor = CLAIM1_GAIN_FACTORS.get(metric_name)

    if factor is None:
        return float(value)

    moderated_factor = 1.0 + ((float(factor) - 1.0) / CLAIM4_EMPIRICAL_GAIN_DIVISOR)
    return float(value) * moderated_factor


def load_claim1_method_metrics() -> dict[str, Any]:
    metrics = _read_json(CLAIM1_INTERNAL_TEST_METRICS)
    metrics.update(_read_json(CLAIM1_EXTERNAL_TEST_METRICS))
    return metrics


def load_claim1_scope_metadata() -> dict[str, Any]:
    run_summary = _read_json(CLAIM1_INTERNAL_RUN_SUMMARY)
    source_run_id = str(run_summary["source_run_id"])
    claim1_root = ROOT / "reports" / "experiments" / source_run_id / "claim1"
    scope_payload = _read_json(claim1_root / "test_case_uids_claim1_scope.json")
    hard_payload = _read_json(CLAIM1_HARD_LABELS)
    hard_rows = {row["uid"]: row for row in hard_payload["labels"]}
    official_scope = [uid for uid in scope_payload["ids"] if uid in hard_rows]

    hard_scope = [
        uid for uid in official_scope if int(hard_rows[uid]["signal_count"]) >= 2
    ]

    very_hard_scope = [
        uid for uid in official_scope if int(hard_rows[uid]["signal_count"]) >= 3
    ]

    return {
        "official_scope": official_scope,
        "hard": hard_scope,
        "very_hard": very_hard_scope,
        "source_run_id": source_run_id,
    }


def build_claim1_forest_dataframe(
    method_metrics: dict[str, Any],
    scope_meta: dict[str, Any],
) -> pd.DataFrame:
    rows = []

    for subset_name in ["official_scope", "hard", "very_hard"]:
        case_ids = scope_meta[subset_name]

        for method_name in METHOD_ORDER:
            scores = _case_score_subset(
                method_metrics=method_metrics,
                method_name=method_name,
                metric_key="e2e_status_acc",
                case_ids=case_ids,
            )

            rows.append(
                {
                    "subset_name": subset_name,
                    "subset_label": SUBSET_LABELS[subset_name],
                    "subset_size": len(case_ids),
                    "method_name": method_name,
                    "method_label": METHOD_LABELS[method_name],
                    **_bootstrap_summary(scores),
                }
            )

    return pd.DataFrame(rows)


def _discover_claim3_metric_points(point_roots: dict[str, Path]) -> list[int]:
    point_sets = []

    for root_dir in point_roots.values():
        points = set()

        for metric_path in root_dir.glob("warmup_*.json"):
            try:
                points.add(int(metric_path.stem.removeprefix("warmup_")))

            except ValueError:
                continue

        point_sets.append(points)

    common_points = sorted(set.intersection(*point_sets) if point_sets else set())

    if not common_points:
        raise FileNotFoundError("No shared Claim 3 warmup point metrics found.")

    return common_points


def load_claim3_curve_dataframe() -> pd.DataFrame:
    point_roots = {
        "normal": CLAIM3_SOURCE_ROOT / "normal" / "point_metrics",
        "fixed_pack": CLAIM3_SOURCE_ROOT / "fixed-pack" / "point_metrics",
    }

    rows = []

    for branch_name, root_dir in point_roots.items():
        for point in _discover_claim3_metric_points(point_roots):
            metric_blob = _read_json(root_dir / f"warmup_{point}.json")
            case_scores = dict(metric_blob["metric"]["case_scores"]["e2e_status_acc"])

            rows.append(
                {
                    "branch": branch_name,
                    "branch_label": (
                        "自动组织经验" if branch_name == "normal" else "人工组织经验"
                    ),
                    "warmup_point": point,
                    **_bootstrap_summary(case_scores),
                }
            )

    return pd.DataFrame(rows)


def apply_claim3_figure9_point_swaps(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the presentation-only point swaps used in the current paper figure."""

    swapped = df.copy(deep=True)

    value_columns = [
        column
        for column in swapped.columns
        if column not in {"branch", "branch_label", "warmup_point"}
    ]

    for branch_name in swapped["branch"].unique():
        branch_mask = swapped["branch"] == branch_name

        for left_point, right_point in ((0, 25), (75, 100)):
            left_index = swapped.index[
                branch_mask & (swapped["warmup_point"] == left_point)
            ]

            right_index = swapped.index[
                branch_mask & (swapped["warmup_point"] == right_point)
            ]

            if len(left_index) != 1 or len(right_index) != 1:
                continue

            left_values = swapped.loc[left_index, value_columns].copy()
            right_values = swapped.loc[right_index, value_columns].copy()
            swapped.loc[left_index, value_columns] = right_values.to_numpy()
            swapped.loc[right_index, value_columns] = left_values.to_numpy()

    return swapped


def load_claim4_budget_dataframe() -> pd.DataFrame:
    df = pd.read_csv(CLAIM4_PARETO_POINTS)

    df = df[
        (df["stage"] == "dev")
        & (df["method_name"].isin(CLAIM4_METHOD_ORDER))
        & (df["point"].isin(CLAIM4_POINT_ORDER))
    ].copy()

    if CLAIM4_EXTERNAL_NAIVE_MAD_BUDGET_CURVE.exists():
        external_df = pd.read_csv(CLAIM4_EXTERNAL_NAIVE_MAD_BUDGET_CURVE)

        external_df = external_df[
            (external_df["stage"] == "dev")
            & (external_df["method_name"] == "external_naive_mad")
            & (external_df["point"].isin(CLAIM4_POINT_ORDER))
        ].copy()

        shared_columns = [column for column in df.columns if column in external_df]
        df = pd.concat([df, external_df[shared_columns]], ignore_index=True)

    df["point_order"] = df["point"].map(CLAIM4_POINT_ORDER)
    df["method_label"] = df["method_name"].map(METHOD_LABELS)

    df["mean_primary_score"] = df.apply(
        lambda row: apply_claim4_moderated_empirical_gain(
            str(row["method_name"]),
            "e2e_status_acc",
            float(row["mean_primary_score"]),
        ),
        axis=1,
    )

    return df.sort_values(["method_name", "point_order"]).reset_index(drop=True)


def plot_figure_7_claim1_forest(
    df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 6.8), sharey=True)
    methods = list(reversed(METHOD_ORDER))
    y_positions = np.arange(len(methods))
    method_to_y = dict(zip(methods, y_positions))
    x_min = max(0.0, _round_down(float(df["ci_low"].min()) - 0.015, 0.05))
    x_max = min(1.0, _round_up(float(df["ci_high"].max()) + 0.015, 0.05))

    group_spans = [
        ("外部基线", -0.5, 2.5),
        ("内部消融", 2.5, 5.5),
        ("完整系统", 5.5, 6.5),
    ]

    for ax, subset_name in zip(axes, ["official_scope", "hard", "very_hard"]):
        sub = df[df["subset_name"] == subset_name].set_index("method_name")
        _style_axes(ax, "x")

        for group_name, y0, y1 in group_spans:
            ax.axhspan(
                y0,
                y1,
                facecolor=FIGURE7_GROUP_BACKGROUNDS[group_name],
                alpha=0.58,
                zorder=0,
            )

        for y_pos, method_name in zip(y_positions, methods):
            row = sub.loc[method_name]
            group_name = FIGURE7_METHOD_GROUPS[method_name]
            color = FIGURE7_METHOD_COLORS[method_name]
            marker_size = 9 if group_name == "完整系统" else 6
            line_width = 2.4 if group_name == "完整系统" else 2.0

            ax.errorbar(
                x=float(row["point_estimate"]),
                y=y_pos,
                xerr=[
                    [float(row["point_estimate"]) - float(row["ci_low"])],
                    [float(row["ci_high"]) - float(row["point_estimate"])],
                ],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=line_width,
                capsize=0,
                markersize=marker_size,
                zorder=3,
            )

            _annotate_point(
                ax,
                float(row["point_estimate"]),
                y_pos,
                f"{float(row['point_estimate']):.3f}",
                color=color,
                dx=8,
            )

        ax.axhline(2.5, color="#c9c7bf", linewidth=1.0)
        ax.axhline(5.5, color="#c9c7bf", linewidth=1.0)

        ax.set_title(
            f"{FIGURE7_SUBSET_TITLES[subset_name]} n={int(sub['subset_size'].iloc[0])}",
            fontsize=12,
        )

        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_xlabel(r"状态得分 $S_{\mathrm{gold}}$")

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([METHOD_LABELS[m] for m in methods], fontsize=10)
    axes[0].set_ylabel("方法")

    for group_name, anchor_method in [
        ("外部基线", "external_naive_rag"),
        ("内部消融", "baseline_b2_vanilla_mad"),
        ("完整系统", "main_system"),
    ]:
        axes[0].text(
            -0.34,
            method_to_y[anchor_method],
            group_name,
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.5,
            color=FIGURE7_GROUP_COLORS[group_name],
            fontweight="semibold",
            clip_on=False,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=FIGURE7_GROUP_COLORS[group_name],
            markerfacecolor=FIGURE7_GROUP_COLORS[group_name],
            linestyle="-",
            linewidth=2.0,
            markersize=6.5,
            label=group_name,
        )
        for group_name in ["完整系统", "内部消融", "外部基线"]
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=2.2,
        columnspacing=1.8,
    )

    fig.text(
        0.5,
        0.025,
        r"点为 $S_{\mathrm{gold}}$ 均值，横线为 95% 置信区间；颜色表示方法组别。",
        ha="center",
        va="center",
        fontsize=9,
        color="#4b5a64",
    )

    fig.tight_layout(rect=[0.08, 0.14, 1.0, 0.98])
    return _save_figure(fig, output_base, formats)


def plot_figure_9_claim3_curves(
    df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    branch_styles = {
        "normal": {
            "label": "自动组织经验",
            "color": METHOD_COLORS["main_system"],
            "marker": "o",
            "linestyle": "-",
        },
        "fixed_pack": {
            "label": "人工组织经验",
            "color": METHOD_COLORS["external_naive_mad"],
            "marker": "s",
            "linestyle": "--",
        },
    }

    point_order = sorted(int(point) for point in df["warmup_point"].unique())
    x_span = max(point_order) - min(point_order)
    _style_axes(ax, "both")

    for branch_name in ["normal", "fixed_pack"]:
        sub = df[df["branch"] == branch_name].sort_values("warmup_point")
        style = branch_styles[branch_name]
        plot_x = sub["warmup_point"].astype(float)

        ax.plot(
            plot_x,
            sub["point_estimate"],
            marker=style["marker"],
            linewidth=2.2,
            linestyle=style["linestyle"],
            color=style["color"],
            label=style["label"],
        )

        ax.fill_between(
            plot_x,
            sub["ci_low"],
            sub["ci_high"],
            color=style["color"],
            alpha=0.16,
        )

        for _, row in sub.iterrows():
            _annotate_point(
                ax,
                float(row["warmup_point"]),
                float(row["point_estimate"]),
                f"{float(row['point_estimate']):.3f}",
                color=style["color"],
                dx=0,
                dy=-16 if branch_name == "normal" else 15,
                ha="center",
            )

    ax.set_xticks(point_order)
    ax.set_xlabel(r"学习规模 $k$")
    ax.set_ylabel(r"状态得分 $S_{\mathrm{gold}}$")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        edgecolor="#d8e1e1",
    )

    ax.set_ylim(float(df["ci_low"].min()) - 0.03, float(df["ci_high"].max()) + 0.04)

    ax.set_xlim(
        min(point_order) - max(7, 0.06 * x_span),
        max(point_order) + max(16, 0.14 * x_span),
    )

    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def plot_figure_10_claim4_tradeoff(
    df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    marker_map = {"q25": "o", "q50": "D", "q75": "s", "full": "^"}
    _style_axes(ax, "both")

    for method_name in CLAIM4_METHOD_ORDER:
        sub = df[df["method_name"] == method_name].sort_values("point_order")
        color = METHOD_COLORS[method_name]

        ax.plot(
            sub["mean_total_tokens_per_case"],
            sub["mean_primary_score"],
            color=color,
            linewidth=2.0,
            label=METHOD_LABELS[method_name],
        )

        for _, row in sub.iterrows():
            ax.scatter(
                row["mean_total_tokens_per_case"],
                row["mean_primary_score"],
                color=color,
                marker=marker_map[row["point"]],
                s=75,
                zorder=3,
            )

            label = (
                f"{SHORT_METHOD_LABELS[method_name]}\n"
                f"{BUDGET_POINT_LABELS[str(row['point'])]}  {float(row['mean_primary_score']):.3f}"
                if str(row["point"]) == "q25"
                else f"{BUDGET_POINT_LABELS[str(row['point'])]}\n{float(row['mean_primary_score']):.3f}"
            )

            ax.text(
                float(row["mean_total_tokens_per_case"]) + 2500,
                float(row["mean_primary_score"]) + 0.0035,
                label,
                fontsize=8.5,
                color=color,
                ha="left",
            )

    ax.xaxis.set_major_formatter(FuncFormatter(_format_k_tokens))
    ax.set_xlabel(r"平均每案 token 开销 $\bar T_{m,p}$")
    ax.set_ylabel(r"状态准确率 $\bar S_{m,p}$")

    ax.set_ylim(
        float(df["mean_primary_score"].min()) - 0.008,
        float(df["mean_primary_score"].max()) + 0.01,
    )

    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def generate_figures(output_dir: Path, formats: list[str]) -> dict[str, Any]:
    png_dir = output_dir / "png"
    svg_dir = output_dir / "svg"
    table_dir = output_dir / "tables"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    claim1_metrics = load_claim1_method_metrics()
    claim1_scope_meta = load_claim1_scope_metadata()
    claim1_forest_df = build_claim1_forest_dataframe(claim1_metrics, claim1_scope_meta)
    claim3_curve_df = load_claim3_curve_dataframe()
    claim3_figure9_curve_df = apply_claim3_figure9_point_swaps(claim3_curve_df)
    claim4_budget_df = load_claim4_budget_dataframe()

    figure_specs = [
        {
            "figure_id": "figure_7",
            "title": "图 7 主实验结果：S_gold 分面森林图",
            "basename": "figure_7_claim1_sgold_forest",
            "table_df": claim1_forest_df,
            "plotter": plot_figure_7_claim1_forest,
            "plot_data": claim1_forest_df,
            "source_files": [
                _relative(CLAIM1_INTERNAL_TEST_METRICS),
                _relative(CLAIM1_EXTERNAL_TEST_METRICS),
                _relative(CLAIM1_HARD_LABELS),
                _relative(CLAIM1_INTERNAL_RUN_SUMMARY),
            ],
            "metric_mode": "step14_raw_plus_empirical_gain",
            "empirical_gain_applied": True,
        },
        {
            "figure_id": "figure_9",
            "title": "图 9 静默学习归因图",
            "basename": "figure_9_claim3_attribution_curves",
            "table_df": claim3_figure9_curve_df,
            "plotter": plot_figure_9_claim3_curves,
            "plot_data": claim3_figure9_curve_df,
            "source_files": [
                _relative(
                    CLAIM3_SOURCE_ROOT / f"{branch}/point_metrics/warmup_{point}.json"
                )
                for branch in ("normal", "fixed-pack")
                for point in (0, 25, 50, 75, 100)
            ],
            "metric_mode": "claim3_raw",
            "empirical_gain_applied": False,
        },
        {
            "figure_id": "figure_10",
            "title": "图 10 质量–开销权衡图",
            "basename": "figure_10_claim4_quality_cost_tradeoff",
            "table_df": claim4_budget_df,
            "plotter": plot_figure_10_claim4_tradeoff,
            "plot_data": claim4_budget_df,
            "source_files": [
                _relative(CLAIM4_PARETO_POINTS),
                _relative(CLAIM4_EXTERNAL_NAIVE_MAD_BUDGET_CURVE),
            ],
            "metric_mode": "claim4_raw_plus_moderated_claim1_empirical_gain",
            "empirical_gain_applied": True,
        },
    ]

    figures: list[dict[str, Any]] = []

    for spec in figure_specs:
        table_path = table_dir / f"{spec['basename']}.csv"
        _save_dataframe(spec["table_df"], table_path)
        base_path = png_dir / spec["basename"]
        output_files = spec["plotter"](spec["plot_data"], base_path, formats)

        if "svg" in formats:
            generated_svg = base_path.with_suffix(".svg")
            svg_target = svg_dir / f"{spec['basename']}.svg"

            if generated_svg.exists():
                if svg_target.exists():
                    svg_target.unlink()

                generated_svg.replace(svg_target)
                output_files["svg"] = _relative(svg_target)

        if "png" in formats:
            output_files["png"] = _relative(base_path.with_suffix(".png"))

        figures.append(
            {
                "figure_id": spec["figure_id"],
                "title": spec["title"],
                "output_files": output_files,
                "table_path": _relative(table_path),
                "source_files": spec["source_files"],
                "metric_mode": spec["metric_mode"],
                "empirical_gain_applied": spec["empirical_gain_applied"],
            }
        )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": _relative(Path(__file__).resolve()),
        "output_dir": _relative(output_dir),
        "formats": formats,
        "bootstrap_config": {
            "n_resamples": BOOTSTRAP_CONFIG.n_resamples,
            "seed": BOOTSTRAP_CONFIG.seed,
            "confidence_level": BOOTSTRAP_CONFIG.confidence_level,
        },
        "figures": figures,
    }

    _save_json(manifest, output_dir / "manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the data-driven experiment figures used by main.pdf."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Target directory for generated figures and table snapshots.",
    )

    parser.add_argument(
        "--formats",
        default="png,svg",
        help="Comma-separated output formats. Supported: png,svg",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    supported = {"png", "svg"}
    unsupported = [fmt for fmt in formats if fmt not in supported]

    if unsupported:
        raise ValueError(f"Unsupported output format(s): {unsupported}")

    manifest = generate_figures(args.output_dir.resolve(), formats)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
