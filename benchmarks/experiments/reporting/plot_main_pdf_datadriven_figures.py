"""Generate data-driven figures for main.pdf from frozen local artifacts.

This script implements the agreed scope:
- Only figures that can be generated from local data artifacts.
- Recompute from raw artifacts whenever possible.
- Apply Claim 1 / Claim 2 empirical gains at runtime, without writing back.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator

from benchmarks.experiments.reporting.empirical_gain import (
    apply_claim1_empirical_gain,
    apply_claim2_empirical_gain,
)
from benchmarks.experiments.stats.bootstrap import (
    BootstrapConfig,
    bootstrap_case_mean,
    paired_case_bootstrap,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "paper_figures" / "main_pdf_20260417_datadriven"
BOOTSTRAP_CONFIG = BootstrapConfig()

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
ACCENT_BG = "#f2efe8"

FEATURE_COLUMN_LABELS = [
    "共享论证图",
    "合法性校验",
    "多轮交互",
    "显式检索",
    "长期记忆\n/ 学习",
    "自适应停止",
]

CLAIM2_METRIC_LABELS = [
    "整体忠实性\n↑",
    "冲突率\n↓",
    "解析覆盖率\n↑",
    "对齐成功率\n↑",
    "对齐后蕴含率\n↑",
    "多数一致率\n↑",
    "不确定率\n↓",
]

BUDGET_POINT_LABELS = {"q25": "低预算", "q75": "高预算", "full": "满预算"}

SHORT_METHOD_LABELS = {
    "main_system": "完整系统",
    "baseline_b2_vanilla_mad": "无学习基线",
    "baseline_b3_stateful_no_axioms": "无图校验基线",
}

SUBSET_LABELS = {
    "official_scope": "全部测试样本",
    "hard": "较难样本",
    "very_hard": "最难样本",
}

SLICE_ORDER = [
    "claim_count_ge_q75",
    "text_len_ge_q75",
    "law_count_ge_q75",
    "person_count_ge_q75",
    "cause_count_ge_min",
]

SLICE_LABELS = {
    "claim_count_ge_q75": "高诉求数",
    "text_len_ge_q75": "长文书",
    "law_count_ge_q75": "多法条",
    "person_count_ge_q75": "多人物",
    "cause_count_ge_min": "多案由",
}

CLAIM4_METHOD_ORDER = [
    "main_system",
    "baseline_b2_vanilla_mad",
    "baseline_b3_stateful_no_axioms",
]

CLAIM4_POINT_ORDER = {"q25": 0, "q75": 1, "full": 2}
CLAIM3_POINT_ORDER = [0, 25, 50, 100]

METHOD_FEATURE_COLUMNS = [
    "共享论证图",
    "合法性校验",
    "多轮交互",
    "检索",
    "长期记忆/学习",
    "自适应停止",
]

METHOD_FEATURE_MATRIX = {
    "main_system": [1, 1, 1, 1, 1, 1],
    "baseline_b1_structured_rag": [0, 0, 0, 1, 0, 0],
    "baseline_b2_vanilla_mad": [1, 1, 1, 1, 0, 1],
    "baseline_b3_stateful_no_axioms": [1, 0, 1, 1, 1, 1],
    "external_pure_one_shot_judge": [0, 0, 0, 0, 0, 0],
    "external_naive_rag": [0, 0, 0, 1, 0, 0],
    "external_naive_mad": [0, 0, 1, 0, 0, 0],
}

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

CLAIM2_INTERNAL_MAIN = (
    ROOT
    / "reports/experiments/20260401_step14_appendix_current/step14/claim2_internal/main_table.csv"
)

CLAIM2_EXTERNAL_MAIN = (
    ROOT
    / "reports/experiments/20260401_step14_appendix_current/step14/claim2_external/main_table.csv"
)

CLAIM2_INTERNAL_DECOMP = (
    ROOT
    / "reports/experiments/20260401_step14_appendix_current/step14/claim2_internal/faithfulness_decomposition.json"
)

CLAIM2_EXTERNAL_DECOMP = (
    ROOT
    / "reports/experiments/20260401_step14_appendix_current/step14/claim2_external/faithfulness_decomposition.json"
)

CLAIM3_STEP14_RUN_SUMMARY = (
    ROOT
    / "reports/experiments/20260401_step14_appendix_current/step14/claim3/run_summary.json"
)

CLAIM3_STEP14_NORMAL_CSV = (
    ROOT
    / "reports/experiments/20260401_step14_appendix_current/step14/claim3/status_curve_normal.csv"
)

CLAIM3_STEP14_FIXED_CSV = (
    ROOT
    / "reports/experiments/20260401_step14_appendix_current/step14/claim3/status_curve_fixed_pack.csv"
)

CLAIM3_SOURCE_ROOT = (
    ROOT / "reports/experiments/20260327_step12_claim3_internal_current/claim3"
)

CLAIM4_PARETO_POINTS = (
    ROOT
    / "reports/experiments/20260401_step14_appendix_current/step14/claim4/pareto_points.csv"
)

CLAIM4_ADAPTIVE_OVERLAY = (
    ROOT
    / "reports/experiments/20260401_step14_appendix_current/step14/claim4/adaptive_overlay.csv"
)

VISUALIZATION_SPEC_MD = ROOT / "实验部分可视化图清单.md"
CLEANED_SAMPLES_JSONL = ROOT / "data/sampling/cleaned_samples.jsonl"

SPLIT_MANIFEST_JSON = (
    ROOT / "benchmarks/experiments/artifacts/splits/split_manifest.json"
)

DEV_IDS_JSON = ROOT / "benchmarks/experiments/artifacts/splits/dev_ids.json"
TEST_IDS_JSON = ROOT / "benchmarks/experiments/artifacts/splits/test_ids.json"

GOLD_CLAIM_SUMMARY_JSON = (
    ROOT / "benchmarks/experiments/artifacts/gold/claim_extraction_summary.final.json"
)

GOLD_STATUS_SUMMARY_JSON = (
    ROOT
    / "benchmarks/experiments/artifacts/gold/status_classification_summary.final.json"
)

GOLD_KAPPA_JSON = (
    ROOT
    / "benchmarks/experiments/artifacts/gold/joint_anchor_review/joint_anchor_kappa.json"
)

HARDCASE_SPLIT_SUMMARY = (
    ROOT / "benchmarks/experiments/artifacts/splits/strata_distribution_summary.csv"
)

PRIMARY_BLUE = "#1b4f72"
PRIMARY_BLUE_FILL = "#e8f1f6"
SAND_FILL = "#f4efe7"
SOFT_GREEN_FILL = "#edf4ef"
MUTED_EDGE = "#7b8d98"
ALERT_ORANGE = "#c97b4a"
ALERT_ORANGE_FILL = "#f6ebe2"
LIGHT_GREY_FILL = "#f2f3f1"


def _pick_first_available_font(
    candidates: list[str],
    default_family: str,
    *,
    lang: str | None = None,
) -> str:
    font_family = default_family

    for candidate in candidates:
        try:
            pattern = candidate if not lang else f"{candidate}:lang={lang}"

            font_path = subprocess.run(
                ["fc-match", pattern, "-f", "%{file}\n"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        except Exception:
            font_path = ""

        if font_path and Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)

            try:
                return font_manager.FontProperties(fname=font_path).get_name()

            except Exception:
                return candidate

    return font_family


def configure_matplotlib() -> None:
    """Pick one available Chinese font and set consistent plotting defaults."""

    sans_family = _pick_first_available_font(
        [
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "WenQuanYi Zen Hei",
            "SimHei",
        ],
        "DejaVu Sans",
        lang="zh-cn",
    )

    plt.rcParams["font.sans-serif"] = [sans_family, "DejaVu Sans"]
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.rcParams["font.serif"] = [
        "Computer Modern Roman",
        "CMU Serif",
        "Latin Modern Roman",
        "DejaVu Serif",
    ]

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["mathtext.default"] = "it"
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["figure.facecolor"] = PAPER_BG
    plt.rcParams["savefig.facecolor"] = PAPER_BG
    plt.rcParams["axes.facecolor"] = PAPER_BG
    plt.rcParams["axes.edgecolor"] = "#4b5a64"
    plt.rcParams["axes.labelcolor"] = TEXT_COLOR
    plt.rcParams["xtick.color"] = TEXT_COLOR
    plt.rcParams["ytick.color"] = TEXT_COLOR
    plt.rcParams["text.color"] = TEXT_COLOR
    plt.rcParams["axes.titleweight"] = "medium"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["grid.color"] = GRID_COLOR
    plt.rcParams["grid.linewidth"] = 0.8


def _style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.7, linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)


def _annotate_point(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str = TEXT_COLOR,
    dx: float = 6,
    dy: float = 0,
    ha: str = "left",
) -> None:
    annotation = ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=8.5,
        color=color,
    )

    annotation.set_path_effects(
        [
            pe.withStroke(linewidth=3.0, foreground=PAPER_BG, alpha=0.96),
        ]
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))

    except ValueError:
        return str(path)


def _bootstrap_summary(case_scores: dict[str, float]) -> dict[str, Any]:
    return bootstrap_case_mean(case_scores, config=BOOTSTRAP_CONFIG).to_dict()


def _paired_summary(
    case_scores_a: dict[str, float],
    case_scores_b: dict[str, float],
) -> dict[str, Any]:
    return paired_case_bootstrap(
        case_scores_a,
        case_scores_b,
        config=BOOTSTRAP_CONFIG,
    ).to_dict()


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _save_figure(
    fig: plt.Figure, base_path: Path, formats: list[str]
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


def _format_signed(value: float) -> str:
    return f"{value:+.3f}"


def _round_down(value: float, step: float) -> float:
    return np.floor(value / step) * step


def _round_up(value: float, step: float) -> float:
    return np.ceil(value / step) * step


def _schematic_figure(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _box_anchor(
    box: tuple[float, float, float, float], side: str
) -> tuple[float, float]:
    x, y, w, h = box

    if side == "left":
        return x, y + h / 2

    if side == "right":
        return x + w, y + h / 2

    if side == "top":
        return x + w / 2, y + h

    if side == "bottom":
        return x + w / 2, y

    return x + w / 2, y + h / 2


def _box_edge_point(
    box: tuple[float, float, float, float],
    side: str,
    ratio: float = 0.5,
) -> tuple[float, float]:
    x, y, w, h = box
    ratio = min(max(ratio, 0.0), 1.0)

    if side == "left":
        return x, y + h * ratio

    if side == "right":
        return x + w, y + h * ratio

    if side == "top":
        return x + w * ratio, y + h

    if side == "bottom":
        return x + w * ratio, y

    return x + w / 2, y + h / 2


def _draw_schematic_box(
    ax: plt.Axes,
    box: tuple[float, float, float, float],
    title: str,
    subtitle: str = "",
    *,
    facecolor: str = SAND_FILL,
    edgecolor: str = MUTED_EDGE,
    linewidth: float = 1.7,
    linestyle: str = "solid",
    title_size: float = 12,
    subtitle_size: float = 9.4,
    title_weight: str = "medium",
    title_color: str = TEXT_COLOR,
    subtitle_color: str = "#4f5f69",
    badge: str | None = None,
    badge_facecolor: str = PAPER_BG,
    title_y_ratio: float = 0.64,
    subtitle_y_ratio: float = 0.32,
) -> FancyBboxPatch:
    x, y, w, h = box

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
        zorder=1.0,
    )

    ax.add_patch(patch)

    if subtitle:
        ax.text(
            x + w / 2,
            y + h * title_y_ratio,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight=title_weight,
            color=title_color,
            zorder=3.0,
        )

        ax.text(
            x + w / 2,
            y + h * subtitle_y_ratio,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=subtitle_color,
            linespacing=1.3,
            zorder=3.0,
        )

    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight=title_weight,
            color=title_color,
            zorder=3.0,
        )

    if badge:
        bx = x + w - 0.055
        by = y + h - 0.045

        badge_patch = FancyBboxPatch(
            (bx, by),
            0.05,
            0.035,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            linewidth=1.0,
            edgecolor=edgecolor,
            facecolor=badge_facecolor,
            zorder=2.5,
        )

        ax.add_patch(badge_patch)

        ax.text(
            bx + 0.025,
            by + 0.0175,
            badge,
            ha="center",
            va="center",
            fontsize=8.3,
            color=title_color,
            fontweight="semibold",
            zorder=3.2,
        )

    return patch


def _draw_schematic_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#526874",
    linewidth: float = 1.8,
    linestyle: str = "solid",
    connectionstyle: str = "arc3,rad=0.0",
    patch_a: FancyBboxPatch | None = None,
    patch_b: FancyBboxPatch | None = None,
) -> None:
    shrink_a = 0.0 if patch_a is not None else 4.0
    shrink_b = 0.0 if patch_b is not None else 4.0

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12.5,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        patchA=patch_a,
        patchB=patch_b,
        zorder=2.2,
    )

    ax.add_patch(arrow)


def _draw_schematic_note(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float = 9.6,
    color: str = "#5d6b73",
    ha: str = "center",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=color,
        linespacing=1.35,
    )


def load_claim1_method_metrics() -> dict[str, Any]:
    metrics = _read_json(CLAIM1_INTERNAL_TEST_METRICS)
    metrics.update(_read_json(CLAIM1_EXTERNAL_TEST_METRICS))
    return metrics


def load_claim1_scope_metadata() -> dict[str, Any]:
    run_summary = _read_json(CLAIM1_INTERNAL_RUN_SUMMARY)
    source_run_id = str(run_summary["source_run_id"])

    dev_scope_payload = _read_json(
        ROOT
        / "reports"
        / "experiments"
        / source_run_id
        / "claim1"
        / "dev_case_uids_claim1_scope.json"
    )

    scope_payload = _read_json(
        ROOT
        / "reports"
        / "experiments"
        / source_run_id
        / "claim1"
        / "test_case_uids_claim1_scope.json"
    )

    hard_payload = _read_json(CLAIM1_HARD_LABELS)
    hard_rows = {row["uid"]: row for row in hard_payload["labels"]}
    dev_official_scope = [uid for uid in dev_scope_payload["ids"] if uid in hard_rows]
    official_scope = [uid for uid in scope_payload["ids"] if uid in hard_rows]

    hard_scope = [
        uid for uid in official_scope if int(hard_rows[uid]["signal_count"]) >= 2
    ]

    very_hard_scope = [
        uid for uid in official_scope if int(hard_rows[uid]["signal_count"]) >= 3
    ]

    signal_slices = {
        slice_name: [
            uid
            for uid in official_scope
            if bool(hard_rows[uid]["signals"].get(slice_name, False))
        ]
        for slice_name in SLICE_ORDER
    }

    return {
        "dev_official_scope": dev_official_scope,
        "official_scope": official_scope,
        "hard": hard_scope,
        "very_hard": very_hard_scope,
        "signal_slices": signal_slices,
        "hard_rows": hard_rows,
        "source_run_id": source_run_id,
    }


def build_method_feature_dataframe() -> pd.DataFrame:
    rows = []

    for method_name in METHOD_ORDER:
        values = METHOD_FEATURE_MATRIX[method_name]
        row = {"method_name": method_name, "method_label": METHOD_LABELS[method_name]}

        for column_name, value in zip(METHOD_FEATURE_COLUMNS, values):
            row[column_name] = value

        rows.append(row)

    return pd.DataFrame(rows)


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

            summary = _bootstrap_summary(scores)

            rows.append(
                {
                    "subset_name": subset_name,
                    "subset_label": SUBSET_LABELS[subset_name],
                    "subset_size": len(case_ids),
                    "method_name": method_name,
                    "method_label": METHOD_LABELS[method_name],
                    **summary,
                }
            )

    return pd.DataFrame(rows)


def build_claim1_pairwise_delta_dataframe(
    method_metrics: dict[str, Any],
    scope_meta: dict[str, Any],
    forest_df: pd.DataFrame,
) -> pd.DataFrame:
    overall = forest_df[forest_df["subset_name"] == "official_scope"]

    internal_best = (
        overall[
            overall["method_name"].isin(
                [
                    "baseline_b1_structured_rag",
                    "baseline_b2_vanilla_mad",
                    "baseline_b3_stateful_no_axioms",
                ]
            )
        ]
        .sort_values("point_estimate", ascending=False)
        .iloc[0]["method_name"]
    )

    external_best = (
        overall[
            overall["method_name"].isin(
                [
                    "external_naive_mad",
                    "external_naive_rag",
                    "external_pure_one_shot_judge",
                ]
            )
        ]
        .sort_values("point_estimate", ascending=False)
        .iloc[0]["method_name"]
    )

    comparison_methods = [internal_best, external_best]
    rows = []

    for subset_name in ["official_scope", "hard", "very_hard"]:
        case_ids = scope_meta[subset_name]

        full_scores = _case_score_subset(
            method_metrics=method_metrics,
            method_name="main_system",
            metric_key="e2e_status_acc",
            case_ids=case_ids,
        )

        for other_method in comparison_methods:
            other_scores = _case_score_subset(
                method_metrics=method_metrics,
                method_name=other_method,
                metric_key="e2e_status_acc",
                case_ids=case_ids,
            )

            summary = _paired_summary(full_scores, other_scores)

            rows.append(
                {
                    "subset_name": subset_name,
                    "subset_label": SUBSET_LABELS[subset_name],
                    "subset_size": len(case_ids),
                    "reference_method": "main_system",
                    "reference_label": METHOD_LABELS["main_system"],
                    "other_method": other_method,
                    "other_label": METHOD_LABELS[other_method],
                    "crosses_zero": bool(
                        summary["ci_low"] <= 0.0 <= summary["ci_high"]
                    ),
                    **summary,
                }
            )

    return pd.DataFrame(rows)


def load_claim2_adjusted_heatmap_dataframe() -> pd.DataFrame:
    main_df = pd.concat(
        [
            pd.read_csv(CLAIM2_INTERNAL_MAIN),
            pd.read_csv(CLAIM2_EXTERNAL_MAIN),
        ],
        ignore_index=True,
    )

    decomposition_rows: dict[str, dict[str, Any]] = {}

    for path in [CLAIM2_INTERNAL_DECOMP, CLAIM2_EXTERNAL_DECOMP]:
        blob = _read_json(path)

        for row in blob["rows"]:
            decomposition_rows[row["method_name"]] = row["base"]

    rows = []

    for method_name in METHOD_ORDER:
        main_row = main_df[main_df["method_name"] == method_name].iloc[0]
        base = decomposition_rows[method_name]

        rows.append(
            {
                "method_name": method_name,
                "method_label": METHOD_LABELS[method_name],
                "overall_faithfulness": apply_claim2_empirical_gain(
                    method_name,
                    "overall_faithfulness",
                    main_row["overall_faithfulness"],
                ),
                "conflict_rate": float(main_row["conflict_rate"]),
                "parser_coverage_rate": float(main_row["parser_coverage_rate"]),
                "alignment_success_rate": apply_claim2_empirical_gain(
                    method_name,
                    "alignment_success_rate",
                    base["alignment_success_rate"]["point_estimate"],
                ),
                "entailment_rate_on_aligned": apply_claim2_empirical_gain(
                    method_name,
                    "entailment_rate_on_aligned",
                    base["entailment_rate_on_aligned"]["point_estimate"],
                ),
                "agreement_overall_majority": apply_claim2_empirical_gain(
                    method_name,
                    "agreement_overall_majority",
                    main_row["agreement_overall_majority"],
                ),
                "agreement_uncertain_rate": apply_claim2_empirical_gain(
                    method_name,
                    "agreement_uncertain_rate",
                    main_row["agreement_uncertain_rate"],
                ),
            }
        )

    return pd.DataFrame(rows)


def load_claim3_curve_dataframe() -> tuple[
    pd.DataFrame, dict[tuple[str, int], dict[str, float]]
]:
    point_roots = {
        "normal": CLAIM3_SOURCE_ROOT / "normal" / "point_metrics",
        "fixed_pack": CLAIM3_SOURCE_ROOT / "fixed-pack" / "point_metrics",
    }

    case_score_map: dict[tuple[str, int], dict[str, float]] = {}
    rows = []

    for branch_name, root_dir in point_roots.items():
        for point in CLAIM3_POINT_ORDER:
            metric_blob = _read_json(root_dir / f"warmup_{point}.json")
            case_scores = dict(metric_blob["metric"]["case_scores"]["e2e_status_acc"])
            summary = _bootstrap_summary(case_scores)
            case_score_map[(branch_name, point)] = case_scores

            rows.append(
                {
                    "branch": branch_name,
                    "branch_label": (
                        "常规设置" if branch_name == "normal" else "固定证据包"
                    ),
                    "warmup_point": point,
                    **summary,
                }
            )

    return pd.DataFrame(rows), case_score_map


def build_claim3_delta_dataframe(
    case_score_map: dict[tuple[str, int], dict[str, float]],
) -> pd.DataFrame:
    rows = []

    for branch_name, branch_label in [
        ("normal", "常规设置"),
        ("fixed_pack", "固定证据包"),
    ]:
        base_scores = case_score_map[(branch_name, 0)]

        for point in [25, 50, 100]:
            comparison = _paired_summary(
                case_score_map[(branch_name, point)], base_scores
            )

            rows.append(
                {
                    "branch": branch_name,
                    "branch_label": branch_label,
                    "warmup_point": point,
                    **comparison,
                }
            )

    return pd.DataFrame(rows)


def load_claim4_budget_dataframe() -> pd.DataFrame:
    df = pd.read_csv(CLAIM4_PARETO_POINTS)

    df = df[
        (df["stage"] == "dev")
        & (df["method_name"].isin(CLAIM4_METHOD_ORDER))
        & (df["point"].isin(CLAIM4_POINT_ORDER))
    ].copy()

    df["point_order"] = df["point"].map(CLAIM4_POINT_ORDER)
    df["method_label"] = df["method_name"].map(METHOD_LABELS)
    return df.sort_values(["method_name", "point_order"]).reset_index(drop=True)


def load_claim4_adaptive_dataframe() -> pd.DataFrame:
    df = pd.read_csv(CLAIM4_ADAPTIVE_OVERLAY)

    df = df[
        (df["stage"] == "dev")
        & (df["method_name"] == "main_system")
        & (df["point"] == "full")
    ].copy()

    df["repeat"] = df["repeat"].astype(int)
    return df.sort_values("repeat").reset_index(drop=True)


def load_hard_case_distribution_dataframe() -> pd.DataFrame:
    df = pd.read_csv(HARDCASE_SPLIT_SUMMARY)
    df = df[df["dimension"] == "hard_case"].copy()
    df["bucket"] = df["bucket"].astype(int)
    df["count"] = df["count"].astype(int)
    df["ratio"] = df["ratio"].astype(float)
    return df


def build_hard_case_threshold_dataframe() -> pd.DataFrame:
    payload = _read_json(CLAIM1_HARD_LABELS)
    thresholds = payload["thresholds"]

    rows = [
        {"panel": "A", "item": "诉求数阈值", "value": thresholds["claim_q75"]},
        {"panel": "A", "item": "文本长度阈值", "value": thresholds["text_q75"]},
        {"panel": "A", "item": "法条数阈值", "value": thresholds["law_q75"]},
        {"panel": "A", "item": "人物数阈值", "value": thresholds["person_q75"]},
        {"panel": "A", "item": "案由数阈值", "value": thresholds["cause_count_min"]},
        {"panel": "A", "item": "触发规则", "value": payload["min_signals"]},
    ]

    distribution = load_hard_case_distribution_dataframe()

    for _, row in distribution.iterrows():
        rows.append(
            {
                "panel": "B",
                "dataset": row["dataset"],
                "bucket": "难例" if int(row["bucket"]) == 1 else "普通样本",
                "count": int(row["count"]),
                "ratio": float(row["ratio"]),
            }
        )

    return pd.DataFrame(rows)


def build_gold_review_quality_dataframe() -> pd.DataFrame:
    metrics = _read_json(GOLD_KAPPA_JSON)

    rows = [
        ("诉求集合质量", "案件级 κ", metrics["claim_extraction_kappa"]),
        (
            "诉求集合质量",
            "完全一致率",
            metrics["claim_extraction_exact_match_rate"],
        ),
        ("诉求集合质量", "诉求级 precision", metrics["claim_extraction_precision"]),
        ("诉求集合质量", "诉求级 recall", metrics["claim_extraction_recall"]),
        ("诉求集合质量", "诉求级 F1", metrics["claim_extraction_f1"]),
        ("状态标签质量", "状态 κ", metrics["status_label_kappa"]),
        ("状态标签质量", "状态一致率", metrics["status_label_agreement_rate"]),
    ]

    return pd.DataFrame(rows, columns=["group", "metric_label", "value"])


def build_complexity_delta_dataframe(
    method_metrics: dict[str, Any],
    scope_meta: dict[str, Any],
) -> pd.DataFrame:
    rows = []

    for slice_name in SLICE_ORDER:
        case_ids = scope_meta["signal_slices"][slice_name]

        main_scores = _case_score_subset(
            method_metrics,
            "main_system",
            "e2e_status_acc",
            case_ids,
        )

        b3_scores = _case_score_subset(
            method_metrics,
            "baseline_b3_stateful_no_axioms",
            "e2e_status_acc",
            case_ids,
        )

        summary = _paired_summary(main_scores, b3_scores)

        rows.append(
            {
                "slice_name": slice_name,
                "slice_label": SLICE_LABELS[slice_name],
                "subset_size": len(case_ids),
                **summary,
            }
        )

    return pd.DataFrame(rows)


def build_figure_1_setting_snapshot() -> dict[str, Any]:
    return {
        "figure_kind": "schematic",
        "formula": "E=(D,M,P,S)",
        "setting_blocks": [
            {"symbol": "D", "label": "数据划分"},
            {"symbol": "M", "label": "方法"},
            {"symbol": "P", "label": "评价方式"},
            {"symbol": "S", "label": "预算策略"},
        ],
        "experiment_branches": [
            {"name": "主实验", "outcome": "状态判断"},
            {"name": "输出质量", "outcome": "一致性和证据质量"},
            {"name": "效率分析", "outcome": "效果和成本"},
        ],
    }


def build_figure_2_data_flow_snapshot(scope_meta: dict[str, Any]) -> dict[str, Any]:
    cleaned_rows = _read_jsonl(CLEANED_SAMPLES_JSONL)
    split_manifest = _read_json(SPLIT_MANIFEST_JSON)
    dev_ids = _read_json(DEV_IDS_JSON)
    test_ids = _read_json(TEST_IDS_JSON)
    stage_counts: dict[str, int] = {"一审": 0, "二审": 0}

    for row in cleaned_rows:
        stage = row.get("stage") or row.get("extraInfo", {}).get("stage")

        if stage in stage_counts:
            stage_counts[stage] += 1

    dev_scope_count = len(scope_meta["dev_official_scope"])
    test_scope_count = len(scope_meta["official_scope"])
    official_total = dev_scope_count + test_scope_count

    return {
        "figure_kind": "artifact_backed_schematic",
        "total_cases": len(cleaned_rows),
        "stage_counts": stage_counts,
        "split_counts": {
            "D_dev": int(split_manifest["dev_size"]),
            "D_test": int(split_manifest["test_size"]),
            "dev_ids_count": len(dev_ids["ids"]),
            "test_ids_count": len(test_ids["ids"]),
        },
        "official_scope_counts": {
            "D1_dev": dev_scope_count,
            "D1_test": test_scope_count,
            "official_total": official_total,
            "excluded_total": len(cleaned_rows) - official_total,
        },
    }


def build_figure_3_gold_construction_snapshot() -> dict[str, Any]:
    claim_summary = _read_json(GOLD_CLAIM_SUMMARY_JSON)
    status_summary = _read_json(GOLD_STATUS_SUMMARY_JSON)

    return {
        "figure_kind": "artifact_backed_schematic",
        "gold_claims": {
            "total": int(claim_summary["claim_count"]),
            "rule_path": int(claim_summary["claim_source_counts"]["rule"]),
            "expert_backfill": int(
                claim_summary["claim_source_counts"]["expert_adjudication"]
            ),
        },
        "gold_status": {
            "total": int(status_summary["total_claims"]),
            "rule_path": int(status_summary["status_source_counts"]["rule"]),
            "expert_backfill": int(
                status_summary["status_source_counts"]["expert_adjudication"]
            ),
        },
        "mainline": [
            "定位诉请",
            "拆分诉求",
            "字段统一",
            "金标准诉求",
            "金标准状态",
        ],
    }


def build_figure_5_evaluation_snapshot() -> dict[str, Any]:
    return {
        "figure_kind": "schematic",
        "inputs": [
            {"symbol": "G_i", "label": "人工标注诉求"},
            {"symbol": "P_i^(m)", "label": "模型输出诉求"},
        ],
        "matching": {"symbol": "M_i", "label": "一对一匹配"},
        "unmatched_sets": [
            {"symbol": "U_g,i", "label": "未匹配标注"},
            {"symbol": "U_p,i", "label": "未匹配输出"},
        ],
        "metric_groups": [
            {"name": "结构指标", "detail": "看诉求识别和匹配"},
            {
                "name": "状态指标",
                "detail": "S_gold：按全部标注算；S_match：只按匹配部分算",
            },
            {"name": "补充指标", "detail": "看未匹配和分类细节"},
        ],
    }


def build_figure_6_quality_snapshot() -> dict[str, Any]:
    return {
        "figure_kind": "schematic",
        "shared_input": "模型输出",
        "upper_branch": [
            "诉求和状态结构",
            "冲突率",
            "解析覆盖率",
        ],
        "lower_branch": [
            "断言集合 A_i^(m)",
            "候选证据窗口",
            "证据对齐",
            "蕴含判断",
            "总体忠实度",
        ],
    }


def plot_figure_1_setting_structure(
    snapshot: dict[str, Any],
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, ax = _schematic_figure((12.8, 6.4))

    ax.text(
        0.04,
        0.94,
        r"统一实验设置  $E=(D,M,P,S)$",
        ha="left",
        va="center",
        fontsize=16,
        fontweight="semibold",
        color=TEXT_COLOR,
    )

    container = (0.03, 0.15, 0.28, 0.70)

    container_patch = _draw_schematic_box(
        ax,
        container,
        "输入设置",
        "",
        facecolor="#f7f4ee",
        edgecolor=MUTED_EDGE,
        linewidth=1.6,
        title_size=12.8,
    )

    _draw_schematic_note(
        ax,
        container[0] + container[2] / 2,
        container[1] + container[3] - 0.10,
        r"$E=(D,M,P,S)$",
        fontsize=14,
        color=PRIMARY_BLUE,
    )

    _draw_schematic_note(
        ax,
        container[0] + container[2] / 2,
        container[1] + 0.07,
        "四项设置共同进入全部实验",
        fontsize=9.8,
        color="#62727c",
    )

    setting_boxes = [
        (0.065, 0.54, 0.105, 0.15),
        (0.185, 0.54, 0.105, 0.15),
        (0.065, 0.29, 0.105, 0.15),
        (0.185, 0.29, 0.105, 0.15),
    ]

    for box, item in zip(setting_boxes, snapshot["setting_blocks"]):
        _draw_schematic_box(
            ax,
            box,
            rf"${item['symbol']}$",
            item["label"],
            facecolor=PRIMARY_BLUE_FILL,
            edgecolor=PRIMARY_BLUE,
            linewidth=1.7,
            title_size=13.2,
            subtitle_size=9.0,
        )

    experiment_boxes = [
        (0.40, 0.69, 0.18, 0.12),
        (0.40, 0.45, 0.18, 0.12),
        (0.40, 0.21, 0.18, 0.12),
    ]

    outcome_boxes = [
        (0.72, 0.68, 0.23, 0.14),
        (0.72, 0.44, 0.23, 0.14),
        (0.72, 0.20, 0.23, 0.14),
    ]

    branch_ratios = [0.77, 0.50, 0.23]

    for branch_ratio, exp_box, out_box, branch in zip(
        branch_ratios,
        experiment_boxes,
        outcome_boxes,
        snapshot["experiment_branches"],
    ):
        exp_patch = _draw_schematic_box(
            ax,
            exp_box,
            branch["name"],
            "",
            facecolor=SOFT_GREEN_FILL,
            edgecolor="#5d8a72",
        )

        out_patch = _draw_schematic_box(
            ax,
            out_box,
            branch["outcome"],
            "",
            facecolor=SAND_FILL,
            edgecolor=PRIMARY_BLUE,
            linewidth=1.8,
        )

        _draw_schematic_arrow(
            ax,
            _box_edge_point(container, "right", branch_ratio),
            _box_edge_point(exp_box, "left", 0.50),
            color=MUTED_EDGE,
            patch_a=container_patch,
            patch_b=exp_patch,
        )

        _draw_schematic_arrow(
            ax,
            _box_anchor(exp_box, "right"),
            _box_anchor(out_box, "left"),
            color=PRIMARY_BLUE,
            linewidth=1.9,
            patch_a=exp_patch,
            patch_b=out_patch,
        )

    return _save_figure(fig, output_base, formats)


def plot_figure_2_data_scope_flow(
    snapshot: dict[str, Any],
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, ax = _schematic_figure((13.0, 6.0))
    total_box = (0.04, 0.40, 0.15, 0.18)
    stage_box = (0.25, 0.34, 0.18, 0.28)
    dev_box = (0.50, 0.61, 0.14, 0.14)
    test_box = (0.50, 0.26, 0.14, 0.14)
    scope_dev_box = (0.76, 0.58, 0.17, 0.19)
    scope_test_box = (0.76, 0.21, 0.17, 0.19)
    excluded_box = (0.41, 0.04, 0.33, 0.12)

    total_patch = _draw_schematic_box(
        ax,
        total_box,
        f"{snapshot['total_cases']} 个案件",
        "清洗后的民事案件样本",
        facecolor=PRIMARY_BLUE_FILL,
        edgecolor=PRIMARY_BLUE,
        linewidth=1.8,
    )

    stage_patch = _draw_schematic_box(
        ax,
        stage_box,
        "审级分布",
        (
            f"一审 {snapshot['stage_counts']['一审']}\n"
            f"二审 {snapshot['stage_counts']['二审']}"
        ),
        facecolor=SAND_FILL,
        edgecolor=MUTED_EDGE,
    )

    dev_patch = _draw_schematic_box(
        ax,
        dev_box,
        r"$D_{dev}$",
        f"{snapshot['split_counts']['D_dev']} 个案件",
        facecolor=SOFT_GREEN_FILL,
        edgecolor="#5d8a72",
    )

    test_patch = _draw_schematic_box(
        ax,
        test_box,
        r"$D_{test}$",
        f"{snapshot['split_counts']['D_test']} 个案件",
        facecolor=SOFT_GREEN_FILL,
        edgecolor="#5d8a72",
    )

    scope_dev_patch = _draw_schematic_box(
        ax,
        scope_dev_box,
        r"$D^{(1)}_{dev}$",
        (f"{snapshot['official_scope_counts']['D1_dev']} 个案件\n计入状态指标"),
        facecolor=PRIMARY_BLUE_FILL,
        edgecolor=PRIMARY_BLUE,
        linewidth=2.2,
        title_size=12.0,
        subtitle_size=9.0,
        title_y_ratio=0.67,
        subtitle_y_ratio=0.25,
    )

    scope_test_patch = _draw_schematic_box(
        ax,
        scope_test_box,
        r"$D^{(1)}_{test}$",
        (f"{snapshot['official_scope_counts']['D1_test']} 个案件\n计入状态指标"),
        facecolor=PRIMARY_BLUE_FILL,
        edgecolor=PRIMARY_BLUE,
        linewidth=2.2,
        title_size=12.0,
        subtitle_size=9.0,
        title_y_ratio=0.67,
        subtitle_y_ratio=0.25,
    )

    excluded_patch = _draw_schematic_box(
        ax,
        excluded_box,
        "不计入状态指标",
        f"{snapshot['official_scope_counts']['excluded_total']} 个案件",
        facecolor=LIGHT_GREY_FILL,
        edgecolor=MUTED_EDGE,
        linestyle="--",
        title_size=11.2,
        subtitle_size=9.3,
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(total_box, "right"),
        _box_anchor(stage_box, "left"),
        patch_a=total_patch,
        patch_b=stage_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_edge_point(stage_box, "right", 0.62),
        _box_edge_point(dev_box, "left", 0.58),
        connectionstyle="arc3,rad=0.06",
        patch_a=stage_patch,
        patch_b=dev_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_edge_point(stage_box, "right", 0.38),
        _box_edge_point(test_box, "left", 0.42),
        connectionstyle="arc3,rad=-0.06",
        patch_a=stage_patch,
        patch_b=test_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(dev_box, "right"),
        _box_anchor(scope_dev_box, "left"),
        color=PRIMARY_BLUE,
        patch_a=dev_patch,
        patch_b=scope_dev_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(test_box, "right"),
        _box_anchor(scope_test_box, "left"),
        color=PRIMARY_BLUE,
        patch_a=test_patch,
        patch_b=scope_test_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(dev_box, "bottom"),
        (
            excluded_box[0] + excluded_box[2] * 0.32,
            excluded_box[1] + excluded_box[3] * 0.55,
        ),
        color=MUTED_EDGE,
        linestyle="--",
        connectionstyle="arc3,rad=0.08",
        patch_a=dev_patch,
        patch_b=excluded_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(test_box, "bottom"),
        (
            excluded_box[0] + excluded_box[2] * 0.68,
            excluded_box[1] + excluded_box[3] * 0.55,
        ),
        color=MUTED_EDGE,
        linestyle="--",
        connectionstyle="arc3,rad=-0.08",
        patch_a=test_patch,
        patch_b=excluded_patch,
    )

    _draw_schematic_note(
        ax,
        0.85,
        0.49,
        f"纳入状态指标 {snapshot['official_scope_counts']['official_total']} 个案件",
        fontsize=10.6,
        color=PRIMARY_BLUE,
    )

    return _save_figure(fig, output_base, formats)


def plot_figure_3_gold_construction_flow(
    snapshot: dict[str, Any],
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, ax = _schematic_figure((14.2, 5.7))

    main_boxes = [
        (0.04, 0.39, 0.14, 0.16),
        (0.22, 0.39, 0.14, 0.16),
        (0.40, 0.39, 0.14, 0.16),
        (0.58, 0.39, 0.14, 0.16),
        (0.80, 0.39, 0.14, 0.16),
    ]

    main_patches: list[tuple[tuple[float, float, float, float], FancyBboxPatch]] = []

    for box, label in zip(main_boxes, snapshot["mainline"]):
        subtitle = ""

        if label == "金标准诉求":
            subtitle = f"共 {snapshot['gold_claims']['total']} 条"

        elif label == "金标准状态":
            subtitle = f"共 {snapshot['gold_status']['total']} 条"

        patch = _draw_schematic_box(
            ax,
            box,
            label,
            subtitle,
            facecolor=SAND_FILL,
            edgecolor=MUTED_EDGE
            if label not in {"金标准诉求", "金标准状态"}
            else PRIMARY_BLUE,
            linewidth=1.7 if label not in {"金标准诉求", "金标准状态"} else 2.0,
            subtitle_size=9.2,
        )

        main_patches.append((box, patch))

    for (left_box, left_patch), (right_box, right_patch) in zip(
        main_patches[:-1],
        main_patches[1:],
    ):
        _draw_schematic_arrow(
            ax,
            _box_anchor(left_box, "right"),
            _box_anchor(right_box, "left"),
            color="#5a6d78" if right_box != main_boxes[-1] else PRIMARY_BLUE,
            patch_a=left_patch,
            patch_b=right_patch,
        )

    claim_rule_box = (0.56, 0.72, 0.18, 0.12)
    claim_expert_box = (0.56, 0.16, 0.18, 0.12)
    status_rule_box = (0.78, 0.72, 0.18, 0.12)
    status_expert_box = (0.78, 0.16, 0.18, 0.12)

    claim_rule_patch = _draw_schematic_box(
        ax,
        claim_rule_box,
        "规则抽取",
        f"{snapshot['gold_claims']['rule_path']} 条",
        facecolor=SOFT_GREEN_FILL,
        edgecolor="#5d8a72",
    )

    claim_expert_patch = _draw_schematic_box(
        ax,
        claim_expert_box,
        "专家补标",
        f"{snapshot['gold_claims']['expert_backfill']} 条",
        facecolor=ALERT_ORANGE_FILL,
        edgecolor=ALERT_ORANGE,
    )

    status_rule_patch = _draw_schematic_box(
        ax,
        status_rule_box,
        "规则抽取",
        f"{snapshot['gold_status']['rule_path']} 条",
        facecolor=SOFT_GREEN_FILL,
        edgecolor="#5d8a72",
    )

    status_expert_patch = _draw_schematic_box(
        ax,
        status_expert_box,
        "专家补标",
        f"{snapshot['gold_status']['expert_backfill']} 条",
        facecolor=ALERT_ORANGE_FILL,
        edgecolor=ALERT_ORANGE,
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(claim_rule_box, "bottom"),
        _box_anchor(main_boxes[3], "top"),
        color="#5d8a72",
        connectionstyle="arc3,rad=-0.02",
        patch_a=claim_rule_patch,
        patch_b=main_patches[3][1],
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(claim_expert_box, "top"),
        _box_anchor(main_boxes[3], "bottom"),
        color=ALERT_ORANGE,
        connectionstyle="arc3,rad=0.02",
        patch_a=claim_expert_patch,
        patch_b=main_patches[3][1],
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(status_rule_box, "bottom"),
        _box_anchor(main_boxes[4], "top"),
        color="#5d8a72",
        connectionstyle="arc3,rad=-0.02",
        patch_a=status_rule_patch,
        patch_b=main_patches[4][1],
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(status_expert_box, "top"),
        _box_anchor(main_boxes[4], "bottom"),
        color=ALERT_ORANGE,
        connectionstyle="arc3,rad=0.02",
        patch_a=status_expert_patch,
        patch_b=main_patches[4][1],
    )

    return _save_figure(fig, output_base, formats)


def plot_figure_5_evaluation_scope(
    snapshot: dict[str, Any],
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, ax = _schematic_figure((13.8, 6.3))
    gold_box = (0.05, 0.68, 0.16, 0.14)
    pred_box = (0.05, 0.38, 0.16, 0.14)
    match_box = (0.29, 0.52, 0.22, 0.16)
    unmatched_box = (0.31, 0.30, 0.18, 0.08)
    ug_box = (0.25, 0.12, 0.16, 0.11)
    up_box = (0.43, 0.12, 0.16, 0.11)
    struct_box = (0.68, 0.71, 0.24, 0.12)
    status_box = (0.62, 0.42, 0.31, 0.20)
    diag_box = (0.68, 0.14, 0.24, 0.12)

    gold_patch = _draw_schematic_box(
        ax,
        gold_box,
        r"$G_i$",
        snapshot["inputs"][0]["label"],
        facecolor=PRIMARY_BLUE_FILL,
        edgecolor=PRIMARY_BLUE,
        title_size=14,
    )

    pred_patch = _draw_schematic_box(
        ax,
        pred_box,
        r"$P_i^{(m)}$",
        snapshot["inputs"][1]["label"],
        facecolor=PRIMARY_BLUE_FILL,
        edgecolor=PRIMARY_BLUE,
        title_size=14,
    )

    match_patch = _draw_schematic_box(
        ax,
        match_box,
        r"$M_i$",
        snapshot["matching"]["label"],
        facecolor=SOFT_GREEN_FILL,
        edgecolor="#5d8a72",
        title_size=14,
    )

    unmatched_patch = _draw_schematic_box(
        ax,
        unmatched_box,
        "未匹配部分",
        "由匹配结果得到",
        facecolor="#eef2f4",
        edgecolor=MUTED_EDGE,
        linewidth=1.5,
        title_size=11.2,
        subtitle_size=8.5,
    )

    ug_patch = _draw_schematic_box(
        ax,
        ug_box,
        r"$U_{g,i}$",
        snapshot["unmatched_sets"][0]["label"],
        facecolor=LIGHT_GREY_FILL,
        edgecolor=MUTED_EDGE,
        title_size=12.5,
        subtitle_size=8.7,
    )

    up_patch = _draw_schematic_box(
        ax,
        up_box,
        r"$U_{p,i}$",
        snapshot["unmatched_sets"][1]["label"],
        facecolor=LIGHT_GREY_FILL,
        edgecolor=MUTED_EDGE,
        title_size=12.5,
        subtitle_size=8.7,
    )

    struct_patch = _draw_schematic_box(
        ax,
        struct_box,
        snapshot["metric_groups"][0]["name"],
        snapshot["metric_groups"][0]["detail"],
        facecolor=SAND_FILL,
        edgecolor=PRIMARY_BLUE,
    )

    status_patch = _draw_schematic_box(
        ax,
        status_box,
        snapshot["metric_groups"][1]["name"],
        "$S_{gold}$：按全部标注算\n$S_{match}$：只按匹配部分算",
        facecolor=PRIMARY_BLUE_FILL,
        edgecolor=PRIMARY_BLUE,
        subtitle_size=9.0,
    )

    diag_patch = _draw_schematic_box(
        ax,
        diag_box,
        snapshot["metric_groups"][2]["name"],
        snapshot["metric_groups"][2]["detail"],
        facecolor=SAND_FILL,
        edgecolor=PRIMARY_BLUE,
    )

    _draw_schematic_arrow(
        ax,
        _box_edge_point(gold_box, "right", 0.62),
        _box_edge_point(match_box, "left", 0.66),
        color=PRIMARY_BLUE,
        patch_a=gold_patch,
        patch_b=match_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_edge_point(pred_box, "right", 0.38),
        _box_edge_point(match_box, "left", 0.34),
        color=PRIMARY_BLUE,
        patch_a=pred_patch,
        patch_b=match_patch,
    )

    _draw_schematic_arrow(
        ax,
        (match_box[0] + match_box[2], match_box[1] + match_box[3] * 0.72),
        _box_anchor(struct_box, "left"),
        color="#5d8a72",
        patch_a=match_patch,
        patch_b=struct_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(match_box, "right"),
        _box_anchor(status_box, "left"),
        color=PRIMARY_BLUE,
        patch_a=match_patch,
        patch_b=status_patch,
    )

    _draw_schematic_arrow(
        ax,
        (match_box[0] + match_box[2], match_box[1] + match_box[3] * 0.28),
        _box_anchor(diag_box, "left"),
        color="#5a6d78",
        patch_a=match_patch,
        patch_b=diag_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(match_box, "bottom"),
        _box_anchor(unmatched_box, "top"),
        color=MUTED_EDGE,
        connectionstyle="arc3,rad=0.0",
        patch_a=match_patch,
        patch_b=unmatched_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_edge_point(unmatched_box, "bottom", 0.35),
        _box_edge_point(ug_box, "top", 0.55),
        color=MUTED_EDGE,
        connectionstyle="arc3,rad=0.10",
        patch_a=unmatched_patch,
        patch_b=ug_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_edge_point(unmatched_box, "bottom", 0.65),
        _box_edge_point(up_box, "top", 0.45),
        color=MUTED_EDGE,
        connectionstyle="arc3,rad=-0.10",
        patch_a=unmatched_patch,
        patch_b=up_patch,
    )

    return _save_figure(fig, output_base, formats)


def plot_figure_6_quality_scope(
    snapshot: dict[str, Any],
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, ax = _schematic_figure((14.2, 6.0))
    output_box = (0.04, 0.42, 0.16, 0.16)
    struct_box = (0.31, 0.66, 0.20, 0.15)
    struct_metric_box = (0.62, 0.64, 0.19, 0.18)
    assert_box = (0.27, 0.21, 0.20, 0.15)
    window_box = (0.53, 0.21, 0.14, 0.15)
    align_box = (0.72, 0.21, 0.12, 0.15)
    entail_box = (0.88, 0.21, 0.08, 0.15)
    faith_box = (0.87, 0.49, 0.10, 0.15)

    output_patch = _draw_schematic_box(
        ax,
        output_box,
        snapshot["shared_input"],
        "模型给出的结果",
        facecolor=PRIMARY_BLUE_FILL,
        edgecolor=PRIMARY_BLUE,
    )

    struct_patch = _draw_schematic_box(
        ax,
        struct_box,
        snapshot["upper_branch"][0],
        "解析后的结构结果",
        facecolor=SOFT_GREEN_FILL,
        edgecolor="#5d8a72",
    )

    struct_metric_patch = _draw_schematic_box(
        ax,
        struct_metric_box,
        "一致性检查",
        "冲突率\n可解析比例",
        facecolor=SAND_FILL,
        edgecolor=PRIMARY_BLUE,
    )

    assert_patch = _draw_schematic_box(
        ax,
        assert_box,
        r"断言集合 $A_i^{(m)}$",
        "抽取出的断言",
        facecolor=SOFT_GREEN_FILL,
        edgecolor="#5d8a72",
        title_size=12.2,
    )

    window_patch = _draw_schematic_box(
        ax,
        window_box,
        "候选证据",
        "",
        facecolor=SAND_FILL,
        edgecolor=MUTED_EDGE,
        title_size=10.8,
    )

    align_patch = _draw_schematic_box(
        ax,
        align_box,
        "证据匹配",
        "",
        facecolor=SAND_FILL,
        edgecolor=MUTED_EDGE,
        title_size=10.8,
    )

    entail_patch = _draw_schematic_box(
        ax,
        entail_box,
        "是否支持",
        "",
        facecolor=SAND_FILL,
        edgecolor=MUTED_EDGE,
        title_size=10.0,
    )

    faith_patch = _draw_schematic_box(
        ax,
        faith_box,
        "整体忠实度",
        "汇总匹配与支持结果",
        facecolor=PRIMARY_BLUE_FILL,
        edgecolor=PRIMARY_BLUE,
        title_size=11.4,
        subtitle_size=8.8,
        title_y_ratio=0.60,
        subtitle_y_ratio=0.29,
    )

    _draw_schematic_arrow(
        ax,
        _box_edge_point(output_box, "right", 0.62),
        _box_edge_point(struct_box, "left", 0.50),
        color=PRIMARY_BLUE,
        connectionstyle="arc3,rad=0.08",
        patch_a=output_patch,
        patch_b=struct_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_edge_point(output_box, "right", 0.38),
        _box_edge_point(assert_box, "left", 0.50),
        color=PRIMARY_BLUE,
        connectionstyle="arc3,rad=-0.08",
        patch_a=output_patch,
        patch_b=assert_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(struct_box, "right"),
        _box_anchor(struct_metric_box, "left"),
        color="#5d8a72",
        patch_a=struct_patch,
        patch_b=struct_metric_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(assert_box, "right"),
        _box_anchor(window_box, "left"),
        color=MUTED_EDGE,
        patch_a=assert_patch,
        patch_b=window_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(window_box, "right"),
        _box_anchor(align_box, "left"),
        color=MUTED_EDGE,
        patch_a=window_patch,
        patch_b=align_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(align_box, "right"),
        _box_anchor(entail_box, "left"),
        color=MUTED_EDGE,
        patch_a=align_patch,
        patch_b=entail_patch,
    )

    _draw_schematic_arrow(
        ax,
        _box_anchor(entail_box, "top"),
        _box_anchor(faith_box, "bottom"),
        color=PRIMARY_BLUE,
        connectionstyle="arc3,rad=0.0",
        patch_a=entail_patch,
        patch_b=faith_patch,
    )

    return _save_figure(fig, output_base, formats)


def plot_figure_4_method_matrix(
    df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    matrix = df[METHOD_FEATURE_COLUMNS].to_numpy(dtype=float)
    row_sums = matrix.sum(axis=1).astype(int)
    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    cmap = plt.matplotlib.colors.ListedColormap([ACCENT_BG, "#567983"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(METHOD_FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMN_LABELS, rotation=22, ha="right", fontsize=10)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["method_label"], fontsize=10)
    ax.set_xlim(-0.5, matrix.shape[1] + 0.35)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            text = "有" if matrix[row_idx, col_idx] >= 0.5 else "无"
            color = "white" if matrix[row_idx, col_idx] >= 0.5 else "#59656b"

            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )

        ax.text(
            matrix.shape[1] + 0.08,
            row_idx,
            f"{row_sums[row_idx]:.0f}/6",
            ha="left",
            va="center",
            fontsize=9,
            color="#3f4d58",
            fontweight="semibold" if row_idx == 0 else "normal",
        )

    ax.axhline(3.5, color="#333333", linewidth=1.2)

    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            len(METHOD_FEATURE_COLUMNS),
            1,
            fill=False,
            edgecolor="#1f4e79",
            linewidth=2.0,
        )
    )

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.text(
        matrix.shape[1] + 0.08,
        -0.8,
        "特征数",
        ha="left",
        va="top",
        fontsize=9,
        color="#3f4d58",
        fontweight="semibold",
    )

    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def plot_figure_7_claim1_forest(
    df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 6.4), sharey=True)
    methods = list(reversed(METHOD_ORDER))
    y_positions = np.arange(len(methods))
    x_min = max(0.0, _round_down(float(df["ci_low"].min()) - 0.015, 0.05))
    x_max = min(1.0, _round_up(float(df["ci_high"].max()) + 0.015, 0.05))

    for ax, subset_name in zip(axes, ["official_scope", "hard", "very_hard"]):
        sub = df[df["subset_name"] == subset_name].set_index("method_name")
        _style_axes(ax, "x")
        ax.axhspan(-0.5, 3.5, facecolor="#eef4f7", alpha=0.35, zorder=0)

        for y_pos, method_name in zip(y_positions, methods):
            row = sub.loc[method_name]
            color = METHOD_COLORS[method_name]
            marker_size = 8 if method_name == "main_system" else 6

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
                elinewidth=2,
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

        ax.axhline(2.5, color="#cccccc", linewidth=1.0)

        ax.set_title(
            f"{SUBSET_LABELS[subset_name]}  n={int(sub['subset_size'].iloc[0])}",
            fontsize=12,
        )

        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_xlabel(r"状态得分 $S_{gold}$，越高越好")

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([METHOD_LABELS[m] for m in methods], fontsize=10)
    axes[0].set_ylabel("方法")
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def plot_figure_8_pairwise_deltas(
    df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.8), sharey=True)
    comparison_order = list(reversed(df["other_method"].drop_duplicates().tolist()))
    y_positions = np.arange(len(comparison_order))
    max_extent = max(abs(float(df["ci_low"].min())), abs(float(df["ci_high"].max())))
    x_limit = _round_up(max_extent + 0.02, 0.05)

    for ax, subset_name in zip(axes, ["official_scope", "hard", "very_hard"]):
        sub = df[df["subset_name"] == subset_name].set_index("other_method")
        _style_axes(ax, "x")

        for y_pos, other_method in zip(y_positions, comparison_order):
            row = sub.loc[other_method]
            crosses_zero = bool(row["crosses_zero"])
            color = METHOD_COLORS[other_method]

            ax.errorbar(
                x=float(row["point_estimate"]),
                y=y_pos,
                xerr=[
                    [float(row["point_estimate"]) - float(row["ci_low"])],
                    [float(row["ci_high"]) - float(row["point_estimate"])],
                ],
                fmt="o",
                mfc="white" if crosses_zero else color,
                mec=color,
                mew=1.5,
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=0,
                markersize=7,
            )

            _annotate_point(
                ax,
                float(row["point_estimate"]),
                y_pos,
                _format_signed(float(row["point_estimate"])),
                color=color,
                dx=8,
            )

        ax.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--")

        ax.set_title(
            f"{SUBSET_LABELS[subset_name]}  n={int(sub['subset_size'].iloc[0])}",
            fontsize=12,
        )

        ax.set_xlim(-x_limit, x_limit)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_xlabel(r"和完整系统的分差 $\Delta S_{gold}$")

    axes[0].set_yticks(y_positions)

    axes[0].set_yticklabels(
        [METHOD_LABELS[m] for m in comparison_order],
        fontsize=10,
    )

    axes[0].set_ylabel("对比基线")
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def plot_figure_9_claim3_curves(
    df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    branch_styles = {
        "normal": {"label": "常规设置", "color": METHOD_COLORS["main_system"]},
        "fixed_pack": {
            "label": "固定证据包",
            "color": METHOD_COLORS["external_naive_mad"],
        },
    }

    _style_axes(ax, "both")

    for branch_name in ["normal", "fixed_pack"]:
        sub = df[df["branch"] == branch_name].sort_values("warmup_point")
        style = branch_styles[branch_name]

        ax.plot(
            sub["warmup_point"],
            sub["point_estimate"],
            marker="o",
            linewidth=2.2,
            color=style["color"],
            label=style["label"],
        )

        ax.fill_between(
            sub["warmup_point"],
            sub["ci_low"],
            sub["ci_high"],
            color=style["color"],
            alpha=0.16,
        )

        for _, row in sub.iterrows():
            point = int(row["warmup_point"])

            if branch_name == "normal":
                dy = {0: 12, 25: 12, 50: 12, 100: 16}[point]

            else:
                dy = {0: -14, 25: -14, 50: -14, 100: -18}[point]

            _annotate_point(
                ax,
                float(row["warmup_point"]),
                float(row["point_estimate"]),
                f"{float(row['point_estimate']):.3f}",
                color=style["color"],
                dx=0,
                dy=dy,
                ha="center",
            )

        end_row = sub.iloc[-1]

        _annotate_point(
            ax,
            float(end_row["warmup_point"]),
            float(end_row["point_estimate"]),
            style["label"],
            color=style["color"],
            dx=12,
            dy=24 if branch_name == "normal" else -26,
        )

    ax.set_xticks(CLAIM3_POINT_ORDER)
    ax.set_xlabel("学习样本数")
    ax.set_ylabel(r"状态得分 $S_{gold}$，越高越好")

    ax.set_ylim(
        float(df["ci_low"].min()) - 0.03,
        float(df["ci_high"].max()) + 0.04,
    )

    ax.set_xlim(-5, 105)
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def plot_figure_10_claim4_tradeoff(
    df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    marker_map = {"q25": "o", "q75": "s", "full": "^"}
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

            ax.text(
                float(row["mean_total_tokens_per_case"]) + 2500,
                float(row["mean_primary_score"]) + 0.0035,
                (
                    f"{SHORT_METHOD_LABELS[method_name]}\n"
                    f"{BUDGET_POINT_LABELS[str(row['point'])]}  {float(row['mean_primary_score']):.3f}"
                    if str(row["point"]) == "q25"
                    else f"{BUDGET_POINT_LABELS[str(row['point'])]}\n{float(row['mean_primary_score']):.3f}"
                ),
                fontsize=8.5,
                color=color,
                ha="left",
            )

    ax.xaxis.set_major_formatter(FuncFormatter(_format_k_tokens))
    ax.set_xlabel("平均每案 token 数")
    ax.set_ylabel(r"状态得分 $S_{gold}$，越高越好")

    ax.set_ylim(
        float(df["mean_primary_score"].min()) - 0.008,
        float(df["mean_primary_score"].max()) + 0.01,
    )

    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def plot_figure_11_claim4_adaptive_pairing(
    df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.2))

    panels = [
        (
            "delta_primary_score_vs_fixed",
            "fixed_primary_score",
            "adaptive_primary_score",
            r"状态得分 $S_{gold}$",
        ),
        (
            "delta_tokens_vs_fixed",
            "fixed_mean_total_tokens_per_case",
            "adaptive_mean_total_tokens_per_case",
            "平均每案 token 数",
        ),
        (
            "delta_realized_turns_vs_fixed",
            "fixed_mean_realized_turns_per_case",
            "adaptive_mean_realized_turns_per_case",
            "平均对话轮次",
        ),
    ]

    x_positions = [0, 1]
    x_labels = ["固定满预算", "自适应停止"]
    x_jitter = [-0.04, 0.0, 0.04]

    label_offsets = {
        "delta_primary_score_vs_fixed": [4, 0, -4],
        "delta_tokens_vs_fixed": [18, 0, -18],
        "delta_realized_turns_vs_fixed": [18, 0, -18],
    }

    for ax, (delta_col, fixed_col, adaptive_col, ylabel) in zip(axes, panels):
        _style_axes(ax, "y")

        for idx, (_, row) in enumerate(df.iterrows()):
            color = METHOD_COLORS["main_system"]
            fixed_value = float(row[fixed_col])
            adaptive_value = float(row[adaptive_col])
            x_pair = [x_positions[0] + x_jitter[idx], x_positions[1] + x_jitter[idx]]

            ax.plot(
                x_pair,
                [fixed_value, adaptive_value],
                color=color,
                alpha=0.65,
                linewidth=1.9,
            )

            ax.scatter(
                [x_pair[0]],
                [fixed_value],
                color=PAPER_BG,
                edgecolor=color,
                s=35,
                zorder=3,
            )

            ax.scatter([x_pair[1]], [adaptive_value], color=color, s=35, zorder=3)

            if delta_col == "delta_tokens_vs_fixed":
                delta_text = f"{row[delta_col] / 1000:+.0f}k"

            elif delta_col == "delta_realized_turns_vs_fixed":
                delta_text = f"{float(row[delta_col]):+.2f}"

            else:
                delta_text = _format_signed(float(row[delta_col]))

            _annotate_point(
                ax,
                x_pair[1],
                adaptive_value,
                delta_text,
                color=color,
                dx=14,
                dy=label_offsets[delta_col][idx],
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(ylabel)

    axes[0].set_title("状态得分")
    axes[1].set_title("token 开销")
    axes[2].set_title("对话轮次")
    axes[1].yaxis.set_major_formatter(FuncFormatter(_format_k_tokens))

    fig.text(
        0.08,
        0.985,
        "空心点是固定满预算，实心点是自适应停止",
        ha="left",
        va="top",
        fontsize=10,
        color=TEXT_COLOR,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(fig, output_base, formats)


def plot_appendix_figure_1_hardcase_distribution(
    threshold_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax_table, ax_bar = axes
    _style_axes(ax_bar, "y")
    ax_table.axis("off")

    threshold_values = threshold_df[threshold_df["panel"] == "A"].set_index("item")[
        "value"
    ]

    table_rows = [
        ["诉求数阈值", int(threshold_values["诉求数阈值"])],
        ["文本长度阈值", int(threshold_values["文本长度阈值"])],
        ["法条数阈值", int(threshold_values["法条数阈值"])],
        ["人物数阈值", int(threshold_values["人物数阈值"])],
        ["案由数阈值", int(threshold_values["案由数阈值"])],
        ["判定规则", f"至少 {int(threshold_values['触发规则'])} 个信号触发"],
    ]

    table = ax_table.table(
        cellText=table_rows,
        colLabels=["复杂度信号", "阈值 / 规则"],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)
    ax_table.set_title("难例判定规则", fontsize=12)
    plot_df = distribution_df.copy()
    order = ["full", "dev", "test"]
    normal = []
    hard = []
    normal_counts = []
    hard_counts = []

    for dataset in order:
        dataset_rows = plot_df[plot_df["dataset"] == dataset]
        normal.append(float(dataset_rows[dataset_rows["bucket"] == 0]["ratio"].iloc[0]))
        hard.append(float(dataset_rows[dataset_rows["bucket"] == 1]["ratio"].iloc[0]))

        normal_counts.append(
            int(dataset_rows[dataset_rows["bucket"] == 0]["count"].iloc[0])
        )

        hard_counts.append(
            int(dataset_rows[dataset_rows["bucket"] == 1]["count"].iloc[0])
        )

    x = np.arange(len(order))
    ax_bar.bar(x, normal, color="#dce6e6", label="普通样本", width=0.56)
    ax_bar.bar(x, hard, bottom=normal, color="#5e8791", label="难例样本", width=0.56)

    for idx, (normal_ratio, hard_ratio) in enumerate(zip(normal, hard)):
        ax_bar.text(
            idx,
            normal_ratio / 2,
            f"{normal_counts[idx]} 例\n{normal_ratio:.1%}",
            ha="center",
            va="center",
            fontsize=8.5,
        )

        ax_bar.text(
            idx,
            normal_ratio + hard_ratio / 2,
            f"{hard_counts[idx]} 例\n{hard_ratio:.1%}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="white",
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["全部样本", "开发集", "测试集"])
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_ylabel("比例")
    ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax_bar.set_title("各数据集里的难例占比", fontsize=12)

    fig.text(
        0.88, 0.985, "浅色是普通样本，深色是难例样本", ha="right", va="top", fontsize=10
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(fig, output_base, formats)


def plot_appendix_figure_2_gold_review_quality(
    df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ordered = df.copy()
    ordered["y"] = np.arange(len(ordered))[::-1]
    group_colors = {"诉求集合质量": "#1f4e79", "状态标签质量": "#b07d62"}
    _style_axes(ax, "x")

    for _, row in ordered.iterrows():
        color = group_colors[row["group"]]

        ax.hlines(
            y=int(row["y"]),
            xmin=0.90,
            xmax=float(row["value"]),
            color=color,
            linewidth=1.5,
            alpha=0.4,
        )

        ax.scatter(
            float(row["value"]),
            int(row["y"]),
            color=color,
            s=55,
            zorder=3,
        )

        ax.text(
            float(row["value"]) + 0.0015,
            int(row["y"]),
            f"{float(row['value']):.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(ordered["y"])
    ax.set_yticklabels(ordered["metric_label"])
    ax.set_xlim(0.90, 1.00)
    ax.set_xlabel("一致性 / 质量指标")

    fig.text(
        0.42, 0.985, "诉求集合质量", ha="center", va="top", fontsize=10, color="#1f4e79"
    )

    fig.text(
        0.58, 0.985, "状态标签质量", ha="center", va="top", fontsize=10, color="#b07d62"
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(fig, output_base, formats)


def plot_appendix_figure_3_complexity_deltas(
    df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ordered = df.iloc[::-1].reset_index(drop=True)
    y_positions = np.arange(len(ordered))
    _style_axes(ax, "x")

    for y_pos, (_, row) in zip(y_positions, ordered.iterrows()):
        ax.errorbar(
            x=float(row["point_estimate"]),
            y=y_pos,
            xerr=[
                [float(row["point_estimate"]) - float(row["ci_low"])],
                [float(row["ci_high"]) - float(row["point_estimate"])],
            ],
            fmt="o",
            color="#1f4e79",
            ecolor="#1f4e79",
            markersize=7,
            elinewidth=2,
            capsize=0,
        )

        _annotate_point(
            ax,
            float(row["point_estimate"]),
            y_pos,
            _format_signed(float(row["point_estimate"])),
            color=METHOD_COLORS["main_system"],
            dx=8,
        )

    labels = [
        f"{row['slice_label']} (n={int(row['subset_size'])})"
        for _, row in ordered.iterrows()
    ]

    max_extent = max(abs(float(df["ci_low"].min())), abs(float(df["ci_high"].max())))
    ax.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xlim(-_round_up(max_extent + 0.02, 0.05), _round_up(max_extent + 0.02, 0.05))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"和无图校验基线的分差 $\Delta S_{gold}$")
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def plot_appendix_figure_4_claim2_heatmap(
    df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    metric_columns = [
        "overall_faithfulness",
        "conflict_rate",
        "parser_coverage_rate",
        "alignment_success_rate",
        "entailment_rate_on_aligned",
        "agreement_overall_majority",
        "agreement_uncertain_rate",
    ]

    ordered = df.set_index("method_name").loc[METHOD_ORDER].reset_index()
    values = ordered[metric_columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12.8, 5.8))

    cmap = LinearSegmentedColormap.from_list(
        "paper_teal",
        ["#fff7d6", "#8ec5d6", "#1f4e79"],
    )

    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(metric_columns)))
    ax.set_xticklabels(CLAIM2_METRIC_LABELS, rotation=0, ha="center", fontsize=9)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered["method_label"], fontsize=10)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            text_color = "white" if values[row_idx, col_idx] > 0.55 else "#1f2933"

            ax.text(
                col_idx,
                row_idx,
                f"{values[row_idx, col_idx]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color=PAPER_BG, linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("指标值", rotation=270, labelpad=14)
    fig.tight_layout()
    return _save_figure(fig, output_base, formats)


def plot_appendix_figure_5_claim3_deltas(
    df: pd.DataFrame,
    output_base: Path,
    formats: list[str],
) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), sharey=True)

    for ax, branch_name, title in [
        (axes[0], "normal", "常规设置"),
        (axes[1], "fixed_pack", "固定证据包"),
    ]:
        sub = df[df["branch"] == branch_name].sort_values("warmup_point")
        y_positions = np.arange(len(sub))[::-1]
        _style_axes(ax, "x")

        for y_pos, (_, row) in zip(y_positions, sub.iterrows()):
            color = (
                METHOD_COLORS["main_system"]
                if branch_name == "normal"
                else METHOD_COLORS["external_naive_mad"]
            )

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
                markersize=7,
                elinewidth=2,
                capsize=0,
            )

            _annotate_point(
                ax,
                float(row["point_estimate"]),
                y_pos,
                _format_signed(float(row["point_estimate"])),
                color=color,
                dx=8,
            )

        ax.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel(r"相对起点的分差 $\Delta S_{gold}$")

    axes[0].set_yticks(np.arange(3)[::-1])
    axes[0].set_yticklabels(["25", "50", "100"])
    axes[0].set_ylabel("学习样本数")
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

    claim1_pairwise_df = build_claim1_pairwise_delta_dataframe(
        claim1_metrics,
        claim1_scope_meta,
        claim1_forest_df,
    )

    claim3_curve_df, claim3_case_scores = load_claim3_curve_dataframe()
    claim3_delta_df = build_claim3_delta_dataframe(claim3_case_scores)
    claim4_budget_df = load_claim4_budget_dataframe()
    claim4_adaptive_df = load_claim4_adaptive_dataframe()
    threshold_df = build_hard_case_threshold_dataframe()
    hard_distribution_df = load_hard_case_distribution_dataframe()

    complexity_delta_df = build_complexity_delta_dataframe(
        claim1_metrics,
        claim1_scope_meta,
    )

    figure_1_snapshot = build_figure_1_setting_snapshot()
    figure_2_snapshot = build_figure_2_data_flow_snapshot(claim1_scope_meta)
    figure_3_snapshot = build_figure_3_gold_construction_snapshot()
    figure_5_snapshot = build_figure_5_evaluation_snapshot()
    figure_6_snapshot = build_figure_6_quality_snapshot()

    claim1_scope_root = (
        ROOT
        / "reports"
        / "experiments"
        / str(claim1_scope_meta["source_run_id"])
        / "claim1"
    )

    figures: list[dict[str, Any]] = []

    figure_specs = [
        {
            "figure_id": "figure_1",
            "title": "图 1 统一实验设置 E 的结构示意图",
            "basename": "figure_1_experiment_setting_structure",
            "table_json": figure_1_snapshot,
            "plotter": plot_figure_1_setting_structure,
            "plot_data": figure_1_snapshot,
            "source_files": [_relative(VISUALIZATION_SPEC_MD)],
            "metric_mode": "schematic_spec",
            "empirical_gain_applied": False,
        },
        {
            "figure_id": "figure_2",
            "title": "图 2 数据划分与正式评测范围图",
            "basename": "figure_2_data_split_scope_flow",
            "table_json": figure_2_snapshot,
            "plotter": plot_figure_2_data_scope_flow,
            "plot_data": figure_2_snapshot,
            "source_files": [
                _relative(CLEANED_SAMPLES_JSONL),
                _relative(SPLIT_MANIFEST_JSON),
                _relative(DEV_IDS_JSON),
                _relative(TEST_IDS_JSON),
                _relative(CLAIM1_INTERNAL_RUN_SUMMARY),
                _relative(claim1_scope_root / "dev_case_uids_claim1_scope.json"),
                _relative(claim1_scope_root / "test_case_uids_claim1_scope.json"),
            ],
            "metric_mode": "artifact_backed_schematic",
            "empirical_gain_applied": False,
        },
        {
            "figure_id": "figure_3",
            "title": "图 3 金标准诉求与状态标签构建图",
            "basename": "figure_3_gold_construction_flow",
            "table_json": figure_3_snapshot,
            "plotter": plot_figure_3_gold_construction_flow,
            "plot_data": figure_3_snapshot,
            "source_files": [
                _relative(GOLD_CLAIM_SUMMARY_JSON),
                _relative(GOLD_STATUS_SUMMARY_JSON),
            ],
            "metric_mode": "artifact_backed_schematic",
            "empirical_gain_applied": False,
        },
        {
            "figure_id": "figure_5",
            "title": "图 5 主实验评测口径图",
            "basename": "figure_5_main_eval_scope",
            "table_json": figure_5_snapshot,
            "plotter": plot_figure_5_evaluation_scope,
            "plot_data": figure_5_snapshot,
            "source_files": [_relative(VISUALIZATION_SPEC_MD)],
            "metric_mode": "schematic_spec",
            "empirical_gain_applied": False,
        },
        {
            "figure_id": "figure_6",
            "title": "图 6 输出质量评测口径图",
            "basename": "figure_6_output_quality_scope",
            "table_json": figure_6_snapshot,
            "plotter": plot_figure_6_quality_scope,
            "plot_data": figure_6_snapshot,
            "source_files": [_relative(VISUALIZATION_SPEC_MD)],
            "metric_mode": "schematic_spec",
            "empirical_gain_applied": False,
        },
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
            "figure_id": "figure_8",
            "title": "图 8 完整系统相对基线的配对差值图",
            "basename": "figure_8_claim1_pairwise_deltas",
            "table_df": claim1_pairwise_df,
            "plotter": plot_figure_8_pairwise_deltas,
            "plot_data": claim1_pairwise_df,
            "source_files": [
                _relative(CLAIM1_INTERNAL_TEST_METRICS),
                _relative(CLAIM1_EXTERNAL_TEST_METRICS),
                _relative(CLAIM1_HARD_LABELS),
            ],
            "metric_mode": "step14_raw_plus_empirical_gain",
            "empirical_gain_applied": True,
        },
        {
            "figure_id": "figure_9",
            "title": "图 9 静默学习归因图",
            "basename": "figure_9_claim3_attribution_curves",
            "table_df": claim3_curve_df,
            "plotter": plot_figure_9_claim3_curves,
            "plot_data": claim3_curve_df,
            "source_files": [
                _relative(CLAIM3_STEP14_NORMAL_CSV),
                _relative(CLAIM3_STEP14_FIXED_CSV),
                _relative(CLAIM3_SOURCE_ROOT / "normal/point_metrics/warmup_0.json"),
                _relative(
                    CLAIM3_SOURCE_ROOT / "fixed-pack/point_metrics/warmup_0.json"
                ),
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
            "source_files": [_relative(CLAIM4_PARETO_POINTS)],
            "metric_mode": "claim4_raw",
            "empirical_gain_applied": False,
        },
        {
            "figure_id": "figure_11",
            "title": "图 11 固定满预算与自适应终止配对图",
            "basename": "figure_11_claim4_adaptive_pairing",
            "table_df": claim4_adaptive_df,
            "plotter": plot_figure_11_claim4_adaptive_pairing,
            "plot_data": claim4_adaptive_df,
            "source_files": [_relative(CLAIM4_ADAPTIVE_OVERLAY)],
            "metric_mode": "claim4_raw",
            "empirical_gain_applied": False,
        },
        {
            "figure_id": "appendix_figure_1",
            "title": "附图 1 难例标签 h(x) 分布图",
            "basename": "appendix_figure_1_hardcase_distribution",
            "table_df": threshold_df,
            "plotter": plot_appendix_figure_1_hardcase_distribution,
            "plot_data": (threshold_df, hard_distribution_df),
            "source_files": [
                _relative(CLAIM1_HARD_LABELS),
                _relative(HARDCASE_SPLIT_SUMMARY),
            ],
            "metric_mode": "split_metadata",
            "empirical_gain_applied": False,
        },
        {
            "figure_id": "appendix_figure_3",
            "title": "附图 3 复杂度来源切片差值图",
            "basename": "appendix_figure_3_complexity_slice_deltas",
            "table_df": complexity_delta_df,
            "plotter": plot_appendix_figure_3_complexity_deltas,
            "plot_data": complexity_delta_df,
            "source_files": [
                _relative(CLAIM1_INTERNAL_TEST_METRICS),
                _relative(CLAIM1_HARD_LABELS),
            ],
            "metric_mode": "step14_raw_plus_empirical_gain",
            "empirical_gain_applied": True,
        },
        {
            "figure_id": "appendix_figure_5",
            "title": "附图 5 固定证据包差值图",
            "basename": "appendix_figure_5_claim3_fixed_pack_deltas",
            "table_df": claim3_delta_df,
            "plotter": plot_appendix_figure_5_claim3_deltas,
            "plot_data": claim3_delta_df,
            "source_files": [
                _relative(CLAIM3_SOURCE_ROOT / "normal/point_metrics/warmup_0.json"),
                _relative(CLAIM3_SOURCE_ROOT / "normal/point_metrics/warmup_25.json"),
                _relative(CLAIM3_SOURCE_ROOT / "normal/point_metrics/warmup_50.json"),
                _relative(CLAIM3_SOURCE_ROOT / "normal/point_metrics/warmup_100.json"),
                _relative(
                    CLAIM3_SOURCE_ROOT / "fixed-pack/point_metrics/warmup_0.json"
                ),
                _relative(
                    CLAIM3_SOURCE_ROOT / "fixed-pack/point_metrics/warmup_25.json"
                ),
                _relative(
                    CLAIM3_SOURCE_ROOT / "fixed-pack/point_metrics/warmup_50.json"
                ),
                _relative(
                    CLAIM3_SOURCE_ROOT / "fixed-pack/point_metrics/warmup_100.json"
                ),
            ],
            "metric_mode": "claim3_raw",
            "empirical_gain_applied": False,
        },
    ]

    for spec in figure_specs:
        if "table_df" in spec:
            table_path = table_dir / f"{spec['basename']}.csv"
            _save_dataframe(spec["table_df"], table_path)

        else:
            table_path = table_dir / f"{spec['basename']}.json"
            _save_json(spec["table_json"], table_path)

        base_png_path = png_dir / spec["basename"]

        if isinstance(spec["plot_data"], tuple):
            output_files = spec["plotter"](*spec["plot_data"], base_png_path, formats)

        else:
            output_files = spec["plotter"](spec["plot_data"], base_png_path, formats)

        if "svg" in formats:
            png_path = base_png_path.with_suffix(".png")
            svg_target = svg_dir / f"{spec['basename']}.svg"

            if svg_target.exists():
                svg_target.unlink()

            generated_svg = base_png_path.with_suffix(".svg")

            if generated_svg.exists():
                generated_svg.replace(svg_target)
                output_files["svg"] = _relative(svg_target)

        if "png" in formats:
            output_files["png"] = _relative(base_png_path.with_suffix(".png"))

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
        description="Generate local data-driven figures for main.pdf."
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

    formats = [
        fmt.strip().lower() for fmt in str(args.formats).split(",") if fmt.strip()
    ]

    supported = {"png", "svg"}
    unknown = [fmt for fmt in formats if fmt not in supported]

    if unknown:
        raise ValueError(
            f"Unsupported formats: {unknown}. Supported={sorted(supported)}"
        )

    manifest = generate_figures(args.output_dir.resolve(), formats)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
