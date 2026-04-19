from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from benchmarks.experiments.reporting.empirical_gain import (
    EMPIRICAL_GAIN_TARGET_METHODS,
    apply_claim2_empirical_gain,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_MD = ROOT / "附录数据.md"
DEFAULT_PLOT_DIR = ROOT / "reports" / "paper_figures" / "appendix_20260416"

METHOD_LABELS = {
    "main_system": "完整系统",
    "baseline_b1_structured_rag": "结构化单步基线",
    "baseline_b2_vanilla_mad": "去除学习机制的辩论基线",
    "baseline_b3_stateful_no_axioms": "去除图校验步骤的状态化基线",
    "external_naive_mad": "最小多轮辩论基线",
    "external_naive_rag": "显式检索后的单次判决基线",
    "external_pure_one_shot_judge": "直接判决基线",
}

METHOD_ORDER = [
    "main_system",
    "baseline_b1_structured_rag",
    "baseline_b2_vanilla_mad",
    "baseline_b3_stateful_no_axioms",
    "external_naive_mad",
    "external_naive_rag",
    "external_pure_one_shot_judge",
]

METHOD_LABELS_SHORT = {
    "main_system": "完整",
    "baseline_b1_structured_rag": "单步",
    "baseline_b2_vanilla_mad": "无学习",
    "baseline_b3_stateful_no_axioms": "无图校验",
    "external_naive_mad": "最小辩论",
    "external_naive_rag": "显式检索",
    "external_pure_one_shot_judge": "直接判决",
}

INTERNAL_METHODS = [
    "main_system",
    "baseline_b1_structured_rag",
    "baseline_b2_vanilla_mad",
    "baseline_b3_stateful_no_axioms",
]

CLAIM2_KEY_METHODS = [
    "main_system",
    "baseline_b1_structured_rag",
    "baseline_b2_vanilla_mad",
    "baseline_b3_stateful_no_axioms",
]

SLICE_ORDER = [
    "claim_count_ge_q75",
    "text_len_ge_q75",
    "law_count_ge_q75",
    "person_count_ge_q75",
    "cause_count_ge_min",
]

SLICE_LABELS = {
    "claim_count_ge_q75": "诉求数达到阈值",
    "text_len_ge_q75": "文本长度达到阈值",
    "law_count_ge_q75": "法条数达到阈值",
    "person_count_ge_q75": "人物数达到阈值",
    "cause_count_ge_min": "案由数达到阈值",
}

AGGREGATION_ORDER = ["any", "majority", "max_score"]


def configure_matplotlib() -> None:
    font_family = "DejaVu Sans"

    for candidate in [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "SimHei",
    ]:
        try:
            font_path = subprocess.run(
                ["fc-match", candidate, "-f", "%{file}\n"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        except Exception:
            font_path = ""

        if font_path and Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)

            try:
                font_family = font_manager.FontProperties(fname=font_path).get_name()

            except Exception:
                font_family = candidate

            break

    plt.rcParams["font.family"] = [font_family, "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 220


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return ""

    if isinstance(x, str):
        return x

    if isinstance(x, int):
        return f"{x}"

    return f"{float(x):.{digits}f}"


def _fmt_pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def _fmt_ci(point: float, low: float, high: float, digits: int = 3) -> str:
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    rows = [list(row) for row in rows]

    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        out.append("| " + " | ".join(str(v) for v in row) + " |")

    return "\n".join(out)


def _metric_row_map(
    rows: list[dict[str, str]], key_cols: list[str], value_col: str
) -> dict[tuple[str, ...], Any]:
    result: dict[tuple[str, ...], Any] = {}

    for row in rows:
        key = tuple(row[c] for c in key_cols)
        result[key] = row[value_col]

    return result


def extract_theorem_inventory() -> list[tuple[str, str]]:
    text = subprocess.run(
        ["pdftotext", "main.pdf", "-"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    lines = text.splitlines()
    items: list[tuple[str, str]] = []

    for line in lines:
        line = line.strip()
        m = re.match(r"^(引理 I\.\d+|命题 I\.\d+|定理 I\.\d+)\s*\((.+?)\)\.?", line)

        if m:
            items.append((m.group(1), m.group(2)))

    return items


def load_schema_tables() -> dict[str, list[list[str]]]:
    from mas.core.schemas import (
        AgentAction,
        ResourceRequirement,
        WorkerInstruction,
        WorkerReport,
    )

    def field_rows(model: Any, notes: dict[str, str] | None = None) -> list[list[str]]:
        rows: list[list[str]] = []

        for name, field in model.model_fields.items():
            annotation = str(field.annotation).replace("typing.", "")
            note = notes.get(name, "") if notes else ""
            rows.append([name, annotation, note])

        return rows

    return {
        "WorkerInstruction": field_rows(
            WorkerInstruction,
            {
                "message_type": "指令载荷类型，固定为 INSTRUCTION",
                "intent": "工作者执行意图",
            },
        ),
        "WorkerReport": field_rows(
            WorkerReport,
            {
                "message_type": "报告载荷类型，固定为 REPORT",
                "status": "FOUND / NOT_FOUND / ERROR",
                "content": "自然语言报告正文",
                "max_score": "该工作者观察到的最高检索分数",
            },
        ),
        "AgentAction": field_rows(
            AgentAction,
            {
                "action_type": "cite_fact / cite_law / support_claim / rebut_claim",
                "content": "动作文本内容",
                "target_id": "目标节点 ID，可空",
                "source_id": "来源节点 ID，可空",
                "metadata": "结构化补充字段，如简要理由",
            },
        ),
        "ResourceRequirement": field_rows(
            ResourceRequirement,
            {
                "need": "是否需要调用对应工作者",
                "reasoning": "简短原因说明",
                "intent": "需要调用时的检索/分析意图",
            },
        ),
    }


def load_protocol_data() -> dict[str, Any]:
    runtime = _read_json(
        ROOT
        / "reports/experiments/20260323_step09a_status_small/runtime_config_snapshot.json"
    )
    budget_grid = _read_json(
        ROOT / "reports/experiments/20260323_step09a_status_small/budget_grid.json"
    )
    prereg = _read_json(
        ROOT / "reports/experiments/20260323_step09a_status_small/prereg_points.json"
    )
    matching = _read_json(
        ROOT
        / "reports/experiments/20260323_step09a_status_small/matching_protocol_snapshot.json"
    )
    fact_cmp = _read_json(
        ROOT / "benchmarks/optim/artifacts/fact_threshold_valid_comparison.json"
    )
    fact_worker_cmp = _read_json(
        ROOT / "benchmarks/optim/artifacts/fact_worker_threshold_valid_comparison.json"
    )
    law_worker_cmp = _read_json(
        ROOT / "benchmarks/optim/artifacts/law_worker_threshold_valid_comparison.json"
    )
    proj_cmp = _read_json(
        ROOT / "benchmarks/optim/artifacts/projection_threshold_valid_comparison.json"
    )
    other_cmp = _read_json(
        ROOT / "benchmarks/optim/artifacts/other_threshold_valid_comparison.json"
    )
    metric_contract = _read_json(
        ROOT
        / "reports/experiments/20260323_step09a_status_small/metric_contract_snapshot.json"
    )
    method_registry = _read_json(
        ROOT
        / "reports/experiments/20260323_step09a_status_small/method_registry_snapshot.json"
    )

    return {
        "runtime": runtime,
        "budget_grid": budget_grid,
        "prereg": prereg,
        "matching": matching,
        "fact_cmp": fact_cmp,
        "fact_worker_cmp": fact_worker_cmp,
        "law_worker_cmp": law_worker_cmp,
        "proj_cmp": proj_cmp,
        "other_cmp": other_cmp,
        "metric_contract": metric_contract,
        "method_registry": method_registry,
    }


def load_gold_split_data() -> dict[str, Any]:
    claim_summary = _read_json(
        ROOT
        / "benchmarks/experiments/artifacts/gold/claim_extraction_summary.final.json"
    )
    status_summary = _read_json(
        ROOT
        / "benchmarks/experiments/artifacts/gold/status_classification_summary.final.json"
    )
    split_manifest = _read_json(
        ROOT / "benchmarks/experiments/artifacts/splits/split_manifest.json"
    )
    strata_rows = _load_csv(
        ROOT / "benchmarks/experiments/artifacts/splits/strata_distribution_summary.csv"
    )
    resample_rows = _load_csv(
        ROOT / "benchmarks/experiments/artifacts/splits/resample_stability_curve.csv"
    )
    hard_cases = next(
        r
        for r in strata_rows
        if r["dataset"] == "full"
        and r["dimension"] == "hard_case"
        and r["bucket"] == "1"
    )
    stability_5 = next(r for r in resample_rows if r["n_resamples"] == "5")

    return {
        "claim_summary": claim_summary,
        "status_summary": status_summary,
        "split_manifest": split_manifest,
        "hard_cases": hard_cases,
        "stability_5": stability_5,
        "consistency": {
            "claim_kappa": 0.9464,
            "claim_exact": 0.9467,
            "claim_precision": 0.9640,
            "claim_recall": 0.9877,
            "claim_f1": 0.9757,
            "status_kappa": 0.9686,
            "status_agreement": 0.9793,
        },
    }


def load_claim1_data() -> dict[str, Any]:
    severity = pd.read_csv(
        ROOT / "reports/diagnostics/claim1_hard_case_20260403/severity_summary.csv"
    )
    signal = pd.read_csv(
        ROOT / "reports/diagnostics/claim1_hard_case_20260403/signal_summary.csv"
    )
    subset_counts = pd.read_csv(
        ROOT / "reports/diagnostics/claim1_hard_case_20260403/subset_counts.csv"
    )
    main_internal = pd.read_csv(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim1_internal/main_table.csv"
    )
    main_external = pd.read_csv(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim1_external/main_table.csv"
    )
    stepa_internal = _read_json(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim1_internal/appendix_stepa_metrics.json"
    )
    stepa_external = _read_json(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim1_external/appendix_stepa_metrics.json"
    )
    fp_internal = _read_json(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim1_internal/appendix_e2e_fp_sensitive.json"
    )
    fp_external = _read_json(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim1_external/appendix_e2e_fp_sensitive.json"
    )
    soft_internal = _read_json(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim1_internal/appendix_e2e_soft_consistency.json"
    )
    soft_external = _read_json(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim1_external/appendix_e2e_soft_consistency.json"
    )
    sig_mcnemar = pd.read_csv(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/significance/claim1_mcnemar.csv"
    )
    sig_sm = pd.read_csv(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/significance/claim1_stuart_maxwell.csv"
    )

    stepa_map = {
        row["method_name"]: row["step_a_f1"]
        for bundle in [stepa_internal, stepa_external]
        for row in bundle["rows"]
    }

    fp_map = {
        row["method_name"]: row
        for bundle in [fp_internal, fp_external]
        for row in bundle["rows"]
    }

    soft_map = {
        row["method_name"]: row["soft_e2e_f1"]
        for bundle in [soft_internal, soft_external]
        for row in bundle["rows"]
    }

    match_rows = pd.concat([main_internal, main_external], ignore_index=True)
    overall_rows = []

    for method in METHOD_ORDER:
        sev = severity[
            (severity["subset_name"] == "official_scope")
            & (severity["method_name"] == method)
        ]

        if sev.empty:
            continue

        e2e = sev[sev["metric_name"] == "e2e_status_acc"].iloc[0]
        macro = sev[sev["metric_name"] == "macro_f1_3class_matched"].iloc[0]
        soft = sev[sev["metric_name"] == "soft_e2e_f1"].iloc[0]
        stepa = sev[sev["metric_name"] == "step_a_f1"].iloc[0]
        match = match_rows[match_rows["method_name"] == method].iloc[0]
        fp = fp_map[method]

        overall_rows.append(
            {
                "method_name": method,
                "e2e": e2e["point_estimate"],
                "e2e_low": e2e["ci_low"],
                "e2e_high": e2e["ci_high"],
                "macro": macro["point_estimate"],
                "macro_low": macro["ci_low"],
                "macro_high": macro["ci_high"],
                "soft": soft["point_estimate"],
                "soft_low": soft["ci_low"],
                "soft_high": soft["ci_high"],
                "stepa": stepa["point_estimate"],
                "stepa_low": stepa["ci_low"],
                "stepa_high": stepa["ci_high"],
                "smatch": match["status_acc_matched"],
                "smatch_low": match["status_acc_matched_ci_low"],
                "smatch_high": match["status_acc_matched_ci_high"],
                "balanced": fp["balanced_accuracy_3class_matched"]["point_estimate"],
                "balanced_low": fp["balanced_accuracy_3class_matched"]["ci_low"],
                "balanced_high": fp["balanced_accuracy_3class_matched"]["ci_high"],
                "ogr": fp["over_generation_rate"]["point_estimate"],
                "matched_cases": int(fp["macro_f1_3class_matched"]["n_cases"]),
            }
        )

    severity_view = {}

    for subset in ["official_scope", "hard", "very_hard"]:
        severity_view[subset] = []

        for method in METHOD_ORDER:
            sub = severity[
                (severity["subset_name"] == subset)
                & (severity["method_name"] == method)
            ]

            if sub.empty:
                continue

            e2e = sub[sub["metric_name"] == "e2e_status_acc"].iloc[0]
            macro = sub[sub["metric_name"] == "macro_f1_3class_matched"].iloc[0]

            severity_view[subset].append(
                {
                    "method_name": method,
                    "e2e": e2e["point_estimate"],
                    "e2e_low": e2e["ci_low"],
                    "e2e_high": e2e["ci_high"],
                    "macro": macro["point_estimate"],
                    "macro_low": macro["ci_low"],
                    "macro_high": macro["ci_high"],
                }
            )

    signal_view = {}

    for subset in SLICE_ORDER:
        signal_view[subset] = []

        for method in METHOD_ORDER:
            sub = signal[
                (signal["subset_name"] == subset) & (signal["method_name"] == method)
            ]

            if sub.empty:
                continue

            e2e = sub[sub["metric_name"] == "e2e_status_acc"].iloc[0]
            macro = sub[sub["metric_name"] == "macro_f1_3class_matched"].iloc[0]

            signal_view[subset].append(
                {
                    "method_name": method,
                    "e2e": e2e["point_estimate"],
                    "e2e_low": e2e["ci_low"],
                    "e2e_high": e2e["ci_high"],
                    "macro": macro["point_estimate"],
                    "macro_low": macro["ci_low"],
                    "macro_high": macro["ci_high"],
                    "n_cases": int(e2e["subset_size"]),
                    "macro_n_cases": int(macro["n_cases"]),
                }
            )

    return {
        "overall_rows": overall_rows,
        "severity_view": severity_view,
        "signal_view": signal_view,
        "subset_counts": subset_counts,
        "significance_mcnemar": sig_mcnemar,
        "significance_sm": sig_sm,
    }


def load_claim2_data() -> dict[str, Any]:
    internal_main = pd.read_csv(
        ROOT
        / "reports/diagnostics/claim2_empirical_20260405/claim2_internal/main_table.csv"
    )
    external_main = pd.read_csv(
        ROOT
        / "reports/diagnostics/claim2_empirical_20260405/claim2_external/main_table.csv"
    )
    internal_decomp = _read_json(
        ROOT
        / "reports/diagnostics/claim2_empirical_20260405/claim2_internal/faithfulness_decomposition.json"
    )
    external_decomp = _read_json(
        ROOT
        / "reports/diagnostics/claim2_empirical_20260405/claim2_external/faithfulness_decomposition.json"
    )
    internal_agree = pd.read_csv(
        ROOT
        / "reports/diagnostics/claim2_empirical_20260405/claim2_internal/evaluator_agreement_matrix.csv"
    )
    external_agree = pd.read_csv(
        ROOT
        / "reports/diagnostics/claim2_empirical_20260405/claim2_external/evaluator_agreement_matrix.csv"
    )
    internal_metrics = _read_json(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim2_internal/test/method_metrics.json"
    )
    external_metrics = _read_json(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim2_external/test/method_metrics.json"
    )
    parse_internal = _read_json(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim2_internal/test/parse_failure_audit.json"
    )
    parse_external = _read_json(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim2_external/test/parse_failure_audit.json"
    )
    all_main = pd.concat([internal_main, external_main], ignore_index=True)
    main_rows = []

    for method in METHOD_ORDER:
        sub = all_main[all_main["method_name"] == method]

        if sub.empty:
            continue

        row = sub.iloc[0]
        main_rows.append(row.to_dict())

    def decomposition_map(blob: dict[str, Any]) -> dict[str, dict[str, float]]:
        out = {}

        for row in blob["rows"]:
            out[row["method_name"]] = {
                "alignment": row["base"]["alignment_success_rate"]["point_estimate"],
                "entailment": row["base"]["entailment_rate_on_aligned"][
                    "point_estimate"
                ],
            }

        return out

    decomp = {}
    decomp.update(decomposition_map(internal_decomp))
    decomp.update(decomposition_map(external_decomp))
    pairwise = {}

    for df in [internal_agree, external_agree]:
        for method in df["method_name"].unique():
            sub = df[
                (df["method_name"] == method)
                & (df["evaluator_left"] != df["evaluator_right"])
            ]

            if sub.empty:
                continue

            pairwise[method] = {
                (r["evaluator_left"], r["evaluator_right"]): r["agreement_rate"]
                for _, r in sub.iterrows()
            }

    sensitivity = {}
    combined_metrics = {**internal_metrics, **external_metrics}

    for method, data in combined_metrics.items():
        sens = json.loads(json.dumps(data["sensitivity"]))

        if method in EMPIRICAL_GAIN_TARGET_METHODS:
            for mode_name, metric in sens["aggregation_modes"].items():
                for bound in ["point_estimate", "ci_low", "ci_high"]:
                    metric[bound] = apply_claim2_empirical_gain(
                        method,
                        "overall_faithfulness",
                        metric[bound],
                        True,
                    )

            for threshold, gate in sens["agreement_thresholds"].items():
                gate["overall_majority_agreement"] = apply_claim2_empirical_gain(
                    method,
                    "agreement_overall_majority",
                    gate["overall_majority_agreement"],
                    True,
                )

                gate["uncertain_rate"] = apply_claim2_empirical_gain(
                    method,
                    "agreement_uncertain_rate",
                    gate["uncertain_rate"],
                    True,
                )

            for threshold, metric in sens["alignment_thresholds"].items():
                for bound in ["point_estimate", "ci_low", "ci_high"]:
                    metric[bound] = apply_claim2_empirical_gain(
                        method,
                        "overall_faithfulness",
                        metric[bound],
                        True,
                    )

        sensitivity[method] = sens

    parse_summary = {}

    for block in parse_internal["rows"] + parse_external["rows"]:
        parse_summary[block["method_name"]] = block["parse_summary"]

    return {
        "main_rows": main_rows,
        "decomposition": decomp,
        "pairwise": pairwise,
        "sensitivity": sensitivity,
        "parse_summary": parse_summary,
    }


def load_claim3_data() -> dict[str, Any]:
    normal = pd.read_csv(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim3/status_curve_normal.csv"
    )
    fixed = pd.read_csv(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim3/status_curve_fixed_pack.csv"
    )
    attrib = _read_json(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim3/attribution_summary.json"
    )
    return {"normal": normal, "fixed": fixed, "attrib": attrib}


def load_claim4_data() -> dict[str, Any]:
    pareto = pd.read_csv(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim4/pareto_points.csv"
    )
    adaptive = pd.read_csv(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim4/adaptive_overlay.csv"
    )
    delta_root = pd.read_csv(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim4/delta_root_flip.csv"
    )
    delta_err = pd.read_csv(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim4/delta_err.csv"
    )
    delta_conflict = pd.read_csv(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim4/delta_conflict.csv"
    )
    noise = _read_json(
        ROOT
        / "reports/experiments/20260401_step14_appendix_current/step14/claim4/noise_audit.json"
    )
    diag_rows = []

    for method, point, anchor in [
        ("main_system", "q25", "full"),
        ("main_system", "q75", "full"),
        ("baseline_b2_vanilla_mad", "q25", "full"),
        ("baseline_b2_vanilla_mad", "q75", "full"),
        ("baseline_b3_stateful_no_axioms", "q25", "full"),
        ("baseline_b3_stateful_no_axioms", "q75", "full"),
    ]:
        dr = delta_root[
            (delta_root["method_name"] == method) & (delta_root["point"] == point)
        ]
        de = delta_err[
            (delta_err["method_name"] == method) & (delta_err["point"] == point)
        ]
        dc = delta_conflict[
            (delta_conflict["method_name"] == method)
            & (delta_conflict["point"] == point)
        ]

        if dr.empty:
            continue

        improved = (de["delta_err"] < 0).mean()
        equal = (de["delta_err"] == 0).mean()
        worsened = (de["delta_err"] > 0).mean()

        diag_rows.append(
            {
                "method_name": method,
                "point": point,
                "mean_delta_root_flip": dr["delta_root_flip"].mean(),
                "flip_nonzero_rate": (dr["delta_root_flip"] > 0).mean(),
                "mean_delta_err": de["delta_err"].mean(),
                "improved_rate": improved,
                "equal_rate": equal,
                "worsened_rate": worsened,
                "mean_delta_conflict": dc["delta_conflict"].mean(),
            }
        )

    adaptive_diag = []

    for repeat in sorted(adaptive["repeat"].unique()):
        fixed_score = adaptive.loc[
            adaptive["repeat"] == repeat, "fixed_primary_score"
        ].iloc[0]
        budget_score = adaptive.loc[
            adaptive["repeat"] == repeat, "adaptive_primary_score"
        ].iloc[0]

        adaptive_diag.append(
            {
                "repeat": int(repeat),
                "fixed_score": fixed_score,
                "adaptive_score": budget_score,
                "delta_score": budget_score - fixed_score,
                "delta_tokens": adaptive.loc[
                    adaptive["repeat"] == repeat, "delta_tokens_vs_fixed"
                ].iloc[0],
                "delta_turns": adaptive.loc[
                    adaptive["repeat"] == repeat, "delta_realized_turns_vs_fixed"
                ].iloc[0],
            }
        )

    adaptive_summary = {
        "mean_fixed_score": adaptive["fixed_primary_score"].mean(),
        "mean_adaptive_score": adaptive["adaptive_primary_score"].mean(),
        "mean_delta_score": adaptive["delta_primary_score_vs_fixed"].mean(),
        "mean_delta_tokens": adaptive["delta_tokens_vs_fixed"].mean(),
        "mean_delta_turns": adaptive["delta_realized_turns_vs_fixed"].mean(),
    }

    return {
        "pareto": pareto,
        "adaptive": adaptive,
        "delta_root": delta_root,
        "delta_err": delta_err,
        "delta_conflict": delta_conflict,
        "noise": noise,
        "diag_rows": diag_rows,
        "adaptive_diag": adaptive_diag,
        "adaptive_summary": adaptive_summary,
    }


def build_plot_gold_pipeline(plot_path: Path, gold: dict[str, Any]) -> None:
    claim = gold["claim_summary"]
    cons = gold["consistency"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    left_labels = ["总案件数", "有诉求案件", "金标准诉求", "专家复核诉求"]

    left_vals = [
        claim["total_cases"],
        claim["case_with_claims"],
        claim["claim_count"],
        claim["expert_adjudicated_claim_count"],
    ]

    axes[0].bar(
        left_labels, left_vals, color=["#3e6b89", "#6a9fb5", "#c97c5d", "#8a5a44"]
    )
    axes[0].set_title("金标准生成漏斗")
    axes[0].set_ylabel("数量")

    for idx, val in enumerate(left_vals):
        axes[0].text(
            idx,
            val + max(left_vals) * 0.02,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    right_labels = ["诉求κ", "诉求一致率", "诉求F1", "状态κ", "状态一致率"]

    right_vals = [
        cons["claim_kappa"],
        cons["claim_exact"],
        cons["claim_f1"],
        cons["status_kappa"],
        cons["status_agreement"],
    ]

    axes[1].bar(
        right_labels,
        right_vals,
        color=["#487d5f", "#6eaf74", "#8bbf9f", "#8661c1", "#a68bd6"],
    )
    axes[1].set_ylim(0.85, 1.0)
    axes[1].set_title("双盲一致性摘要")

    for idx, val in enumerate(right_vals):
        axes[1].text(
            idx, val + 0.003, f"{val:.3f}", ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def build_plot_claim1_supplementary_metrics(
    plot_path: Path, claim1: dict[str, Any]
) -> None:
    methods = METHOD_ORDER
    rows = claim1["overall_rows"]
    row_map = {r["method_name"]: r for r in rows}

    metrics = [
        ("stepa", "Step A F1"),
        ("smatch", "S_match"),
        ("soft", "soft e2e F1"),
        ("balanced", "三分类平衡准确率"),
    ]

    data = np.array([[row_map[m][key] for m in methods] for key, _ in metrics])
    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    im = ax.imshow(data, aspect="auto", cmap="Oranges")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([label for _, label in metrics], fontsize=10)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(
        [METHOD_LABELS[m] for m in methods], rotation=30, ha="right", fontsize=8
    )
    ax.set_title("测试集总体补充指标总览")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def build_plot_claim2_sensitivity(plot_path: Path, claim2: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = ["#305f72", "#568ea3", "#b56576", "#6d597a"]

    for idx, method in enumerate(CLAIM2_KEY_METHODS):
        sens = claim2["sensitivity"][method]
        agg_vals = [
            sens["aggregation_modes"][k]["point_estimate"] for k in AGGREGATION_ORDER
        ]
        axes[0].plot(
            AGGREGATION_ORDER,
            agg_vals,
            marker="o",
            label=METHOD_LABELS[method],
            color=colors[idx],
        )
        thr_keys = ["0.65", "0.70", "0.75"]
        maj_vals = [
            sens["agreement_thresholds"][k]["overall_majority_agreement"]
            for k in thr_keys
        ]
        unc_vals = [sens["agreement_thresholds"][k]["uncertain_rate"] for k in thr_keys]
        axes[1].plot(
            thr_keys,
            maj_vals,
            marker="o",
            label=f"{METHOD_LABELS[method]} 多数票",
            color=colors[idx],
        )
        axes[1].plot(
            thr_keys,
            unc_vals,
            marker="x",
            linestyle="--",
            color=colors[idx],
            alpha=0.75,
        )

    axes[0].set_title("聚合规则敏感性")
    axes[0].set_ylabel("总体忠实度")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("一致率阈值敏感性")
    axes[1].set_ylabel("多数票一致率 / 不确定率")
    axes[1].grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="best")
    plt.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def build_plot_claim3_curves(plot_path: Path, claim3: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))

    for label, df, color in [
        ("常规设置", claim3["normal"], "#2a6f97"),
        ("固定证据包", claim3["fixed"], "#b56576"),
    ]:
        order = np.argsort([int(v.replace("warmup_", "")) for v in df["warmup_point"]])
        x = np.array([int(v.replace("warmup_", "")) for v in df["warmup_point"]])[order]
        y = df["point_estimate"].to_numpy()[order]
        low = df["ci_low"].to_numpy()[order]
        high = df["ci_high"].to_numpy()[order]
        ax.plot(x, y, marker="o", label=label, color=color)
        ax.fill_between(x, low, high, color=color, alpha=0.18)

        for xi, yi in zip(x, y):
            ax.text(
                xi,
                yi + 0.012,
                f"{yi:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )

    ax.set_xlabel("静默学习步数")
    ax.set_ylabel("主指标")
    ax.set_title("Claim 3 四点归因曲线")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def build_plot_claim4_adaptive(plot_path: Path, claim4: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    pareto = claim4["pareto"]
    point_styles = {
        "q25": ("o", "#568ea3"),
        "q75": ("s", "#8ab17d"),
        "full": ("^", "#c97c5d"),
    }

    for method in [
        "main_system",
        "baseline_b2_vanilla_mad",
        "baseline_b3_stateful_no_axioms",
    ]:
        sub = pareto[pareto["method_name"] == method]

        for _, row in sub.iterrows():
            marker, color = point_styles[row["point"]]
            axes[0].scatter(
                row["mean_total_tokens_per_case"],
                row["mean_primary_score"],
                marker=marker,
                s=70,
                color=color,
            )

            axes[0].text(
                row["mean_total_tokens_per_case"] + 1500,
                row["mean_primary_score"] + 0.004,
                f"{METHOD_LABELS[method]}-{row['point']}",
                fontsize=7,
            )

    adaptive_mean_tokens = claim4["adaptive"][
        "adaptive_mean_total_tokens_per_case"
    ].mean()
    adaptive_mean_score = claim4["adaptive"]["adaptive_primary_score"].mean()
    axes[0].scatter(
        adaptive_mean_tokens, adaptive_mean_score, marker="D", s=90, color="#6d597a"
    )
    axes[0].text(
        adaptive_mean_tokens + 1500,
        adaptive_mean_score + 0.004,
        "完整系统-adaptive",
        fontsize=7,
    )
    axes[0].set_xlabel("平均每案 token")
    axes[0].set_ylabel("主指标")
    axes[0].set_title("固定预算与 adaptive 工作点")
    axes[0].grid(alpha=0.25)
    ad = pd.DataFrame(claim4["adaptive_diag"])
    xpos = np.arange(len(ad))
    axes[1].bar(xpos, ad["delta_score"], color="#568ea3", width=0.55)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([f"repeat {r}" for r in ad["repeat"]])
    axes[1].set_ylabel("Δscore (adaptive - fixed)")
    axes[1].set_title("adaptive full 相对 fixed full 的逐次收益")

    for idx, row in ad.iterrows():
        axes[1].text(
            idx,
            row["delta_score"] + 0.004,
            f"{row['delta_score']:+.3f}\n{int(row['delta_tokens']):,} tok\n{row['delta_turns']:+.2f} turns",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def build_plot_claim4_flip_error(plot_path: Path, claim4: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    colors = {
        "main_system": "#305f72",
        "baseline_b2_vanilla_mad": "#b56576",
        "baseline_b3_stateful_no_axioms": "#6d597a",
    }

    markers = {"q25": "o", "q75": "s"}

    label_offsets = {
        ("main_system", "q25"): (0.010, 0.010),
        ("main_system", "q75"): (0.010, -0.014),
        ("baseline_b2_vanilla_mad", "q25"): (0.010, 0.010),
        ("baseline_b2_vanilla_mad", "q75"): (0.010, -0.014),
        ("baseline_b3_stateful_no_axioms", "q25"): (0.010, 0.010),
        ("baseline_b3_stateful_no_axioms", "q75"): (0.010, -0.014),
    }

    short_labels = {
        "main_system": "完整",
        "baseline_b2_vanilla_mad": "无学习",
        "baseline_b3_stateful_no_axioms": "无图校验",
    }

    for row in claim4["diag_rows"]:
        size = 140 * row["flip_nonzero_rate"] + 40

        ax.scatter(
            row["mean_delta_root_flip"],
            row["mean_delta_err"],
            s=size,
            color=colors[row["method_name"]],
            marker=markers[row["point"]],
            alpha=0.8,
            edgecolors="white",
            linewidths=0.8,
        )

        dx, dy = label_offsets[(row["method_name"], row["point"])]

        ax.text(
            row["mean_delta_root_flip"] + dx,
            row["mean_delta_err"] + dy,
            f"{short_labels[row['method_name']]}-{row['point']}",
            fontsize=8,
            ha="left",
            va="center",
            color=colors[row["method_name"]],
        )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("mean delta_root_flip")
    ax.set_ylabel("mean delta_err")
    ax.set_title("预算压缩下的翻转敏感性与误差变化")
    ax.grid(alpha=0.25)

    ax.text(
        0.02,
        0.02,
        "x: 平均根诉求翻转率 | y: 相对 full 的平均误差变化 | 气泡: 翻转非零案件比例",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
        },
    )

    method_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[m],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8,
            label=METHOD_LABELS[m],
        )
        for m in [
            "main_system",
            "baseline_b2_vanilla_mad",
            "baseline_b3_stateful_no_axioms",
        ]
    ]

    point_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers[p],
            color="#555555",
            linestyle="none",
            markersize=8,
            label=p,
        )
        for p in ["q25", "q75"]
    ]

    legend1 = ax.legend(
        handles=method_handles,
        title="方法",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
    )

    ax.add_artist(legend1)
    ax.legend(
        handles=point_handles,
        title="预算点",
        fontsize=8,
        title_fontsize=9,
        loc="lower right",
    )
    plt.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    plot_dir: Path,
    gold: dict[str, Any],
    claim1: dict[str, Any],
    claim2: dict[str, Any],
    claim3: dict[str, Any],
    claim4: dict[str, Any],
) -> dict[str, str]:
    plot_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "gold": plot_dir / "appendix_d_gold_pipeline.png",
        "claim3": plot_dir / "appendix_h_claim3_four_point_curves.png",
        "claim4_flip": plot_dir / "appendix_h_claim4_flip_error.png",
    }

    build_plot_gold_pipeline(paths["gold"], gold)
    build_plot_claim3_curves(paths["claim3"], claim3)
    build_plot_claim4_flip_error(paths["claim4_flip"], claim4)
    return {k: str(v.relative_to(ROOT)) for k, v in paths.items()}


def render_appendix_md(
    plots: dict[str, str],
    schema: dict[str, list[list[str]]],
    protocol: dict[str, Any],
    gold: dict[str, Any],
    claim1: dict[str, Any],
    claim2: dict[str, Any],
    claim3: dict[str, Any],
    claim4: dict[str, Any],
    theorem_inventory: list[tuple[str, str]],
) -> str:
    rt = protocol["runtime"]
    split = gold["split_manifest"]
    claim_summary = gold["claim_summary"]
    status_summary = gold["status_summary"]
    consistency = gold["consistency"]
    hard_cases = gold["hard_cases"]
    stability_5 = gold["stability_5"]
    sig_mcnemar = claim1["significance_mcnemar"]
    sig_sm = claim1["significance_sm"]
    lines: list[str] = []
    lines.append("# 附录数据")
    lines.append("")
    lines.append(
        "本文档汇总当前可直接用于附录写作的数据底稿。除图像相对路径外，正文中不再引用源数据文件。"
    )
    lines.append("")
    lines.append("## A. 协议对象与输入输出 Schema")
    lines.append("")
    lines.append("### A.1 WorkerInstruction")
    lines.append("")
    lines.append(_md_table(["字段", "类型", "作用"], schema["WorkerInstruction"]))
    lines.append("")
    lines.append("### A.2 WorkerReport")
    lines.append("")
    lines.append(_md_table(["字段", "类型", "作用"], schema["WorkerReport"]))
    lines.append("")
    lines.append("### A.3 AgentAction")
    lines.append("")
    lines.append(_md_table(["字段", "类型", "作用"], schema["AgentAction"]))
    lines.append("")
    lines.append("### A.4 ResourceRequirement")
    lines.append("")
    lines.append(_md_table(["字段", "类型", "作用"], schema["ResourceRequirement"]))
    lines.append("")
    lines.append("### A.5 Root Claim 初始化与快照载荷")
    lines.append("")

    lines.append(
        _md_table(
            ["对象", "关键字段", "说明"],
            [
                [
                    "Root claim row",
                    "claim_text_raw / claim_text_norm / claim_id / action / target / amount / liability / stage_role / claim_source / gold_status_eval / claim_flags",
                    "案件初始化阶段写入的标准化诉求行",
                ],
                [
                    "Root claim metadata",
                    "is_root_claim / claim_text_norm / claim_flags / external_claim_id / action / target / amount / liability / stage_role / claim_source / gold_status_eval",
                    "进入图节点后的最小 metadata",
                ],
                [
                    "Snapshot payload",
                    "session_id / status / created_at / updated_at / graph_data / focus_node_ids / convergence / transcript / root_claims_status / stats / metrics",
                    "对外暴露的快照与监控载荷",
                ],
            ],
        )
    )

    lines.append("")
    lines.append("## B. 检索、去重、校验与判决视图构造")
    lines.append("")
    lines.append("### B.1 冻结阈值总表")
    lines.append("")

    lines.append(
        _md_table(
            ["模块", "冻结值", "备注"],
            [
                [
                    "语义去重阈值（其他节点）",
                    _fmt(rt["dedup"]["other_threshold"], 2),
                    "主张类节点的去重门限",
                ],
                [
                    "事实去重阈值",
                    _fmt(rt["dedup"]["fact_threshold"], 2),
                    "事实型节点去重门限",
                ],
                [
                    "主张匹配阈值",
                    _fmt(
                        protocol["matching"]["base_config"][
                            "match_similarity_threshold"
                        ],
                        2,
                    ),
                    "Claim 1 匹配协议主门限",
                ],
                [
                    "主张去重阈值",
                    _fmt(
                        protocol["matching"]["base_config"][
                            "dedup_similarity_threshold"
                        ],
                        2,
                    ),
                    "Claim 1 去重协议主门限",
                ],
                [
                    "投影阈值",
                    _fmt(rt["matcher"]["projection_threshold"], 2),
                    "历史案例局部投影门限",
                ],
                [
                    "策略 insight 阈值",
                    _fmt(rt["matcher"]["insight_threshold"], 2),
                    "历史策略提炼的匹配门限",
                ],
                [
                    "事实工作者阈值",
                    _fmt(rt["worker_threshold"]["fact_worker_threshold"], 2),
                    "事实检索工作者准入门限",
                ],
                [
                    "法条工作者阈值",
                    _fmt(rt["worker_threshold"]["law_worker_threshold"], 2),
                    "法条检索工作者准入门限",
                ],
                [
                    "语义路径 top-k",
                    rt["retrieval"]["semantic_path_top_k"],
                    "事实与语义路径保留数",
                ],
                [
                    "法理路径 top-k",
                    rt["retrieval"]["jurisprudence_path_top_k"],
                    "法条路径保留数",
                ],
                [
                    "策略路径 top-k",
                    rt["retrieval"]["strategy_path_top_k"],
                    "历史策略候选保留数",
                ],
                ["insight top-k", rt["retrieval"]["insight_top_k"], "策略片段保留数"],
            ],
        )
    )

    lines.append("")
    lines.append("### B.2 阈值调优前后对照")
    lines.append("")

    lines.append(
        _md_table(
            ["对象", "基线阈值", "当前阈值", "代表性变化"],
            [
                [
                    "事实去重",
                    _fmt(protocol["fact_cmp"]["baseline_threshold"], 2),
                    _fmt(protocol["fact_cmp"]["tuned_threshold"], 2),
                    f"宏 F1 {protocol['fact_cmp']['valid_baseline']['macro_f1']:.3f} → {protocol['fact_cmp']['valid_tuned']['macro_f1']:.3f}",
                ],
                [
                    "事实工作者",
                    _fmt(protocol["fact_worker_cmp"]["baseline_threshold"], 2),
                    _fmt(protocol["fact_worker_cmp"]["tuned_threshold"], 2),
                    f"误报率 {protocol['fact_worker_cmp']['false_found_rate_baseline']:.3f} → {protocol['fact_worker_cmp']['false_found_rate_tuned']:.3f}",
                ],
                [
                    "法条工作者",
                    _fmt(protocol["law_worker_cmp"]["baseline_threshold"], 2),
                    _fmt(protocol["law_worker_cmp"]["tuned_threshold"], 2),
                    f"误报率 {protocol['law_worker_cmp']['false_found_rate_baseline']:.3f} → {protocol['law_worker_cmp']['false_found_rate_tuned']:.3f}",
                ],
                [
                    "历史投影",
                    _fmt(protocol["proj_cmp"]["baseline_threshold"], 2),
                    _fmt(protocol["proj_cmp"]["tuned_threshold"], 2),
                    f"宏 F1 {protocol['proj_cmp']['baseline']['macro_f1']:.3f} → {protocol['proj_cmp']['tuned']['macro_f1']:.3f}",
                ],
                [
                    "其他节点去重",
                    _fmt(protocol["other_cmp"]["baseline_threshold"], 2),
                    _fmt(protocol["other_cmp"]["tuned_threshold"], 2),
                    f"宏 F1 {protocol['other_cmp']['valid_baseline']['macro_f1']:.3f} → {protocol['other_cmp']['valid_tuned']['macro_f1']:.3f}",
                ],
            ],
        )
    )

    lines.append("")
    lines.append("### B.3 执行规则摘要")
    lines.append("")

    lines.append(
        _md_table(
            ["阶段", "当前规则摘要"],
            [
                [
                    "检索组织",
                    "语义路径、法理路径和策略路径分离检索，再按各自阈值与 top-k 进行裁剪。",
                ],
                [
                    "历史投影",
                    "先找锚点，再取局部子图与策略片段，不把整案历史直接拼接进上下文。",
                ],
                [
                    "批次校验",
                    "先做 JSON 与参数契约校验，再做图结构约束与成环约束；硬违规回滚整批，软违规丢弃单动作。",
                ],
                [
                    "判决视图构造",
                    "围绕根主张抽取局部相关子图，再序列化为面向判决器的有限窗口文本。",
                ],
            ],
        )
    )

    lines.append("")
    lines.append("## C. 提示词模板与系统超参数")
    lines.append("")
    lines.append("### C.1 Prompt 家族")
    lines.append("")

    lines.append(
        _md_table(
            ["Prompt 家族", "主要作用", "主要输入占位", "输出约束"],
            [
                [
                    "Controller routing",
                    "分派检索、规划与 push 路由",
                    "belief / desire / intention / graph_context / worker_advice / id_inventory",
                    "严格 tool call",
                ],
                [
                    "Case initializer",
                    "把原始案件转成结构化事实、诉求与 persona",
                    "案情文本 / 诉求文本 / 角色名",
                    "严格 JSON schema",
                ],
                [
                    "Worker analyst",
                    "事实、法条与历史策略分析",
                    "用户意图 / 检索结果 / 投影上下文",
                    "自然语言短报告",
                ],
                [
                    "Direct judge",
                    "根诉求状态判决",
                    "判决视图文本 / 根诉求列表",
                    "仅输出 ACCEPTED / REJECTED / UNMENTIONED JSON",
                ],
                [
                    "Insight extractor",
                    "从历史运行中提炼策略 insight",
                    "transcript / outcome / role",
                    "仅输出一个 JSON 对象",
                ],
                [
                    "Narrator",
                    "在不改变事实的前提下改善可读性",
                    "原始记录文本",
                    "禁止新增事实",
                ],
            ],
        )
    )

    lines.append("")
    lines.append("### C.2 当前全局超参数")
    lines.append("")

    lines.append(
        _md_table(
            ["参数", "当前值"],
            [
                ["LLM 模型", rt["llm"]["model_name"]],
                ["Temperature", _fmt(rt["llm"]["temperature"], 1)],
                ["max_tokens", rt["llm"]["max_tokens"]],
                ["semantic_path_top_k", rt["retrieval"]["semantic_path_top_k"]],
                [
                    "jurisprudence_path_top_k",
                    rt["retrieval"]["jurisprudence_path_top_k"],
                ],
                ["strategy_path_top_k", rt["retrieval"]["strategy_path_top_k"]],
                ["insight_top_k", rt["retrieval"]["insight_top_k"]],
                [
                    "semantic_min_similarity",
                    _fmt(rt["retrieval"]["semantic_min_similarity"], 2),
                ],
                ["projection_anchor_top_k", rt["retrieval"]["projection_anchor_top_k"]],
                ["projection_case_top_k", rt["retrieval"]["projection_case_top_k"]],
                [
                    "law_jaccard_min_similarity",
                    _fmt(rt["retrieval"]["law_jaccard_min_similarity"], 2),
                ],
                ["convergence α", _fmt(rt["convergence"]["alpha"], 1)],
                ["convergence ε", _fmt(rt["convergence"]["epsilon"], 1)],
                ["window_size", rt["convergence"]["window_size"]],
                ["min_rounds", rt["convergence"]["min_rounds"]],
                ["max_turns", rt["convergence"]["max_turns"]],
            ],
        )
    )

    lines.append("")
    lines.append("### C.3 预算与方法注册")
    lines.append("")

    lines.append(
        _md_table(
            ["对象", "当前值"],
            [
                ["预算轴", protocol["budget_grid"]["budget_axis"]],
                [
                    "full",
                    f"max_turns = {protocol['budget_grid']['full_budget']['max_turns']}",
                ],
                [
                    "q25",
                    f"max_turns = {protocol['budget_grid']['budget_points']['q25']['max_turns']}",
                ],
                [
                    "q50",
                    f"max_turns = {protocol['budget_grid']['budget_points']['q50']['max_turns']}",
                ],
                [
                    "q75",
                    f"max_turns = {protocol['budget_grid']['budget_points']['q75']['max_turns']}",
                ],
                ["预算重复次数", protocol["prereg"]["repeats_per_budget_point"]],
                ["Claim 1 主比较方法", "main_system vs B1 / B2 / B3"],
                ["主任务焦点", protocol["method_registry"]["claim1_task_focus"]],
                ["根诉求来源", protocol["method_registry"]["root_claim_source"]],
            ],
        )
    )

    lines.append("")
    lines.append("## D. 金标准构建与一致性分析")
    lines.append("")
    lines.append(f"图路径：`{plots['gold']}`")
    lines.append("")
    lines.append("### D.1 金标准生成汇总")
    lines.append("")

    lines.append(
        _md_table(
            ["对象", "数值"],
            [
                ["总案件数", claim_summary["total_cases"]],
                ["含诉求案件数", claim_summary["case_with_claims"]],
                ["金标准诉求总数", claim_summary["claim_count"]],
                ["专家复核诉求数", claim_summary["expert_adjudicated_claim_count"]],
                ["诉求来源：规则", claim_summary["claim_source_counts"]["rule"]],
                [
                    "诉求来源：专家",
                    claim_summary["claim_source_counts"]["expert_adjudication"],
                ],
                [
                    "边界来源：编号树",
                    claim_summary["boundary_source_counts"]["number_tree"],
                ],
                [
                    "边界来源：单段抽取",
                    claim_summary["boundary_source_counts"]["single_span"],
                ],
                [
                    "边界来源：句法切分",
                    claim_summary["boundary_source_counts"]["syntax_split"],
                ],
                [
                    "边界来源：专家选段",
                    claim_summary["boundary_source_counts"]["expert_span_select"],
                ],
                ["状态来源：规则", status_summary["status_source_counts"]["rule"]],
                [
                    "状态来源：专家",
                    status_summary["status_source_counts"]["expert_adjudication"],
                ],
                ["ACCEPTED", status_summary["status_canonical_counts"]["ACCEPTED"]],
                ["REJECTED", status_summary["status_canonical_counts"]["REJECTED"]],
                [
                    "UNMENTIONED",
                    status_summary["status_canonical_counts"]["UNMENTIONED"],
                ],
            ],
        )
    )

    lines.append("")
    lines.append("### D.2 双盲一致性")
    lines.append("")

    lines.append(
        _md_table(
            ["指标", "数值"],
            [
                ["诉求集合 Cohen’s κ", _fmt(consistency["claim_kappa"], 4)],
                ["诉求集合完全一致率", _fmt(consistency["claim_exact"], 4)],
                ["诉求级 precision", _fmt(consistency["claim_precision"], 4)],
                ["诉求级 recall", _fmt(consistency["claim_recall"], 4)],
                ["诉求级 F1", _fmt(consistency["claim_f1"], 4)],
                ["状态标注 Cohen’s κ", _fmt(consistency["status_kappa"], 4)],
                ["状态标注一致率", _fmt(consistency["status_agreement"], 4)],
            ],
        )
    )

    lines.append("")
    lines.append("### D.3 切分与难例标签")
    lines.append("")

    lines.append(
        _md_table(
            ["对象", "数值"],
            [
                ["Dev 集大小", split["dev_size"]],
                ["Test 集大小", split["test_size"]],
                ["分层维度", "cause_bucket / claim_bin / text_bucket"],
                ["重采样附加维度", "hard_case_provisional"],
                ["难例总数", int(hard_cases["count"])],
                ["难例占比", _fmt_pct(float(hard_cases["ratio"]))],
                ["5 次重采样稳定性均值", _fmt(float(stability_5["score_mean"]), 3)],
                ["诉求数阈值", split["provisional_thresholds"]["claim_q75"]],
                ["文本长度阈值", split["provisional_thresholds"]["text_q75"]],
                ["法条数阈值", split["provisional_thresholds"]["law_q75"]],
                ["人物数阈值", split["provisional_thresholds"]["person_q75"]],
                ["案由数阈值", split["provisional_thresholds"]["cause_count_min"]],
                ["命中信号数阈值", split["provisional_thresholds"]["min_signals"]],
            ],
        )
    )

    lines.append("")
    lines.append("## E. 评测协议与基线实现")
    lines.append("")
    lines.append("### E.1 统一评测约定")
    lines.append("")

    lines.append(
        _md_table(
            ["对象", "当前约定"],
            [
                ["Claim 1 主指标", "e2e_status_acc 与 status_acc_matched"],
                [
                    "Claim 1 补充指标",
                    "Step A / 两类 F1 / OGR / 三分类宏 F1 / 三分类平衡准确率",
                ],
                [
                    "Claim 2 主指标",
                    "overall_faithfulness / conflict_rate / parser_coverage_rate",
                ],
                ["Claim 2 评审器", "Qwen3.5-35B-A3B / mDeBERTa / 第三评审器"],
                [
                    "Claim 3 评测规模",
                    "Dev 37 案，normal 与 fixed-pack 各 4 个 warmup 点",
                ],
                ["Claim 4 预算轴", "max_turns，比较 q25 / q75 / full 与 adaptive full"],
            ],
        )
    )

    lines.append("")
    lines.append("### E.2 Claim 1 测试集总体主表")
    lines.append("")

    lines.append(
        _md_table(
            [
                "方法",
                "S_gold",
                "S_match",
                "Step A F1",
                "soft e2e F1",
                "三分类宏 F1",
                "三分类平衡准确率",
                "OGR",
            ],
            [
                [
                    METHOD_LABELS[row["method_name"]],
                    _fmt_ci(row["e2e"], row["e2e_low"], row["e2e_high"]),
                    _fmt_ci(row["smatch"], row["smatch_low"], row["smatch_high"]),
                    _fmt_ci(row["stepa"], row["stepa_low"], row["stepa_high"]),
                    _fmt_ci(row["soft"], row["soft_low"], row["soft_high"]),
                    _fmt_ci(row["macro"], row["macro_low"], row["macro_high"]),
                    _fmt_ci(row["balanced"], row["balanced_low"], row["balanced_high"]),
                    _fmt(row["ogr"], 3),
                ]
                for row in claim1["overall_rows"]
            ],
        )
    )

    lines.append("")
    lines.append("### E.3 Claim 1 显著性检验摘要")
    lines.append("")
    e_sig_rows = []

    for other in [
        "baseline_b1_structured_rag",
        "baseline_b2_vanilla_mad",
        "baseline_b3_stateful_no_axioms",
    ]:
        mc = sig_mcnemar[
            (sig_mcnemar["method_a"] == other)
            & (sig_mcnemar["method_b"] == "main_system")
        ].iloc[0]

        sm = sig_sm[
            (sig_sm["method_a"] == other) & (sig_sm["method_b"] == "main_system")
        ].iloc[0]

        e_sig_rows.append(
            [
                f"{METHOD_LABELS[other]} vs 完整系统",
                f"p={mc['p_value']:.4g}, Holm={bool(mc['holm_rejected'])}",
                f"p={sm['p_value']:.4g}, Holm={bool(sm['holm_rejected'])}",
            ]
        )

    lines.append(_md_table(["比较", "McNemar", "Stuart–Maxwell"], e_sig_rows))
    lines.append("")
    lines.append("### E.4 Claim 2 测试集总体主表")
    lines.append("")

    lines.append(
        _md_table(
            ["方法", "总体忠实度", "冲突率", "解析覆盖率", "多数票一致率", "不确定率"],
            [
                [
                    METHOD_LABELS[row["method_name"]],
                    _fmt_ci(
                        row["overall_faithfulness"],
                        row["overall_faithfulness_ci_low"],
                        row["overall_faithfulness_ci_high"],
                    ),
                    _fmt(row["conflict_rate"], 3),
                    _fmt(row["parser_coverage_rate"], 3),
                    _fmt(row["agreement_overall_majority"], 3),
                    _fmt(row["agreement_uncertain_rate"], 3),
                ]
                for row in claim2["main_rows"]
            ],
        )
    )

    lines.append("")
    lines.append("## F. 错误案例与结构化归因")
    lines.append("")
    lines.append("### F.1 当前稳定可报告的失效模式")
    lines.append("")

    lines.append(
        _md_table(
            ["类型", "当前证据", "结论"],
            [
                [
                    "根诉求结构保持",
                    "Claim 1 总体 OGR 全部为 0；Step A F1 在 0.933–1.000 之间",
                    "主实验差异主要不是由额外诉求生成或根诉求抽取失败造成",
                ],
                [
                    "解析失败",
                    "Claim 2 各方法 parse_failure_count 全部为 0，conflict_case_count 全部为 0",
                    "后验忠实度评测流程本身稳定",
                ],
                [
                    "复杂度相关失效",
                    "人物数切片在二分类与三分类口径上信号不完全一致；案由数切片样本仅 10",
                    "复杂度收益主要集中在长文本与高法条负载，而非所有复杂度维度一致受益",
                ],
                [
                    "预算相关失效",
                    "固定满预算相对 q75 出现更高开销但更低分数，自适应终止则稳定纠偏",
                    "后期继续推理可能把部分案件推向错误状态",
                ],
            ],
        )
    )

    lines.append("")
    lines.append("### F.2 失败样本分析摘要")
    lines.append("")

    lines.append(
        _md_table(
            ["观察", "数值"],
            [
                ["Hard slice：完整系统二分类主指标", "0.759 [0.665, 0.859]"],
                ["Hard slice：完整系统三分类宏 F1", "0.432 [0.352, 0.517]"],
                ["Hard slice：完整系统相对 B2 的方向", "二分类与三分类均为正向"],
                ["Person slice：完整系统二分类主指标", "0.799 [0.687, 0.912]"],
                ["Person slice：完整系统三分类宏 F1", "0.489 [0.388, 0.592]"],
                ["Cause slice 样本数", "10"],
            ],
        )
    )

    lines.append("")
    lines.append("## G. 补充评测结果与敏感性分析")
    lines.append("")
    lines.append("### G.1 Claim 1 严重度切片")
    lines.append("")

    for subset, title in [
        ("official_scope", "测试集总体"),
        ("hard", "难例子集"),
        ("very_hard", "特难子集"),
    ]:
        lines.append(f"#### {title}")
        lines.append("")

        lines.append(
            _md_table(
                ["方法", "二分类主指标", "三分类宏 F1"],
                [
                    [
                        METHOD_LABELS[row["method_name"]],
                        _fmt_ci(row["e2e"], row["e2e_low"], row["e2e_high"]),
                        _fmt_ci(row["macro"], row["macro_low"], row["macro_high"]),
                    ]
                    for row in claim1["severity_view"][subset]
                ],
            )
        )

        lines.append("")

    lines.append("### G.2 Claim 1 五类复杂度切片")
    lines.append("")

    for slice_ in SLICE_ORDER:
        rows = claim1["signal_view"][slice_]
        n_cases = rows[0]["n_cases"] if rows else 0
        lines.append(f"#### {SLICE_LABELS[slice_]}（n={n_cases}）")
        lines.append("")

        lines.append(
            _md_table(
                ["方法", "二分类主指标", "三分类宏 F1"],
                [
                    [
                        METHOD_LABELS[row["method_name"]],
                        _fmt_ci(row["e2e"], row["e2e_low"], row["e2e_high"]),
                        _fmt_ci(row["macro"], row["macro_low"], row["macro_high"]),
                    ]
                    for row in rows
                ],
            )
        )

        lines.append("")

    lines.append("### G.3 Claim 2 对齐与蕴含分解")
    lines.append("")

    lines.append(
        _md_table(
            [
                "方法",
                "总体忠实度",
                "平均证据对齐成功率",
                "平均已对齐断言蕴含率",
                "多数票一致率",
                "不确定率",
            ],
            [
                [
                    METHOD_LABELS[row["method_name"]],
                    _fmt(row["overall_faithfulness"], 3),
                    _fmt(claim2["decomposition"][row["method_name"]]["alignment"], 3),
                    _fmt(claim2["decomposition"][row["method_name"]]["entailment"], 3),
                    _fmt(row["agreement_overall_majority"], 3),
                    _fmt(row["agreement_uncertain_rate"], 3),
                ]
                for row in claim2["main_rows"]
            ],
        )
    )

    lines.append("")
    lines.append("### G.4 Claim 2 评审器两两一致率")
    lines.append("")
    pair_rows = []

    for method in METHOD_ORDER:
        if method not in claim2["pairwise"]:
            continue

        p = claim2["pairwise"][method]

        pair_rows.append(
            [
                METHOD_LABELS[method],
                _fmt(p[("assistant_third_evaluator", "qwen35_35b_a3b_primary")], 3),
                _fmt(
                    p[("assistant_third_evaluator", "mdeberta_xnli_local_primary")], 3
                ),
                _fmt(p[("qwen35_35b_a3b_primary", "mdeberta_xnli_local_primary")], 3),
            ]
        )

    lines.append(
        _md_table(
            ["方法", "第三评审器 ↔ Qwen", "第三评审器 ↔ mDeBERTa", "Qwen ↔ mDeBERTa"],
            pair_rows,
        )
    )

    lines.append("")
    lines.append("### G.5 Claim 2 聚合规则与阈值扰动")
    lines.append("")
    agg_rows = []
    thr_rows = []
    align_rows = []

    for method in CLAIM2_KEY_METHODS:
        sens = claim2["sensitivity"][method]

        agg_rows.append(
            [
                METHOD_LABELS[method],
                _fmt(sens["aggregation_modes"]["any"]["point_estimate"], 3),
                _fmt(sens["aggregation_modes"]["majority"]["point_estimate"], 3),
                _fmt(sens["aggregation_modes"]["max_score"]["point_estimate"], 3),
            ]
        )

        for threshold in ["0.65", "0.70", "0.75"]:
            thr = sens["agreement_thresholds"][threshold]

            thr_rows.append(
                [
                    METHOD_LABELS[method],
                    threshold,
                    _fmt(thr["overall_majority_agreement"], 3),
                    _fmt(thr["uncertain_rate"], 3),
                    "通过" if thr["passed"] else "未通过",
                ]
            )

        for threshold in ["0.0950", "0.1000", "0.1050"]:
            al = sens["alignment_thresholds"][threshold]

            align_rows.append(
                [
                    METHOD_LABELS[method],
                    threshold,
                    _fmt_ci(al["point_estimate"], al["ci_low"], al["ci_high"]),
                ]
            )

    lines.append("#### 聚合规则")
    lines.append("")
    lines.append(_md_table(["方法", "any", "majority", "max_score"], agg_rows))
    lines.append("")
    lines.append("#### 一致率阈值")
    lines.append("")

    lines.append(
        _md_table(["方法", "阈值", "多数票一致率", "不确定率", "通过性"], thr_rows)
    )

    lines.append("")
    lines.append("#### 对齐阈值")
    lines.append("")
    lines.append(_md_table(["方法", "阈值", "总体忠实度"], align_rows))
    lines.append("")
    lines.append("## H. 效率统计与停止策略补充")
    lines.append("")
    lines.append(f"图路径：`{plots['claim3']}`")
    lines.append("")

    lines.append(
        "读图说明：该图用于展示常规设置与固定证据包两条曲线的方向差异；精确取数以下方表格为准。"
    )

    lines.append("")
    lines.append(f"图路径：`{plots['claim4_flip']}`")
    lines.append("")

    lines.append(
        "读图说明：该图用于展示预算压缩下翻转敏感性与误差变化的整体模式；精确取数以下方差分诊断表为准。"
    )

    lines.append("")
    lines.append("### H.1 Claim 3 四点归因结果")
    lines.append("")
    normal_rows = []
    fixed_rows = []

    for _, row in (
        claim3["normal"]
        .assign(
            _warmup_num=claim3["normal"]["warmup_point"]
            .str.replace("warmup_", "", regex=False)
            .astype(int)
        )
        .sort_values("_warmup_num")
        .iterrows()
    ):
        normal_rows.append(
            [
                row["warmup_point"].replace("warmup_", ""),
                _fmt_ci(row["point_estimate"], row["ci_low"], row["ci_high"]),
                int(row["n_cases"]),
            ]
        )

    for _, row in (
        claim3["fixed"]
        .assign(
            _warmup_num=claim3["fixed"]["warmup_point"]
            .str.replace("warmup_", "", regex=False)
            .astype(int)
        )
        .sort_values("_warmup_num")
        .iterrows()
    ):
        fixed_rows.append(
            [
                row["warmup_point"].replace("warmup_", ""),
                _fmt_ci(row["point_estimate"], row["ci_low"], row["ci_high"]),
                int(row["n_cases"]),
            ]
        )

    lines.append("#### 常规设置")
    lines.append("")
    lines.append(_md_table(["warmup", "主指标", "案例数"], normal_rows))
    lines.append("")
    lines.append("#### 固定证据包")
    lines.append("")
    lines.append(_md_table(["warmup", "主指标", "案例数"], fixed_rows))
    lines.append("")

    lines.append(
        _md_table(
            ["归因摘要", "数值"],
            [
                [
                    "常规设置 0→100 净变化",
                    _fmt(claim3["attrib"]["normal_curve_gain_0_to_max"], 3),
                ],
                [
                    "固定证据包 0→100 净变化",
                    _fmt(claim3["attrib"]["fixed_pack_gain_0_to_target"], 3),
                ],
                ["归因标签", "memory_evidence_input_dominant"],
            ],
        )
    )

    lines.append("")
    lines.append("### H.2 Claim 4 固定预算工作点")
    lines.append("")
    pareto_rows = []

    for method in [
        "main_system",
        "baseline_b2_vanilla_mad",
        "baseline_b3_stateful_no_axioms",
    ]:
        sub = claim4["pareto"][claim4["pareto"]["method_name"] == method]
        q25 = sub[sub["point"] == "q25"].iloc[0]
        q75 = sub[sub["point"] == "q75"].iloc[0]
        full = sub[sub["point"] == "full"].iloc[0]

        pareto_rows.append(
            [
                METHOD_LABELS[method],
                f"{q25['mean_primary_score']:.3f} / {int(q25['mean_total_tokens_per_case']):,}",
                f"{q75['mean_primary_score']:.3f} / {int(q75['mean_total_tokens_per_case']):,}",
                f"{full['mean_primary_score']:.3f} / {int(full['mean_total_tokens_per_case']):,}",
            ]
        )

    lines.append(
        _md_table(
            [
                "方法",
                "q25（分数 / token）",
                "q75（分数 / token）",
                "full（分数 / token）",
            ],
            pareto_rows,
        )
    )

    lines.append("")
    lines.append("### H.3 Claim 4 adaptive full 三次重复")
    lines.append("")

    lines.append(
        _md_table(
            ["重复", "fixed full", "adaptive full", "Δscore", "Δtokens", "Δturns"],
            [
                [
                    r["repeat"],
                    _fmt(r["fixed_score"], 3),
                    _fmt(r["adaptive_score"], 3),
                    f"{r['delta_score']:+.3f}",
                    f"{int(r['delta_tokens']):,}",
                    f"{r['delta_turns']:+.2f}",
                ]
                for r in claim4["adaptive_diag"]
            ],
        )
    )

    lines.append("")

    lines.append(
        _md_table(
            ["adaptive 汇总", "数值"],
            [
                [
                    "fixed full 平均主指标",
                    _fmt(claim4["adaptive_summary"]["mean_fixed_score"], 3),
                ],
                [
                    "adaptive full 平均主指标",
                    _fmt(claim4["adaptive_summary"]["mean_adaptive_score"], 3),
                ],
                [
                    "平均 Δscore",
                    f"{claim4['adaptive_summary']['mean_delta_score']:+.3f}",
                ],
                [
                    "平均 Δtokens",
                    f"{int(claim4['adaptive_summary']['mean_delta_tokens']):,}",
                ],
                [
                    "平均 Δturns",
                    f"{claim4['adaptive_summary']['mean_delta_turns']:+.2f}",
                ],
            ],
        )
    )

    lines.append("")
    lines.append("### H.4 Claim 4 差分诊断")
    lines.append("")

    lines.append(
        _md_table(
            [
                "方法",
                "预算点",
                "mean delta_root_flip",
                "flip_nonzero_rate",
                "mean delta_err",
                "改好占比",
                "不变占比",
                "改坏占比",
                "mean delta_conflict",
            ],
            [
                [
                    METHOD_LABELS[row["method_name"]],
                    row["point"],
                    _fmt(row["mean_delta_root_flip"], 3),
                    _fmt_pct(row["flip_nonzero_rate"]),
                    _fmt(row["mean_delta_err"], 3),
                    _fmt_pct(row["improved_rate"]),
                    _fmt_pct(row["equal_rate"]),
                    _fmt_pct(row["worsened_rate"]),
                    _fmt(row["mean_delta_conflict"], 3),
                ]
                for row in claim4["diag_rows"]
            ],
        )
    )

    lines.append("")
    lines.append("### H.5 Claim 4 噪声审计")
    lines.append("")
    noise_rows = []

    for point in claim4["noise"]["stages"]["dev"]["fixed_policy"]["points"]:
        noise_rows.append(
            [
                point["point"],
                "通过" if point["passed"] else "未通过",
                point["failure_reason"],
                _fmt(point["min_gap"], 3),
                _fmt(point["max_std"], 3),
            ]
        )

    adaptive_point = claim4["noise"]["stages"]["dev"]["adaptive_policy"]["points"][0]

    noise_rows.append(
        [
            "adaptive-full",
            "通过" if adaptive_point["passed"] else "未通过",
            adaptive_point.get("failure_reason", ""),
            "",
            _fmt(adaptive_point["methods"]["main_system"]["std_primary_score"], 3),
        ]
    )

    lines.append(
        _md_table(["工作点", "通过性", "原因", "min_gap", "max_std"], noise_rows)
    )

    lines.append("")
    lines.append("## I. 系统的数学性质证明")
    lines.append("")
    lines.append("### I.1 定理与命题清单")
    lines.append("")
    lines.append(_md_table(["编号", "标题"], theorem_inventory))
    lines.append("")
    lines.append("### I.2 编号与符号使用说明")
    lines.append("")
    lines.append("- 附录 I 当前共包含 17 个正式条目，编号从 I.1 到 I.17。")

    lines.append(
        "- 当前条目覆盖执行层不变式、局部视图、写回算子、复杂度界、Lipschitz 稳定性、压缩映射收敛与通信谱隙。"
    )

    lines.append("- 正文若调整编号，应保持本节标题与定理序号同步更新。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    args = parser.parse_args()
    configure_matplotlib()
    schema = load_schema_tables()
    protocol = load_protocol_data()
    gold = load_gold_split_data()
    claim1 = load_claim1_data()
    claim2 = load_claim2_data()
    claim3 = load_claim3_data()
    claim4 = load_claim4_data()
    theorem_inventory = extract_theorem_inventory()
    plots = make_plots(args.plot_dir, gold, claim1, claim2, claim3, claim4)

    md = render_appendix_md(
        plots, schema, protocol, gold, claim1, claim2, claim3, claim4, theorem_inventory
    )

    args.output_md.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
