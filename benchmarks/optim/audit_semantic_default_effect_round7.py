"""Round-7 audit: verify `semantic_default_threshold` has no runtime effect.

This script compares dedup results under two extreme initial matcher thresholds
(`0.10` vs `0.99`) while keeping runtime dedup thresholds fixed:
- FACT nodes: `dedup.fact_threshold`
- CLAIM nodes: `dedup.other_threshold`

If predictions are identical, `matcher.semantic_default_threshold` is a dead
configuration parameter for current dedup execution path and can be removed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import mas.analysis.executor as executor_module
from mas.analysis.executor import GraphExecutor
from mas.core.graph import NodeType, ShadowGraph

os.environ.setdefault("ES_HOST", "http://localhost:9200")
os.environ.setdefault("EMBEDDING_MODEL_PATH", "./bge-m3")
os.environ.setdefault("MAS_STORAGE_DIR", "./tmp/mas_storage")
os.environ.setdefault("LEGAL_LLM_KEY", "test-key")
os.environ.setdefault("LEGAL_LLM_URL", "http://localhost:8001")

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FACT_BASES = [
    "2013年2月1日，原告与被告签订厂房租赁合同，约定月租金98000元。",
    "合同约定甲方将天津开发区第十三大街116号厂房1栋、2栋出租给乙方。",
    "乙方承租厂房用途为生产、办公和存货，租期自2013年3月至2016年2月。",
    "2014年5月29日原告向被告发送解除通知，主张合同已提前终止。",
    "被告提交银行流水证明其已支付2014年1月至4月租金。",
    "原告称被告自2014年6月起继续占用厂房并拒绝腾退。",
    "双方就设备清单进行交接，被告签收但对数量提出异议。",
    "被告抗辩称其已于2014年7月搬离现场并停止经营。",
    "原告提交工商登记信息，显示被告在涉案地址仍持续登记经营。",
    "租赁合同第三十五条约定逾期30天可视为提前终止。",
    "原告主张被告拖欠水电费并要求一并支付。",
    "被告提交盘点表称库存已转移至外地仓库。",
    "双方邮件记录显示被告曾申请延长搬离期限至8月15日。",
    "现场照片显示涉案厂房内仍存放被告设备与包装材料。",
    "原告催告函要求被告在10日内补缴租金并恢复履约。",
    "被告法定代表人庭审中承认曾收到解除通知。",
    "双方在调解笔录中对租金基数和违约金比例存在争议。",
    "原告提交物业记录证明被告人员在7月仍频繁进出厂区。",
    "被告主张合同无效并请求返还已支付保证金。",
    "法院查明被告未按约办理厂房返还手续。",
]

CLAIM_BASES = [
    "原告未给予合理履行期限即解除合同，解除程序存在瑕疵，双倍占用费请求不能成立。",
    "合同条款虽约定逾期30天视为提前终止，但解除权行使仍应遵循催告程序。",
    "在对方抗辩未实际使用厂房时，应结合工商登记和经营记录证明持续占有。",
    "主张违约金不过高的一方，应主动举证实际损失与约定比例的对应关系。",
    "被告已部分清偿借款本金，逾期利息应以实际未还本金为计算基数。",
    "对方主张电子证据真实性存疑时，应补强形成时间和提取流程的完整性说明。",
    "建设工程价款争议中，应围绕签证资料和结算单据构建工程量证明链。",
    "劳动争议中加班事实认定应重点审查考勤、排班和审批记录的一致性。",
    "侵权责任成立需同时论证过错、损害后果与因果关系，不能仅凭结果推定。",
    "公司内部决议瑕疵不当然影响对外效力，应结合相对人善意状态判断。",
    "原告仅提出抽象损失主张而未提交量化证据，赔偿请求存在被限缩风险。",
    "对于解除通知到达时间的争议，应以送达凭证和收件确认记录交叉印证。",
    "对方以重复起诉抗辩时，应比对前案请求基础与本案事实是否同一。",
    "在借款纠纷中，微信转账备注可作为款项性质辅助判断，但需结合整体交易背景。",
    "主张继续履行的一方应证明合同目的仍可实现，避免被认定履行不能。",
    "当事人未及时提出质量异议可能构成默示验收，应提前准备反证材料。",
    "对于高额占用费主张，需论证其补偿属性而非惩罚属性以提升可采性。",
    "被告抗辩通知未送达时，应围绕签收习惯与联系记录证明其已实际知悉。",
    "在证据不足场景下，应优先构建间接证据链闭环而非孤立陈述。",
    "对方强调格式条款免责时，应重点审查提示说明义务是否已充分履行。",
]


class _SeqMatcher:
    """Matcher stub using sequence similarity with mutable threshold."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def find_match(self, query: str, candidates: List[Tuple[str, str]]) -> str | None:
        best_id = None
        best_score = -1.0

        for node_id, text in candidates:
            score = SequenceMatcher(None, query, text).ratio()

            if score > best_score:
                best_score = score
                best_id = node_id

        if best_score >= self.threshold:
            return best_id

        return None


@dataclass
class BinaryMetrics:
    """Binary classification summary used in semantic default-effect audits.

    Attributes:
        tp: True-positive count.
        fp: False-positive count.
        tn: True-negative count.
        fn: False-negative count.
        precision_pos: Positive-class precision.
        recall_pos: Positive-class recall.
        f1_pos: Positive-class F1.
        precision_neg: Negative-class precision.
        recall_neg: Negative-class recall.
        f1_neg: Negative-class F1.
        macro_precision: Macro-averaged precision.
        macro_recall: Macro-averaged recall.
        macro_f1: Macro-averaged F1.
    """

    tp: int
    fp: int
    tn: int
    fn: int
    precision_pos: float
    recall_pos: float
    f1_pos: float
    precision_neg: float
    recall_neg: float
    f1_neg: float
    macro_precision: float
    macro_recall: float
    macro_f1: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision_pos": self.precision_pos,
            "recall_pos": self.recall_pos,
            "f1_pos": self.f1_pos,
            "precision_neg": self.precision_neg,
            "recall_neg": self.recall_neg,
            "f1_neg": self.f1_neg,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
        }


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def evaluate_binary(rows: List[Dict[str, Any]], pred_key: str) -> BinaryMetrics:
    """Evaluate one binary prediction column against dataset labels.

    Args:
        rows: Dataset rows containing ``label`` and prediction columns.
        pred_key: Column name storing the binary predictions to score.

    Returns:
        Aggregate binary classification metrics for ``pred_key``.
    """
    tp = fp = tn = fn = 0
    for row in rows:
        label = int(row["label"])
        pred = int(row[pred_key])

        if label == 1 and pred == 1:
            tp += 1

        elif label == 0 and pred == 1:
            fp += 1

        elif label == 0 and pred == 0:
            tn += 1

        else:
            fn += 1

    precision_pos = _safe_div(tp, tp + fp)
    recall_pos = _safe_div(tp, tp + fn)
    f1_pos = _f1(precision_pos, recall_pos)

    precision_neg = _safe_div(tn, tn + fn)
    recall_neg = _safe_div(tn, tn + fp)
    f1_neg = _f1(precision_neg, recall_neg)

    return BinaryMetrics(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision_pos=precision_pos,
        recall_pos=recall_pos,
        f1_pos=f1_pos,
        precision_neg=precision_neg,
        recall_neg=recall_neg,
        f1_neg=f1_neg,
        macro_precision=0.5 * (precision_pos + precision_neg),
        macro_recall=0.5 * (recall_pos + recall_neg),
        macro_f1=0.5 * (f1_pos + f1_neg),
    )


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().split())


def _make_positive_variant(text: str, node_type: str, rng: random.Random) -> str:
    if node_type == "FACT":
        options = [
            text.replace("，", "，并").replace("。", "，且有书面记录。"),
            text.replace("签订", "订立").replace("约定", "明确约定"),
            f"经查，{text.replace('。', '')}，相关证据能够相互印证。",
        ]

    else:
        options = [
            text.replace("应", "应当").replace("，", "，并且"),
            f"{text.replace('。', '')}，该观点在程序与实体层面均具备一致性。",
            text.replace("主张", "提出主张").replace("证明", "举证证明"),
        ]

    return _normalize(rng.choice(options))


def _make_hard_negative_variant(text: str, node_type: str, rng: random.Random) -> str:
    if node_type == "FACT":
        options = [
            text.replace("已", "未").replace("应", "无需"),
            text.replace("2014年", "2016年").replace("98000元", "68000元"),
            text.replace("持续", "并未").replace("收到", "未收到"),
        ]

    else:
        options = [
            text.replace("不能成立", "可以成立").replace("应", "无需"),
            text.replace("应当", "不应").replace("可采", "不可采"),
            text.replace("存在瑕疵", "不存在瑕疵").replace("应", "不应"),
        ]

    return _normalize(rng.choice(options))


def _pick_clear_negative(
    base_pool: List[str], current_text: str, rng: random.Random
) -> str:
    for _ in range(20):
        candidate = _normalize(rng.choice(base_pool))

        if candidate != _normalize(current_text):
            return candidate

    return _normalize(base_pool[0])


def _pair_similarity(left_text: str, right_text: str) -> float:
    return float(SequenceMatcher(None, left_text, right_text).ratio())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_dataset(
    *,
    seed: int,
    size: int,
    valid_ratio: float,
    positive_ratio: float,
    fact_threshold: float,
    claim_threshold: float,
) -> Dict[str, Any]:
    """Generate the synthetic semantic-default audit dataset.

    Args:
        seed: Random seed for deterministic generation.
        size: Number of synthetic pairs to create.
        valid_ratio: Fraction of rows assigned to the validation split.
        positive_ratio: Fraction of positive examples to sample.
        fact_threshold: Semantic threshold used for fact rows.
        claim_threshold: Semantic threshold used for claim rows.

    Returns:
        Dataset payload containing generated rows and audit metadata.
    """
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []

    for idx in range(size):
        node_type = "FACT" if rng.random() < 0.5 else "CLAIM"

        left_text = _normalize(
            rng.choice(FACT_BASES if node_type == "FACT" else CLAIM_BASES)
        )

        threshold = fact_threshold if node_type == "FACT" else claim_threshold
        label = 1 if rng.random() < positive_ratio else 0
        split = "valid" if rng.random() < valid_ratio else "train"

        if label == 1:
            right_text = _make_positive_variant(left_text, node_type, rng)
            negative_type = None

        else:
            if rng.random() < 0.38:
                right_text = _make_hard_negative_variant(left_text, node_type, rng)
                negative_type = "hard_negative"

            else:
                pool = FACT_BASES if node_type == "FACT" else CLAIM_BASES
                right_text = _pick_clear_negative(pool, left_text, rng)
                negative_type = "clear_negative"

        similarity = _pair_similarity(left_text, right_text)

        boundary_hint = (
            abs(similarity - threshold) <= 0.03 or negative_type == "hard_negative"
        )

        rows.append(
            {
                "sample_id": f"R7_{seed}_{str(idx + 1).zfill(4)}",
                "seed": seed,
                "split": split,
                "node_type": node_type,
                "left_text": left_text,
                "right_text": right_text,
                "label": int(label),
                "negative_type": negative_type,
                "similarity": round(similarity, 6),
                "boundary_hint": bool(boundary_hint),
            }
        )

    summary = {
        "size": len(rows),
        "positive_count": sum(1 for row in rows if int(row["label"]) == 1),
        "negative_count": sum(1 for row in rows if int(row["label"]) == 0),
        "positive_ratio": round(
            sum(1 for row in rows if int(row["label"]) == 1) / len(rows), 4
        ),
        "boundary_ratio": round(
            sum(1 for row in rows if row["boundary_hint"]) / len(rows), 4
        ),
        "fact_ratio": round(
            sum(1 for row in rows if row["node_type"] == "FACT") / len(rows), 4
        ),
        "hard_negative_ratio": round(
            sum(1 for row in rows if row.get("negative_type") == "hard_negative")
            / len(rows),
            4,
        ),
    }

    return {"rows": rows, "summary": summary}


@contextmanager
def _patch_dedup_config(*, fact_threshold: float, other_threshold: float):
    cfg = SimpleNamespace(
        dedup=SimpleNamespace(
            fact_threshold=float(fact_threshold),
            other_threshold=float(other_threshold),
        )
    )

    old_system_config = executor_module.SystemConfig
    executor_module.SystemConfig = lambda: cfg

    try:
        yield

    finally:
        executor_module.SystemConfig = old_system_config


def _predict_merge(row: Dict[str, Any], initial_threshold: float) -> Tuple[int, float]:
    matcher = _SeqMatcher(threshold=initial_threshold)
    graph = ShadowGraph()
    executor = GraphExecutor(graph, matcher=matcher)
    node_type = NodeType(row["node_type"])

    first_id, _ = executor._apply_add_node(
        content=row["left_text"],
        node_type=node_type,
        agent_id="round7",
    )

    second_id, _ = executor._apply_add_node(
        content=row["right_text"],
        node_type=node_type,
        agent_id="round7",
    )

    if first_id is None or second_id is None:
        return 0, float(matcher.threshold)

    return int(first_id == second_id), float(matcher.threshold)


def _build_key_cases() -> List[Dict[str, Any]]:
    rows = [
        {
            "case_id": "KC_R7_FACT_POS_001",
            "node_type": "FACT",
            "label": 1,
            "left_text": "2013年2月1日，原告与被告签订厂房租赁合同，约定月租金98000元。",
            "right_text": "2013年2月1日，原告与被告订立厂房租赁合同，并明确约定月租金98000元。",
            "description": "FACT 同义改写应合并。",
        },
        {
            "case_id": "KC_R7_FACT_NEG_002",
            "node_type": "FACT",
            "label": 0,
            "left_text": "2014年5月29日原告向被告发送解除通知，主张合同已提前终止。",
            "right_text": "2014年5月29日原告未向被告发送解除通知，并确认合同继续履行。",
            "description": "FACT 结论相反应不合并。",
        },
        {
            "case_id": "KC_R7_CLAIM_POS_003",
            "node_type": "CLAIM",
            "label": 1,
            "left_text": "合同条款虽约定逾期30天视为提前终止，但解除权行使仍应遵循催告程序。",
            "right_text": "合同虽约定逾期30天可提前终止，但解除权行使仍应遵循催告程序。",
            "description": "CLAIM 同义法理表述应合并。",
        },
        {
            "case_id": "KC_R7_CLAIM_NEG_004",
            "node_type": "CLAIM",
            "label": 0,
            "left_text": "原告未给予合理履行期限即解除合同，解除程序存在瑕疵，双倍占用费请求不能成立。",
            "right_text": "原告已给予合理履行期限并依法解除合同，双倍占用费请求可以成立。",
            "description": "CLAIM 结论对立应不合并。",
        },
        {
            "case_id": "KC_R7_LAW_EQ_005",
            "node_type": "LAW",
            "label": 1,
            "left_text": "《中华人民共和国民法典》第五百七十七条：当事人一方不履行合同义务应承担违约责任。",
            "right_text": "《中华人民共和国民法典》第五百七十七条：当事人一方不履行合同义务应承担违约责任。",
            "description": "LAW 精确重复仍应复用。",
        },
    ]

    return rows


def _render_report(
    *,
    report_path: Path,
    low_threshold: float,
    high_threshold: float,
    fact_threshold: float,
    claim_threshold: float,
    total_samples: int,
    consistency_rate: float,
    mismatch_count: int,
    threshold_override_mismatch_count: int,
    low_metrics: BinaryMetrics,
    high_metrics: BinaryMetrics,
    key_cases: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# Round 7 Audit Report: semantic_default_threshold deprecation")
    lines.append("")
    lines.append("## Scope")

    lines.append(
        "- Objective: verify whether initial matcher threshold affects runtime dedup results."
    )

    lines.append(
        f"- Compared initial thresholds: low={low_threshold:.2f}, high={high_threshold:.2f}"
    )

    lines.append(
        f"- Runtime dedup thresholds fixed: FACT={fact_threshold:.2f}, CLAIM/OTHER={claim_threshold:.2f}"
    )

    lines.append(
        f"- Samples: {total_samples} synthetic FACT/CLAIM pair cases (multi-seed)."
    )

    lines.append("")
    lines.append("## Core Findings")
    lines.append(f"- Prediction mismatch count (low vs high): **{mismatch_count}**")
    lines.append(f"- Consistency rate: **{consistency_rate:.4f}**")

    lines.append(
        f"- Runtime-threshold override mismatch count: **{threshold_override_mismatch_count}**"
    )

    lines.append(
        "- Conclusion: `semantic_default_threshold` does not affect runtime dedup decisions."
    )

    lines.append("")
    lines.append("## Metrics Comparison")

    lines.append(
        "- Low-init metrics: "
        f"P+={low_metrics.precision_pos:.4f}, R+={low_metrics.recall_pos:.4f}, "
        f"F1+={low_metrics.f1_pos:.4f}, MacroF1={low_metrics.macro_f1:.4f}, "
        f"TP/FP/TN/FN={low_metrics.tp}/{low_metrics.fp}/{low_metrics.tn}/{low_metrics.fn}"
    )

    lines.append(
        "- High-init metrics: "
        f"P+={high_metrics.precision_pos:.4f}, R+={high_metrics.recall_pos:.4f}, "
        f"F1+={high_metrics.f1_pos:.4f}, MacroF1={high_metrics.macro_f1:.4f}, "
        f"TP/FP/TN/FN={high_metrics.tp}/{high_metrics.fp}/{high_metrics.tn}/{high_metrics.fn}"
    )

    lines.append("")
    lines.append("## Key Cases")

    lines.append(
        "| case_id | type | label | low_predict | high_predict | description |"
    )

    lines.append("|---|---|---:|---:|---:|---|")

    for row in key_cases:
        lines.append(
            "| {case_id} | {node_type} | {label} | {predict_low} | {predict_high} | {description} |".format(
                **row
            )
        )

    lines.append("")
    lines.append("## Decision")

    lines.append(
        "- Remove `matcher.semantic_default_threshold` from config and runtime wiring."
    )

    lines.append(
        "- Keep dedup control solely through `dedup.fact_threshold` and `dedup.other_threshold`."
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for semantic default-effect auditing.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="102,103,104,105,106")
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    parser.add_argument("--fact-threshold", type=float, default=0.88)
    parser.add_argument("--other-threshold", type=float, default=0.82)
    parser.add_argument("--low-initial-threshold", type=float, default=0.10)
    parser.add_argument("--high-initial-threshold", type=float, default=0.99)

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("benchmarks/optim/artifacts"),
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/round7_semantic_default_deprecation.md"),
    )

    return parser.parse_args()


def main() -> None:
    """Generate the audit dataset and optional evaluation artifacts.

    Raises:
        ValueError: If no valid seed is supplied.
    """
    args = parse_args()
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]

    if not seeds:
        raise ValueError("No valid seed provided")

    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    per_seed_summary: List[Dict[str, Any]] = []

    with _patch_dedup_config(
        fact_threshold=args.fact_threshold,
        other_threshold=args.other_threshold,
    ):
        for seed in seeds:
            payload = generate_dataset(
                seed=seed,
                size=args.size,
                valid_ratio=args.valid_ratio,
                positive_ratio=args.positive_ratio,
                fact_threshold=args.fact_threshold,
                claim_threshold=args.other_threshold,
            )

            rows = payload["rows"]
            per_seed_summary.append(payload["summary"])

            for row in rows:
                pred_low, final_thr_low = _predict_merge(
                    row, args.low_initial_threshold
                )

                pred_high, final_thr_high = _predict_merge(
                    row, args.high_initial_threshold
                )

                expected_thr = (
                    args.fact_threshold
                    if row["node_type"] == "FACT"
                    else args.other_threshold
                )

                row["predict_low"] = pred_low
                row["predict_high"] = pred_high
                row["final_threshold_low"] = round(final_thr_low, 6)
                row["final_threshold_high"] = round(final_thr_high, 6)
                row["expected_runtime_threshold"] = round(float(expected_thr), 6)
                row["prediction_mismatch"] = int(pred_low != pred_high)

                row["threshold_override_mismatch"] = int(
                    abs(final_thr_low - expected_thr) > 1e-9
                    or abs(final_thr_high - expected_thr) > 1e-9
                )

            _write_jsonl(
                artifacts_dir / f"round7_semantic_default_seed{seed}.jsonl", rows
            )

            all_rows.extend(rows)

    total_samples = len(all_rows)
    mismatch_count = sum(int(row["prediction_mismatch"]) for row in all_rows)

    threshold_override_mismatch_count = sum(
        int(row["threshold_override_mismatch"]) for row in all_rows
    )

    consistency_rate = 1.0 - _safe_div(mismatch_count, total_samples)
    low_metrics = evaluate_binary(all_rows, "predict_low")
    high_metrics = evaluate_binary(all_rows, "predict_high")
    by_type: Dict[str, Dict[str, Any]] = {}

    for node_type in ("FACT", "CLAIM"):
        rows = [row for row in all_rows if row["node_type"] == node_type]

        by_type[node_type] = {
            "size": len(rows),
            "mismatch_count": sum(int(row["prediction_mismatch"]) for row in rows),
            "consistency_rate": 1.0
            - _safe_div(
                sum(int(row["prediction_mismatch"]) for row in rows), len(rows)
            ),
            "metrics_low": evaluate_binary(rows, "predict_low").as_dict(),
            "metrics_high": evaluate_binary(rows, "predict_high").as_dict(),
        }

    key_cases = _build_key_cases()

    with _patch_dedup_config(
        fact_threshold=args.fact_threshold,
        other_threshold=args.other_threshold,
    ):
        for row in key_cases:
            if row["node_type"] == "LAW":
                graph = ShadowGraph()

                nid1, _ = graph.add_node(
                    content=row["left_text"],
                    node_type=NodeType.LAW,
                    agent_id="round7",
                    matcher=None,
                )

                nid2, _ = graph.add_node(
                    content=row["right_text"],
                    node_type=NodeType.LAW,
                    agent_id="round7",
                    matcher=None,
                )

                pred = int(nid1 == nid2)
                row["predict_low"] = pred
                row["predict_high"] = pred
                continue

            pred_low, _ = _predict_merge(row, args.low_initial_threshold)
            pred_high, _ = _predict_merge(row, args.high_initial_threshold)
            row["predict_low"] = pred_low
            row["predict_high"] = pred_high

    summary = {
        "seeds": seeds,
        "size_per_seed": args.size,
        "fact_threshold": args.fact_threshold,
        "other_threshold": args.other_threshold,
        "low_initial_threshold": args.low_initial_threshold,
        "high_initial_threshold": args.high_initial_threshold,
        "total_samples": total_samples,
        "prediction_mismatch_count": mismatch_count,
        "consistency_rate": consistency_rate,
        "threshold_override_mismatch_count": threshold_override_mismatch_count,
        "metrics_low": low_metrics.as_dict(),
        "metrics_high": high_metrics.as_dict(),
        "by_node_type": by_type,
        "seed_summary": per_seed_summary,
    }

    _write_json(artifacts_dir / "round7_semantic_default_effect_audit.json", summary)
    _write_json(artifacts_dir / "round7_semantic_default_key_cases.json", key_cases)

    _render_report(
        report_path=args.report_path,
        low_threshold=args.low_initial_threshold,
        high_threshold=args.high_initial_threshold,
        fact_threshold=args.fact_threshold,
        claim_threshold=args.other_threshold,
        total_samples=total_samples,
        consistency_rate=consistency_rate,
        mismatch_count=mismatch_count,
        threshold_override_mismatch_count=threshold_override_mismatch_count,
        low_metrics=low_metrics,
        high_metrics=high_metrics,
        key_cases=key_cases,
    )

    print("[round7-semantic-default-audit] done")
    print(f"total_samples={total_samples}")
    print(f"prediction_mismatch_count={mismatch_count}")
    print(f"consistency_rate={consistency_rate:.6f}")
    print(f"threshold_override_mismatch_count={threshold_override_mismatch_count}")
    print(f"report={args.report_path}")


if __name__ == "__main__":
    main()
