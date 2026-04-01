"""Tune `matcher.projection_threshold` with fully synthetic AIGC-style data.

This script intentionally does not use ES or historical databases. All labeled
pairs are generated in-process and then scored with the same embedding/cosine
stack used by runtime semantic matching.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from benchmarks.optim.generate_projection_dataset import generate_projection_dataset

os.environ.setdefault("ES_HOST", "http://localhost:9200")
os.environ.setdefault("EMBEDDING_MODEL_PATH", "./bge-m3")
os.environ.setdefault("MAS_STORAGE_DIR", "./tmp/mas_storage")
os.environ.setdefault("LEGAL_LLM_KEY", "test-key")
os.environ.setdefault("LEGAL_LLM_URL", "http://localhost:8001")
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FP_COST = 8
FN_COST = 2


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    arr1 = np.asarray(vec1, dtype=float)
    arr2 = np.asarray(vec2, dtype=float)
    norm1 = float(np.linalg.norm(arr1))
    norm2 = float(np.linalg.norm(arr2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(arr1, arr2) / (norm1 * norm2))


class RuntimeEmbedding:
    """Lightweight embedding wrapper aligned with runtime model usage.

    Attributes:
        _func: SentenceTransformer embedding callable used for inference.
    """

    def __init__(self, model_path: str | None = None):
        from chromadb.utils import embedding_functions

        resolved_model_path = model_path or os.getenv(
            "EMBEDDING_MODEL_PATH", "./bge-m3"
        )

        self._func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=resolved_model_path
        )

    def embed_query(self, text: str) -> List[float]:
        return self._func([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in one forward pass."""
        return self._func(texts)


@dataclass
class EvalMetrics:
    """Binary classification metrics for one threshold.

    Attributes:
        threshold: Evaluated projection threshold.
        tp: Count of true positive pairs.
        fp: Count of false positive pairs.
        tn: Count of true negative pairs.
        fn: Count of false negative pairs.
        precision_pos: Positive-class precision.
        recall_pos: Positive-class recall.
        f1_pos: Positive-class F1.
        precision_neg: Negative-class precision.
        recall_neg: Negative-class recall.
        f1_neg: Negative-class F1.
        macro_precision: Macro-averaged precision.
        macro_recall: Macro-averaged recall.
        macro_f1: Macro-averaged F1.
        cost: Weighted false-positive / false-negative cost.
    """

    threshold: float
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
    cost: float

    def as_dict(self) -> Dict[str, Any]:
        """Serialize metrics to plain dict for JSON output."""
        return {
            "threshold": self.threshold,
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
            "cost": self.cost,
        }


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def evaluate_threshold(
    scored_pairs: List[Dict[str, Any]],
    threshold: float,
) -> EvalMetrics:
    """Evaluate one threshold against scored query/candidate pairs.

    Args:
        scored_pairs: Query/candidate rows with binary labels and similarities.
        threshold: Decision threshold applied to cosine similarity.

    Returns:
        Metrics for the supplied threshold.
    """
    tp = fp = tn = fn = 0

    for row in scored_pairs:
        predicted = 1 if row["similarity"] >= threshold else 0
        label = int(row["label"])

        if label == 1 and predicted == 1:
            tp += 1

        elif label == 0 and predicted == 1:
            fp += 1

        elif label == 0 and predicted == 0:
            tn += 1

        else:
            fn += 1

    precision_pos = _safe_div(tp, tp + fp)
    recall_pos = _safe_div(tp, tp + fn)
    f1_pos = _f1(precision_pos, recall_pos)
    precision_neg = _safe_div(tn, tn + fn)
    recall_neg = _safe_div(tn, tn + fp)
    f1_neg = _f1(precision_neg, recall_neg)
    macro_precision = 0.5 * (precision_pos + precision_neg)
    macro_recall = 0.5 * (recall_pos + recall_neg)
    macro_f1 = 0.5 * (f1_pos + f1_neg)
    cost = FP_COST * fp + FN_COST * fn

    return EvalMetrics(
        threshold=threshold,
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
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        cost=cost,
    )


def aggregate_seed_metrics(
    metrics_by_seed: List[EvalMetrics],
    threshold: float,
) -> Dict[str, Any]:
    """Aggregate per-seed metrics into one comparable row.

    Args:
        metrics_by_seed: Per-seed evaluation outputs for one threshold.
        threshold: Threshold shared by the per-seed metrics.

    Returns:
        JSON-serializable aggregate row for scan ranking.

    Raises:
        ValueError: If ``metrics_by_seed`` is empty.
    """
    if not metrics_by_seed:
        raise ValueError("metrics_by_seed cannot be empty")

    def mean(field: str) -> float:
        return float(statistics.mean(getattr(m, field) for m in metrics_by_seed))

    def stdev(field: str) -> float:
        values = [getattr(m, field) for m in metrics_by_seed]
        return float(statistics.pstdev(values))

    return {
        "threshold": threshold,
        "mean_cost": mean("cost"),
        "mean_macro_f1": mean("macro_f1"),
        "mean_macro_precision": mean("macro_precision"),
        "mean_macro_recall": mean("macro_recall"),
        "mean_f1_pos": mean("f1_pos"),
        "mean_precision_pos": mean("precision_pos"),
        "mean_recall_pos": mean("recall_pos"),
        "std_cost": stdev("cost"),
        "std_macro_f1": stdev("macro_f1"),
        "sum_tp": int(sum(m.tp for m in metrics_by_seed)),
        "sum_fp": int(sum(m.fp for m in metrics_by_seed)),
        "sum_tn": int(sum(m.tn for m in metrics_by_seed)),
        "sum_fn": int(sum(m.fn for m in metrics_by_seed)),
    }


def select_best_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the best threshold row using F1-first and Recall tie-break.

    Args:
        rows: Ranked scan rows produced by ``aggregate_seed_metrics``.

    Returns:
        The best-performing scan row under the configured tie-breakers.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("rows cannot be empty")

    ranked = sorted(
        rows,
        key=lambda row: (
            -row["mean_f1_pos"],
            -row["mean_recall_pos"],
            -row["mean_macro_f1"],
            -row["mean_precision_pos"],
            -row["threshold"],
        ),
    )

    return ranked[0]


def threshold_range(start: float, end: float, step: float) -> List[float]:
    """Build a closed float range with stable rounding.

    Args:
        start: First threshold in the scan.
        end: Final threshold in the scan.
        step: Increment applied between thresholds.

    Returns:
        Rounded threshold values including both scan endpoints.
    """
    values: List[float] = []
    cursor = start

    while cursor <= end + 1e-9:
        values.append(round(cursor, 2))
        cursor += step

    return values


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def _score_pairs_with_embedding(
    dataset_rows: List[Dict[str, Any]],
    embedding: RuntimeEmbedding,
    shared_cache: Dict[str, List[float]] | None = None,
) -> List[Dict[str, Any]]:
    """Compute cosine similarity for each pair using runtime embedding stack."""
    text_cache: Dict[str, List[float]] = (
        shared_cache if shared_cache is not None else {}
    )

    unique_texts: List[str] = []
    seen: set[str] = set()

    for row in dataset_rows:
        query = str(row["query"])
        candidate = str(row["candidate"])

        if query not in text_cache and query not in seen:
            unique_texts.append(query)
            seen.add(query)

        if candidate not in text_cache and candidate not in seen:
            unique_texts.append(candidate)
            seen.add(candidate)

    batch_size = 256

    for start in range(0, len(unique_texts), batch_size):
        batch = unique_texts[start : start + batch_size]
        vectors = embedding.embed_texts(batch)

        for text, vec in zip(batch, vectors):
            text_cache[text] = vec

    scored_rows: List[Dict[str, Any]] = []

    for row in dataset_rows:
        q_vec = text_cache[str(row["query"])]
        c_vec = text_cache[str(row["candidate"])]
        sim = _cosine_similarity(q_vec, c_vec)

        scored_rows.append(
            {
                "pair_id": row["pair_id"],
                "label": int(row["label"]),
                "split": row["split"],
                "difficulty": row["difficulty"],
                "sample_type": row["sample_type"],
                "issue_id": row["issue_id"],
                "domain": row["domain"],
                "similarity": float(sim),
            }
        )

    return scored_rows


def _build_key_cases() -> List[Dict[str, Any]]:
    """Construct fixed key positive/negative cases for semantic spot checks."""
    return [
        {
            "case_id": "POS_01",
            "label": 1,
            "description": "同争点改写：解除通知到达效力",
            "query": "原告主张解除函已经送达被告登记地址，合同自通知到达时解除。",
            "candidate": "围绕租赁合同解除，出租方强调通知生效即发生解除效力，并据此请求腾退。",
        },
        {
            "case_id": "POS_02",
            "label": 1,
            "description": "同争点改写：逾期腾退双倍占用费",
            "query": "承租人合同终止后继续占有厂房，应按约支付双倍占有使用费。",
            "candidate": "在租赁终止后的占有期间，承租方应依约承担加倍使用费用。",
        },
        {
            "case_id": "POS_03",
            "label": 1,
            "description": "同争点改写：加班工资证据闭环",
            "query": "员工提交考勤和工作安排记录，可以证明长期加班事实。",
            "candidate": "企业延时用工是否成立，应以考勤数据和任务指令相互印证。",
        },
        {
            "case_id": "POS_04",
            "label": 1,
            "description": "同争点改写：借贷交付与合意",
            "query": "借贷关系成立需同时考虑资金交付与借款合意。",
            "candidate": "认定民间借贷应兼顾出借款项实际交付和双方借款意思表示。",
        },
        {
            "case_id": "POS_05",
            "label": 1,
            "description": "同争点改写：股权价款付款节点",
            "query": "股权转让价款已到约定节点，受让方应履行付款义务。",
            "candidate": "在交割条件满足后，受让方不得无故迟延支付股权转让款。",
        },
        {
            "case_id": "NEG_01",
            "label": 0,
            "description": "高词面重合但结论对立：双倍占用费",
            "query": "双倍占用费系合同终止后费用安排，不应随意调减。",
            "candidate": "双倍占用费具有惩罚属性，明显过高，应按市场租金下调。",
        },
        {
            "case_id": "NEG_02",
            "label": 0,
            "description": "同领域不同争点：租赁解除 vs 电费证据",
            "query": "解除通知到达后合同终止，承租方应腾退厂房。",
            "candidate": "电费公式中的线损参数未核验，原告应补充原始发票证明。",
        },
        {
            "case_id": "NEG_03",
            "label": 0,
            "description": "跨领域词面重合：违约金 vs 竞业限制",
            "query": "买方逾期付款应承担约定违约金。",
            "candidate": "企业未支付竞业补偿，劳动者不应继续受竞业条款约束。",
        },
        {
            "case_id": "NEG_04",
            "label": 0,
            "description": "同域冲突：医疗因果否认",
            "query": "鉴定意见支持医疗行为与损害后果存在因果关系。",
            "candidate": "基础疾病才是主要原因，鉴定并未达到高度盖然性标准。",
        },
        {
            "case_id": "NEG_05",
            "label": 0,
            "description": "同争点冲突：加班事实否认",
            "query": "企业安排延时工作却未支付加班工资。",
            "candidate": "员工提交的打卡记录无法证明存在有效加班审批。",
        },
    ]


def _score_key_cases(
    *,
    embedding: RuntimeEmbedding,
    baseline_threshold: float,
    tuned_threshold: float,
) -> List[Dict[str, Any]]:
    cases = _build_key_cases()
    all_texts: List[str] = []

    for row in cases:
        all_texts.append(row["query"])
        all_texts.append(row["candidate"])

    dedup_texts = list(dict.fromkeys(all_texts))
    vectors = embedding.embed_texts(dedup_texts)
    text_cache: Dict[str, List[float]] = dict(zip(dedup_texts, vectors))
    rows: List[Dict[str, Any]] = []

    for row in cases:
        q_vec = text_cache[row["query"]]
        c_vec = text_cache[row["candidate"]]
        sim = _cosine_similarity(q_vec, c_vec)

        rows.append(
            {
                **row,
                "similarity": round(float(sim), 4),
                "baseline_predict": int(sim >= baseline_threshold),
                "tuned_predict": int(sim >= tuned_threshold),
            }
        )

    return rows


def _format_metrics_row(metrics: EvalMetrics) -> str:
    return (
        f"cost={metrics.cost:.1f}, macroP={metrics.macro_precision:.4f}, "
        f"macroR={metrics.macro_recall:.4f}, macroF1={metrics.macro_f1:.4f}, "
        f"P+={metrics.precision_pos:.4f}, R+={metrics.recall_pos:.4f}, F1+={metrics.f1_pos:.4f}, "
        f"FP={metrics.fp}, FN={metrics.fn}"
    )


def _render_report(
    *,
    report_path: Path,
    seeds: List[int],
    dataset_size: int,
    baseline_threshold: float,
    tuned_threshold: float,
    train_best_row: Dict[str, Any],
    valid_baseline: EvalMetrics,
    valid_tuned: EvalMetrics,
    key_cases: List[Dict[str, Any]],
    scan_rows: List[Dict[str, Any]],
) -> None:
    """Write markdown report for round-1 review gate."""
    recall_delta = valid_tuned.recall_pos - valid_baseline.recall_pos
    f1_delta = valid_tuned.f1_pos - valid_baseline.f1_pos
    lines: List[str] = []
    lines.append("# Round 1 Tuning Report: projection_threshold")
    lines.append("")
    lines.append("## Scope and Constraints")
    lines.append("- Parameter: `mas/config.py` -> `matcher.projection_threshold`")
    lines.append("- Baseline: {:.2f}".format(baseline_threshold))

    lines.append(
        "- Data policy: fully synthetic AIGC-like debate text, no ES, no historical DB sampling"
    )

    lines.append(
        "- Objective: direct F1 + Recall tuning (F1+ primary, Recall+ tie-break)"
    )

    lines.append("")
    lines.append("## Dataset Setup")
    lines.append(f"- Seeds: {seeds}")

    lines.append(
        f"- Per-seed size: {dataset_size} labeled pairs (50% positive / 50% negative)"
    )

    lines.append("- Hard samples: 50% in positives, 60% in negatives")
    lines.append("- Split: train/valid = 80/20")
    lines.append("")
    lines.append("## Threshold Search")
    lines.append("- Rough scan: 0.45 to 0.85, step 0.05")

    lines.append(
        "- Fine scan: around rough best +/-0.04, step 0.01, then aggregate by F1+/Recall+/macro metrics"
    )

    lines.append(
        "- Selection rule: maximize mean F1+; tie-break by mean Recall+, then macro F1, then Precision+"
    )

    lines.append("")
    lines.append("## Selected Threshold")
    lines.append(f"- Recommended `projection_threshold`: **{tuned_threshold:.2f}**")

    lines.append(
        "- Train aggregate: "
        + (
            f"mean_macroF1={train_best_row['mean_macro_f1']:.4f}, "
            f"mean_F1+={train_best_row['mean_f1_pos']:.4f}, "
            f"mean_R+={train_best_row['mean_recall_pos']:.4f}, "
            f"mean_P+={train_best_row['mean_precision_pos']:.4f}"
        )
    )

    lines.append("")
    lines.append("## Validation Comparison")

    lines.append(
        f"- Baseline ({baseline_threshold:.2f}): {_format_metrics_row(valid_baseline)}"
    )

    lines.append(f"- Tuned ({tuned_threshold:.2f}): {_format_metrics_row(valid_tuned)}")
    lines.append(f"- F1+ delta: {f1_delta:+.4f}")
    lines.append(f"- Recall+ delta: {recall_delta:+.4f}")
    lines.append("")
    lines.append("## Key Positive/Negative Cases")
    lines.append("| case_id | label | similarity | baseline | tuned | description |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for row in key_cases:
        lines.append(
            "| {case_id} | {label} | {similarity:.4f} | {baseline_predict} | {tuned_predict} | {description} |".format(
                **row
            )
        )

    lines.append("")
    lines.append("## Config Diff")
    lines.append("```diff")
    lines.append("-    projection_threshold: float = {:.2f}".format(baseline_threshold))
    lines.append("+    projection_threshold: float = {:.2f}".format(tuned_threshold))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")

    lines.append(
        "- Scan details are saved to `benchmarks/optim/artifacts/projection_threshold_scan_train.json`."
    )

    lines.append(
        "- Per-seed scored pairs are saved under `benchmarks/optim/artifacts/projection_similarity_seed*.jsonl`."
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary_payload = {
        "seeds": seeds,
        "dataset_size": dataset_size,
        "baseline_threshold": baseline_threshold,
        "recommended_threshold": tuned_threshold,
        "train_best_row": train_best_row,
        "valid_baseline": valid_baseline.as_dict(),
        "valid_tuned": valid_tuned.as_dict(),
        "top_scan_rows": scan_rows[:10],
    }

    _write_json(
        Path("benchmarks/optim/artifacts/projection_round1_summary.json"),
        summary_payload,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for projection-threshold tuning.

    Returns:
        Parsed command-line namespace for dataset generation, scoring, and
        report output paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated random seeds.",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=600,
        help="Per-seed dataset size.",
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )

    parser.add_argument(
        "--baseline-threshold",
        type=float,
        default=0.58,
        help="Current baseline threshold in config.",
    )

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("benchmarks/optim/artifacts"),
        help="Directory for generated artifacts.",
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/round1_projection_threshold.md"),
        help="Markdown report output path.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional embedding model path override.",
    )

    parser.add_argument(
        "--reuse-similarity",
        action="store_true",
        help="Reuse existing per-seed similarity JSONL files if present.",
    )

    return parser.parse_args()


def main() -> None:
    """Run projection-threshold tuning and export ranked candidates.

    Raises:
        ValueError: If no valid seeds are supplied or the configured dataset
            size is too small for this tuning round.
    """
    args = parse_args()
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]

    if not seeds:
        raise ValueError("No valid seed provided")

    if args.size < 300:
        raise ValueError("--size should be >= 300 for this tuning round")

    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    embedding = RuntimeEmbedding(model_path=args.model_path)
    global_embedding_cache: Dict[str, List[float]] = {}
    seed_datasets: Dict[int, List[Dict[str, Any]]] = {}
    seed_scored: Dict[int, List[Dict[str, Any]]] = {}

    for seed in seeds:
        dataset_path = artifacts_dir / f"projection_dataset_seed{seed}.jsonl"
        scored_path = artifacts_dir / f"projection_similarity_seed{seed}.jsonl"

        dataset = generate_projection_dataset(
            seed=seed,
            total_pairs=args.size,
            valid_ratio=args.valid_ratio,
        )

        _write_jsonl(dataset_path, dataset)

        if args.reuse_similarity and scored_path.exists():
            scored = _load_jsonl(scored_path)

        else:
            scored = _score_pairs_with_embedding(
                dataset,
                embedding,
                shared_cache=global_embedding_cache,
            )

            _write_jsonl(scored_path, scored)

        seed_datasets[seed] = dataset
        seed_scored[seed] = scored

    train_rows_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    valid_rows_by_seed: Dict[int, List[Dict[str, Any]]] = {}

    for seed, rows in seed_scored.items():
        train_rows_by_seed[seed] = [row for row in rows if row["split"] == "train"]
        valid_rows_by_seed[seed] = [row for row in rows if row["split"] == "valid"]

    train_pool = [row for rows in train_rows_by_seed.values() for row in rows]
    rough_thresholds = threshold_range(0.45, 0.85, 0.05)

    rough_evals: List[EvalMetrics] = [
        evaluate_threshold(train_pool, threshold) for threshold in rough_thresholds
    ]

    rough_best = max(
        rough_evals,
        key=lambda row: (
            row.f1_pos,
            row.recall_pos,
            row.macro_f1,
            row.precision_pos,
            row.threshold,
        ),
    )

    fine_start = max(0.0, round(rough_best.threshold - 0.04, 2))
    fine_end = min(1.0, round(rough_best.threshold + 0.04, 2))
    fine_thresholds = threshold_range(fine_start, fine_end, 0.01)
    scan_thresholds = sorted(set(rough_thresholds + fine_thresholds))
    scan_rows: List[Dict[str, Any]] = []

    for threshold in scan_thresholds:
        per_seed_metrics = [
            evaluate_threshold(train_rows_by_seed[seed], threshold) for seed in seeds
        ]

        row = aggregate_seed_metrics(per_seed_metrics, threshold)
        scan_rows.append(row)

    scan_rows.sort(
        key=lambda row: (
            -row["mean_f1_pos"],
            -row["mean_recall_pos"],
            -row["mean_macro_f1"],
            -row["mean_precision_pos"],
            -row["threshold"],
        )
    )

    best_row = select_best_row(scan_rows)
    tuned_threshold = float(best_row["threshold"])

    baseline_valid_metrics = evaluate_threshold(
        [row for rows in valid_rows_by_seed.values() for row in rows],
        args.baseline_threshold,
    )

    tuned_valid_metrics = evaluate_threshold(
        [row for rows in valid_rows_by_seed.values() for row in rows],
        tuned_threshold,
    )

    key_cases = _score_key_cases(
        embedding=embedding,
        baseline_threshold=args.baseline_threshold,
        tuned_threshold=tuned_threshold,
    )

    _write_json(artifacts_dir / "projection_threshold_scan_train.json", scan_rows)

    _write_json(
        artifacts_dir / "projection_threshold_valid_comparison.json",
        {
            "baseline": baseline_valid_metrics.as_dict(),
            "tuned": tuned_valid_metrics.as_dict(),
            "baseline_threshold": args.baseline_threshold,
            "tuned_threshold": tuned_threshold,
        },
    )

    _write_json(artifacts_dir / "projection_key_cases.json", key_cases)

    _render_report(
        report_path=args.report_path,
        seeds=seeds,
        dataset_size=args.size,
        baseline_threshold=args.baseline_threshold,
        tuned_threshold=tuned_threshold,
        train_best_row=best_row,
        valid_baseline=baseline_valid_metrics,
        valid_tuned=tuned_valid_metrics,
        key_cases=key_cases,
        scan_rows=scan_rows,
    )

    print("[projection-tuning] done")
    print(f"baseline_threshold={args.baseline_threshold:.2f}")
    print(f"recommended_threshold={tuned_threshold:.2f}")
    print(f"valid_baseline_f1_pos={baseline_valid_metrics.f1_pos:.4f}")
    print(f"valid_tuned_f1_pos={tuned_valid_metrics.f1_pos:.4f}")
    print(f"valid_baseline_recall_pos={baseline_valid_metrics.recall_pos:.4f}")
    print(f"valid_tuned_recall_pos={tuned_valid_metrics.recall_pos:.4f}")
    print(f"report={args.report_path}")


if __name__ == "__main__":
    main()
