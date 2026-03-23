"""Tune `matcher.insight_threshold` on synthetic retrieval+dedup insight tasks.

This script uses only self-constructed data and F1/Recall-driven selection.
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

from benchmarks.optim.generate_insight_dataset import (
    SIDE_D,
    SIDE_P,
    generate_insight_dataset,
)

os.environ.setdefault("ES_HOST", "http://localhost:9200")
os.environ.setdefault("EMBEDDING_MODEL_PATH", "./bge-m3")
os.environ.setdefault("MAS_STORAGE_DIR", "./tmp/mas_storage")
os.environ.setdefault("LEGAL_LLM_KEY", "test-key")
os.environ.setdefault("LEGAL_LLM_URL", "http://localhost:8001")
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    arr1 = np.asarray(vec1, dtype=float)
    arr2 = np.asarray(vec2, dtype=float)
    norm1 = float(np.linalg.norm(arr1))
    norm2 = float(np.linalg.norm(arr2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(arr1, arr2) / (norm1 * norm2))


class RuntimeEmbedding:
    """Lightweight embedding wrapper aligned with runtime model usage."""

    def __init__(self, model_path: str | None = None):
        from chromadb.utils import embedding_functions

        resolved_model_path = model_path or os.getenv(
            "EMBEDDING_MODEL_PATH", "./bge-m3"
        )

        self._func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=resolved_model_path
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._func(texts)


@dataclass
class BinaryMetrics:
    """Binary metrics for one threshold and one task."""

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


def evaluate_binary(rows: List[Dict[str, Any]], threshold: float) -> BinaryMetrics:
    tp = fp = tn = fn = 0

    for row in rows:
        label = int(row["label"])
        pred = 1 if float(row["similarity"]) >= threshold else 0

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
    macro_precision = 0.5 * (precision_pos + precision_neg)
    macro_recall = 0.5 * (recall_pos + recall_neg)
    macro_f1 = 0.5 * (f1_pos + f1_neg)

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
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
    )


def contamination_rate(rows: List[Dict[str, Any]], threshold: float) -> float:
    """Measure wrong-side positive rate among predicted positive retrieval hits."""
    positives = 0
    contaminated = 0

    for row in rows:
        if float(row["similarity"]) < threshold:
            continue

        positives += 1
        query_side = row["query_side"]
        insight_side = row["insight_side"]

        if query_side == SIDE_P and insight_side == SIDE_D:
            contaminated += 1

        if query_side == SIDE_D and insight_side == SIDE_P:
            contaminated += 1

    return _safe_div(contaminated, positives)


def threshold_range(start: float, end: float, step: float) -> List[float]:
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


def _score_retrieval_rows(
    rows: List[Dict[str, Any]],
    embedding: RuntimeEmbedding,
    cache: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    texts: List[str] = []
    seen: set[str] = set()

    for row in rows:
        context = row["context"]
        insight = row["insight"]

        if context not in cache and context not in seen:
            seen.add(context)
            texts.append(context)

        if insight not in cache and insight not in seen:
            seen.add(insight)
            texts.append(insight)

    batch_size = 256

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors = embedding.embed_texts(batch)

        for text, vec in zip(batch, vectors):
            cache[text] = vec

    scored: List[Dict[str, Any]] = []

    for row in rows:
        sim = _cosine_similarity(cache[row["context"]], cache[row["insight"]])

        scored.append(
            {
                **row,
                "similarity": float(sim),
            }
        )

    return scored


def _score_dedup_rows(
    rows: List[Dict[str, Any]],
    embedding: RuntimeEmbedding,
    cache: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    texts: List[str] = []
    seen: set[str] = set()

    for row in rows:
        left = row["existing_insight"]
        right = row["new_insight"]

        if left not in cache and left not in seen:
            seen.add(left)
            texts.append(left)

        if right not in cache and right not in seen:
            seen.add(right)
            texts.append(right)

    batch_size = 256

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors = embedding.embed_texts(batch)

        for text, vec in zip(batch, vectors):
            cache[text] = vec

    scored: List[Dict[str, Any]] = []

    for row in rows:
        sim = _cosine_similarity(
            cache[row["existing_insight"]], cache[row["new_insight"]]
        )

        scored.append(
            {
                **row,
                "similarity": float(sim),
            }
        )

    return scored


def _aggregate_scan_row(
    *,
    threshold: float,
    retrieval_metrics: List[BinaryMetrics],
    dedup_metrics: List[BinaryMetrics],
    retrieval_contamination: List[float],
) -> Dict[str, Any]:
    def mean_metric(metrics: List[BinaryMetrics], field: str) -> float:
        return float(statistics.mean(getattr(m, field) for m in metrics))

    return {
        "threshold": threshold,
        "retrieval_mean_f1_pos": mean_metric(retrieval_metrics, "f1_pos"),
        "retrieval_mean_recall_pos": mean_metric(retrieval_metrics, "recall_pos"),
        "retrieval_mean_precision_pos": mean_metric(retrieval_metrics, "precision_pos"),
        "retrieval_mean_macro_f1": mean_metric(retrieval_metrics, "macro_f1"),
        "retrieval_mean_contamination_rate": float(
            statistics.mean(retrieval_contamination)
        ),
        "dedup_mean_f1_pos": mean_metric(dedup_metrics, "f1_pos"),
        "dedup_mean_recall_pos": mean_metric(dedup_metrics, "recall_pos"),
        "dedup_mean_precision_pos": mean_metric(dedup_metrics, "precision_pos"),
        "dedup_mean_macro_f1": mean_metric(dedup_metrics, "macro_f1"),
        "retrieval_sum_fp": int(sum(m.fp for m in retrieval_metrics)),
        "retrieval_sum_fn": int(sum(m.fn for m in retrieval_metrics)),
        "dedup_sum_fp": int(sum(m.fp for m in dedup_metrics)),
        "dedup_sum_fn": int(sum(m.fn for m in dedup_metrics)),
    }


def _rank_key(row: Dict[str, Any]) -> tuple:
    return (
        -row["retrieval_mean_f1_pos"],
        -row["retrieval_mean_recall_pos"],
        -row["dedup_mean_f1_pos"],
        -row["retrieval_mean_macro_f1"],
        row["retrieval_mean_contamination_rate"],
        -row["threshold"],
    )


def _select_best_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("rows cannot be empty")

    return sorted(rows, key=_rank_key)[0]


def _style_rates(
    retrieval_rows: List[Dict[str, Any]],
    dedup_rows: List[Dict[str, Any]],
) -> Dict[str, float]:
    merged = retrieval_rows + dedup_rows

    if not merged:
        return {
            "style_compliance_rate": 0.0,
            "actionable_insight_rate": 0.0,
        }

    compliant = sum(1 for row in merged if bool(row.get("quality_pass")))

    actionable = sum(
        1
        for row in merged
        if bool(row.get("has_trigger"))
        and bool(row.get("has_action"))
        and bool(row.get("has_goal"))
    )

    return {
        "style_compliance_rate": compliant / len(merged),
        "actionable_insight_rate": actionable / len(merged),
    }


def _score_key_cases(
    *,
    key_cases: List[Dict[str, Any]],
    embedding: RuntimeEmbedding,
    cache: Dict[str, List[float]],
    baseline_threshold: float,
    tuned_threshold: float,
) -> List[Dict[str, Any]]:
    texts: List[str] = []

    for row in key_cases:
        texts.append(row["context"])
        texts.append(row["candidate"])

    dedup = list(dict.fromkeys(texts))
    missing = [text for text in dedup if text not in cache]

    if missing:
        vectors = embedding.embed_texts(missing)

        for text, vec in zip(missing, vectors):
            cache[text] = vec

    result: List[Dict[str, Any]] = []

    for row in key_cases:
        sim = _cosine_similarity(cache[row["context"]], cache[row["candidate"]])

        result.append(
            {
                **row,
                "similarity": round(float(sim), 4),
                "baseline_predict": int(sim >= baseline_threshold),
                "tuned_predict": int(sim >= tuned_threshold),
            }
        )

    return result


def _format_metric(label: str, metrics: BinaryMetrics) -> str:
    return (
        f"{label}: macroP={metrics.macro_precision:.4f}, macroR={metrics.macro_recall:.4f}, "
        f"macroF1={metrics.macro_f1:.4f}, P+={metrics.precision_pos:.4f}, "
        f"R+={metrics.recall_pos:.4f}, F1+={metrics.f1_pos:.4f}, FP={metrics.fp}, FN={metrics.fn}"
    )


def _render_report(
    *,
    report_path: Path,
    seeds: List[int],
    retrieval_size: int,
    dedup_size: int,
    baseline_threshold: float,
    tuned_threshold: float,
    best_row: Dict[str, Any],
    retrieval_valid_baseline: BinaryMetrics,
    retrieval_valid_tuned: BinaryMetrics,
    dedup_valid_baseline: BinaryMetrics,
    dedup_valid_tuned: BinaryMetrics,
    contamination_baseline: float,
    contamination_tuned: float,
    style_rates: Dict[str, float],
    key_cases: List[Dict[str, Any]],
    scan_rows: List[Dict[str, Any]],
) -> None:
    retrieval_f1_delta = retrieval_valid_tuned.f1_pos - retrieval_valid_baseline.f1_pos

    retrieval_recall_delta = (
        retrieval_valid_tuned.recall_pos - retrieval_valid_baseline.recall_pos
    )

    dedup_f1_delta = dedup_valid_tuned.f1_pos - dedup_valid_baseline.f1_pos
    lines: List[str] = []
    lines.append("# Round 2 Tuning Report: insight_threshold")
    lines.append("")
    lines.append("## Scope and Constraints")
    lines.append("- Parameter: `mas/config.py` -> `matcher.insight_threshold`")
    lines.append(f"- Baseline: {baseline_threshold:.2f}")

    lines.append(
        "- Data policy: fully synthetic AIGC-like insight text, no ES, no historical DB sampling"
    )

    lines.append(
        "- Objective: F1+ primary, Recall+ tie-break; side-contamination monitored"
    )

    lines.append("")
    lines.append("## Dataset Setup")
    lines.append(f"- Seeds: {seeds}")
    lines.append(f"- Retrieval pairs per seed: {retrieval_size}")
    lines.append(f"- Dedup pairs per seed: {dedup_size}")
    lines.append("- Split: train/valid = 80/20")
    lines.append(f"- Style compliance rate: {style_rates['style_compliance_rate']:.2%}")

    lines.append(
        f"- Actionable insight rate: {style_rates['actionable_insight_rate']:.2%}"
    )

    lines.append("")
    lines.append("## Threshold Search")
    lines.append("- Rough scan: 0.50 to 0.90, step 0.05")
    lines.append("- Fine scan: around rough best +/-0.04, step 0.01")

    lines.append(
        "- Selection rule: maximize retrieval F1+, tie-break by retrieval Recall+, dedup F1+, contamination rate"
    )

    lines.append("")
    lines.append("## Selected Threshold")
    lines.append(f"- Recommended `insight_threshold`: **{tuned_threshold:.2f}**")

    lines.append(
        "- Train aggregate: "
        f"retrieval_F1+={best_row['retrieval_mean_f1_pos']:.4f}, "
        f"retrieval_R+={best_row['retrieval_mean_recall_pos']:.4f}, "
        f"dedup_F1+={best_row['dedup_mean_f1_pos']:.4f}, "
        f"contamination={best_row['retrieval_mean_contamination_rate']:.4f}"
    )

    lines.append("")
    lines.append("## Validation Comparison")
    lines.append(f"- {_format_metric('Retrieval baseline', retrieval_valid_baseline)}")
    lines.append(f"- {_format_metric('Retrieval tuned', retrieval_valid_tuned)}")
    lines.append(f"- Retrieval F1+ delta: {retrieval_f1_delta:+.4f}")
    lines.append(f"- Retrieval Recall+ delta: {retrieval_recall_delta:+.4f}")
    lines.append(f"- Retrieval contamination baseline: {contamination_baseline:.4f}")
    lines.append(f"- Retrieval contamination tuned: {contamination_tuned:.4f}")
    lines.append(f"- {_format_metric('Dedup baseline', dedup_valid_baseline)}")
    lines.append(f"- {_format_metric('Dedup tuned', dedup_valid_tuned)}")
    lines.append(f"- Dedup F1+ delta: {dedup_f1_delta:+.4f}")
    lines.append("")
    lines.append("## Key Positive/Negative Cases")

    lines.append(
        "| case_id | category | label | similarity | baseline | tuned | description |"
    )

    lines.append("|---|---|---:|---:|---:|---:|---|")

    for row in key_cases:
        lines.append(
            "| {case_id} | {category} | {label} | {similarity:.4f} | {baseline_predict} | {tuned_predict} | {description} |".format(
                **row
            )
        )

    lines.append("")
    lines.append("## Config Diff")
    lines.append("```diff")
    lines.append("-    insight_threshold: float = {:.2f}".format(baseline_threshold))
    lines.append("+    insight_threshold: float = {:.2f}".format(tuned_threshold))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")

    lines.append(
        "- Scan details: `benchmarks/optim/artifacts/insight_threshold_scan_train.json`."
    )

    lines.append(
        "- Validation summary: `benchmarks/optim/artifacts/insight_threshold_valid_comparison.json`."
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "seeds": seeds,
        "retrieval_size": retrieval_size,
        "dedup_size": dedup_size,
        "baseline_threshold": baseline_threshold,
        "recommended_threshold": tuned_threshold,
        "best_row": best_row,
        "retrieval_valid_baseline": retrieval_valid_baseline.as_dict(),
        "retrieval_valid_tuned": retrieval_valid_tuned.as_dict(),
        "dedup_valid_baseline": dedup_valid_baseline.as_dict(),
        "dedup_valid_tuned": dedup_valid_tuned.as_dict(),
        "contamination_baseline": contamination_baseline,
        "contamination_tuned": contamination_tuned,
        "style_rates": style_rates,
        "top_scan_rows": scan_rows[:10],
    }

    _write_json(Path("benchmarks/optim/artifacts/insight_round2_summary.json"), summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="52,53,54,55,56")
    parser.add_argument("--retrieval-size", type=int, default=720)
    parser.add_argument("--dedup-size", type=int, default=480)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--baseline-threshold", type=float, default=0.70)

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("benchmarks/optim/artifacts"),
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/round2_insight_threshold.md"),
    )

    parser.add_argument("--model-path", type=str, default=None)

    parser.add_argument(
        "--reuse-similarity",
        action="store_true",
        help="Reuse similarity files if present.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]

    if not seeds:
        raise ValueError("No valid seed provided")

    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    embedding = RuntimeEmbedding(model_path=args.model_path)
    embedding_cache: Dict[str, List[float]] = {}
    retrieval_scored_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    dedup_scored_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    key_cases = None
    style_rates_rows_retrieval: List[Dict[str, Any]] = []
    style_rates_rows_dedup: List[Dict[str, Any]] = []

    for seed in seeds:
        payload = generate_insight_dataset(
            seed=seed,
            retrieval_size=args.retrieval_size,
            dedup_size=args.dedup_size,
            valid_ratio=args.valid_ratio,
        )

        retrieval_rows = payload["retrieval_pairs"]
        dedup_rows = payload["dedup_pairs"]
        style_rates_rows_retrieval.extend(retrieval_rows)
        style_rates_rows_dedup.extend(dedup_rows)

        if key_cases is None:
            key_cases = payload["key_cases"]

        retrieval_data_path = artifacts_dir / f"insight_retrieval_seed{seed}.jsonl"
        dedup_data_path = artifacts_dir / f"insight_dedup_seed{seed}.jsonl"

        retrieval_scored_path = (
            artifacts_dir / f"insight_similarity_retrieval_seed{seed}.jsonl"
        )

        dedup_scored_path = artifacts_dir / f"insight_similarity_dedup_seed{seed}.jsonl"
        _write_jsonl(retrieval_data_path, retrieval_rows)
        _write_jsonl(dedup_data_path, dedup_rows)

        if args.reuse_similarity and retrieval_scored_path.exists():
            retrieval_scored = _load_jsonl(retrieval_scored_path)

        else:
            retrieval_scored = _score_retrieval_rows(
                retrieval_rows,
                embedding,
                embedding_cache,
            )

            _write_jsonl(retrieval_scored_path, retrieval_scored)

        if args.reuse_similarity and dedup_scored_path.exists():
            dedup_scored = _load_jsonl(dedup_scored_path)

        else:
            dedup_scored = _score_dedup_rows(
                dedup_rows,
                embedding,
                embedding_cache,
            )

            _write_jsonl(dedup_scored_path, dedup_scored)

        retrieval_scored_by_seed[seed] = retrieval_scored
        dedup_scored_by_seed[seed] = dedup_scored

    retrieval_train_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    retrieval_valid_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    dedup_train_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    dedup_valid_by_seed: Dict[int, List[Dict[str, Any]]] = {}

    for seed in seeds:
        retrieval_train_by_seed[seed] = [
            row for row in retrieval_scored_by_seed[seed] if row["split"] == "train"
        ]

        retrieval_valid_by_seed[seed] = [
            row for row in retrieval_scored_by_seed[seed] if row["split"] == "valid"
        ]

        dedup_train_by_seed[seed] = [
            row for row in dedup_scored_by_seed[seed] if row["split"] == "train"
        ]

        dedup_valid_by_seed[seed] = [
            row for row in dedup_scored_by_seed[seed] if row["split"] == "valid"
        ]

    rough_thresholds = threshold_range(0.50, 0.90, 0.05)
    rough_rows: List[Dict[str, Any]] = []

    for threshold in rough_thresholds:
        retrieval_metrics = [
            evaluate_binary(retrieval_train_by_seed[seed], threshold) for seed in seeds
        ]

        dedup_metrics = [
            evaluate_binary(dedup_train_by_seed[seed], threshold) for seed in seeds
        ]

        contamination = [
            contamination_rate(retrieval_train_by_seed[seed], threshold)
            for seed in seeds
        ]

        rough_rows.append(
            _aggregate_scan_row(
                threshold=threshold,
                retrieval_metrics=retrieval_metrics,
                dedup_metrics=dedup_metrics,
                retrieval_contamination=contamination,
            )
        )

    rough_best = sorted(rough_rows, key=_rank_key)[0]
    fine_start = max(0.0, round(rough_best["threshold"] - 0.04, 2))
    fine_end = min(1.0, round(rough_best["threshold"] + 0.04, 2))

    scan_thresholds = sorted(
        set(rough_thresholds + threshold_range(fine_start, fine_end, 0.01))
    )

    scan_rows: List[Dict[str, Any]] = []

    for threshold in scan_thresholds:
        retrieval_metrics = [
            evaluate_binary(retrieval_train_by_seed[seed], threshold) for seed in seeds
        ]

        dedup_metrics = [
            evaluate_binary(dedup_train_by_seed[seed], threshold) for seed in seeds
        ]

        contamination = [
            contamination_rate(retrieval_train_by_seed[seed], threshold)
            for seed in seeds
        ]

        scan_rows.append(
            _aggregate_scan_row(
                threshold=threshold,
                retrieval_metrics=retrieval_metrics,
                dedup_metrics=dedup_metrics,
                retrieval_contamination=contamination,
            )
        )

    scan_rows.sort(key=_rank_key)
    best_row = _select_best_row(scan_rows)
    tuned_threshold = float(best_row["threshold"])

    retrieval_valid_all = [
        row for seed in seeds for row in retrieval_valid_by_seed[seed]
    ]

    dedup_valid_all = [row for seed in seeds for row in dedup_valid_by_seed[seed]]

    retrieval_valid_baseline = evaluate_binary(
        retrieval_valid_all, args.baseline_threshold
    )

    retrieval_valid_tuned = evaluate_binary(retrieval_valid_all, tuned_threshold)
    dedup_valid_baseline = evaluate_binary(dedup_valid_all, args.baseline_threshold)
    dedup_valid_tuned = evaluate_binary(dedup_valid_all, tuned_threshold)

    contamination_baseline = contamination_rate(
        retrieval_valid_all, args.baseline_threshold
    )

    contamination_tuned = contamination_rate(retrieval_valid_all, tuned_threshold)
    style_rates = _style_rates(style_rates_rows_retrieval, style_rates_rows_dedup)

    key_case_rows = _score_key_cases(
        key_cases=key_cases or [],
        embedding=embedding,
        cache=embedding_cache,
        baseline_threshold=args.baseline_threshold,
        tuned_threshold=tuned_threshold,
    )

    _write_json(artifacts_dir / "insight_threshold_scan_train.json", scan_rows)

    _write_json(
        artifacts_dir / "insight_threshold_valid_comparison.json",
        {
            "baseline_threshold": args.baseline_threshold,
            "tuned_threshold": tuned_threshold,
            "retrieval_baseline": retrieval_valid_baseline.as_dict(),
            "retrieval_tuned": retrieval_valid_tuned.as_dict(),
            "dedup_baseline": dedup_valid_baseline.as_dict(),
            "dedup_tuned": dedup_valid_tuned.as_dict(),
            "contamination_baseline": contamination_baseline,
            "contamination_tuned": contamination_tuned,
            "style_rates": style_rates,
        },
    )

    _write_json(artifacts_dir / "insight_key_cases.json", key_case_rows)

    _render_report(
        report_path=args.report_path,
        seeds=seeds,
        retrieval_size=args.retrieval_size,
        dedup_size=args.dedup_size,
        baseline_threshold=args.baseline_threshold,
        tuned_threshold=tuned_threshold,
        best_row=best_row,
        retrieval_valid_baseline=retrieval_valid_baseline,
        retrieval_valid_tuned=retrieval_valid_tuned,
        dedup_valid_baseline=dedup_valid_baseline,
        dedup_valid_tuned=dedup_valid_tuned,
        contamination_baseline=contamination_baseline,
        contamination_tuned=contamination_tuned,
        style_rates=style_rates,
        key_cases=key_case_rows,
        scan_rows=scan_rows,
    )

    print("[insight-tuning] done")
    print(f"baseline_threshold={args.baseline_threshold:.2f}")
    print(f"recommended_threshold={tuned_threshold:.2f}")
    print(f"retrieval_valid_baseline_f1_pos={retrieval_valid_baseline.f1_pos:.4f}")
    print(f"retrieval_valid_tuned_f1_pos={retrieval_valid_tuned.f1_pos:.4f}")

    print(
        f"retrieval_valid_baseline_recall_pos={retrieval_valid_baseline.recall_pos:.4f}"
    )

    print(f"retrieval_valid_tuned_recall_pos={retrieval_valid_tuned.recall_pos:.4f}")
    print(f"report={args.report_path}")


if __name__ == "__main__":
    main()
