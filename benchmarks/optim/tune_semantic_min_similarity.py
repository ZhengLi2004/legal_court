"""Tune `retrieval.semantic_min_similarity` on mixed ES+synthetic semantic-path data.

Runtime-aligned scoring path:
1) embed `query_context` and `candidate_context` with runtime embedding stack,
2) cosine similarity,
3) gate by `similarity >= threshold`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from benchmarks.optim.generate_semantic_path_dataset import (
    generate_semantic_path_dataset,
)

load_dotenv(override=True)

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class BinaryMetrics:
    """Binary classification summary for semantic-min-similarity sweeps.

    Attributes:
        tp: Count of positive rows predicted as found.
        fp: Count of negative rows predicted as found.
        tn: Count of negative rows predicted as not found.
        fn: Count of positive rows predicted as not found.
        precision_pos: Positive-class precision.
        recall_pos: Positive-class recall.
        f1_pos: Positive-class F1.
        precision_neg: Negative-class precision.
        recall_neg: Negative-class recall.
        f1_neg: Negative-class F1.
        macro_precision: Mean precision across the two classes.
        macro_recall: Mean recall across the two classes.
        macro_f1: Mean F1 across the two classes.
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


def evaluate_binary(rows: List[Dict[str, Any]], threshold: float) -> BinaryMetrics:
    """Evaluate one semantic similarity threshold against labeled pairs.

    Args:
        rows: Scored pair rows with binary labels and similarities.
        threshold: Decision threshold applied to cosine similarity.

    Returns:
        Binary classification metrics for the supplied threshold.
    """
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


def false_found_rate(rows: List[Dict[str, Any]], threshold: float) -> float:
    """Measure false-positive rate among negative rows at a threshold.

    Args:
        rows: Scored pair rows with binary labels and similarities.
        threshold: Decision threshold applied to cosine similarity.

    Returns:
        Share of negative rows predicted as found.
    """
    negatives = [row for row in rows if int(row["label"]) == 0]

    if not negatives:
        return 0.0

    false_found = sum(1 for row in negatives if float(row["similarity"]) >= threshold)
    return _safe_div(false_found, len(negatives))


def false_found_rate_by_type(
    rows: List[Dict[str, Any]],
    threshold: float,
    negative_type: str,
) -> float:
    """Measure false-positive rate on one negative subtype.

    Args:
        rows: Scored pair rows with labels and negative subtype annotations.
        threshold: Decision threshold applied to cosine similarity.
        negative_type: Negative subtype to isolate before scoring.

    Returns:
        Share of subtype rows predicted as found.
    """
    negatives = [
        row
        for row in rows
        if int(row["label"]) == 0 and str(row.get("negative_type", "")) == negative_type
    ]

    if not negatives:
        return 0.0

    false_found = sum(1 for row in negatives if float(row["similarity"]) >= threshold)
    return _safe_div(false_found, len(negatives))


def threshold_range(start: float, end: float, step: float) -> List[float]:
    """Build an inclusive floating-point threshold grid.

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

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def _cosine(vec1: List[float], vec2: List[float]) -> float:
    arr1 = np.asarray(vec1, dtype=float)
    arr2 = np.asarray(vec2, dtype=float)
    norm1 = float(np.linalg.norm(arr1))
    norm2 = float(np.linalg.norm(arr2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(arr1, arr2) / (norm1 * norm2))


class SemanticPairScorer:
    """Vector scorer aligned with runtime embedding stack.

    Attributes:
        embedding: SentenceTransformer embedding wrapper aligned with runtime.
        cache: Text-to-vector cache reused across scoring calls.
    """

    def __init__(self, model_path: str):
        self.embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_path
        )

        self.cache: Dict[str, List[float]] = {}

    def _embed_missing(self, texts: List[str], batch_size: int = 256) -> None:
        missing = [text for text in texts if text not in self.cache]

        if not missing:
            return

        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            vectors = self.embedding(batch)

            for text, vec in zip(batch, vectors):
                self.cache[text] = vec

    def score_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = list(
            {str(row["query_context"]) for row in rows}
            | {str(row["candidate_context"]) for row in rows}
        )

        self._embed_missing(texts)
        scored: List[Dict[str, Any]] = []

        for row in rows:
            q = str(row["query_context"])
            c = str(row["candidate_context"])
            sim = _cosine(self.cache[q], self.cache[c])
            scored.append({**row, "similarity": sim})

        return scored


def _aggregate_scan_row(
    *,
    threshold: float,
    metrics_by_seed: List[BinaryMetrics],
    false_found_by_seed: List[float],
    hard_negative_by_seed: List[float],
) -> Dict[str, Any]:
    def mean_metric(field: str) -> float:
        return float(statistics.mean(getattr(m, field) for m in metrics_by_seed))

    def std_metric(field: str) -> float:
        return float(statistics.pstdev(getattr(m, field) for m in metrics_by_seed))

    return {
        "threshold": threshold,
        "mean_f1_pos": mean_metric("f1_pos"),
        "mean_recall_pos": mean_metric("recall_pos"),
        "mean_precision_pos": mean_metric("precision_pos"),
        "mean_macro_f1": mean_metric("macro_f1"),
        "mean_false_found_rate": float(statistics.mean(false_found_by_seed)),
        "mean_hard_negative_false_found_rate": float(
            statistics.mean(hard_negative_by_seed)
        ),
        "std_f1_pos": std_metric("f1_pos"),
        "std_recall_pos": std_metric("recall_pos"),
        "sum_tp": int(sum(m.tp for m in metrics_by_seed)),
        "sum_fp": int(sum(m.fp for m in metrics_by_seed)),
        "sum_tn": int(sum(m.tn for m in metrics_by_seed)),
        "sum_fn": int(sum(m.fn for m in metrics_by_seed)),
    }


def _rank_key(row: Dict[str, Any]) -> tuple:
    return (
        -row["mean_f1_pos"],
        -row["mean_recall_pos"],
        -row["mean_macro_f1"],
        -row["mean_precision_pos"],
        row["std_f1_pos"],
        row["threshold"],
    )


def _select_best_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("rows cannot be empty")

    return sorted(rows, key=_rank_key)[0]


def _build_manual_review_candidates(
    *,
    rows: List[Dict[str, Any]],
    baseline_threshold: float,
    tuned_threshold: float,
    limit: int = 120,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for row in rows:
        score = float(row["similarity"])
        baseline_predict = int(score >= baseline_threshold)
        tuned_predict = int(score >= tuned_threshold)
        near_baseline = abs(score - baseline_threshold) <= 0.03
        near_tuned = abs(score - tuned_threshold) <= 0.03
        disagree = baseline_predict != tuned_predict
        boundary = bool(row.get("boundary_hint", False))

        if not (near_baseline or near_tuned or disagree or boundary):
            continue

        reason_parts = []

        if disagree:
            reason_parts.append("prediction_flip")

        if near_baseline or near_tuned:
            reason_parts.append("near_threshold")

        if boundary:
            reason_parts.append("boundary_hint")

        candidates.append(
            {
                "sample_id": row["sample_id"],
                "seed": row["seed"],
                "split": row["split"],
                "label": int(row["label"]),
                "pair_type": row.get("pair_type"),
                "negative_type": row.get("negative_type"),
                "similarity": round(score, 4),
                "baseline_predict": baseline_predict,
                "tuned_predict": tuned_predict,
                "review_label": int(row["label"]),
                "review_reason": ",".join(reason_parts),
            }
        )

    candidates.sort(
        key=lambda row: (
            abs(float(row["similarity"]) - tuned_threshold),
            abs(float(row["similarity"]) - baseline_threshold),
            -float(row["similarity"]),
        )
    )

    return candidates[:limit]


def _score_key_cases(
    *,
    key_cases: List[Dict[str, Any]],
    scorer: SemanticPairScorer,
    baseline_threshold: float,
    tuned_threshold: float,
) -> List[Dict[str, Any]]:
    rows = [
        {
            "sample_id": row["case_id"],
            "query_context": row["query_context"],
            "candidate_context": row["candidate_context"],
            "label": row["label"],
        }
        for row in key_cases
    ]

    scored_rows = scorer.score_rows(rows)
    scored_by_id = {row["sample_id"]: row for row in scored_rows}
    scored: List[Dict[str, Any]] = []

    for row in key_cases:
        score = float(scored_by_id[row["case_id"]]["similarity"])

        scored.append(
            {
                **row,
                "similarity": round(score, 4),
                "baseline_predict": int(score >= baseline_threshold),
                "tuned_predict": int(score >= tuned_threshold),
            }
        )

    return scored


def _render_report(
    *,
    report_path: Path,
    baseline_threshold: float,
    tuned_threshold: float,
    best_row: Dict[str, Any],
    valid_baseline: BinaryMetrics,
    valid_tuned: BinaryMetrics,
    false_found_baseline: float,
    false_found_tuned: float,
    hard_baseline: float,
    hard_tuned: float,
    key_cases: List[Dict[str, Any]],
    review_candidates_count: int,
) -> None:
    lines: List[str] = []
    lines.append("# Round 10 Tuning Report: semantic_min_similarity")
    lines.append("")
    lines.append("## Scope and Constraints")
    lines.append("- Parameter: `mas/config.py` -> `retrieval.semantic_min_similarity`")

    lines.append(
        "- Runtime gate: `LegalGMemory.retrieve_memory` uses `similarity >= threshold`."
    )

    lines.append(f"- Baseline: {baseline_threshold:.2f}")
    lines.append("- Data policy: ES+synthetic mixed semantic pairs.")
    lines.append("")
    lines.append("## Threshold Search")
    lines.append("- Rough scan: 0.00 to 0.90, step 0.05")
    lines.append("- Fine scan: around rough best +/-0.06, step 0.01")
    lines.append("- Selection: max F1+, tie-break Recall+.")
    lines.append("")
    lines.append("## Selected Threshold")
    lines.append(f"- Recommended `semantic_min_similarity`: **{tuned_threshold:.2f}**")

    lines.append(
        "- Train aggregate: "
        f"F1+={best_row['mean_f1_pos']:.4f}, "
        f"Recall+={best_row['mean_recall_pos']:.4f}, "
        f"Precision+={best_row['mean_precision_pos']:.4f}, "
        f"MacroF1={best_row['mean_macro_f1']:.4f}, "
        f"FalseFound={best_row['mean_false_found_rate']:.4f}, "
        f"HardNegFalseFound={best_row['mean_hard_negative_false_found_rate']:.4f}"
    )

    lines.append("")
    lines.append("## Validation Comparison")

    lines.append(
        "- Baseline: "
        f"macroP={valid_baseline.macro_precision:.4f}, "
        f"macroR={valid_baseline.macro_recall:.4f}, "
        f"macroF1={valid_baseline.macro_f1:.4f}, "
        f"P+={valid_baseline.precision_pos:.4f}, "
        f"R+={valid_baseline.recall_pos:.4f}, "
        f"F1+={valid_baseline.f1_pos:.4f}, "
        f"FP={valid_baseline.fp}, FN={valid_baseline.fn}"
    )

    lines.append(
        "- Tuned: "
        f"macroP={valid_tuned.macro_precision:.4f}, "
        f"macroR={valid_tuned.macro_recall:.4f}, "
        f"macroF1={valid_tuned.macro_f1:.4f}, "
        f"P+={valid_tuned.precision_pos:.4f}, "
        f"R+={valid_tuned.recall_pos:.4f}, "
        f"F1+={valid_tuned.f1_pos:.4f}, "
        f"FP={valid_tuned.fp}, FN={valid_tuned.fn}"
    )

    lines.append(f"- F1+ delta: {valid_tuned.f1_pos - valid_baseline.f1_pos:+.4f}")

    lines.append(
        f"- Recall+ delta: {valid_tuned.recall_pos - valid_baseline.recall_pos:+.4f}"
    )

    lines.append(
        f"- Precision+ delta: {valid_tuned.precision_pos - valid_baseline.precision_pos:+.4f}"
    )

    lines.append(f"- False-found rate baseline: {false_found_baseline:.4f}")
    lines.append(f"- False-found rate tuned: {false_found_tuned:.4f}")

    lines.append(
        "- Hard-negative false-found baseline/tuned/delta: "
        f"{hard_baseline:.4f} -> {hard_tuned:.4f} ({hard_tuned - hard_baseline:+.4f})"
    )

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

    lines.append(
        "-    semantic_min_similarity: float = {:.2f}".format(baseline_threshold)
    )

    lines.append("+    semantic_min_similarity: float = {:.2f}".format(tuned_threshold))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")

    lines.append(
        "- Scan details: `benchmarks/optim/artifacts/semantic_min_similarity_scan_train.json`."
    )

    lines.append(
        "- Validation summary: `benchmarks/optim/artifacts/semantic_min_similarity_valid_comparison.json`."
    )

    lines.append(
        "- Manual review template: `benchmarks/optim/artifacts/round10_semantic_min_similarity_manual_review_candidates.jsonl`."
    )

    lines.append(f"- Manual review candidates exported: {review_candidates_count}.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for semantic-min-similarity tuning.

    Returns:
        Parsed command-line namespace for dataset generation, scoring, and
        report output paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="112,113,114,115,116")
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--positive-ratio", type=float, default=0.45)
    parser.add_argument("--baseline-threshold", type=float, default=0.00)

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("benchmarks/optim/artifacts"),
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/round10_semantic_min_similarity.md"),
    )

    parser.add_argument("--es-host", type=str, default=None)

    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("EMBEDDING_MODEL_PATH", "./bge-m3"),
    )

    parser.add_argument(
        "--reuse-scored",
        action="store_true",
        help="Reuse scored per-seed files when present.",
    )

    return parser.parse_args()


def main() -> None:
    """Run semantic-min-similarity tuning and export diagnostic artifacts.

    Raises:
        RuntimeError: If semantic-path key cases cannot be constructed.
        ValueError: If no valid seeds are supplied.
    """
    args = parse_args()
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]

    if not seeds:
        raise ValueError("No valid seed provided")

    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    scorer = SemanticPairScorer(model_path=args.model_path)
    scored_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []
    key_cases_payload: Optional[List[Dict[str, Any]]] = None

    for seed in seeds:
        payload = generate_semantic_path_dataset(
            seed=seed,
            size=args.size,
            valid_ratio=args.valid_ratio,
            positive_ratio=args.positive_ratio,
            es_host=args.es_host,
        )

        rows = payload["rows"]
        summary_rows.append(payload["summary"])

        if key_cases_payload is None:
            key_cases_payload = payload["key_cases"]

        raw_path = artifacts_dir / f"semantic_path_seed{seed}.jsonl"
        summary_path = artifacts_dir / f"semantic_path_seed{seed}.summary.json"
        _write_jsonl(raw_path, rows)
        _write_json(summary_path, payload["summary"])
        scored_path = artifacts_dir / f"semantic_path_scored_seed{seed}.jsonl"

        if args.reuse_scored and scored_path.exists():
            scored_rows = _load_jsonl(scored_path)

        else:
            scored_rows = scorer.score_rows(rows)
            _write_jsonl(scored_path, scored_rows)

        scored_by_seed[seed] = scored_rows

    if key_cases_payload is None:
        raise RuntimeError(
            "Failed to build key cases for semantic_min_similarity tuning."
        )

    rough_thresholds = threshold_range(0.00, 0.90, 0.05)
    rough_scan_rows: List[Dict[str, Any]] = []

    for threshold in rough_thresholds:
        metrics_by_seed: List[BinaryMetrics] = []
        false_found_by_seed: List[float] = []
        hard_by_seed: List[float] = []

        for seed in seeds:
            train_rows = [
                row for row in scored_by_seed[seed] if row["split"] == "train"
            ]

            metrics_by_seed.append(evaluate_binary(train_rows, threshold))
            false_found_by_seed.append(false_found_rate(train_rows, threshold))

            hard_by_seed.append(
                false_found_rate_by_type(
                    train_rows,
                    threshold,
                    negative_type="hard_semantic",
                )
            )

        rough_scan_rows.append(
            _aggregate_scan_row(
                threshold=threshold,
                metrics_by_seed=metrics_by_seed,
                false_found_by_seed=false_found_by_seed,
                hard_negative_by_seed=hard_by_seed,
            )
        )

    rough_best = _select_best_row(rough_scan_rows)
    fine_start = max(0.00, round(float(rough_best["threshold"]) - 0.06, 2))
    fine_end = min(0.95, round(float(rough_best["threshold"]) + 0.06, 2))
    fine_thresholds = threshold_range(fine_start, fine_end, 0.01)
    fine_scan_rows: List[Dict[str, Any]] = []

    for threshold in fine_thresholds:
        metrics_by_seed = []
        false_found_by_seed = []
        hard_by_seed = []

        for seed in seeds:
            train_rows = [
                row for row in scored_by_seed[seed] if row["split"] == "train"
            ]

            metrics_by_seed.append(evaluate_binary(train_rows, threshold))
            false_found_by_seed.append(false_found_rate(train_rows, threshold))

            hard_by_seed.append(
                false_found_rate_by_type(
                    train_rows,
                    threshold,
                    negative_type="hard_semantic",
                )
            )

        fine_scan_rows.append(
            _aggregate_scan_row(
                threshold=threshold,
                metrics_by_seed=metrics_by_seed,
                false_found_by_seed=false_found_by_seed,
                hard_negative_by_seed=hard_by_seed,
            )
        )

    all_scan_rows = rough_scan_rows + [
        row for row in fine_scan_rows if row not in rough_scan_rows
    ]

    all_scan_rows.sort(key=lambda row: row["threshold"])
    best_row = _select_best_row(all_scan_rows)
    tuned_threshold = float(best_row["threshold"])
    all_rows = [row for seed in seeds for row in scored_by_seed[seed]]
    valid_rows = [row for row in all_rows if row["split"] == "valid"]
    valid_baseline = evaluate_binary(valid_rows, args.baseline_threshold)
    valid_tuned = evaluate_binary(valid_rows, tuned_threshold)
    false_found_baseline = false_found_rate(valid_rows, args.baseline_threshold)
    false_found_tuned = false_found_rate(valid_rows, tuned_threshold)

    hard_baseline = false_found_rate_by_type(
        valid_rows,
        args.baseline_threshold,
        negative_type="hard_semantic",
    )

    hard_tuned = false_found_rate_by_type(
        valid_rows,
        tuned_threshold,
        negative_type="hard_semantic",
    )

    review_candidates = _build_manual_review_candidates(
        rows=valid_rows,
        baseline_threshold=args.baseline_threshold,
        tuned_threshold=tuned_threshold,
        limit=120,
    )

    _write_jsonl(
        artifacts_dir
        / "round10_semantic_min_similarity_manual_review_candidates.jsonl",
        review_candidates,
    )

    key_cases_scored = _score_key_cases(
        key_cases=key_cases_payload,
        scorer=scorer,
        baseline_threshold=args.baseline_threshold,
        tuned_threshold=tuned_threshold,
    )

    _write_json(
        artifacts_dir / "semantic_min_similarity_key_cases.json", key_cases_scored
    )

    _write_json(
        artifacts_dir / "semantic_min_similarity_scan_train.json", all_scan_rows
    )

    _write_json(
        artifacts_dir / "semantic_min_similarity_valid_comparison.json",
        {
            "baseline_threshold": args.baseline_threshold,
            "tuned_threshold": tuned_threshold,
            "valid_baseline": valid_baseline.as_dict(),
            "valid_tuned": valid_tuned.as_dict(),
            "false_found_rate_baseline": false_found_baseline,
            "false_found_rate_tuned": false_found_tuned,
            "hard_negative_false_found_rate_baseline": hard_baseline,
            "hard_negative_false_found_rate_tuned": hard_tuned,
            "review_candidates_count": len(review_candidates),
        },
    )

    summary_payload = {
        "seeds": seeds,
        "size_per_seed": args.size,
        "baseline_threshold": args.baseline_threshold,
        "recommended_threshold": tuned_threshold,
        "best_row": best_row,
        "valid_baseline": valid_baseline.as_dict(),
        "valid_tuned": valid_tuned.as_dict(),
        "false_found_rate_baseline": false_found_baseline,
        "false_found_rate_tuned": false_found_tuned,
        "hard_negative_false_found_rate_baseline": hard_baseline,
        "hard_negative_false_found_rate_tuned": hard_tuned,
        "review_candidates_count": len(review_candidates),
        "dataset_summary_samples": summary_rows,
        "top_scan_rows": sorted(all_scan_rows, key=_rank_key)[:10],
    }

    _write_json(
        artifacts_dir / "semantic_min_similarity_round10_summary.json", summary_payload
    )

    _render_report(
        report_path=args.report_path,
        baseline_threshold=args.baseline_threshold,
        tuned_threshold=tuned_threshold,
        best_row=best_row,
        valid_baseline=valid_baseline,
        valid_tuned=valid_tuned,
        false_found_baseline=false_found_baseline,
        false_found_tuned=false_found_tuned,
        hard_baseline=hard_baseline,
        hard_tuned=hard_tuned,
        key_cases=key_cases_scored,
        review_candidates_count=len(review_candidates),
    )

    print("[semantic-min-sim-tuning] done")
    print(f"baseline_threshold={args.baseline_threshold:.2f}")
    print(f"recommended_threshold={tuned_threshold:.2f}")
    print(f"valid_baseline_f1_pos={valid_baseline.f1_pos:.4f}")
    print(f"valid_tuned_f1_pos={valid_tuned.f1_pos:.4f}")
    print(f"valid_baseline_recall_pos={valid_baseline.recall_pos:.4f}")
    print(f"valid_tuned_recall_pos={valid_tuned.recall_pos:.4f}")
    print(f"false_found_rate_baseline={false_found_baseline:.4f}")
    print(f"false_found_rate_tuned={false_found_tuned:.4f}")
    print(f"hard_negative_false_found_rate_baseline={hard_baseline:.4f}")
    print(f"hard_negative_false_found_rate_tuned={hard_tuned:.4f}")
    print(f"report={args.report_path}")


if __name__ == "__main__":
    main()
