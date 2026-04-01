"""Tune `worker_threshold.law_worker_threshold` on synthetic Round-6 data.

Runtime-aligned scoring path:
1) multiple query decomposition (pre-generated queries per sample),
2) vector retrieval over synthetic law corpus (`top_k=3`),
3) deduplicate + rerank by `(law_name, article_id)`,
4) `max_score = top1(_score) - 1.0`,
5) gate by `max_score >= threshold`.
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

from benchmarks.optim.generate_law_worker_dataset import generate_law_worker_dataset

load_dotenv(override=True)
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class BinaryMetrics:
    """Binary classification summary for law-worker threshold sweeps.

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
    """Evaluate one law-worker score threshold against labeled samples.

    Args:
        rows: Scored dataset rows with binary labels and ``max_score`` values.
        threshold: Decision threshold applied to ``max_score``.

    Returns:
        Binary classification metrics for the supplied threshold.
    """
    tp = fp = tn = fn = 0

    for row in rows:
        label = int(row["label"])
        pred = 1 if float(row["max_score"]) >= threshold else 0

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
        rows: Scored dataset rows containing labels and ``max_score``.
        threshold: Decision threshold applied to ``max_score``.

    Returns:
        Share of negative rows predicted as found.
    """
    negatives = [row for row in rows if int(row["label"]) == 0]

    if not negatives:
        return 0.0

    false_found = sum(1 for row in negatives if float(row["max_score"]) >= threshold)
    return _safe_div(false_found, len(negatives))


def false_found_rate_by_type(
    rows: List[Dict[str, Any]],
    threshold: float,
    negative_type: str,
) -> float:
    """Measure false-positive rate on one negative subtype.

    Args:
        rows: Scored dataset rows containing labels and subtype annotations.
        threshold: Decision threshold applied to ``max_score``.
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

    false_found = sum(1 for row in negatives if float(row["max_score"]) >= threshold)
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


class SyntheticLawScorer:
    """Runtime-aligned vector scorer over synthetic law corpus.

    Attributes:
        embedding: SentenceTransformer embedding wrapper aligned with runtime.
        embedding_cache: Text to vector cache for queries and law snippets.
        search_cache: Query string to ranked retrieval hits cache.
        corpus_docs: Current synthetic law corpus in document form.
        corpus_matrix: Dense matrix of corpus embeddings.
        corpus_norms: Cached vector norms for cosine scoring.
    """

    def __init__(self, model_path: str):
        self.embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_path
        )

        self.embedding_cache: Dict[str, List[float]] = {}
        self.search_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.corpus_docs: List[Dict[str, Any]] = []
        self.corpus_matrix: Optional[np.ndarray] = None
        self.corpus_norms: Optional[np.ndarray] = None

    @staticmethod
    def _doc_text(doc: Dict[str, Any]) -> str:
        return (
            f"《{doc.get('law_name', '')}》{doc.get('article_id', '')} "
            f"{doc.get('content', '')}"
        ).strip()

    def _embed_texts(self, texts: List[str], batch_size: int = 256) -> None:
        missing = [text for text in texts if text not in self.embedding_cache]

        if not missing:
            return

        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            vectors = self.embedding(batch)

            for text, vec in zip(batch, vectors):
                self.embedding_cache[text] = vec

    def set_corpus(self, corpus_docs: List[Dict[str, Any]]) -> None:
        self.corpus_docs = corpus_docs
        corpus_texts = [self._doc_text(doc) for doc in corpus_docs]
        self._embed_texts(corpus_texts)

        vectors = np.array(
            [self.embedding_cache[text] for text in corpus_texts], dtype=float
        )

        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1e-12
        self.corpus_matrix = vectors
        self.corpus_norms = norms
        self.search_cache = {}

    def embed_queries(self, queries: List[str], batch_size: int = 256) -> None:
        self._embed_texts(queries, batch_size=batch_size)

    def search_query(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        key = f"{top_k}::{query}"

        if key in self.search_cache:
            return self.search_cache[key]

        if self.corpus_matrix is None or self.corpus_norms is None:
            raise RuntimeError("Corpus not initialized. Call set_corpus first.")

        if query not in self.embedding_cache:
            self._embed_texts([query])

        qvec = np.array(self.embedding_cache[query], dtype=float)
        qnorm = np.linalg.norm(qvec)

        if qnorm == 0:
            qnorm = 1e-12

        sims = (self.corpus_matrix @ qvec) / (self.corpus_norms * qnorm)
        top_k = min(top_k, len(self.corpus_docs))

        if top_k <= 0:
            self.search_cache[key] = []
            return []

        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        hits: List[Dict[str, Any]] = []

        for idx in top_idx:
            sim = float(sims[idx])
            doc = self.corpus_docs[int(idx)]

            hits.append(
                {
                    "_id": doc.get("doc_id", ""),
                    "_score": sim + 1.0,
                    "_source": {
                        "law_name": doc.get("law_name", ""),
                        "article_id": doc.get("article_id", ""),
                        "content": doc.get("content", ""),
                        "domain": doc.get("domain", ""),
                    },
                }
            )

        self.search_cache[key] = hits
        return hits

    @staticmethod
    def deduplicate_and_rerank(
        all_hits_list: List[List[Dict[str, Any]]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        unique_map: Dict[str, Dict[str, Any]] = {}

        for hits in all_hits_list:
            for hit in hits:
                src = hit.get("_source", {})
                key = f"{src.get('law_name', '')}_{src.get('article_id', '')}"
                score = float(hit.get("_score", 0.0))

                if key not in unique_map or score > float(
                    unique_map[key].get("_score", 0.0)
                ):
                    unique_map[key] = hit

        merged = list(unique_map.values())
        merged.sort(key=lambda row: float(row.get("_score", 0.0)), reverse=True)
        return merged[:top_k]


def _score_dataset_rows(
    *,
    rows: List[Dict[str, Any]],
    scorer: SyntheticLawScorer,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []

    for row in rows:
        query_hits = [scorer.search_query(query, top_k=3) for query in row["queries"]]
        final_hits = scorer.deduplicate_and_rerank(query_hits, top_k=3)
        max_score = float(final_hits[0].get("_score", 1.0) - 1.0) if final_hits else 0.0
        top_law_name = ""
        top_article_id = ""

        if final_hits:
            src = final_hits[0].get("_source", {})
            top_law_name = str(src.get("law_name", ""))
            top_article_id = str(src.get("article_id", ""))

        scored.append(
            {
                **row,
                "max_score": max_score,
                "retrieved_count": len(final_hits),
                "top_law_name": top_law_name,
                "top_article_id": top_article_id,
            }
        )

    return scored


def _apply_manual_review(
    rows: List[Dict[str, Any]],
    manual_review_rows: List[Dict[str, Any]],
) -> int:
    manual_map: Dict[str, int] = {}

    for row in manual_review_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        label = row.get("review_label")

        if sample_id and label in (0, 1):
            manual_map[sample_id] = int(label)

    updated = 0

    for row in rows:
        sample_id = str(row.get("sample_id", ""))

        if sample_id in manual_map:
            review_label = manual_map[sample_id]

            if int(row["label"]) != review_label:
                updated += 1

            row["label"] = review_label
            row["reviewed"] = True

        else:
            row["reviewed"] = False

    return updated


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
        row["mean_hard_negative_false_found_rate"],
        row["mean_false_found_rate"],
        -row["mean_macro_f1"],
        -row["mean_precision_pos"],
        row["std_f1_pos"],
        -row["threshold"],
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
        score = float(row["max_score"])
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
                "intent": row["intent"],
                "label": int(row["label"]),
                "negative_type": row.get("negative_type"),
                "max_score": round(score, 4),
                "baseline_predict": baseline_predict,
                "tuned_predict": tuned_predict,
                "review_label": int(row["label"]),
                "review_reason": ",".join(reason_parts),
            }
        )

    candidates.sort(
        key=lambda row: (
            abs(float(row["max_score"]) - tuned_threshold),
            abs(float(row["max_score"]) - baseline_threshold),
            -float(row["max_score"]),
        )
    )

    return candidates[:limit]


def _score_key_cases(
    *,
    key_cases: List[Dict[str, Any]],
    scorer: SyntheticLawScorer,
    baseline_threshold: float,
    tuned_threshold: float,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []

    for row in key_cases:
        query_hits = [scorer.search_query(query, top_k=3) for query in row["queries"]]
        final_hits = scorer.deduplicate_and_rerank(query_hits, top_k=3)
        max_score = float(final_hits[0].get("_score", 1.0) - 1.0) if final_hits else 0.0

        scored.append(
            {
                **row,
                "max_score": round(max_score, 4),
                "baseline_predict": int(max_score >= baseline_threshold),
                "tuned_predict": int(max_score >= tuned_threshold),
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
    manual_override_count: int,
    review_candidates_count: int,
) -> None:
    lines: List[str] = []
    lines.append("# Round 6 Tuning Report: law_worker_threshold")
    lines.append("")
    lines.append("## Scope and Constraints")

    lines.append(
        "- Parameter: `mas/config.py` -> `worker_threshold.law_worker_threshold`"
    )

    lines.append(
        "- Runtime gate: `LawWorker` uses `max_score >= threshold` for FOUND/NOT_FOUND."
    )

    lines.append(f"- Baseline: {baseline_threshold:.2f}")
    lines.append("- Data policy: synthetic-only (law corpus + intent/query samples).")

    lines.append(
        "- Labeling policy: auto labeling with manual-review candidate export."
    )

    lines.append("")
    lines.append("## Threshold Search")
    lines.append("- Rough scan: 0.35 to 0.90, step 0.05")
    lines.append("- Fine scan: around rough best +/-0.06, step 0.01")

    lines.append(
        "- Selection: max F1+, tie-break Recall+, then hard-negative false-found and macro F1"
    )

    lines.append("")
    lines.append("## Selected Threshold")
    lines.append(f"- Recommended `law_worker_threshold`: **{tuned_threshold:.2f}**")

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
        f"- False-found rate delta: {false_found_tuned - false_found_baseline:+.4f}"
    )

    lines.append(
        "- Hard-negative false-found baseline/tuned/delta: "
        f"{hard_baseline:.4f} -> {hard_tuned:.4f} ({hard_tuned - hard_baseline:+.4f})"
    )

    lines.append("")
    lines.append("## Key Positive/Negative Cases")

    lines.append(
        "| case_id | category | label | max_score | baseline | tuned | description |"
    )

    lines.append("|---|---|---:|---:|---:|---:|---|")

    for row in key_cases:
        lines.append(
            "| {case_id} | {category} | {label} | {max_score:.4f} | {baseline_predict} | {tuned_predict} | {description} |".format(
                **row
            )
        )

    lines.append("")
    lines.append("## Config Diff")
    lines.append("```diff")
    lines.append("-    law_worker_threshold: float = {:.2f}".format(baseline_threshold))
    lines.append("+    law_worker_threshold: float = {:.2f}".format(tuned_threshold))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")

    lines.append(
        "- Scan details: `benchmarks/optim/artifacts/law_worker_threshold_scan_train.json`."
    )

    lines.append(
        "- Validation summary: `benchmarks/optim/artifacts/law_worker_threshold_valid_comparison.json`."
    )

    lines.append(
        "- Manual review template: `benchmarks/optim/artifacts/round6_law_worker_manual_review_candidates.jsonl`."
    )

    lines.append(f"- Manual overrides applied: {manual_override_count}.")
    lines.append(f"- Manual review candidates exported: {review_candidates_count}.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for law-worker threshold tuning.

    Returns:
        Parsed command-line namespace for dataset generation, scoring, and
        report output paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="92,93,94,95,96")
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--positive-ratio", type=float, default=0.45)
    parser.add_argument("--baseline-threshold", type=float, default=0.60)

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("benchmarks/optim/artifacts"),
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/round6_law_worker_threshold.md"),
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("EMBEDDING_MODEL_PATH", "./bge-m3"),
    )

    parser.add_argument(
        "--manual-review-file",
        type=Path,
        default=Path(
            "benchmarks/optim/artifacts/round6_law_worker_manual_review.jsonl"
        ),
    )

    parser.add_argument(
        "--reuse-scored",
        action="store_true",
        help="Reuse scored per-seed files if present.",
    )

    return parser.parse_args()


def main() -> None:
    """Run law-worker threshold tuning and export diagnostic artifacts.

    Raises:
        RuntimeError: If key cases cannot be built or scoring prerequisites are
            missing.
        ValueError: If no valid seeds are supplied.
    """
    args = parse_args()
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]

    if not seeds:
        raise ValueError("No valid seed provided")

    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    scorer = SyntheticLawScorer(model_path=args.model_path)
    scored_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []
    key_cases_payload: Optional[List[Dict[str, Any]]] = None
    first_seed_corpus: Optional[List[Dict[str, Any]]] = None

    for seed in seeds:
        payload = generate_law_worker_dataset(
            seed=seed,
            size=args.size,
            valid_ratio=args.valid_ratio,
            positive_ratio=args.positive_ratio,
        )

        rows = payload["rows"]
        corpus = payload["law_corpus"]
        summary_rows.append(payload["summary"])

        if key_cases_payload is None:
            key_cases_payload = payload["key_cases"]
            first_seed_corpus = corpus

        raw_path = artifacts_dir / f"law_worker_seed{seed}.jsonl"
        _write_jsonl(raw_path, rows)

        _write_json(
            artifacts_dir / f"law_worker_seed{seed}.summary.json",
            payload["summary"],
        )

        _write_json(artifacts_dir / f"law_worker_corpus_seed{seed}.json", corpus)
        scored_path = artifacts_dir / f"law_worker_scored_seed{seed}.jsonl"

        if args.reuse_scored and scored_path.exists():
            scored_rows = _load_jsonl(scored_path)

        else:
            scorer.set_corpus(corpus)
            all_queries = list({query for row in rows for query in row["queries"]})
            scorer.embed_queries(all_queries)
            scored_rows = _score_dataset_rows(rows=rows, scorer=scorer)
            _write_jsonl(scored_path, scored_rows)

        scored_by_seed[seed] = scored_rows

    if key_cases_payload is None or first_seed_corpus is None:
        raise RuntimeError("Failed to build key cases for Round-6.")

    all_scored_rows = [row for seed in seeds for row in scored_by_seed[seed]]
    manual_review_rows = _load_jsonl(args.manual_review_file)
    manual_override_count = _apply_manual_review(all_scored_rows, manual_review_rows)
    reviewed_map = {row["sample_id"]: row for row in all_scored_rows}

    for seed in seeds:
        scored_by_seed[seed] = [
            reviewed_map[row["sample_id"]] for row in scored_by_seed[seed]
        ]

    rough_thresholds = threshold_range(0.35, 0.90, 0.05)
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
                    negative_type="law_present_no_retrieval",
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
    fine_start = max(0.30, round(float(rough_best["threshold"]) - 0.06, 2))
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
                    negative_type="law_present_no_retrieval",
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
    valid_rows = [row for row in all_scored_rows if row["split"] == "valid"]
    valid_baseline = evaluate_binary(valid_rows, args.baseline_threshold)
    valid_tuned = evaluate_binary(valid_rows, tuned_threshold)
    false_found_baseline = false_found_rate(valid_rows, args.baseline_threshold)
    false_found_tuned = false_found_rate(valid_rows, tuned_threshold)

    hard_baseline = false_found_rate_by_type(
        valid_rows, args.baseline_threshold, negative_type="law_present_no_retrieval"
    )

    hard_tuned = false_found_rate_by_type(
        valid_rows, tuned_threshold, negative_type="law_present_no_retrieval"
    )

    review_candidates = _build_manual_review_candidates(
        rows=valid_rows,
        baseline_threshold=args.baseline_threshold,
        tuned_threshold=tuned_threshold,
        limit=120,
    )

    _write_jsonl(
        artifacts_dir / "round6_law_worker_manual_review_candidates.jsonl",
        review_candidates,
    )

    scorer.set_corpus(first_seed_corpus)

    key_case_queries = list(
        {query for row in key_cases_payload for query in row["queries"]}
    )

    scorer.embed_queries(key_case_queries)

    key_cases_scored = _score_key_cases(
        key_cases=key_cases_payload,
        scorer=scorer,
        baseline_threshold=args.baseline_threshold,
        tuned_threshold=tuned_threshold,
    )

    _write_json(artifacts_dir / "law_worker_key_cases.json", key_cases_scored)
    _write_json(artifacts_dir / "law_worker_threshold_scan_train.json", all_scan_rows)

    _write_json(
        artifacts_dir / "law_worker_threshold_valid_comparison.json",
        {
            "baseline_threshold": args.baseline_threshold,
            "tuned_threshold": tuned_threshold,
            "valid_baseline": valid_baseline.as_dict(),
            "valid_tuned": valid_tuned.as_dict(),
            "false_found_rate_baseline": false_found_baseline,
            "false_found_rate_tuned": false_found_tuned,
            "hard_negative_false_found_rate_baseline": hard_baseline,
            "hard_negative_false_found_rate_tuned": hard_tuned,
            "manual_override_count": manual_override_count,
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
        "manual_override_count": manual_override_count,
        "review_candidates_count": len(review_candidates),
        "dataset_summary_samples": summary_rows,
        "top_scan_rows": sorted(all_scan_rows, key=_rank_key)[:10],
    }

    _write_json(artifacts_dir / "law_worker_round6_summary.json", summary_payload)

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
        manual_override_count=manual_override_count,
        review_candidates_count=len(review_candidates),
    )

    print("[law-worker-threshold-tuning] done")
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
