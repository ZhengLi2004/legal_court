"""Tune `worker_threshold.fact_worker_threshold` with ES+synthetic mixed data.

Runtime-aligned scoring path:
1) multiple query decomposition (pre-generated queries per sample),
2) ES vector search per query (`top_k=3`),
3) deduplicate+rereank by case title,
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

from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from benchmarks.optim.generate_fact_worker_es_dataset import (
    generate_fact_worker_es_dataset,
)

load_dotenv(override=True)
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class BinaryMetrics:
    """Binary classification summary for fact-worker threshold sweeps.

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


class RuntimeESScorer:
    """Search ES with runtime-equivalent embedding and cosine script score.

    Attributes:
        es: Elasticsearch client used for retrieval calls.
        embedding: SentenceTransformer embedding wrapper aligned with runtime.
        embedding_cache: Query text to embedding vector cache.
        search_cache: Query text to ES hit list cache.
    """

    def __init__(self, *, es_host: str, model_path: str):
        self.es = Elasticsearch(es_host, request_timeout=30)

        self.embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_path
        )

        self.embedding_cache: Dict[str, List[float]] = {}
        self.search_cache: Dict[str, List[Dict[str, Any]]] = {}

    def ping(self) -> bool:
        return bool(self.es.ping())

    def embed_queries(self, queries: List[str], batch_size: int = 256) -> None:
        missing = [q for q in queries if q not in self.embedding_cache]

        if not missing:
            return

        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            vectors = self.embedding(batch)

            for text, vec in zip(batch, vectors):
                self.embedding_cache[text] = vec

    def search_query(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        key = f"{top_k}::{query}"

        if key in self.search_cache:
            return self.search_cache[key]

        vector = self.embedding_cache.get(query)

        if vector is None:
            vector = self.embedding([query])[0]
            self.embedding_cache[query] = vector

        body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'combined_vector') + 1.0",
                        "params": {"query_vector": vector},
                    },
                }
            },
            "_source": ["case_title", "analysis", "facts", "focus", "source_type"],
        }

        response = self.es.search(
            index="rag_legal_cases", body=body, request_timeout=30
        )

        hits = response.get("hits", {}).get("hits", [])
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
                source = hit.get("_source", {})
                key = str(source.get("case_title") or hit.get("_id"))
                score = float(hit.get("_score", 0.0))

                if key not in unique_map or score > float(
                    unique_map[key].get("_score", 0.0)
                ):
                    unique_map[key] = hit

        merged = list(unique_map.values())
        merged.sort(key=lambda row: float(row.get("_score", 0.0)), reverse=True)
        return merged[:top_k]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def evaluate_binary(rows: List[Dict[str, Any]], threshold: float) -> BinaryMetrics:
    """Evaluate a fact-worker score threshold against labeled samples.

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

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def _apply_manual_review(
    rows: List[Dict[str, Any]],
    manual_review_rows: List[Dict[str, Any]],
) -> int:
    manual_map: Dict[str, int] = {}

    for row in manual_review_rows:
        sample_id = str(row.get("sample_id", "")).strip()

        if not sample_id:
            continue

        label = row.get("review_label")

        if label in (0, 1):
            manual_map[sample_id] = int(label)

    updated = 0

    for row in rows:
        sample_id = str(row.get("sample_id", ""))

        if sample_id in manual_map:
            reviewed = manual_map[sample_id]

            if int(row["label"]) != reviewed:
                updated += 1

            row["label"] = reviewed
            row["reviewed"] = True

        else:
            row["reviewed"] = False

    return updated


def _score_dataset_rows(
    *,
    rows: List[Dict[str, Any]],
    scorer: RuntimeESScorer,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []

    for row in rows:
        query_hits = [scorer.search_query(query, top_k=3) for query in row["queries"]]
        final_hits = scorer.deduplicate_and_rerank(query_hits, top_k=3)
        max_score = float(final_hits[0].get("_score", 1.0) - 1.0) if final_hits else 0.0
        top_case_title = ""
        top_source_type = ""

        if final_hits:
            source = final_hits[0].get("_source", {})
            top_case_title = str(source.get("case_title", ""))
            top_source_type = str(source.get("source_type", ""))

        scored.append(
            {
                **row,
                "max_score": max_score,
                "retrieved_count": len(final_hits),
                "top_case_title": top_case_title,
                "top_source_type": top_source_type,
            }
        )

    return scored


def _aggregate_scan_row(
    *,
    threshold: float,
    metrics_by_seed: List[BinaryMetrics],
    false_found_by_seed: List[float],
    law_false_found_by_seed: List[float],
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
        "mean_law_centric_false_found_rate": float(
            statistics.mean(law_false_found_by_seed)
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
        row["mean_false_found_rate"],
        row["mean_law_centric_false_found_rate"],
        -row["mean_macro_f1"],
        -row["mean_precision_pos"],
        row["std_f1_pos"],
        -row["threshold"],
    )


def _select_best_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("rows cannot be empty")

    return sorted(rows, key=_rank_key)[0]


def _score_key_cases(
    *,
    key_cases: List[Dict[str, Any]],
    scorer: RuntimeESScorer,
    baseline_threshold: float,
    tuned_threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for row in key_cases:
        query_hits = [scorer.search_query(query, top_k=3) for query in row["queries"]]
        final_hits = scorer.deduplicate_and_rerank(query_hits, top_k=3)
        max_score = float(final_hits[0].get("_score", 1.0) - 1.0) if final_hits else 0.0

        rows.append(
            {
                **row,
                "max_score": round(max_score, 4),
                "baseline_predict": int(max_score >= baseline_threshold),
                "tuned_predict": int(max_score >= tuned_threshold),
            }
        )

    return rows


def _build_manual_review_candidates(
    *,
    rows: List[Dict[str, Any]],
    baseline_threshold: float,
    max_candidates: int = 120,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for row in rows:
        score = float(row["max_score"])
        label = int(row["label"])
        baseline_pred = 1 if score >= baseline_threshold else 0
        is_borderline = abs(score - baseline_threshold) <= 0.04
        is_wrong = baseline_pred != label
        is_risky_neg = label == 0 and baseline_pred == 1

        if not (is_borderline or is_wrong or is_risky_neg):
            continue

        candidates.append(
            {
                "sample_id": row["sample_id"],
                "seed": row["seed"],
                "split": row["split"],
                "auto_label": label,
                "baseline_predict": baseline_pred,
                "max_score": round(score, 4),
                "source": row.get("source"),
                "source_type": row.get("source_type"),
                "negative_type": row.get("negative_type"),
                "intent": row["intent"],
                "queries": row["queries"],
                "review_label": None,
                "review_note": "",
            }
        )

    candidates.sort(
        key=lambda row: (
            0 if row["baseline_predict"] != row["auto_label"] else 1,
            abs(float(row["max_score"]) - baseline_threshold),
            -float(row["max_score"]),
        )
    )

    return candidates[:max_candidates]


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
    size_per_seed: int,
    baseline_threshold: float,
    tuned_threshold: float,
    best_row: Dict[str, Any],
    valid_baseline: BinaryMetrics,
    valid_tuned: BinaryMetrics,
    false_found_baseline: float,
    false_found_tuned: float,
    law_false_found_baseline: float,
    law_false_found_tuned: float,
    summary_rows: List[Dict[str, Any]],
    key_cases: List[Dict[str, Any]],
    review_candidates_count: int,
    manual_override_count: int,
) -> None:
    f1_delta = valid_tuned.f1_pos - valid_baseline.f1_pos
    recall_delta = valid_tuned.recall_pos - valid_baseline.recall_pos
    precision_delta = valid_tuned.precision_pos - valid_baseline.precision_pos
    ffr_delta = false_found_tuned - false_found_baseline
    law_ffr_delta = law_false_found_tuned - law_false_found_baseline
    lines: List[str] = []
    lines.append("# Round 5 Tuning Report: fact_worker_threshold")
    lines.append("")
    lines.append("## Scope and Constraints")

    lines.append(
        "- Parameter: `mas/config.py` -> `worker_threshold.fact_worker_threshold`"
    )

    lines.append(
        "- Runtime gate: `FactWorker` uses `max_score >= threshold` for FOUND/NOT_FOUND."
    )

    lines.append(f"- Baseline: {baseline_threshold:.2f}")
    lines.append("- Data policy: ES real samples + synthetic negatives/positives mix.")

    lines.append(
        "- Labeling policy: auto labeling with manual-review candidate export."
    )

    lines.append("")
    lines.append("## Dataset Setup")
    lines.append(f"- Seeds: {seeds}")
    lines.append(f"- Samples per seed: {size_per_seed}")
    lines.append("- Split: train/valid = 80/20")

    lines.append(
        f"- Manual review candidates exported: {review_candidates_count}; manual overrides applied: {manual_override_count}"
    )

    lines.append("")
    lines.append("## Threshold Search")
    lines.append("- Rough scan: 0.40 to 0.85, step 0.05")
    lines.append("- Fine scan: around rough best +/-0.06, step 0.01")

    lines.append(
        "- Selection: max F1+, tie-break Recall+, then false-found rate and macro F1"
    )

    lines.append("")
    lines.append("## Selected Threshold")
    lines.append(f"- Recommended `fact_worker_threshold`: **{tuned_threshold:.2f}**")

    lines.append(
        "- Train aggregate: "
        f"F1+={best_row['mean_f1_pos']:.4f}, "
        f"Recall+={best_row['mean_recall_pos']:.4f}, "
        f"Precision+={best_row['mean_precision_pos']:.4f}, "
        f"MacroF1={best_row['mean_macro_f1']:.4f}, "
        f"FalseFound={best_row['mean_false_found_rate']:.4f}"
    )

    lines.append("")
    lines.append("## Validation Comparison")
    lines.append(f"- {_format_metric('Baseline', valid_baseline)}")
    lines.append(f"- {_format_metric('Tuned', valid_tuned)}")
    lines.append(f"- F1+ delta: {f1_delta:+.4f}")
    lines.append(f"- Recall+ delta: {recall_delta:+.4f}")
    lines.append(f"- Precision+ delta: {precision_delta:+.4f}")
    lines.append(f"- False-found rate baseline: {false_found_baseline:.4f}")
    lines.append(f"- False-found rate tuned: {false_found_tuned:.4f}")
    lines.append(f"- False-found rate delta: {ffr_delta:+.4f}")

    lines.append(
        f"- Law-centric false-found baseline/tuned/delta: {law_false_found_baseline:.4f} -> {law_false_found_tuned:.4f} ({law_ffr_delta:+.4f})"
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

    lines.append(
        "-    fact_worker_threshold: float = {:.2f}".format(baseline_threshold)
    )

    lines.append("+    fact_worker_threshold: float = {:.2f}".format(tuned_threshold))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")

    lines.append(
        "- Scan details: `benchmarks/optim/artifacts/fact_worker_threshold_scan_train.json`."
    )

    lines.append(
        "- Validation summary: `benchmarks/optim/artifacts/fact_worker_threshold_valid_comparison.json`."
    )

    lines.append(
        "- Manual review template: `benchmarks/optim/artifacts/round5_fact_worker_manual_review_candidates.jsonl`."
    )

    if summary_rows:
        lines.append(
            f"- Seed summary example: positive_ratio={summary_rows[0]['positive_ratio']:.4f}, es_source_ratio={summary_rows[0]['es_source_ratio']:.4f}."
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for fact-worker threshold tuning.

    Returns:
        Parsed command-line namespace for dataset generation, scoring, and
        report output paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="82,83,84,85,86")
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
        default=Path("reports/round5_fact_worker_threshold.md"),
    )

    parser.add_argument(
        "--es-host", type=str, default=os.getenv("ES_HOST", "http://127.0.0.1:9200")
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
            "benchmarks/optim/artifacts/round5_fact_worker_manual_review.jsonl"
        ),
    )

    parser.add_argument(
        "--reuse-scored",
        action="store_true",
        help="Reuse scored per-seed files if present.",
    )

    return parser.parse_args()


def main() -> None:
    """Run fact-worker threshold tuning and export diagnostic artifacts.

    Raises:
        RuntimeError: If Elasticsearch is unreachable or key payloads cannot
            be constructed.
        ValueError: If no valid seeds are supplied.
    """
    args = parse_args()
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]

    if not seeds:
        raise ValueError("No valid seed provided")

    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    scorer = RuntimeESScorer(es_host=args.es_host, model_path=args.model_path)

    if not scorer.ping():
        raise RuntimeError(f"Elasticsearch not reachable at {args.es_host}")

    scored_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []
    key_cases: Optional[List[Dict[str, Any]]] = None
    all_queries: List[str] = []
    raw_rows_by_seed: Dict[int, List[Dict[str, Any]]] = {}

    for seed in seeds:
        payload = generate_fact_worker_es_dataset(
            seed=seed,
            size=args.size,
            valid_ratio=args.valid_ratio,
            positive_ratio=args.positive_ratio,
            es_host=args.es_host,
        )

        rows = payload["rows"]
        summary_rows.append(payload["summary"])

        if key_cases is None:
            key_cases = payload["key_cases"]

        raw_rows_by_seed[seed] = rows
        all_queries.extend(query for row in rows for query in row["queries"])
        data_path = artifacts_dir / f"fact_worker_seed{seed}.jsonl"
        summary_path = artifacts_dir / f"fact_worker_seed{seed}.summary.json"
        _write_jsonl(data_path, rows)
        _write_json(summary_path, payload["summary"])

    dedup_queries = list(dict.fromkeys(q for q in all_queries if q))
    scorer.embed_queries(dedup_queries)

    for seed in seeds:
        scored_path = artifacts_dir / f"fact_worker_scored_seed{seed}.jsonl"

        if args.reuse_scored and scored_path.exists():
            scored = _load_jsonl(scored_path)

        else:
            scored = _score_dataset_rows(rows=raw_rows_by_seed[seed], scorer=scorer)
            _write_jsonl(scored_path, scored)

        scored_by_seed[seed] = scored

    manual_rows = _load_jsonl(args.manual_review_file)
    manual_override_count = 0

    for seed in seeds:
        manual_override_count += _apply_manual_review(scored_by_seed[seed], manual_rows)

    train_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    valid_by_seed: Dict[int, List[Dict[str, Any]]] = {}

    for seed in seeds:
        train_by_seed[seed] = [
            row for row in scored_by_seed[seed] if row["split"] == "train"
        ]

        valid_by_seed[seed] = [
            row for row in scored_by_seed[seed] if row["split"] == "valid"
        ]

    review_candidates = _build_manual_review_candidates(
        rows=[row for seed in seeds for row in scored_by_seed[seed]],
        baseline_threshold=args.baseline_threshold,
        max_candidates=120,
    )

    _write_jsonl(
        artifacts_dir / "round5_fact_worker_manual_review_candidates.jsonl",
        review_candidates,
    )

    rough_thresholds = threshold_range(0.40, 0.85, 0.05)
    rough_rows: List[Dict[str, Any]] = []

    for threshold in rough_thresholds:
        metrics_by_seed = [
            evaluate_binary(train_by_seed[seed], threshold) for seed in seeds
        ]

        ffr_by_seed = [
            false_found_rate(train_by_seed[seed], threshold) for seed in seeds
        ]

        law_ffr_by_seed = [
            false_found_rate_by_type(train_by_seed[seed], threshold, "law_centric")
            for seed in seeds
        ]

        rough_rows.append(
            _aggregate_scan_row(
                threshold=threshold,
                metrics_by_seed=metrics_by_seed,
                false_found_by_seed=ffr_by_seed,
                law_false_found_by_seed=law_ffr_by_seed,
            )
        )

    rough_best = _select_best_row(rough_rows)
    fine_start = max(0.0, round(rough_best["threshold"] - 0.06, 2))
    fine_end = min(1.0, round(rough_best["threshold"] + 0.06, 2))

    scan_thresholds = sorted(
        set(rough_thresholds + threshold_range(fine_start, fine_end, 0.01))
    )

    scan_rows: List[Dict[str, Any]] = []

    for threshold in scan_thresholds:
        metrics_by_seed = [
            evaluate_binary(train_by_seed[seed], threshold) for seed in seeds
        ]

        ffr_by_seed = [
            false_found_rate(train_by_seed[seed], threshold) for seed in seeds
        ]

        law_ffr_by_seed = [
            false_found_rate_by_type(train_by_seed[seed], threshold, "law_centric")
            for seed in seeds
        ]

        scan_rows.append(
            _aggregate_scan_row(
                threshold=threshold,
                metrics_by_seed=metrics_by_seed,
                false_found_by_seed=ffr_by_seed,
                law_false_found_by_seed=law_ffr_by_seed,
            )
        )

    scan_rows.sort(key=_rank_key)
    best_row = _select_best_row(scan_rows)
    tuned_threshold = float(best_row["threshold"])
    valid_all = [row for seed in seeds for row in valid_by_seed[seed]]
    valid_baseline = evaluate_binary(valid_all, args.baseline_threshold)
    valid_tuned = evaluate_binary(valid_all, tuned_threshold)
    false_found_baseline = false_found_rate(valid_all, args.baseline_threshold)
    false_found_tuned = false_found_rate(valid_all, tuned_threshold)

    law_false_found_baseline = false_found_rate_by_type(
        valid_all, args.baseline_threshold, "law_centric"
    )

    law_false_found_tuned = false_found_rate_by_type(
        valid_all, tuned_threshold, "law_centric"
    )

    key_case_rows = _score_key_cases(
        key_cases=key_cases or [],
        scorer=scorer,
        baseline_threshold=args.baseline_threshold,
        tuned_threshold=tuned_threshold,
    )

    _write_json(artifacts_dir / "fact_worker_threshold_scan_train.json", scan_rows)

    _write_json(
        artifacts_dir / "fact_worker_threshold_valid_comparison.json",
        {
            "baseline_threshold": args.baseline_threshold,
            "tuned_threshold": tuned_threshold,
            "valid_baseline": valid_baseline.as_dict(),
            "valid_tuned": valid_tuned.as_dict(),
            "false_found_rate_baseline": false_found_baseline,
            "false_found_rate_tuned": false_found_tuned,
            "law_centric_false_found_rate_baseline": law_false_found_baseline,
            "law_centric_false_found_rate_tuned": law_false_found_tuned,
            "manual_override_count": manual_override_count,
            "review_candidates_count": len(review_candidates),
        },
    )

    _write_json(artifacts_dir / "fact_worker_key_cases.json", key_case_rows)

    _write_json(
        artifacts_dir / "fact_worker_round5_summary.json",
        {
            "seeds": seeds,
            "size_per_seed": args.size,
            "baseline_threshold": args.baseline_threshold,
            "recommended_threshold": tuned_threshold,
            "best_row": best_row,
            "valid_baseline": valid_baseline.as_dict(),
            "valid_tuned": valid_tuned.as_dict(),
            "false_found_rate_baseline": false_found_baseline,
            "false_found_rate_tuned": false_found_tuned,
            "law_centric_false_found_rate_baseline": law_false_found_baseline,
            "law_centric_false_found_rate_tuned": law_false_found_tuned,
            "manual_override_count": manual_override_count,
            "review_candidates_count": len(review_candidates),
            "dataset_summary_samples": summary_rows,
            "top_scan_rows": scan_rows[:10],
        },
    )

    _render_report(
        report_path=args.report_path,
        seeds=seeds,
        size_per_seed=args.size,
        baseline_threshold=args.baseline_threshold,
        tuned_threshold=tuned_threshold,
        best_row=best_row,
        valid_baseline=valid_baseline,
        valid_tuned=valid_tuned,
        false_found_baseline=false_found_baseline,
        false_found_tuned=false_found_tuned,
        law_false_found_baseline=law_false_found_baseline,
        law_false_found_tuned=law_false_found_tuned,
        summary_rows=summary_rows,
        key_cases=key_case_rows,
        review_candidates_count=len(review_candidates),
        manual_override_count=manual_override_count,
    )

    print("[fact-worker-threshold-tuning] done")
    print(f"baseline_threshold={args.baseline_threshold:.2f}")
    print(f"recommended_threshold={tuned_threshold:.2f}")
    print(f"valid_baseline_f1_pos={valid_baseline.f1_pos:.4f}")
    print(f"valid_tuned_f1_pos={valid_tuned.f1_pos:.4f}")
    print(f"valid_baseline_recall_pos={valid_baseline.recall_pos:.4f}")
    print(f"valid_tuned_recall_pos={valid_tuned.recall_pos:.4f}")
    print(f"false_found_rate_baseline={false_found_baseline:.4f}")
    print(f"false_found_rate_tuned={false_found_tuned:.4f}")
    print(f"report={args.report_path}")


if __name__ == "__main__":
    main()
