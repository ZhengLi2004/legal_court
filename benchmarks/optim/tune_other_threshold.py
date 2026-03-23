"""Tune `dedup.other_threshold` for CLAIM-node semantic deduplication.

This round focuses on LLM-style CLAIM content generated in controller
`VERIFY_AND_DECIDE` actions.
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

from benchmarks.optim.generate_claim_dedup_dataset import generate_claim_dedup_dataset

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
    """Embedding wrapper aligned with runtime model usage."""

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


def false_merge_risk(rows: List[Dict[str, Any]], threshold: float) -> float:
    """False merge risk on contradictory high-overlap negative subset."""
    hard_negatives = [
        row for row in rows if row.get("risk_bucket") == "contradictory_high_overlap"
    ]

    if not hard_negatives:
        return 0.0

    risky_fp = sum(1 for row in hard_negatives if float(row["similarity"]) >= threshold)
    return _safe_div(risky_fp, len(hard_negatives))


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


def _score_rows(
    rows: List[Dict[str, Any]],
    embedding: RuntimeEmbedding,
    cache: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    texts: List[str] = []
    seen: set[str] = set()

    for row in rows:
        left = row["existing_claim"]
        right = row["new_claim"]

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
        sim = _cosine_similarity(cache[row["existing_claim"]], cache[row["new_claim"]])
        scored.append({**row, "similarity": float(sim)})

    return scored


def _aggregate_scan_row(
    *,
    threshold: float,
    metrics_by_seed: List[BinaryMetrics],
    false_merge_risk_by_seed: List[float],
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
        "mean_false_merge_risk": float(statistics.mean(false_merge_risk_by_seed)),
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
        row["mean_false_merge_risk"],
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
    embedding: RuntimeEmbedding,
    cache: Dict[str, List[float]],
    baseline_threshold: float,
    tuned_threshold: float,
) -> List[Dict[str, Any]]:
    texts: List[str] = []

    for row in key_cases:
        texts.append(row["existing_claim"])
        texts.append(row["new_claim"])

    missing = [text for text in dict.fromkeys(texts) if text not in cache]

    if missing:
        vectors = embedding.embed_texts(missing)

        for text, vec in zip(missing, vectors):
            cache[text] = vec

    result: List[Dict[str, Any]] = []

    for row in key_cases:
        sim = _cosine_similarity(cache[row["existing_claim"]], cache[row["new_claim"]])

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
    size_per_seed: int,
    baseline_threshold: float,
    tuned_threshold: float,
    style_rate: float,
    boundary_ratio: float,
    hard_negative_ratio: float,
    best_row: Dict[str, Any],
    valid_baseline: BinaryMetrics,
    valid_tuned: BinaryMetrics,
    risk_baseline: float,
    risk_tuned: float,
    key_cases: List[Dict[str, Any]],
) -> None:
    f1_delta = valid_tuned.f1_pos - valid_baseline.f1_pos
    recall_delta = valid_tuned.recall_pos - valid_baseline.recall_pos
    precision_delta = valid_tuned.precision_pos - valid_baseline.precision_pos
    risk_delta = risk_tuned - risk_baseline
    lines: List[str] = []
    lines.append("# Round 4 Tuning Report: other_threshold")
    lines.append("")
    lines.append("## Scope and Constraints")
    lines.append("- Parameter: `mas/config.py` -> `dedup.other_threshold`")

    lines.append(
        "- Runtime semantic scope in this round: CLAIM dedup (viewpoint nodes)."
    )

    lines.append("- Baseline: {:.2f}".format(baseline_threshold))

    lines.append(
        "- Data policy: fully synthetic LLM-style claim text, no ES, no historical DB sampling"
    )

    lines.append(
        "- Objective: F1+ primary, Recall+ tie-break, with hard-negative false-merge risk audit"
    )

    lines.append("")
    lines.append("## VERIFY_AND_DECIDE Style Alignment")
    lines.append("- Claims are generated in controller-action content style:")
    lines.append("  - legal conclusion + reason chain + procedural/legal basis")
    lines.append("  - short/long mixed outputs and multi-sentence legal arguments")
    lines.append("")
    lines.append("## Dataset Setup")
    lines.append(f"- Seeds: {seeds}")
    lines.append(f"- Pairs per seed: {size_per_seed}")
    lines.append("- Split: train/valid = 80/20")
    lines.append(f"- Style compliance rate: {style_rate:.2%}")
    lines.append(f"- Boundary sample ratio: {boundary_ratio:.2%}")

    lines.append(
        f"- Contradictory high-overlap negative ratio: {hard_negative_ratio:.2%}"
    )

    lines.append("")
    lines.append("## Threshold Search")
    lines.append("- Rough scan: 0.70 to 0.98, step 0.03")
    lines.append("- Fine scan: around rough best +/-0.06, step 0.01")

    lines.append(
        "- Selection: max F1+, tie-break Recall+, then false-merge risk, macro F1 and precision"
    )

    lines.append("")
    lines.append("## Selected Threshold")
    lines.append(f"- Recommended `other_threshold`: **{tuned_threshold:.2f}**")

    lines.append(
        "- Train aggregate: "
        f"F1+={best_row['mean_f1_pos']:.4f}, "
        f"Recall+={best_row['mean_recall_pos']:.4f}, "
        f"Precision+={best_row['mean_precision_pos']:.4f}, "
        f"MacroF1={best_row['mean_macro_f1']:.4f}, "
        f"hard-neg-risk={best_row['mean_false_merge_risk']:.4f}"
    )

    lines.append("")
    lines.append("## Validation Comparison")
    lines.append(f"- {_format_metric('Baseline', valid_baseline)}")
    lines.append(f"- {_format_metric('Tuned', valid_tuned)}")
    lines.append(f"- F1+ delta: {f1_delta:+.4f}")
    lines.append(f"- Recall+ delta: {recall_delta:+.4f}")
    lines.append(f"- Precision+ delta: {precision_delta:+.4f}")
    lines.append(f"- Hard-negative false-merge risk baseline: {risk_baseline:.4f}")
    lines.append(f"- Hard-negative false-merge risk tuned: {risk_tuned:.4f}")
    lines.append(f"- Hard-negative false-merge risk delta: {risk_delta:+.4f}")
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
    lines.append("-    other_threshold: float = {:.2f}".format(baseline_threshold))
    lines.append("+    other_threshold: float = {:.2f}".format(tuned_threshold))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")

    lines.append(
        "- Scan details: `benchmarks/optim/artifacts/other_threshold_scan_train.json`."
    )

    lines.append(
        "- Validation summary: `benchmarks/optim/artifacts/other_threshold_valid_comparison.json`."
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="72,73,74,75,76")
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--baseline-threshold", type=float, default=0.90)

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("benchmarks/optim/artifacts"),
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/round4_other_threshold.md"),
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
    cache: Dict[str, List[float]] = {}
    scored_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    key_cases: List[Dict[str, Any]] = []
    style_flags: List[bool] = []
    boundary_flags: List[bool] = []
    hard_negative_flags: List[bool] = []

    for seed in seeds:
        payload = generate_claim_dedup_dataset(
            seed=seed,
            size=args.size,
            valid_ratio=args.valid_ratio,
        )

        rows = payload["pairs"]

        if not key_cases:
            key_cases = payload["key_cases"]

        style_flags.extend(bool(row.get("style_pass")) for row in rows)
        boundary_flags.extend(row.get("noise_tag") == "boundary" for row in rows)

        hard_negative_flags.extend(
            row.get("risk_bucket") == "contradictory_high_overlap" for row in rows
        )

        data_path = artifacts_dir / f"claim_dedup_seed{seed}.jsonl"
        scored_path = artifacts_dir / f"claim_similarity_seed{seed}.jsonl"
        _write_jsonl(data_path, rows)

        if args.reuse_similarity and scored_path.exists():
            scored = _load_jsonl(scored_path)

        else:
            scored = _score_rows(rows, embedding=embedding, cache=cache)
            _write_jsonl(scored_path, scored)

        scored_by_seed[seed] = scored

    train_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    valid_by_seed: Dict[int, List[Dict[str, Any]]] = {}

    for seed in seeds:
        train_by_seed[seed] = [
            row for row in scored_by_seed[seed] if row["split"] == "train"
        ]

        valid_by_seed[seed] = [
            row for row in scored_by_seed[seed] if row["split"] == "valid"
        ]

    rough_thresholds = threshold_range(0.70, 0.98, 0.03)
    rough_rows: List[Dict[str, Any]] = []

    for threshold in rough_thresholds:
        metrics_by_seed = [
            evaluate_binary(train_by_seed[seed], threshold) for seed in seeds
        ]

        risk_by_seed = [
            false_merge_risk(train_by_seed[seed], threshold) for seed in seeds
        ]

        rough_rows.append(
            _aggregate_scan_row(
                threshold=threshold,
                metrics_by_seed=metrics_by_seed,
                false_merge_risk_by_seed=risk_by_seed,
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

        risk_by_seed = [
            false_merge_risk(train_by_seed[seed], threshold) for seed in seeds
        ]

        scan_rows.append(
            _aggregate_scan_row(
                threshold=threshold,
                metrics_by_seed=metrics_by_seed,
                false_merge_risk_by_seed=risk_by_seed,
            )
        )

    scan_rows.sort(key=_rank_key)
    best_row = _select_best_row(scan_rows)
    tuned_threshold = float(best_row["threshold"])
    valid_all = [row for seed in seeds for row in valid_by_seed[seed]]
    valid_baseline = evaluate_binary(valid_all, args.baseline_threshold)
    valid_tuned = evaluate_binary(valid_all, tuned_threshold)
    risk_baseline = false_merge_risk(valid_all, args.baseline_threshold)
    risk_tuned = false_merge_risk(valid_all, tuned_threshold)
    style_rate = float(sum(style_flags)) / max(1, len(style_flags))
    boundary_ratio = float(sum(boundary_flags)) / max(1, len(boundary_flags))

    hard_negative_ratio = float(sum(hard_negative_flags)) / max(
        1, len(hard_negative_flags)
    )

    key_case_rows = _score_key_cases(
        key_cases=key_cases,
        embedding=embedding,
        cache=cache,
        baseline_threshold=args.baseline_threshold,
        tuned_threshold=tuned_threshold,
    )

    _write_json(artifacts_dir / "other_threshold_scan_train.json", scan_rows)

    _write_json(
        artifacts_dir / "other_threshold_valid_comparison.json",
        {
            "baseline_threshold": args.baseline_threshold,
            "tuned_threshold": tuned_threshold,
            "valid_baseline": valid_baseline.as_dict(),
            "valid_tuned": valid_tuned.as_dict(),
            "hard_negative_false_merge_risk_baseline": risk_baseline,
            "hard_negative_false_merge_risk_tuned": risk_tuned,
            "style_compliance_rate": style_rate,
            "boundary_ratio": boundary_ratio,
            "hard_negative_ratio": hard_negative_ratio,
        },
    )

    _write_json(artifacts_dir / "claim_key_cases.json", key_case_rows)
    _write_json(
        artifacts_dir / "claim_round4_summary.json",
        {
            "seeds": seeds,
            "size_per_seed": args.size,
            "baseline_threshold": args.baseline_threshold,
            "recommended_threshold": tuned_threshold,
            "best_row": best_row,
            "valid_baseline": valid_baseline.as_dict(),
            "valid_tuned": valid_tuned.as_dict(),
            "hard_negative_false_merge_risk_baseline": risk_baseline,
            "hard_negative_false_merge_risk_tuned": risk_tuned,
            "style_compliance_rate": style_rate,
            "boundary_ratio": boundary_ratio,
            "hard_negative_ratio": hard_negative_ratio,
            "top_scan_rows": scan_rows[:10],
        },
    )

    _render_report(
        report_path=args.report_path,
        seeds=seeds,
        size_per_seed=args.size,
        baseline_threshold=args.baseline_threshold,
        tuned_threshold=tuned_threshold,
        style_rate=style_rate,
        boundary_ratio=boundary_ratio,
        hard_negative_ratio=hard_negative_ratio,
        best_row=best_row,
        valid_baseline=valid_baseline,
        valid_tuned=valid_tuned,
        risk_baseline=risk_baseline,
        risk_tuned=risk_tuned,
        key_cases=key_case_rows,
    )

    print("[other-threshold-tuning] done")
    print(f"baseline_threshold={args.baseline_threshold:.2f}")
    print(f"recommended_threshold={tuned_threshold:.2f}")
    print(f"valid_baseline_f1_pos={valid_baseline.f1_pos:.4f}")
    print(f"valid_tuned_f1_pos={valid_tuned.f1_pos:.4f}")
    print(f"valid_baseline_recall_pos={valid_baseline.recall_pos:.4f}")
    print(f"valid_tuned_recall_pos={valid_tuned.recall_pos:.4f}")
    print(f"hard_negative_false_merge_risk_baseline={risk_baseline:.4f}")
    print(f"hard_negative_false_merge_risk_tuned={risk_tuned:.4f}")
    print(f"report={args.report_path}")


if __name__ == "__main__":
    main()
