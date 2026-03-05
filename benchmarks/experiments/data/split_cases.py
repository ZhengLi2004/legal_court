"""Build Step 05 Dev/Test split and strict Dev-resample stability evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.data.step05_features import get_case_features, q_value

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))


def _parse_resample_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]

    if not seeds:
        raise ValueError("At least one resample seed is required")

    return seeds


def _parse_positive_int_list(raw: str, *, name: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]

    if not values:
        raise ValueError(f"{name} must contain at least one positive integer")

    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain only positive integers")

    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values: {values}")

    return values


def _bucket_text_len(value: int, q25: int, q50: int, q75: int) -> str:
    if value <= q25:
        return "Q1"

    if value <= q50:
        return "Q2"

    if value <= q75:
        return "Q3"

    return "Q4"


def _stable_hash_bucket(text: str, mod: int) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    bucket_idx = int(digest[:8], 16) % mod
    return f"CAUSE_H{bucket_idx:02d}"


def _build_quantile_edges(values: list[int], *, bins: int) -> list[float]:
    if bins <= 0:
        raise ValueError("bins must be positive")

    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        raise ValueError("Cannot build quantile edges from empty values")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    return [float(x) for x in np.quantile(arr, quantiles)]


def _bin_by_edges(value: int | float, edges: list[float]) -> str:
    if len(edges) < 2:
        return "B01"

    idx = int(np.searchsorted(np.asarray(edges, dtype=float), value, side="right") - 1)
    idx = max(0, min(idx, len(edges) - 2))
    return f"B{idx + 1:02d}"


def _make_distribution(values: list[Any]) -> dict[Any, float]:
    counts = Counter(values)
    total = len(values)

    if total == 0:
        return {}

    return {key: count / total for key, count in counts.items()}


def _tv_distance(base_counts: dict[str, int], sample_counts: dict[str, int]) -> float:
    base_total = sum(base_counts.values())
    sample_total = sum(sample_counts.values())

    if base_total <= 0 or sample_total <= 0:
        raise ValueError("Cannot compute TV distance on empty counts")

    keys = set(base_counts) | set(sample_counts)

    return 0.5 * sum(
        abs(
            base_counts.get(key, 0) / base_total
            - sample_counts.get(key, 0) / sample_total
        )
        for key in keys
    )


def _score_tv(base_dist: dict[Any, float], sample_dist: dict[Any, float]) -> float:
    keys = set(base_dist) | set(sample_dist)

    tv = 0.5 * sum(
        abs(base_dist.get(key, 0.0) - sample_dist.get(key, 0.0)) for key in keys
    )

    return 1.0 - tv


def _bootstrap_mean_ci(
    values: list[float],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    if len(values) == 1:
        only = float(values[0])
        return only, only

    arr = np.asarray(values, dtype=float)
    n = arr.size
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, n, size=(n_bootstrap, n))
    sample_means = arr[sample_indices].mean(axis=1)
    low, high = np.quantile(sample_means, [0.025, 0.975])
    return float(low), float(high)


def _allocate_dev_quota(
    *,
    counts_by_stratum: dict[str, int],
    dev_size: int,
    total_size: int,
) -> dict[str, int]:
    if dev_size <= 0:
        raise ValueError("dev_size must be positive")

    raw_quota = {
        key: counts_by_stratum[key] * dev_size / total_size for key in counts_by_stratum
    }

    floor_quota = {key: int(math.floor(value)) for key, value in raw_quota.items()}
    assigned = sum(floor_quota.values())
    remaining = dev_size - assigned

    if remaining < 0:
        raise ValueError("Invalid quota allocation: floor sum exceeds target")

    ranked = sorted(
        counts_by_stratum.keys(),
        key=lambda key: (
            raw_quota[key] - floor_quota[key],
            counts_by_stratum[key],
            key,
        ),
        reverse=True,
    )

    idx = 0

    while remaining > 0:
        key = ranked[idx % len(ranked)]

        if floor_quota[key] < counts_by_stratum[key]:
            floor_quota[key] += 1
            remaining -= 1

        idx += 1

        if idx > len(ranked) * 10 and remaining > 0:
            raise ValueError("Quota allocation failed to converge")

    if sum(floor_quota.values()) != dev_size:
        raise ValueError("Quota allocation does not sum to dev_size")

    return floor_quota


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_records(
    *,
    features_by_uid: dict[str, dict[str, Any]],
    q25: int,
    q50: int,
    q75: int,
    cause_bucket_mode: str,
    cause_hash_mod: int | None,
    top_cause_set: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for uid, feature in features_by_uid.items():
        claim_bin = "multi" if feature["claim_count"] >= 2 else "single"

        if cause_bucket_mode == "hash":
            if cause_hash_mod is None:
                raise ValueError("cause_hash_mod is required for hash mode")

            cause_bucket = _stable_hash_bucket(feature["sample_cause"], cause_hash_mod)

        elif cause_bucket_mode == "topk_other":
            cause_bucket = (
                feature["sample_cause"]
                if feature["sample_cause"] in top_cause_set
                else "OTHER"
            )

        else:
            raise ValueError(f"Unsupported cause_bucket_mode: {cause_bucket_mode}")

        text_bucket = _bucket_text_len(feature["text_len"], q25, q50, q75)

        records.append(
            {
                **feature,
                "uid": uid,
                "claim_bin": claim_bin,
                "cause_bucket": cause_bucket,
                "text_bucket": text_bucket,
            }
        )

    return records


def _select_dev_ids(
    *,
    records: list[dict[str, Any]],
    dev_size: int,
    total_size: int,
    split_seed: int,
) -> list[str]:
    rows_by_cause: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in records:
        rows_by_cause[row["cause_bucket"]].append(row)

    cause_quota = _allocate_dev_quota(
        counts_by_stratum={key: len(value) for key, value in rows_by_cause.items()},
        dev_size=dev_size,
        total_size=total_size,
    )

    split_rng = np.random.default_rng(split_seed)
    dev_ids: list[str] = []

    for cause_bucket in sorted(rows_by_cause.keys()):
        cause_rows = rows_by_cause[cause_bucket]
        cause_target = cause_quota[cause_bucket]

        if cause_target == 0:
            continue

        rows_by_secondary: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for row in cause_rows:
            secondary_key = "|".join([row["claim_bin"], row["text_bucket"]])
            rows_by_secondary[secondary_key].append(row)

        secondary_quota = _allocate_dev_quota(
            counts_by_stratum={
                key: len(value) for key, value in rows_by_secondary.items()
            },
            dev_size=cause_target,
            total_size=len(cause_rows),
        )

        for secondary_key in sorted(rows_by_secondary.keys()):
            rows = rows_by_secondary[secondary_key]
            quota = secondary_quota[secondary_key]

            if quota == 0:
                continue

            sampled = split_rng.choice(len(rows), size=quota, replace=False)
            dev_ids.extend(rows[int(idx)]["uid"] for idx in sampled)

    dev_id_set = set(dev_ids)

    if len(dev_id_set) != dev_size:
        raise ValueError(
            f"Dev split must contain {dev_size} unique ids, got {len(dev_id_set)}"
        )

    return sorted(dev_id_set)


def _compute_provisional_thresholds(
    *,
    dev_records: list[dict[str, Any]],
    min_signals: int,
    cause_count_min: int,
) -> dict[str, int]:
    return {
        "claim_q75": int(q_value((row["claim_count"] for row in dev_records), 0.75)),
        "text_q75": int(q_value((row["text_len"] for row in dev_records), 0.75)),
        "law_q75": int(q_value((row["law_count"] for row in dev_records), 0.75)),
        "person_q75": int(q_value((row["person_count"] for row in dev_records), 0.75)),
        "cause_count_min": int(cause_count_min),
        "min_signals": int(min_signals),
    }


def _provisional_hard_case(
    row: dict[str, Any],
    thresholds: dict[str, int],
) -> int:
    signal_count = 0
    signal_count += int(row["claim_count"] >= thresholds["claim_q75"])
    signal_count += int(row["text_len"] >= thresholds["text_q75"])
    signal_count += int(row["law_count"] >= thresholds["law_q75"])
    signal_count += int(row["person_count"] >= thresholds["person_q75"])
    signal_count += int(row["cause_count"] >= thresholds["cause_count_min"])
    return int(signal_count >= thresholds["min_signals"])


def _build_dev_rows_by_joint_strata(
    *,
    dev_records: list[dict[str, Any]],
    provisional_thresholds: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in dev_records:
        provisional_hard_case = _provisional_hard_case(row, provisional_thresholds)

        joint_stratum = "|".join(
            [
                row["cause_bucket"],
                row["claim_bin"],
                row["text_bucket"],
                f"hard_{provisional_hard_case}",
            ]
        )

        rows_by_stratum[joint_stratum].append(row)

    return rows_by_stratum


def _joint_sparsity_stats(
    rows_by_stratum: dict[str, list[dict[str, Any]]],
    *,
    dev_size: int,
) -> dict[str, float | int]:
    unique_count = len(rows_by_stratum)
    singleton_count = sum(1 for rows in rows_by_stratum.values() if len(rows) == 1)
    unique_ratio = unique_count / dev_size if dev_size else 0.0
    singleton_ratio = singleton_count / unique_count if unique_count else 0.0

    return {
        "unique_joint_strata": unique_count,
        "singleton_joint_strata": singleton_count,
        "joint_unique_ratio": unique_ratio,
        "joint_singleton_ratio": singleton_ratio,
    }


def _format_candidate_diagnostics(rows: list[dict[str, Any]]) -> str:
    header = (
        "candidate_mod | unique_joint_strata | singleton_joint_strata | "
        "joint_unique_ratio | joint_singleton_ratio | passes"
    )

    lines = [header]

    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row["candidate_mod"]),
                    str(row["unique_joint_strata"]),
                    str(row["singleton_joint_strata"]),
                    f"{float(row['joint_unique_ratio']):.4f}",
                    f"{float(row['joint_singleton_ratio']):.4f}",
                    str(bool(row["passes"])),
                ]
            )
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        default="data/sampling/cleaned_samples.jsonl",
        help="Input JSONL data path.",
    )

    parser.add_argument(
        "--output-dir",
        default="benchmarks/experiments/artifacts/splits",
        help="Output directory for split artifacts.",
    )

    parser.add_argument(
        "--report-dir",
        default="reports/experiments/step05/appendix",
        help="Directory for appendix figures.",
    )

    parser.add_argument("--dev-size", type=int, default=50)
    parser.add_argument("--cause-topk", type=int, default=20)
    parser.add_argument("--cause-hash-mod", type=int, default=16)

    parser.add_argument(
        "--cause-hash-candidates",
        default="16,12,10,8,6,4",
        help="Comma-separated candidates for adaptive hash bucket mod.",
    )

    parser.add_argument(
        "--joint-unique-ratio-max",
        type=float,
        default=0.8,
        help="Maximum allowed unique_joint_strata / dev_size for adaptive hash selection.",
    )

    parser.add_argument(
        "--joint-singleton-ratio-max",
        type=float,
        default=0.5,
        help="Maximum allowed singleton_joint_strata / unique_joint_strata for adaptive hash selection.",
    )

    parser.add_argument("--split-seed", type=int, default=20260350)

    parser.add_argument(
        "--resample-seeds",
        default="20260351,20260352,20260353,20260354,20260355",
        help="Comma-separated seeds used for 5 stratified bootstrap resamples.",
    )

    parser.add_argument("--bootstrap-b", type=int, default=2000)

    parser.add_argument(
        "--min-signals",
        type=int,
        default=2,
        help="Signal threshold used by Dev provisional hard-case labels.",
    )

    parser.add_argument(
        "--cause-count-min",
        type=int,
        default=2,
        help="Minimum cause_count signal for Dev provisional hard-case labels.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not (0.0 < args.joint_unique_ratio_max <= 1.0):
        raise ValueError("joint_unique_ratio_max must be in (0, 1]")

    if not (0.0 <= args.joint_singleton_ratio_max <= 1.0):
        raise ValueError("joint_singleton_ratio_max must be in [0, 1]")

    raw_cases = load_cases_from_jsonl(args.input)

    features_by_uid = {
        row["uid"]: row for row in (get_case_features(case) for case in raw_cases)
    }

    total_size = len(features_by_uid)

    if total_size != len(raw_cases):
        raise ValueError("Duplicate uid detected in input dataset")

    if args.dev_size <= 0 or args.dev_size >= total_size:
        raise ValueError(
            f"dev_size must be in [1, {total_size - 1}], got {args.dev_size}"
        )

    global_text_q25 = int(
        q_value((row["text_len"] for row in features_by_uid.values()), 0.25)
    )

    global_text_q50 = int(
        q_value((row["text_len"] for row in features_by_uid.values()), 0.50)
    )

    global_text_q75 = int(
        q_value((row["text_len"] for row in features_by_uid.values()), 0.75)
    )

    cause_counts = Counter(row["sample_cause"] for row in features_by_uid.values())
    unique_cause_count = len(cause_counts)
    max_cause_freq = max(cause_counts.values()) if cause_counts else 0
    unique_ratio = unique_cause_count / total_size if total_size else 0.0
    use_hash_cause_bucket = max_cause_freq <= 2 or unique_ratio > 0.5
    top_causes: list[str] = []
    top_cause_set: set[str] = set()
    records: list[dict[str, Any]]
    dev_ids_sorted: list[str]
    provisional_thresholds: dict[str, int]
    dev_rows_by_stratum: dict[str, list[dict[str, Any]]]
    hash_candidate_diagnostics: list[dict[str, Any]] = []
    cause_bucket_mode = "topk_other"
    cause_hash_mod_selected: int | None = None
    cause_hash_candidates: list[int] = []

    if use_hash_cause_bucket:
        cause_bucket_mode = "hash_adaptive"

        cause_hash_candidates = _parse_positive_int_list(
            args.cause_hash_candidates,
            name="cause_hash_candidates",
        )

        selected_payload: dict[str, Any] | None = None

        for candidate_mod in cause_hash_candidates:
            candidate_records = _build_records(
                features_by_uid=features_by_uid,
                q25=global_text_q25,
                q50=global_text_q50,
                q75=global_text_q75,
                cause_bucket_mode="hash",
                cause_hash_mod=candidate_mod,
                top_cause_set=set(),
            )

            candidate_dev_ids = _select_dev_ids(
                records=candidate_records,
                dev_size=args.dev_size,
                total_size=total_size,
                split_seed=args.split_seed,
            )

            candidate_record_by_uid = {row["uid"]: row for row in candidate_records}

            candidate_dev_records = [
                candidate_record_by_uid[uid] for uid in candidate_dev_ids
            ]

            candidate_thresholds = _compute_provisional_thresholds(
                dev_records=candidate_dev_records,
                min_signals=args.min_signals,
                cause_count_min=args.cause_count_min,
            )

            candidate_rows_by_stratum = _build_dev_rows_by_joint_strata(
                dev_records=candidate_dev_records,
                provisional_thresholds=candidate_thresholds,
            )

            sparsity = _joint_sparsity_stats(
                candidate_rows_by_stratum, dev_size=args.dev_size
            )

            passes = (
                float(sparsity["joint_unique_ratio"]) <= args.joint_unique_ratio_max
                and float(sparsity["joint_singleton_ratio"])
                <= args.joint_singleton_ratio_max
            )

            hash_candidate_diagnostics.append(
                {
                    "candidate_mod": candidate_mod,
                    **sparsity,
                    "passes": passes,
                }
            )

            if selected_payload is None and passes:
                selected_payload = {
                    "cause_hash_mod_selected": candidate_mod,
                    "records": candidate_records,
                    "dev_ids_sorted": candidate_dev_ids,
                    "provisional_thresholds": candidate_thresholds,
                    "dev_rows_by_stratum": candidate_rows_by_stratum,
                    "joint_sparsity": sparsity,
                }

        if selected_payload is None:
            diagnostics_table = _format_candidate_diagnostics(
                hash_candidate_diagnostics
            )

            raise ValueError(
                "No cause_hash_mod candidate satisfies joint strata sparsity constraints.\n"
                f"joint_unique_ratio_max={args.joint_unique_ratio_max}, "
                f"joint_singleton_ratio_max={args.joint_singleton_ratio_max}\n"
                f"{diagnostics_table}"
            )

        cause_hash_mod_selected = int(selected_payload["cause_hash_mod_selected"])
        records = selected_payload["records"]
        dev_ids_sorted = selected_payload["dev_ids_sorted"]
        provisional_thresholds = selected_payload["provisional_thresholds"]
        dev_rows_by_stratum = selected_payload["dev_rows_by_stratum"]
        joint_sparsity = selected_payload["joint_sparsity"]

    else:
        top_causes = [
            cause
            for cause, _ in sorted(
                cause_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[: args.cause_topk]
        ]

        top_cause_set = set(top_causes)

        records = _build_records(
            features_by_uid=features_by_uid,
            q25=global_text_q25,
            q50=global_text_q50,
            q75=global_text_q75,
            cause_bucket_mode="topk_other",
            cause_hash_mod=None,
            top_cause_set=top_cause_set,
        )

        dev_ids_sorted = _select_dev_ids(
            records=records,
            dev_size=args.dev_size,
            total_size=total_size,
            split_seed=args.split_seed,
        )

        record_by_uid = {row["uid"]: row for row in records}
        dev_records = [record_by_uid[uid] for uid in dev_ids_sorted]

        provisional_thresholds = _compute_provisional_thresholds(
            dev_records=dev_records,
            min_signals=args.min_signals,
            cause_count_min=args.cause_count_min,
        )

        dev_rows_by_stratum = _build_dev_rows_by_joint_strata(
            dev_records=dev_records,
            provisional_thresholds=provisional_thresholds,
        )

        joint_sparsity = _joint_sparsity_stats(
            dev_rows_by_stratum, dev_size=args.dev_size
        )

    dev_id_set = set(dev_ids_sorted)
    all_ids = set(features_by_uid.keys())
    test_id_set = all_ids - dev_id_set

    if dev_id_set & test_id_set:
        raise ValueError("Dev/Test overlap detected")

    if len(dev_id_set) + len(test_id_set) != total_size:
        raise ValueError("Dev/Test size mismatch")

    test_ids_sorted = sorted(test_id_set)

    _write_json(
        output_dir / "dev_ids.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(Path(args.input).resolve()),
            "split_seed": args.split_seed,
            "dev_size": args.dev_size,
            "ids": dev_ids_sorted,
        },
    )

    _write_json(
        output_dir / "test_ids.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(Path(args.input).resolve()),
            "split_seed": args.split_seed,
            "test_size": len(test_ids_sorted),
            "ids": test_ids_sorted,
        },
    )

    record_by_uid = {row["uid"]: row for row in records}
    dev_records = [record_by_uid[uid] for uid in dev_ids_sorted]
    feature_score_name = "feature_preservation_score"

    text_len_edges = _build_quantile_edges(
        [int(row["text_len"]) for row in dev_records],
        bins=10,
    )

    dev_baseline_distributions = {
        "claim_count": _make_distribution(
            [int(row["claim_count"]) for row in dev_records]
        ),
        "law_count": _make_distribution([int(row["law_count"]) for row in dev_records]),
        "person_count": _make_distribution(
            [int(row["person_count"]) for row in dev_records]
        ),
        "text_len_bin": _make_distribution(
            [_bin_by_edges(int(row["text_len"]), text_len_edges) for row in dev_records]
        ),
    }

    baseline_joint_counts = {
        key: len(rows) for key, rows in dev_rows_by_stratum.items()
    }

    resample_seeds = _parse_resample_seeds(args.resample_seeds)
    resample_results: list[dict[str, Any]] = []
    stratum_keys = sorted(dev_rows_by_stratum.keys())

    for idx, seed in enumerate(resample_seeds, start=1):
        rng = np.random.default_rng(seed)
        sampled_rows: list[dict[str, Any]] = []

        for stratum_key in stratum_keys:
            sampled_count = baseline_joint_counts[stratum_key]
            stratum_rows = dev_rows_by_stratum[stratum_key]

            chosen = rng.choice(
                len(stratum_rows), size=int(sampled_count), replace=True
            )

            sampled_rows.extend(stratum_rows[int(i)] for i in chosen)

        if len(sampled_rows) != args.dev_size:
            raise ValueError(
                f"Resample size mismatch for seed={seed}: {len(sampled_rows)}"
            )

        sampled_ids = [row["uid"] for row in sampled_rows]

        sampled_joint_counts = Counter(
            "|".join(
                [
                    row["cause_bucket"],
                    row["claim_bin"],
                    row["text_bucket"],
                    f"hard_{_provisional_hard_case(row, provisional_thresholds)}",
                ]
            )
            for row in sampled_rows
        )

        joint_tv = _tv_distance(baseline_joint_counts, sampled_joint_counts)
        exact_match = dict(sampled_joint_counts) == baseline_joint_counts

        if joint_tv != 0.0 or not exact_match:
            raise ValueError(
                f"Strict joint strata preservation failed for seed={seed}: "
                f"tv={joint_tv}, exact_match={exact_match}"
            )

        resample_claim_q75 = int(
            q_value((row["claim_count"] for row in sampled_rows), 0.75)
        )

        resample_text_q75 = int(
            q_value((row["text_len"] for row in sampled_rows), 0.75)
        )

        resample_law_q75 = int(
            q_value((row["law_count"] for row in sampled_rows), 0.75)
        )

        resample_person_q75 = int(
            q_value((row["person_count"] for row in sampled_rows), 0.75)
        )

        dim_scores = {}

        for dim_name, extractor in (
            ("claim_count", lambda row: int(row["claim_count"])),
            ("law_count", lambda row: int(row["law_count"])),
            ("person_count", lambda row: int(row["person_count"])),
            (
                "text_len_bin",
                lambda row: _bin_by_edges(int(row["text_len"]), text_len_edges),
            ),
        ):
            base_dist = dev_baseline_distributions[dim_name]
            sample_dist = _make_distribution([extractor(row) for row in sampled_rows])
            dim_scores[dim_name] = _score_tv(base_dist, sample_dist)

        overall_score = float(np.mean(list(dim_scores.values())))

        payload = {
            "resample_index": idx,
            "seed": seed,
            "size": len(sampled_rows),
            "unique_size": len(set(sampled_ids)),
            "sampled_ids": sampled_ids,
            "parameter_vector": {
                "claim_q75": resample_claim_q75,
                "text_q75": resample_text_q75,
                "law_q75": resample_law_q75,
                "person_q75": resample_person_q75,
            },
            "hard_case_source": "dev_provisional",
            "resample_strictness": "exact_joint_ratio",
            "joint_strata_tv_vs_dev": joint_tv,
            "joint_strata_exact_match": exact_match,
            "cause_hash_mod_selected": cause_hash_mod_selected,
            "score_name": feature_score_name,
            "dimension_scores": dim_scores,
            "overall_score": overall_score,
        }

        _write_json(output_dir / f"dev_resample_{idx}.json", payload)
        resample_results.append(payload)

    curve_rows: list[dict[str, Any]] = []

    for n in range(1, len(resample_results) + 1):
        subset = resample_results[:n]
        scores = [float(row["overall_score"]) for row in subset]
        score_mean = float(np.mean(scores))

        ci_low, ci_high = _bootstrap_mean_ci(
            scores,
            n_bootstrap=args.bootstrap_b,
            seed=20260399 + n,
        )

        claim_vals = [row["parameter_vector"]["claim_q75"] for row in subset]
        text_vals = [row["parameter_vector"]["text_q75"] for row in subset]
        law_vals = [row["parameter_vector"]["law_q75"] for row in subset]
        person_vals = [row["parameter_vector"]["person_q75"] for row in subset]

        def _summary(values: list[int], prefix: str) -> dict[str, float]:
            p20 = float(np.quantile(values, 0.2))
            p80 = float(np.quantile(values, 0.8))
            median = float(np.median(values))

            return {
                f"{prefix}_median": median,
                f"{prefix}_p20": p20,
                f"{prefix}_p80": p80,
                f"{prefix}_width": p80 - p20,
            }

        row = {
            "n_resamples": n,
            "score_mean": score_mean,
            "score_ci_low": ci_low,
            "score_ci_high": ci_high,
        }

        row.update(_summary(claim_vals, "claim_q75"))
        row.update(_summary(text_vals, "text_q75"))
        row.update(_summary(law_vals, "law_q75"))
        row.update(_summary(person_vals, "person_q75"))
        curve_rows.append(row)

    curve_path = output_dir / "resample_stability_curve.csv"

    with curve_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)

    x = [row["n_resamples"] for row in curve_rows]
    score_mean = [row["score_mean"] for row in curve_rows]
    score_low = [row["score_ci_low"] for row in curve_rows]
    score_high = [row["score_ci_high"] for row in curve_rows]
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(x, score_mean, marker="o", label="Feature Preservation Score")
    ax1.fill_between(x, score_low, score_high, alpha=0.2, label="95% CI")
    ax1.set_title("Step 05 Resample Stability")
    ax1.set_ylabel("Score")
    ax1.set_xticks(x)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")
    ax2 = plt.subplot(2, 1, 2)

    ax2.plot(
        x,
        [row["claim_q75_width"] for row in curve_rows],
        marker="o",
        label="claim_q75_width",
    )

    ax2.plot(
        x,
        [row["text_q75_width"] for row in curve_rows],
        marker="o",
        label="text_q75_width",
    )

    ax2.plot(
        x,
        [row["law_q75_width"] for row in curve_rows],
        marker="o",
        label="law_q75_width",
    )

    ax2.plot(
        x,
        [row["person_q75_width"] for row in curve_rows],
        marker="o",
        label="person_q75_width",
    )

    ax2.set_xlabel("Number of Resamples Included")
    ax2.set_ylabel("P80 - P20")
    ax2.set_xticks(x)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plot_path = report_dir / "resample_stability.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol_version": "step05_v3_strict_joint_resample",
        "input_path": str(Path(args.input).resolve()),
        "dev_size": args.dev_size,
        "test_size": len(test_ids_sorted),
        "split_seed": args.split_seed,
        "resample_seeds": resample_seeds,
        "resample_strictness": "exact_joint_ratio",
        "split_strata_dimensions": ["cause_bucket", "claim_bin", "text_bucket"],
        "resample_strata_dimensions": [
            "cause_bucket",
            "claim_bin",
            "text_bucket",
            "hard_case_provisional",
        ],
        "resample_hard_case_source": "dev_provisional",
        "provisional_thresholds": provisional_thresholds,
        "joint_strata_sparsity": joint_sparsity,
        "joint_sparsity_thresholds": {
            "joint_unique_ratio_max": args.joint_unique_ratio_max,
            "joint_singleton_ratio_max": args.joint_singleton_ratio_max,
        },
        "text_quantiles": {
            "q25": global_text_q25,
            "q50": global_text_q50,
            "q75": global_text_q75,
        },
        "score_name": feature_score_name,
        "cause_bucket_mode": cause_bucket_mode,
        "cause_hash_mod": cause_hash_mod_selected if use_hash_cause_bucket else None,
        "cause_hash_mod_selected": cause_hash_mod_selected
        if use_hash_cause_bucket
        else None,
        "cause_hash_mod_candidates": cause_hash_candidates
        if use_hash_cause_bucket
        else [],
        "hash_candidate_diagnostics": hash_candidate_diagnostics,
        "cause_longtail_stats": {
            "unique_cause_count": unique_cause_count,
            "max_cause_frequency": max_cause_freq,
            "unique_ratio": unique_ratio,
        },
        "cause_topk": args.cause_topk,
        "top_cause_values": top_causes,
    }

    _write_json(output_dir / "split_manifest.json", summary)
    print(f"saved: {output_dir / 'dev_ids.json'}")
    print(f"saved: {output_dir / 'test_ids.json'}")
    print(f"saved: {curve_path}")
    print(f"saved: {plot_path}")
    print(f"saved: {output_dir / 'split_manifest.json'}")

    if cause_hash_mod_selected is not None:
        print(f"summary: cause_hash_mod_selected={cause_hash_mod_selected}")

    print(
        f"summary: dev={args.dev_size}, test={len(test_ids_sorted)}, total={total_size}"
    )


if __name__ == "__main__":
    main()
