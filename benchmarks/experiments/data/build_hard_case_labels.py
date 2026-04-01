"""Build Step 05 hard-case labels from Dev-resample median thresholds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from benchmarks.experiments.data.loader import load_cases_from_jsonl
from benchmarks.experiments.data.step05_features import get_case_features


@dataclass(frozen=True)
class RuleThresholds:
    """Thresholds used by hard-case labeling rules.

    Attributes:
        claim_q75: Q75 threshold for claim counts.
        text_q75: Q75 threshold for text lengths.
        law_q75: Q75 threshold for law citation counts.
        person_q75: Q75 threshold for person-name counts.
        cause_count_min: Minimum distinct-cause threshold.
    """

    claim_q75: int
    text_q75: int
    law_q75: int
    person_q75: int
    cause_count_min: int = 2


def _label_one(
    feature: dict[str, Any],
    thresholds: RuleThresholds,
    *,
    min_signals: int,
) -> dict[str, Any]:
    signals = {
        "claim_count_ge_q75": feature["claim_count"] >= thresholds.claim_q75,
        "text_len_ge_q75": feature["text_len"] >= thresholds.text_q75,
        "law_count_ge_q75": feature["law_count"] >= thresholds.law_q75,
        "person_count_ge_q75": feature["person_count"] >= thresholds.person_q75,
        "cause_count_ge_min": feature["cause_count"] >= thresholds.cause_count_min,
    }

    signal_count = sum(1 for value in signals.values() if value)

    return {
        "uid": feature["uid"],
        "hard_case": 1 if signal_count >= min_signals else 0,
        "signal_count": signal_count,
        "signals": signals,
        "features": {
            "sample_cause": feature["sample_cause"],
            "claim_count": feature["claim_count"],
            "text_len": feature["text_len"],
            "law_count": feature["law_count"],
            "person_count": feature["person_count"],
            "cause_count": feature["cause_count"],
        },
    }


def _hard_case_set(labels: list[dict[str, Any]]) -> set[str]:
    return {row["uid"] for row in labels if int(row["hard_case"]) == 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0

    return len(a & b) / len(a | b)


def _stable_hash_bucket(text: str, mod: int) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    bucket_idx = int(digest[:8], 16) % mod
    return f"CAUSE_H{bucket_idx:02d}"


def _bucket_text_len(value: int, q25: int, q50: int, q75: int) -> str:
    if value <= q25:
        return "Q1"

    if value <= q50:
        return "Q2"

    if value <= q75:
        return "Q3"

    return "Q4"


def _collect_distribution_rows(
    records: list[dict[str, Any]],
    *,
    dataset_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    dims = {
        "cause_bucket": [row["cause_bucket"] for row in records],
        "claim_bin": [row["claim_bin"] for row in records],
        "text_bucket": [row["text_bucket"] for row in records],
        "hard_case": [row["hard_case"] for row in records],
        "joint_stratum": [row["joint_stratum"] for row in records],
    }

    for dim_name, values in dims.items():
        counts = Counter(values)
        total = len(values)

        for bucket, count in sorted(
            counts.items(), key=lambda item: (-item[1], str(item[0]))
        ):
            rows.append(
                {
                    "dataset": dataset_name,
                    "dimension": dim_name,
                    "bucket": str(bucket),
                    "count": count,
                    "ratio": count / total if total else 0.0,
                }
            )

    return rows


def _load_thresholds_from_curve(
    curve_path: Path, cause_count_min: int
) -> tuple[RuleThresholds, int]:
    with curve_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise ValueError(f"Curve file is empty: {curve_path}")

    rows_sorted = sorted(rows, key=lambda row: int(row["n_resamples"]))
    last_row = rows_sorted[-1]

    for key in (
        "claim_q75_median",
        "text_q75_median",
        "law_q75_median",
        "person_q75_median",
    ):
        if key not in last_row:
            raise ValueError(f"Missing column `{key}` in curve file: {curve_path}")

    thresholds = RuleThresholds(
        claim_q75=int(round(float(last_row["claim_q75_median"]))),
        text_q75=int(round(float(last_row["text_q75_median"]))),
        law_q75=int(round(float(last_row["law_q75_median"]))),
        person_q75=int(round(float(last_row["person_q75_median"]))),
        cause_count_min=cause_count_min,
    )

    n_resamples = int(last_row["n_resamples"])

    if (
        min(
            thresholds.claim_q75,
            thresholds.text_q75,
            thresholds.law_q75,
            thresholds.person_q75,
        )
        <= 0
    ):
        raise ValueError(f"Invalid median thresholds from curve: {thresholds}")

    return thresholds, n_resamples


def _build_perturbation_rows(
    *,
    features: list[dict[str, Any]],
    base_thresholds: RuleThresholds,
    min_signals: int,
    text_perturb_ratio: float,
) -> list[dict[str, Any]]:
    base_labels = [
        _label_one(row, base_thresholds, min_signals=min_signals) for row in features
    ]

    base_set = _hard_case_set(base_labels)
    total = len(base_labels)

    scenarios: list[tuple[str, RuleThresholds, int]] = [
        ("base", base_thresholds, min_signals),
        (
            "claim_q75_minus1",
            RuleThresholds(
                claim_q75=max(1, base_thresholds.claim_q75 - 1),
                text_q75=base_thresholds.text_q75,
                law_q75=base_thresholds.law_q75,
                person_q75=base_thresholds.person_q75,
                cause_count_min=base_thresholds.cause_count_min,
            ),
            min_signals,
        ),
        (
            "claim_q75_plus1",
            RuleThresholds(
                claim_q75=base_thresholds.claim_q75 + 1,
                text_q75=base_thresholds.text_q75,
                law_q75=base_thresholds.law_q75,
                person_q75=base_thresholds.person_q75,
                cause_count_min=base_thresholds.cause_count_min,
            ),
            min_signals,
        ),
        (
            "text_q75_minus5pct",
            RuleThresholds(
                claim_q75=base_thresholds.claim_q75,
                text_q75=max(
                    1, int(round(base_thresholds.text_q75 * (1.0 - text_perturb_ratio)))
                ),
                law_q75=base_thresholds.law_q75,
                person_q75=base_thresholds.person_q75,
                cause_count_min=base_thresholds.cause_count_min,
            ),
            min_signals,
        ),
        (
            "text_q75_plus5pct",
            RuleThresholds(
                claim_q75=base_thresholds.claim_q75,
                text_q75=max(
                    1, int(round(base_thresholds.text_q75 * (1.0 + text_perturb_ratio)))
                ),
                law_q75=base_thresholds.law_q75,
                person_q75=base_thresholds.person_q75,
                cause_count_min=base_thresholds.cause_count_min,
            ),
            min_signals,
        ),
        (
            "law_q75_minus1",
            RuleThresholds(
                claim_q75=base_thresholds.claim_q75,
                text_q75=base_thresholds.text_q75,
                law_q75=max(1, base_thresholds.law_q75 - 1),
                person_q75=base_thresholds.person_q75,
                cause_count_min=base_thresholds.cause_count_min,
            ),
            min_signals,
        ),
        (
            "law_q75_plus1",
            RuleThresholds(
                claim_q75=base_thresholds.claim_q75,
                text_q75=base_thresholds.text_q75,
                law_q75=base_thresholds.law_q75 + 1,
                person_q75=base_thresholds.person_q75,
                cause_count_min=base_thresholds.cause_count_min,
            ),
            min_signals,
        ),
        (
            "person_q75_minus1",
            RuleThresholds(
                claim_q75=base_thresholds.claim_q75,
                text_q75=base_thresholds.text_q75,
                law_q75=base_thresholds.law_q75,
                person_q75=max(1, base_thresholds.person_q75 - 1),
                cause_count_min=base_thresholds.cause_count_min,
            ),
            min_signals,
        ),
        (
            "person_q75_plus1",
            RuleThresholds(
                claim_q75=base_thresholds.claim_q75,
                text_q75=base_thresholds.text_q75,
                law_q75=base_thresholds.law_q75,
                person_q75=base_thresholds.person_q75 + 1,
                cause_count_min=base_thresholds.cause_count_min,
            ),
            min_signals,
        ),
        ("min_signals_minus1", base_thresholds, max(1, min_signals - 1)),
        ("min_signals_plus1", base_thresholds, min_signals + 1),
    ]

    rows: list[dict[str, Any]] = []

    for scenario_name, thresholds, scenario_min_signals in scenarios:
        labels = [
            _label_one(row, thresholds, min_signals=scenario_min_signals)
            for row in features
        ]

        hard_set = _hard_case_set(labels)

        changed = sum(
            1
            for left, right in zip(base_labels, labels)
            if int(left["hard_case"]) != int(right["hard_case"])
        )

        rows.append(
            {
                "scenario": scenario_name,
                "min_signals": scenario_min_signals,
                "claim_q75": thresholds.claim_q75,
                "text_q75": thresholds.text_q75,
                "law_q75": thresholds.law_q75,
                "person_q75": thresholds.person_q75,
                "cause_count_min": thresholds.cause_count_min,
                "hard_case_count": len(hard_set),
                "hard_case_ratio": len(hard_set) / total,
                "changed_count_vs_base": changed,
                "changed_ratio_vs_base": changed / total,
                "hard_set_jaccard_vs_base": _jaccard(base_set, hard_set),
            }
        )

    return rows


def _write_rulebook(
    *,
    path: Path,
    rule_version: str,
    thresholds: RuleThresholds,
    min_signals: int,
    total_cases: int,
    hard_cases: int,
    curve_path: Path,
    n_resamples: int,
    split_manifest: dict[str, Any],
) -> None:
    lines = [
        "# Step 05 Hard Case Rulebook",
        "",
        f"- 规则版本: `{rule_version}`",
        f"- 生成时间: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- 协议版本: `{split_manifest.get('protocol_version', 'unknown')}`",
        f"- 阈值来源: `Dev重采样中位数 (n={n_resamples})`",
        f"- 稳定性曲线路径: `{curve_path.resolve()}`",
        f"- 总样本数: `{total_cases}`",
        f"- 难例样本数: `{hard_cases}`",
        f"- 难例占比: `{hard_cases / total_cases:.4f}`",
        "",
        "## 标签规则",
        "",
        f"- 命中信号数 `>= {min_signals}` 判为难例。",
        f"- 信号1: `claim_count >= {thresholds.claim_q75}`",
        f"- 信号2: `text_len >= {thresholds.text_q75}`",
        f"- 信号3: `law_count >= {thresholds.law_q75}`",
        f"- 信号4: `person_count >= {thresholds.person_q75}`",
        f"- 信号5: `cause_count >= {thresholds.cause_count_min}`",
        "",
        "## 合规说明",
        "",
        "- 仅使用输入侧可得特征，不使用任何测试标签。",
        "- 分层策略在 Dev 上确定，Test 仅按同规则打标。",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def _load_id_set(path: Path, key: str) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("ids")

    if not isinstance(ids, list):
        raise ValueError(f"Invalid `{key}` file, `ids` must be a list: {path}")

    id_set = {str(item).strip() for item in ids}

    if "" in id_set:
        raise ValueError(f"Invalid `{key}` file, empty id found: {path}")

    return id_set


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for hard-case label generation.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        default="data/sampling/cleaned_samples.jsonl",
        help="Input JSONL file path.",
    )

    parser.add_argument(
        "--output-dir",
        default="benchmarks/experiments/artifacts/splits",
        help="Output directory for split artifacts.",
    )

    parser.add_argument(
        "--curve-path",
        default="benchmarks/experiments/artifacts/splits/resample_stability_curve.csv",
        help="Path to resample_stability_curve.csv.",
    )

    parser.add_argument(
        "--dev-ids",
        default="benchmarks/experiments/artifacts/splits/dev_ids.json",
        help="Path to dev_ids.json.",
    )

    parser.add_argument(
        "--test-ids",
        default="benchmarks/experiments/artifacts/splits/test_ids.json",
        help="Path to test_ids.json.",
    )

    parser.add_argument(
        "--split-manifest",
        default="benchmarks/experiments/artifacts/splits/split_manifest.json",
        help="Path to split_manifest.json generated by split_cases.py.",
    )

    parser.add_argument(
        "--rule-version",
        default="step05_v2",
        help="Rule version tag written into output artifacts.",
    )

    parser.add_argument(
        "--min-signals",
        type=int,
        default=2,
        help="Minimum positive signals to label a case as hard.",
    )

    parser.add_argument(
        "--cause-count-min",
        type=int,
        default=2,
        help="Minimum cause_count signal used by hard-case rule.",
    )

    parser.add_argument(
        "--text-perturb-ratio",
        type=float,
        default=0.05,
        help="Perturbation ratio for text length threshold stress tests.",
    )

    return parser.parse_args()


def main() -> None:
    """Build hard-case labels and accompanying audit artifacts for Step 05."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = Path(args.curve_path)
    dev_ids_path = Path(args.dev_ids)
    test_ids_path = Path(args.test_ids)
    split_manifest_path = Path(args.split_manifest)

    if args.min_signals <= 0:
        raise ValueError("min_signals must be positive")

    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    protocol_version = str(split_manifest.get("protocol_version", "")).strip()

    if protocol_version not in {
        "step05_v2_no_leakage",
        "step05_v3_strict_joint_resample",
    }:
        raise ValueError(
            "split_manifest protocol_version must be one of "
            "`step05_v2_no_leakage` or `step05_v3_strict_joint_resample`, "
            f"got {split_manifest.get('protocol_version')}"
        )

    thresholds, n_resamples = _load_thresholds_from_curve(
        curve_path, args.cause_count_min
    )

    raw_cases = load_cases_from_jsonl(args.input)
    features = [get_case_features(case) for case in raw_cases]
    features_by_uid = {row["uid"]: row for row in features}

    if len(features_by_uid) != len(features):
        raise ValueError("Duplicate uid detected in input dataset")

    dev_ids = _load_id_set(dev_ids_path, "dev_ids")
    test_ids = _load_id_set(test_ids_path, "test_ids")

    if dev_ids & test_ids:
        raise ValueError("Dev/Test overlap detected")

    all_ids = set(features_by_uid.keys())

    if dev_ids | test_ids != all_ids:
        raise ValueError("Dev/Test ids do not cover full dataset")

    labels = [
        _label_one(row, thresholds, min_signals=args.min_signals) for row in features
    ]

    hard_case_count = sum(int(row["hard_case"]) for row in labels)
    hard_case_ratio = hard_case_count / len(labels)
    hard_case_by_uid = {row["uid"]: int(row["hard_case"]) for row in labels}

    labels_payload = {
        "rule_version": args.rule_version,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol_version": split_manifest.get("protocol_version"),
        "input_path": str(Path(args.input).resolve()),
        "curve_path": str(curve_path.resolve()),
        "dev_ids_path": str(dev_ids_path.resolve()),
        "test_ids_path": str(test_ids_path.resolve()),
        "split_manifest_path": str(split_manifest_path.resolve()),
        "n_resamples_used": n_resamples,
        "total_cases": len(labels),
        "min_signals": int(args.min_signals),
        "thresholds": {
            "claim_q75": thresholds.claim_q75,
            "text_q75": thresholds.text_q75,
            "law_q75": thresholds.law_q75,
            "person_q75": thresholds.person_q75,
            "cause_count_min": thresholds.cause_count_min,
        },
        "threshold_source": "dev_resample_median",
        "hard_case_count": hard_case_count,
        "hard_case_ratio": hard_case_ratio,
        "labels": labels,
    }

    labels_path = output_dir / "hard_case_labels.json"

    labels_path.write_text(
        json.dumps(labels_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rulebook_path = output_dir / "hard_case_rulebook.md"

    _write_rulebook(
        path=rulebook_path,
        rule_version=args.rule_version,
        thresholds=thresholds,
        min_signals=args.min_signals,
        total_cases=len(labels),
        hard_cases=hard_case_count,
        curve_path=curve_path,
        n_resamples=n_resamples,
        split_manifest=split_manifest,
    )

    perturb_rows = _build_perturbation_rows(
        features=features,
        base_thresholds=thresholds,
        min_signals=args.min_signals,
        text_perturb_ratio=args.text_perturb_ratio,
    )

    perturb_path = output_dir / "hard_case_perturbation.csv"

    with perturb_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(perturb_rows[0].keys()))
        writer.writeheader()
        writer.writerows(perturb_rows)

    text_q = split_manifest.get("text_quantiles") or {}

    try:
        text_q25 = int(text_q["q25"])
        text_q50 = int(text_q["q50"])
        text_q75 = int(text_q["q75"])

    except Exception as exc:  # noqa: BLE001
        raise ValueError("split_manifest missing text_quantiles q25/q50/q75") from exc

    cause_bucket_mode = str(split_manifest.get("cause_bucket_mode", "")).strip()

    if cause_bucket_mode not in {"hash16", "hash_adaptive", "topk_other"}:
        raise ValueError(
            f"Unsupported cause_bucket_mode in split_manifest: {cause_bucket_mode}"
        )

    cause_hash_mod = split_manifest.get("cause_hash_mod_selected")

    if cause_hash_mod is None:
        cause_hash_mod = split_manifest.get("cause_hash_mod")

    if cause_bucket_mode in {"hash16", "hash_adaptive"}:
        if cause_hash_mod is None:
            raise ValueError(
                "split_manifest missing cause_hash_mod(_selected) for hash mode"
            )

        cause_hash_mod = int(cause_hash_mod)

        if cause_hash_mod <= 0:
            raise ValueError(f"Invalid cause_hash_mod: {cause_hash_mod}")

    top_cause_values = split_manifest.get("top_cause_values") or []
    top_cause_set = {str(item) for item in top_cause_values}
    records: list[dict[str, Any]] = []

    for feature in features:
        uid = feature["uid"]
        claim_bin = "multi" if int(feature["claim_count"]) >= 2 else "single"

        if cause_bucket_mode in {"hash16", "hash_adaptive"}:
            cause_bucket = _stable_hash_bucket(
                str(feature["sample_cause"]), cause_hash_mod
            )

        else:
            cause_text = str(feature["sample_cause"])
            cause_bucket = cause_text if cause_text in top_cause_set else "OTHER"

        text_bucket = _bucket_text_len(
            int(feature["text_len"]), text_q25, text_q50, text_q75
        )

        hard_case = hard_case_by_uid[uid]

        joint_stratum = "|".join(
            [cause_bucket, claim_bin, text_bucket, f"hard_{hard_case}"]
        )

        records.append(
            {
                "uid": uid,
                "cause_bucket": cause_bucket,
                "claim_bin": claim_bin,
                "text_bucket": text_bucket,
                "hard_case": hard_case,
                "joint_stratum": joint_stratum,
            }
        )

    record_by_uid = {row["uid"]: row for row in records}
    dev_records = [record_by_uid[uid] for uid in sorted(dev_ids)]
    test_records = [record_by_uid[uid] for uid in sorted(test_ids)]
    dist_rows: list[dict[str, Any]] = []
    dist_rows.extend(_collect_distribution_rows(records, dataset_name="full"))
    dist_rows.extend(_collect_distribution_rows(dev_records, dataset_name="dev"))
    dist_rows.extend(_collect_distribution_rows(test_records, dataset_name="test"))
    dist_path = output_dir / "strata_distribution_summary.csv"

    with dist_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dist_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dist_rows)

    print(f"saved: {labels_path}")
    print(f"saved: {rulebook_path}")
    print(f"saved: {perturb_path}")
    print(f"saved: {dist_path}")

    print(
        "summary:",
        f"hard_case_count={hard_case_count}",
        f"hard_case_ratio={hard_case_ratio:.4f}",
        f"threshold_source=dev_resample_median(n={n_resamples})",
    )


if __name__ == "__main__":
    main()
