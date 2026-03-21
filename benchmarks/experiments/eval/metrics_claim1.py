"""Step 08 Claim 1 metric helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from benchmarks.experiments.eval._metric_utils import (
    STATUS_CLASSES_2,
    STATUS_CLASSES_3,
    balanced_accuracy_for_case,
    collapse_status_eval,
    confusion_matrix,
    f1_from_counts,
    macro_f1_for_case,
    normalize_status_eval,
    require_row_field,
    safe_div,
)
from benchmarks.experiments.eval.matching import CaseMatchBundle
from benchmarks.experiments.stats.bootstrap import (
    BootstrapConfig,
    bootstrap_case_mean,
)


def _gold_status_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}

    for row in rows:
        uid = str(row.get("uid", "") or "")
        claim_id = str(row.get("claim_id", "") or "")

        if not uid or not claim_id:
            raise ValueError("gold_status_rows must contain non-empty uid and claim_id")

        key = (uid, claim_id)

        if key in lookup:
            raise ValueError(
                f"Duplicate gold status row for uid={uid} claim_id={claim_id}"
            )

        lookup[key] = dict(row)

    return lookup


def _serialize_summary(
    case_scores: Mapping[str, float], config: BootstrapConfig
) -> dict[str, Any]:
    return asdict(bootstrap_case_mean(case_scores, config=config))


def evaluate_claim1_metrics(
    case_bundles: Sequence[CaseMatchBundle],
    gold_status_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_config: BootstrapConfig | None = None,
    family_name: str = "claim1",
) -> dict[str, Any]:
    config = bootstrap_config or BootstrapConfig()
    gold_lookup = _gold_status_lookup(gold_status_rows)
    method_names = sorted({bundle.method_name for bundle in case_bundles})

    if len(method_names) > 1:
        raise ValueError(
            f"evaluate_claim1_metrics expects one method at a time, got {method_names}"
        )

    scenario_names = sorted({bundle.scenario_name for bundle in case_bundles})

    if len(scenario_names) > 1:
        raise ValueError(
            f"evaluate_claim1_metrics expects one scenario at a time, got {scenario_names}"
        )

    case_scores: dict[str, dict[str, float]] = {
        "step_a_precision": {},
        "step_a_recall": {},
        "step_a_f1": {},
        "e2e_status_acc": {},
        "status_acc_matched": {},
        "e2e_f1_fp_sensitive": {},
        "soft_e2e_f1": {},
        "over_generation_rate": {},
        "macro_f1_3class_matched": {},
        "balanced_accuracy_3class_matched": {},
    }

    matched_true_2: list[str] = []
    matched_pred_2: list[str] = []
    matched_true_3: list[str] = []
    matched_pred_3: list[str] = []
    gold_claim_count = 0
    prediction_claim_count = 0
    matched_gold_count = 0
    root_claim_case_count = 0

    for bundle in case_bundles:
        uid = str(bundle.uid)
        gold_count = len(bundle.gold_rows)
        pred_count = len(bundle.prediction_rows_deduped)
        matched_count = len(bundle.matched_pairs)
        unmatched_gold = len(bundle.unmatched_gold_rows)
        unmatched_pred = len(bundle.unmatched_prediction_rows)
        gold_claim_count += gold_count
        prediction_claim_count += pred_count
        matched_gold_count += matched_count
        case_scores["over_generation_rate"][uid] = safe_div(unmatched_pred, pred_count)

        for gold_row in bundle.gold_rows:
            gold_claim_id = require_row_field(gold_row, "claim_id")

            if (uid, gold_claim_id) not in gold_lookup:
                raise ValueError(
                    f"Missing gold status row for uid={uid} claim_id={gold_claim_id}"
                )

        if gold_count:
            root_claim_case_count += 1

            case_scores["step_a_precision"][uid] = safe_div(
                matched_count, matched_count + unmatched_pred
            )

            case_scores["step_a_recall"][uid] = safe_div(
                matched_count, matched_count + unmatched_gold
            )

            case_scores["step_a_f1"][uid] = f1_from_counts(
                matched_count, unmatched_pred, unmatched_gold
            )

        correct_status_2 = 0
        mismatch_status_2 = 0
        case_true_3: list[str] = []
        case_pred_3: list[str] = []

        for pair in bundle.matched_pairs:
            gold_claim_id = require_row_field(pair.gold_row, "claim_id")
            gold_row = gold_lookup.get((uid, gold_claim_id))

            if gold_row is None:
                raise ValueError(
                    f"Missing gold status row for uid={uid} claim_id={gold_claim_id}"
                )

            gold_status_3 = normalize_status_eval(
                require_row_field(gold_row, "status_eval")
            )

            pred_status_3 = normalize_status_eval(
                require_row_field(pair.prediction_row, "status_eval")
            )

            gold_status_2 = collapse_status_eval(gold_status_3)
            pred_status_2 = collapse_status_eval(pred_status_3)
            matched_true_2.append(gold_status_2)
            matched_pred_2.append(pred_status_2)
            matched_true_3.append(gold_status_3)
            matched_pred_3.append(pred_status_3)
            case_true_3.append(gold_status_3)
            case_pred_3.append(pred_status_3)

            if gold_status_2 == pred_status_2:
                correct_status_2 += 1

            else:
                mismatch_status_2 += 1

        if gold_count:
            case_scores["e2e_status_acc"][uid] = safe_div(correct_status_2, gold_count)

        if matched_count:
            case_scores["status_acc_matched"][uid] = safe_div(
                correct_status_2, matched_count
            )

            case_scores["macro_f1_3class_matched"][uid] = macro_f1_for_case(
                case_true_3,
                case_pred_3,
                classes=STATUS_CLASSES_3,
            )

            case_scores["balanced_accuracy_3class_matched"][uid] = (
                balanced_accuracy_for_case(
                    case_true_3,
                    case_pred_3,
                    classes=STATUS_CLASSES_3,
                )
            )

        if gold_count:
            case_scores["e2e_f1_fp_sensitive"][uid] = f1_from_counts(
                correct_status_2,
                unmatched_pred + mismatch_status_2,
                unmatched_gold + mismatch_status_2,
            )

            case_scores["soft_e2e_f1"][uid] = f1_from_counts(
                correct_status_2,
                unmatched_pred,
                unmatched_gold + mismatch_status_2,
            )

    main_metrics = {
        "e2e_status_acc": _serialize_summary(case_scores["e2e_status_acc"], config),
        "status_acc_matched": _serialize_summary(
            case_scores["status_acc_matched"], config
        ),
    }

    appendix_metrics = {
        "step_a_precision": _serialize_summary(case_scores["step_a_precision"], config),
        "step_a_recall": _serialize_summary(case_scores["step_a_recall"], config),
        "step_a_f1": _serialize_summary(case_scores["step_a_f1"], config),
        "e2e_f1_fp_sensitive": _serialize_summary(
            case_scores["e2e_f1_fp_sensitive"], config
        ),
        "soft_e2e_f1": _serialize_summary(case_scores["soft_e2e_f1"], config),
        "over_generation_rate": _serialize_summary(
            case_scores["over_generation_rate"], config
        ),
        "macro_f1_3class_matched": _serialize_summary(
            case_scores["macro_f1_3class_matched"], config
        ),
        "balanced_accuracy_3class_matched": _serialize_summary(
            case_scores["balanced_accuracy_3class_matched"], config
        ),
    }

    diagnostics = {
        "case_count": len(case_bundles),
        "root_claim_case_count": root_claim_case_count,
        "zero_gold_case_count": len(case_bundles) - root_claim_case_count,
        "matched_case_count": len(case_scores["status_acc_matched"]),
        "gold_claim_count": gold_claim_count,
        "prediction_claim_count": prediction_claim_count,
        "matched_gold_count": matched_gold_count,
        "matched_gold_coverage": safe_div(matched_gold_count, gold_claim_count),
        "status_confusion_2class": confusion_matrix(
            matched_true_2, matched_pred_2, classes=STATUS_CLASSES_2
        ),
        "status_confusion_3class": confusion_matrix(
            matched_true_3, matched_pred_3, classes=STATUS_CLASSES_3
        ),
    }

    return {
        "family_name": family_name,
        "method_name": method_names[0] if method_names else "",
        "scenario_name": scenario_names[0] if scenario_names else "",
        "bootstrap": asdict(config),
        "main_metrics": main_metrics,
        "appendix_metrics": appendix_metrics,
        "case_scores": {
            name: dict(sorted(values.items())) for name, values in case_scores.items()
        },
        "diagnostics": diagnostics,
        "hypothesis_tests": {
            "family_name": family_name,
            "available": False,
            "reason": "single_method_evaluation_only",
        },
    }
