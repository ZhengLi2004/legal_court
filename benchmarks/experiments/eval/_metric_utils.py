"""Shared helpers for Step 08 metric modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

STATUS_CLASSES_3 = ("VALIDATED", "DEFEATED", "HYPOTHETICAL")
STATUS_CLASSES_2 = ("VALIDATED", "DEFEATED")


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return safe_div(2.0 * precision * recall, precision + recall)


def normalize_status_eval(label: str, *, field_name: str = "status_eval") -> str:
    value = str(label or "")

    if value not in STATUS_CLASSES_3:
        raise ValueError(f"Unsupported `{field_name}` label: {value!r}")

    return value


def collapse_status_eval(label: str) -> str:
    normalized = normalize_status_eval(label)
    return "DEFEATED" if normalized == "HYPOTHETICAL" else normalized


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    classes: Sequence[str],
) -> dict[str, dict[str, int]]:
    matrix = {
        str(true_label): {str(pred_label): 0 for pred_label in classes}
        for true_label in classes
    }

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[str(true_label)][str(pred_label)] += 1

    return matrix


def macro_f1_for_case(
    y_true: Sequence[str], y_pred: Sequence[str], *, classes: Sequence[str]
) -> float:
    active = [label for label in classes if label in y_true or label in y_pred]

    if not active:
        return 0.0

    scores = []

    for label in active:
        tp = sum(
            1 for gold, pred in zip(y_true, y_pred) if gold == label and pred == label
        )

        fp = sum(
            1 for gold, pred in zip(y_true, y_pred) if gold != label and pred == label
        )

        fn = sum(
            1 for gold, pred in zip(y_true, y_pred) if gold == label and pred != label
        )

        scores.append(f1_from_counts(tp, fp, fn))

    return float(sum(scores) / len(scores))


def balanced_accuracy_for_case(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    classes: Sequence[str],
) -> float:
    active = [label for label in classes if label in y_true]

    if not active:
        return 0.0

    recalls = []

    for label in active:
        tp = sum(
            1 for gold, pred in zip(y_true, y_pred) if gold == label and pred == label
        )

        fn = sum(
            1 for gold, pred in zip(y_true, y_pred) if gold == label and pred != label
        )

        recalls.append(safe_div(tp, tp + fn))

    return float(sum(recalls) / len(recalls))


def validate_case_score_mapping(
    case_scores: Mapping[str, float],
    *,
    name: str,
) -> dict[str, float]:
    if not case_scores:
        raise ValueError(f"{name} must not be empty")

    normalized: dict[str, float] = {}

    for uid, value in case_scores.items():
        key = str(uid)
        score = float(value)

        if score != score or score in (float("inf"), float("-inf")):
            raise ValueError(f"{name} contains non-finite value for uid={key}")

        normalized[key] = score

    return dict(sorted(normalized.items()))


def validate_nested_case_score_mapping(
    nested_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    name: str,
) -> dict[str, dict[str, dict[str, float]]]:
    if not nested_scores:
        raise ValueError(f"{name} must not be empty")

    normalized: dict[str, dict[str, dict[str, float]]] = {}

    for outer_key, inner in nested_scores.items():
        if not inner:
            raise ValueError(f"{name}[{outer_key!r}] must not be empty")

        normalized_inner: dict[str, dict[str, float]] = {}

        for point_key, case_scores in inner.items():
            normalized_inner[str(point_key)] = validate_case_score_mapping(
                case_scores,
                name=f"{name}[{outer_key!r}][{point_key!r}]",
            )

        normalized[str(outer_key)] = dict(sorted(normalized_inner.items()))

    return dict(sorted(normalized.items()))


def require_row_field(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name, "")

    if value in (None, ""):
        raise ValueError(
            f"Missing non-empty `{field_name}` for uid={row.get('uid', '')} claim_id={row.get('claim_id', '')}"
        )

    return str(value)
