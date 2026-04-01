"""Shared helpers for Step 08 metric modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

STATUS_CLASSES_3 = ("VALIDATED", "DEFEATED", "HYPOTHETICAL")
STATUS_CLASSES_2 = ("VALIDATED", "DEFEATED")


def safe_div(numerator: float, denominator: float) -> float:
    """Return a safe division result, defaulting to ``0.0`` for zero denominators.

    Args:
        numerator: Dividend.
        denominator: Divisor that may be zero.

    Returns:
        ``numerator / denominator`` when possible, otherwise ``0.0``.
    """

    return float(numerator / denominator) if denominator else 0.0


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """Compute F1 directly from true-positive, false-positive, and false-negative counts.

    Args:
        tp: True-positive count.
        fp: False-positive count.
        fn: False-negative count.

    Returns:
        F1 score derived from precision and recall.
    """

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return safe_div(2.0 * precision * recall, precision + recall)


def normalize_status_eval(label: str, *, field_name: str = "status_eval") -> str:
    """Validate and normalize one three-class status label.

    Args:
        label: Raw status label.
        field_name: Field name used in validation errors.

    Returns:
        Normalized three-class status label.

    Raises:
        ValueError: If the label is outside the supported status universe.
    """

    value = str(label or "")

    if value not in STATUS_CLASSES_3:
        raise ValueError(f"Unsupported `{field_name}` label: {value!r}")

    return value


def collapse_status_eval(label: str) -> str:
    """Map ``HYPOTHETICAL`` into ``DEFEATED`` for the official 2-class view.

    Args:
        label: Raw or normalized three-class status label.

    Returns:
        Two-class status label used by the official Claim 1 metric view.
    """

    normalized = normalize_status_eval(label)
    return "DEFEATED" if normalized == "HYPOTHETICAL" else normalized


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    classes: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Build a dense confusion matrix for an explicit class ordering.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels aligned with ``y_true``.
        classes: Explicit class ordering to materialize.

    Returns:
        Nested true-label -> predicted-label count matrix.
    """

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
    """Compute macro F1 over the labels active in one case.

    Args:
        y_true: Gold labels for one case.
        y_pred: Predicted labels aligned with ``y_true``.
        classes: Full class universe used to select active labels.

    Returns:
        Case-level macro F1 over active labels.
    """

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
    """Compute balanced accuracy over the gold-active labels in one case.

    Args:
        y_true: Gold labels for one case.
        y_pred: Predicted labels aligned with ``y_true``.
        classes: Full class universe used to select gold-active labels.

    Returns:
        Case-level balanced accuracy over gold-active labels.
    """

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
    """Normalize one case-score mapping and reject empty or non-finite values.

    Args:
        case_scores: Case-score mapping to validate.
        name: Human-readable field name used in validation errors.

    Returns:
        Sorted normalized case-score mapping.

    Raises:
        ValueError: If the mapping is empty or contains non-finite scores.
    """

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
    """Normalize a nested ``series -> point -> case`` score mapping.

    Args:
        nested_scores: Nested score mapping keyed by series and point.
        name: Human-readable field name used in validation errors.

    Returns:
        Fully normalized nested case-score mapping.

    Raises:
        ValueError: If any nested mapping is empty or contains invalid scores.
    """

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
    """Read one required row field or raise with case-specific context.

    Args:
        row: Data row expected to contain the field.
        field_name: Required field name.

    Returns:
        Field value coerced to string.

    Raises:
        ValueError: If the field is missing or empty.
    """

    value = row.get(field_name, "")

    if value in (None, ""):
        raise ValueError(
            f"Missing non-empty `{field_name}` for uid={row.get('uid', '')} claim_id={row.get('claim_id', '')}"
        )

    return str(value)
