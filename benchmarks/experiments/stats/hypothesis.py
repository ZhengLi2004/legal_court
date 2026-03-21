"""Hypothesis testing helpers for benchmark experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import binomtest, chi2


@dataclass(frozen=True)
class HolmBonferroniItem:
    label: str
    p_value: float
    threshold: float
    rejected: bool


@dataclass(frozen=True)
class HolmBonferroniResult:
    alpha: float
    items: tuple[HolmBonferroniItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "items": [asdict(item) for item in self.items],
            "any_rejected": any(item.rejected for item in self.items),
        }


@dataclass(frozen=True)
class McNemarResult:
    statistic: float | None
    p_value: float
    n: int
    valid: bool
    reason_if_invalid: str
    b: int
    c: int
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StuartMaxwellResult:
    statistic: float | None
    p_value: float
    n: int
    valid: bool
    reason_if_invalid: str
    degrees_of_freedom: int
    classes: tuple[str, ...]
    contingency: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def holm_bonferroni(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> HolmBonferroniResult:
    ordered = sorted(
        ((str(label), float(p_value)) for label, p_value in p_values.items()),
        key=lambda item: (item[1], item[0]),
    )

    items: list[HolmBonferroniItem] = []
    still_rejecting = True
    total = len(ordered)

    for idx, (label, p_value) in enumerate(ordered, start=1):
        threshold = alpha / (total - idx + 1) if total else alpha
        rejected = still_rejecting and p_value <= threshold

        if not rejected:
            still_rejecting = False

        items.append(
            HolmBonferroniItem(
                label=label, p_value=p_value, threshold=threshold, rejected=rejected
            )
        )

    return HolmBonferroniResult(alpha=alpha, items=tuple(items))


def mcnemar_test(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    *,
    exact: bool | None = None,
) -> McNemarResult:
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("y_true, pred_a, and pred_b must have the same length")

    b = 0
    c = 0

    for gold, a, b_pred in zip(y_true, pred_a, pred_b):
        a_correct = a == gold
        b_correct = b_pred == gold

        if a_correct and not b_correct:
            b += 1

        elif b_correct and not a_correct:
            c += 1

    discordant = b + c

    if discordant == 0:
        return McNemarResult(
            statistic=0.0,
            p_value=1.0,
            n=len(y_true),
            valid=False,
            reason_if_invalid="no_discordant_pairs",
            b=b,
            c=c,
            exact=bool(exact if exact is not None else True),
        )

    use_exact = discordant < 25 if exact is None else bool(exact)

    if use_exact:
        p_value = float(
            binomtest(min(b, c), n=discordant, p=0.5, alternative="two-sided").pvalue
        )

        statistic: float | None = None

    else:
        statistic = float(((abs(b - c) - 1.0) ** 2) / discordant)
        p_value = float(1.0 - chi2.cdf(statistic, df=1))

    return McNemarResult(
        statistic=statistic,
        p_value=p_value,
        n=len(y_true),
        valid=True,
        reason_if_invalid="",
        b=b,
        c=c,
        exact=use_exact,
    )


def stuart_maxwell_test(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    *,
    classes: Sequence[str],
) -> StuartMaxwellResult:
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have the same length")

    class_list = tuple(str(label) for label in classes)
    class_to_idx = {label: idx for idx, label in enumerate(class_list)}
    size = len(class_list)

    if size < 2:
        raise ValueError("stuart_maxwell_test requires at least two classes")

    table = np.zeros((size, size), dtype=np.int64)

    for left, right in zip(labels_a, labels_b):
        if left not in class_to_idx or right not in class_to_idx:
            raise ValueError("labels must be contained in `classes`")

        table[class_to_idx[left], class_to_idx[right]] += 1

    row_totals = table.sum(axis=1)
    col_totals = table.sum(axis=0)
    active = [idx for idx in range(size) if row_totals[idx] > 0 or col_totals[idx] > 0]

    if len(active) < 2:
        return StuartMaxwellResult(
            statistic=0.0,
            p_value=1.0,
            n=len(labels_a),
            valid=False,
            reason_if_invalid="fewer_than_two_active_classes",
            degrees_of_freedom=0,
            classes=class_list,
            contingency=tuple(
                tuple(int(value) for value in row) for row in table.tolist()
            ),
        )

    reduced = table[np.ix_(active, active)].astype(np.float64)
    row_totals = reduced.sum(axis=1)
    col_totals = reduced.sum(axis=0)
    k = reduced.shape[0]
    d = row_totals[:-1] - col_totals[:-1]
    s = np.zeros((k - 1, k - 1), dtype=np.float64)

    for i in range(k - 1):
        for j in range(k - 1):
            if i == j:
                s[i, i] = row_totals[i] + col_totals[i] - 2.0 * reduced[i, i]

            else:
                s[i, j] = -(reduced[i, j] + reduced[j, i])

    rank = int(np.linalg.matrix_rank(s))

    if rank == 0:
        return StuartMaxwellResult(
            statistic=0.0,
            p_value=1.0,
            n=len(labels_a),
            valid=False,
            reason_if_invalid="singular_covariance",
            degrees_of_freedom=0,
            classes=class_list,
            contingency=tuple(
                tuple(int(value) for value in row) for row in table.tolist()
            ),
        )

    statistic = float(d.T @ np.linalg.pinv(s) @ d)
    p_value = float(1.0 - chi2.cdf(statistic, df=rank))

    return StuartMaxwellResult(
        statistic=statistic,
        p_value=p_value,
        n=len(labels_a),
        valid=True,
        reason_if_invalid="",
        degrees_of_freedom=rank,
        classes=class_list,
        contingency=tuple(tuple(int(value) for value in row) for row in table.tolist()),
    )
