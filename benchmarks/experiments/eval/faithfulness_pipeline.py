"""Offline faithfulness helpers for Step 11.

This module keeps Step 11 experimentation detached from runtime execution.
It provides deterministic interfaces and fake-friendly components that can be
wired into a future Claim 2 runner after Step 10 completes.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Protocol

from benchmarks.experiments.eval.consistency_parser import ConsistencyParseResult

VALID_EVAL_LABELS = {"E", "N", "C"}


@dataclass(frozen=True)
class EvidenceWindow:
    window_id: str
    section: str
    text: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimAssertion:
    assertion_id: str
    case_uid: str
    claim_id: str
    claim_text: str
    status_eval: str
    assertion_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentHit:
    window: EvidenceWindow
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"window": self.window.to_dict(), "score": self.score}


@dataclass(frozen=True)
class AlignmentResult:
    assertion: ClaimAssertion
    aligned_hits: tuple[AlignmentHit, ...]
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "assertion": self.assertion.to_dict(),
            "aligned_hits": [item.to_dict() for item in self.aligned_hits],
            "success": self.success,
        }


@dataclass(frozen=True)
class AgreementMatrix:
    pairwise: dict[str, dict[str, float]]
    overall_majority_agreement: float
    uncertain_rate: float
    item_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgreementGateResult:
    passed: bool
    threshold: float
    overall_majority_agreement: float
    uncertain_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssertionFaithfulness:
    assertion_id: str
    case_uid: str
    claim_id: str
    aligned: bool
    final_label: str
    window_majority_labels: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FaithfulnessSummary:
    alignment_success_rate: float
    entailment_rate_on_aligned: float
    overall_faithfulness: float
    uncertain_rate: float
    aggregation_mode: str
    agreement_matrix: AgreementMatrix
    per_case_results: dict[str, dict[str, float]]
    per_assertion_results: tuple[AssertionFaithfulness, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alignment_success_rate": self.alignment_success_rate,
            "entailment_rate_on_aligned": self.entailment_rate_on_aligned,
            "overall_faithfulness": self.overall_faithfulness,
            "uncertain_rate": self.uncertain_rate,
            "aggregation_mode": self.aggregation_mode,
            "agreement_matrix": self.agreement_matrix.to_dict(),
            "per_case_results": self.per_case_results,
            "per_assertion_results": [
                item.to_dict() for item in self.per_assertion_results
            ],
        }


class EvidenceAligner(Protocol):
    threshold: float

    def score(self, assertion_text: str, window: EvidenceWindow) -> float: ...


class NliLabeler(Protocol):
    name: str

    def label(self, assertion: ClaimAssertion, window: EvidenceWindow) -> str: ...


def _normalize_tokens(text: str) -> set[str]:
    raw = str(text or "")

    tokens = {
        token.strip().lower()
        for token in re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_.]+", raw)
        if token.strip()
    }

    return tokens


class DeterministicOverlapAligner:
    """Simple deterministic aligner for tests and offline scaffolding."""

    def __init__(self, *, threshold: float = 0.1) -> None:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be within [0, 1]")

        self.threshold = float(threshold)

    def score(self, assertion_text: str, window: EvidenceWindow) -> float:
        left = _normalize_tokens(assertion_text)
        right = _normalize_tokens(window.text)

        if not left or not right:
            return 0.0

        overlap = len(left & right)
        union = len(left | right)
        return float(overlap / union) if union else 0.0


def extract_evidence_windows(case_payload: Mapping[str, Any]) -> list[EvidenceWindow]:
    """Extract deterministic evidence windows from case content sections."""
    content = case_payload.get("content") or {}

    if not isinstance(content, Mapping):
        raise ValueError("case_payload['content'] must be a mapping")

    windows: list[EvidenceWindow] = []

    for section, raw_text in content.items():
        text = str(raw_text or "").strip()

        if not text:
            continue

        windows.append(
            EvidenceWindow(
                window_id=f"{section}:0",
                section=str(section),
                text=text,
                start=0,
                end=len(text),
            )
        )

    return windows


def build_claim_assertions(
    parsed_result: ConsistencyParseResult,
) -> list[ClaimAssertion]:
    """Build one status-conditioned assertion per parsed claim/status pair."""
    claims_by_id = {claim.claim_id: claim for claim in parsed_result.parsed_claims}
    assertions: list[ClaimAssertion] = []

    for claim_id, statuses in sorted(parsed_result.claim_status_sets.items()):
        claim = claims_by_id.get(claim_id)

        if claim is None:
            continue

        for status_eval in statuses:
            assertions.append(
                ClaimAssertion(
                    assertion_id=f"{claim_id}::{status_eval}",
                    case_uid=parsed_result.case_uid,
                    claim_id=claim_id,
                    claim_text=claim.claim_text_raw,
                    status_eval=status_eval,
                    assertion_text=_build_assertion_text(
                        claim.claim_text_raw, status_eval
                    ),
                )
            )

    return assertions


def _build_assertion_text(claim_text: str, status_eval: str) -> str:
    if status_eval == "VALIDATED":
        return f"法院支持了该诉请：{claim_text}"

    if status_eval == "DEFEATED":
        return f"法院未支持该诉请：{claim_text}"

    if status_eval == "HYPOTHETICAL":
        return f"法院未明确处理或仅部分处理该诉请：{claim_text}"

    raise ValueError(f"Unsupported status_eval: {status_eval!r}")


def align_assertions(
    assertions: Sequence[ClaimAssertion],
    windows: Sequence[EvidenceWindow],
    aligner: EvidenceAligner,
    *,
    top_k: int = 3,
) -> list[AlignmentResult]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    results: list[AlignmentResult] = []

    for assertion in assertions:
        scored = [
            AlignmentHit(
                window=window,
                score=float(aligner.score(assertion.assertion_text, window)),
            )
            for window in windows
        ]

        scored.sort(key=lambda item: (-item.score, item.window.window_id))

        chosen = tuple(
            item for item in scored[:top_k] if item.score >= aligner.threshold
        )

        results.append(
            AlignmentResult(
                assertion=assertion,
                aligned_hits=chosen,
                success=bool(chosen),
            )
        )

    return results


def _majority_vote(labels: Sequence[str]) -> str:
    if not labels:
        return "U"

    counts = Counter(labels)
    most_common = counts.most_common()
    best_count = most_common[0][1]
    winners = sorted(label for label, count in most_common if count == best_count)

    if len(winners) != 1:
        return "U"

    return winners[0]


def compute_agreement_matrix(votes: Mapping[str, Sequence[str]]) -> AgreementMatrix:
    if not votes:
        raise ValueError("votes must not be empty")

    normalized = {str(name): list(labels) for name, labels in votes.items()}
    lengths = {len(labels) for labels in normalized.values()}

    if len(lengths) != 1:
        raise ValueError("All evaluator vote sequences must have the same length")

    item_count = lengths.pop()

    if item_count == 0:
        raise ValueError("votes must contain at least one item")

    evaluator_names = sorted(normalized)
    pairwise = {name: {} for name in evaluator_names}

    for left, right in combinations(evaluator_names, 2):
        matches = sum(
            1
            for left_label, right_label in zip(normalized[left], normalized[right])
            if left_label == right_label
        )

        rate = float(matches / item_count)
        pairwise[left][right] = rate
        pairwise[right][left] = rate

    for name in evaluator_names:
        pairwise[name][name] = 1.0

    majority_scores: list[float] = []
    uncertain_items = 0

    for index in range(item_count):
        labels = [normalized[name][index] for name in evaluator_names]
        counts = Counter(labels)
        best = counts.most_common()
        max_count = best[0][1]
        winners = [label for label, count in best if count == max_count]
        majority_scores.append(float(max_count / len(evaluator_names)))

        if len(winners) != 1:
            uncertain_items += 1

    return AgreementMatrix(
        pairwise={
            name: dict(sorted(row.items())) for name, row in sorted(pairwise.items())
        },
        overall_majority_agreement=float(sum(majority_scores) / len(majority_scores)),
        uncertain_rate=float(uncertain_items / item_count),
        item_count=item_count,
    )


def validate_agreement_gate(
    summary: AgreementMatrix | FaithfulnessSummary,
    *,
    threshold: float = 0.70,
) -> AgreementGateResult:
    matrix = (
        summary.agreement_matrix
        if isinstance(summary, FaithfulnessSummary)
        else summary
    )

    return AgreementGateResult(
        passed=bool(matrix.overall_majority_agreement >= threshold),
        threshold=float(threshold),
        overall_majority_agreement=matrix.overall_majority_agreement,
        uncertain_rate=matrix.uncertain_rate,
    )


def score_faithfulness(
    alignment_results: Sequence[AlignmentResult],
    labelers: Sequence[NliLabeler],
    *,
    aggregation: str = "any",
) -> FaithfulnessSummary:
    if aggregation.lower() != "any":
        raise ValueError("Only aggregation='any' is supported in the offline scaffold")

    if not labelers:
        raise ValueError("labelers must not be empty")

    labeler_names = [str(labeler.name) for labeler in labelers]

    if len(set(labeler_names)) != len(labeler_names):
        raise ValueError("labelers must have unique names")

    vote_sequences: dict[str, list[str]] = {name: [] for name in labeler_names}
    per_assertion_results: list[AssertionFaithfulness] = []
    per_case_counts: dict[str, Counter[str]] = {}
    aligned_count = 0
    entailed_count = 0
    uncertain_count = 0

    for result in alignment_results:
        case_counter = per_case_counts.setdefault(result.assertion.case_uid, Counter())
        case_counter["assertion_count"] += 1

        if not result.success:
            per_assertion_results.append(
                AssertionFaithfulness(
                    assertion_id=result.assertion.assertion_id,
                    case_uid=result.assertion.case_uid,
                    claim_id=result.assertion.claim_id,
                    aligned=False,
                    final_label="U",
                    window_majority_labels={},
                )
            )

            continue

        aligned_count += 1
        case_counter["aligned_count"] += 1
        window_majority_labels: dict[str, str] = {}

        for hit in result.aligned_hits:
            labels_for_window: list[str] = []

            for labeler in labelers:
                label = str(labeler.label(result.assertion, hit.window)).strip()

                if label not in VALID_EVAL_LABELS:
                    raise ValueError(
                        f"Unsupported evaluator label {label!r} from {labeler.name}"
                    )

                labels_for_window.append(label)
                vote_sequences[str(labeler.name)].append(label)

            window_majority_labels[hit.window.window_id] = _majority_vote(
                labels_for_window
            )

        majority_values = list(window_majority_labels.values())

        if "E" in majority_values:
            final_label = "E"
            entailed_count += 1
            case_counter["entailed_count"] += 1

        elif majority_values and all(label == "C" for label in majority_values):
            final_label = "C"
            case_counter["contradicted_count"] += 1

        else:
            final_label = "U"
            uncertain_count += 1
            case_counter["uncertain_count"] += 1

        per_assertion_results.append(
            AssertionFaithfulness(
                assertion_id=result.assertion.assertion_id,
                case_uid=result.assertion.case_uid,
                claim_id=result.assertion.claim_id,
                aligned=True,
                final_label=final_label,
                window_majority_labels=dict(sorted(window_majority_labels.items())),
            )
        )

    if any(vote_sequences.values()):
        agreement_matrix = compute_agreement_matrix(vote_sequences)

    else:
        agreement_matrix = AgreementMatrix(
            pairwise={
                name: {other: 0.0 for other in labeler_names} for name in labeler_names
            },
            overall_majority_agreement=0.0,
            uncertain_rate=1.0,
            item_count=0,
        )

    total_assertions = len(alignment_results)

    alignment_success_rate = (
        float(aligned_count / total_assertions) if total_assertions else 0.0
    )

    entailment_rate_on_aligned = (
        float(entailed_count / aligned_count) if aligned_count else 0.0
    )

    aligned_uncertain_rate = (
        float(uncertain_count / aligned_count) if aligned_count else 0.0
    )

    per_case_results: dict[str, dict[str, float]] = {}

    for case_uid, counter in sorted(per_case_counts.items()):
        aligned = float(counter.get("aligned_count", 0))
        assertions = float(counter.get("assertion_count", 0))
        entailed = float(counter.get("entailed_count", 0))
        case_alignment_rate = aligned / assertions if assertions else 0.0
        case_entail_rate = entailed / aligned if aligned else 0.0
        per_case_results[case_uid] = {
            "assertion_count": assertions,
            "aligned_count": aligned,
            "entailed_count": entailed,
            "uncertain_count": float(counter.get("uncertain_count", 0)),
            "contradicted_count": float(counter.get("contradicted_count", 0)),
            "alignment_success_rate": case_alignment_rate,
            "entailment_rate_on_aligned": case_entail_rate,
            "overall_faithfulness": case_alignment_rate * case_entail_rate,
        }

    return FaithfulnessSummary(
        alignment_success_rate=alignment_success_rate,
        entailment_rate_on_aligned=entailment_rate_on_aligned,
        overall_faithfulness=alignment_success_rate * entailment_rate_on_aligned,
        uncertain_rate=aligned_uncertain_rate,
        aggregation_mode="any",
        agreement_matrix=agreement_matrix,
        per_case_results=per_case_results,
        per_assertion_results=tuple(per_assertion_results),
    )
