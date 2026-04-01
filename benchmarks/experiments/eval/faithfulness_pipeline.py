"""Offline faithfulness helpers for Step 11.

This module keeps Step 11 experimentation detached from runtime execution.
It provides deterministic interfaces and fake-friendly components that can be
wired into a future Claim 2 runner after Step 10 completes.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Protocol

from benchmarks.experiments.eval.consistency_parser import ConsistencyParseResult
from mas.infrastructure.embedding import cosine_similarity

VALID_EVAL_LABELS = {"E", "N", "C"}


@dataclass(frozen=True)
class EvidenceWindow:
    """One evidence span extracted from the source judgment text.

    Attributes:
        window_id: Stable identifier within one case.
        section: Source section name from the case payload.
        text: Extracted evidence text.
        start: Character start offset within the source section.
        end: Character end offset within the source section.
    """

    window_id: str
    section: str
    text: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimAssertion:
    """Status-conditioned natural-language assertion derived from one claim.

    Attributes:
        assertion_id: Stable assertion identifier.
        case_uid: Case identifier owning the claim.
        claim_id: Gold claim identifier.
        claim_text: Original claim text.
        status_eval: Status label that conditions the assertion text.
        assertion_text: Natural-language assertion evaluated for faithfulness.
    """

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
    """One scored evidence-window candidate for an assertion.

    Attributes:
        window: Candidate evidence window.
        score: Alignment score assigned by the aligner.
    """

    window: EvidenceWindow
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"window": self.window.to_dict(), "score": self.score}


@dataclass(frozen=True)
class AlignmentResult:
    """Alignment outcome for one assertion after threshold filtering.

    Attributes:
        assertion: Assertion being aligned.
        aligned_hits: Candidate hits that survived threshold filtering.
        success: Whether at least one aligned hit was retained.
        candidate_hits: Top-k candidate hits before threshold filtering.
    """

    assertion: ClaimAssertion
    aligned_hits: tuple[AlignmentHit, ...]
    success: bool
    candidate_hits: tuple[AlignmentHit, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "assertion": self.assertion.to_dict(),
            "aligned_hits": [item.to_dict() for item in self.aligned_hits],
            "success": self.success,
            "candidate_hits": [item.to_dict() for item in self.candidate_hits],
        }


@dataclass(frozen=True)
class AgreementMatrix:
    """Pairwise and majority-agreement statistics for evaluator votes.

    Attributes:
        pairwise: Pairwise agreement matrix keyed by evaluator name.
        overall_majority_agreement: Mean majority-agreement score per item.
        uncertain_rate: Fraction of items with no unique majority label.
        item_count: Number of labeled items included in the matrix.
    """

    pairwise: dict[str, dict[str, float]]
    overall_majority_agreement: float
    uncertain_rate: float
    item_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgreementGateResult:
    """Pass/fail view of the evaluator agreement gate.

    Attributes:
        passed: Whether the gate threshold was met.
        threshold: Agreement threshold used by the gate.
        overall_majority_agreement: Observed majority-agreement value.
        uncertain_rate: Observed uncertain-item rate.
    """

    passed: bool
    threshold: float
    overall_majority_agreement: float
    uncertain_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssertionFaithfulness:
    """Final faithfulness label and vote trace for one assertion.

    Attributes:
        assertion_id: Assertion identifier.
        case_uid: Owning case id.
        claim_id: Gold claim identifier.
        aligned: Whether the assertion aligned to at least one evidence window.
        final_label: Final aggregated label for the assertion.
        window_majority_labels: Majority label emitted per aligned window.
    """

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
    """Case-level and assertion-level Claim 2 faithfulness aggregates.

    Attributes:
        alignment_success_rate: Fraction of assertions with aligned evidence.
        entailment_rate_on_aligned: Fraction of aligned assertions labeled entailment.
        overall_faithfulness: Product-style overall faithfulness score.
        uncertain_rate: Fraction of aligned assertions with uncertain final labels.
        aggregation_mode: Case-level aggregation mode used for scoring.
        agreement_matrix: Agreement statistics over evaluator votes.
        per_case_results: Per-case aggregate scores.
        per_assertion_results: Per-assertion final labels and trace info.
        vote_sequences: Raw evaluator vote sequences keyed by evaluator name.
    """

    alignment_success_rate: float
    entailment_rate_on_aligned: float
    overall_faithfulness: float
    uncertain_rate: float
    aggregation_mode: str
    agreement_matrix: AgreementMatrix
    per_case_results: dict[str, dict[str, float]]
    per_assertion_results: tuple[AssertionFaithfulness, ...]
    vote_sequences: dict[str, tuple[str, ...]] = field(default_factory=dict)

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
            "vote_sequences": {
                name: list(labels)
                for name, labels in sorted(self.vote_sequences.items())
            },
        }


class EvidenceAligner(Protocol):
    """Protocol for scoring one assertion against one evidence window.

    Attributes:
        threshold: Score threshold used to decide whether a hit is aligned.
    """

    threshold: float

    def score(self, assertion_text: str, window: EvidenceWindow) -> float: ...


class NliLabeler(Protocol):
    """Protocol for labeling one assertion-window pair as E/N/C.

    Attributes:
        name: Stable evaluator name written into vote traces.
    """

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
    """Simple deterministic aligner for tests and offline scaffolding.

    Attributes:
        threshold: Minimum score required for an aligned hit.
    """

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


class EmbeddingCosineAligner:
    """Embedding-based cosine aligner for offline Step 11 evaluation.

    Attributes:
        embedding_func: Embedding provider used for text encoding.
        threshold: Minimum score required for an aligned hit.
        _cache: In-memory text-to-vector cache.
    """

    def __init__(self, embedding_func: Any, *, threshold: float = 0.1) -> None:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be within [0, 1]")

        if embedding_func is None:
            raise ValueError("embedding_func must be provided")

        self.embedding_func = embedding_func
        self.threshold = float(threshold)
        self._cache: dict[str, list[float]] = {}

    def _embed(self, text: str) -> list[float]:
        key = str(text or "")

        if key not in self._cache:
            self._cache[key] = list(self.embedding_func.embed_query(key))

        return self._cache[key]

    def score(self, assertion_text: str, window: EvidenceWindow) -> float:
        left = self._embed(assertion_text)
        right = self._embed(window.text)
        return float(cosine_similarity(left, right))


def extract_evidence_windows(case_payload: Mapping[str, Any]) -> list[EvidenceWindow]:
    """Extract deterministic evidence windows from case content sections.

    Args:
        case_payload: Case payload containing a `content` mapping.

    Returns:
        A deterministic list of section-level evidence windows.

    Raises:
        ValueError: If `case_payload['content']` is not a mapping.
    """
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
    """Build one status-conditioned assertion per parsed claim/status pair.

    Args:
        parsed_result: Parsed Claim 2 consistency result for one method output.

    Returns:
        One assertion per parsed claim and observed status label.
    """
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
    """Score every assertion against candidate evidence windows and keep top hits.

    Args:
        assertions: Assertions to align.
        windows: Evidence windows extracted from the case text.
        aligner: Aligner used to score assertion-window pairs.
        top_k: Maximum number of candidate windows retained per assertion.

    Returns:
        Alignment results for all assertions in input order.

    Raises:
        ValueError: If `top_k` is not positive.
    """

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
        candidate_hits = tuple(scored[:top_k])
        chosen = tuple(
            item for item in candidate_hits if item.score >= aligner.threshold
        )

        results.append(
            AlignmentResult(
                assertion=assertion,
                aligned_hits=chosen,
                success=bool(chosen),
                candidate_hits=candidate_hits,
            )
        )

    return results


def rethreshold_alignment_results(
    alignment_results: Sequence[AlignmentResult],
    *,
    threshold: float,
) -> list[AlignmentResult]:
    """Re-evaluate stored alignment candidates under a new score threshold.

    Args:
        alignment_results: Previously computed alignment results with candidates.
        threshold: New score threshold applied to stored candidate hits.

    Returns:
        Updated alignment results under the new threshold.

    Raises:
        ValueError: If `threshold` falls outside `[0, 1]`.
    """

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be within [0, 1]")

    updated: list[AlignmentResult] = []

    for result in alignment_results:
        candidates = result.candidate_hits or result.aligned_hits
        aligned_hits = tuple(item for item in candidates if item.score >= threshold)

        updated.append(
            AlignmentResult(
                assertion=result.assertion,
                aligned_hits=aligned_hits,
                success=bool(aligned_hits),
                candidate_hits=tuple(candidates),
            )
        )

    return updated


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
    """Compute pairwise agreement and majority consistency over evaluator votes.

    Args:
        votes: Mapping from evaluator name to ordered vote sequences.

    Returns:
        Pairwise agreement rates plus majority-agreement aggregates.

    Raises:
        ValueError: If vote sequences are empty or length-mismatched.
    """

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
    """Apply the majority-agreement threshold to one matrix or summary object.

    Args:
        summary: Agreement matrix or full faithfulness summary.
        threshold: Minimum acceptable majority-agreement score.

    Returns:
        A pass/fail view of the agreement gate.
    """

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
    """Aggregate evaluator votes into Claim 2 faithfulness statistics.

    Args:
        alignment_results: Alignment outcomes for all assertions.
        labelers: Evaluators used to label aligned assertion-window pairs.
        aggregation: Case-level aggregation mode.

    Returns:
        Faithfulness summary with agreement, per-case, and per-assertion outputs.

    Raises:
        ValueError: If aggregation mode is unsupported or labelers are invalid.
    """

    normalized_aggregation = str(aggregation or "").strip().lower()

    if normalized_aggregation not in {"any", "majority", "max_score"}:
        raise ValueError("aggregation must be one of {'any', 'majority', 'max_score'}")

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

        final_label = _aggregate_window_labels(
            majority_values,
            aggregation=normalized_aggregation,
        )

        if final_label == "E":
            entailed_count += 1
            case_counter["entailed_count"] += 1

        elif final_label == "C":
            case_counter["contradicted_count"] += 1

        else:
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
        aggregation_mode=normalized_aggregation,
        agreement_matrix=agreement_matrix,
        per_case_results=per_case_results,
        per_assertion_results=tuple(per_assertion_results),
        vote_sequences={
            name: tuple(labels) for name, labels in sorted(vote_sequences.items())
        },
    )


def _aggregate_window_labels(labels: Sequence[str], *, aggregation: str) -> str:
    values = [str(label or "").strip() for label in labels if str(label or "").strip()]

    if not values:
        return "U"

    if aggregation == "any":
        if "E" in values:
            return "E"

        if all(label == "C" for label in values):
            return "C"

        return "U"

    if aggregation == "majority":
        majority = _majority_vote(values)

        if majority in {"E", "C"}:
            return majority

        return "U"

    if aggregation == "max_score":
        top_label = values[0]
        return top_label if top_label in {"E", "C"} else "U"

    raise ValueError(f"Unsupported aggregation: {aggregation!r}")
