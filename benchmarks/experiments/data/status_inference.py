"""Gold status inference helpers for Step 06A."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from cn2an import cn2an
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

STATUS_CANONICAL = ("ACCEPTED", "REJECTED", "UNMENTIONED")

STATUS_EVAL_MAP = {
    "ACCEPTED": "VALIDATED",
    "REJECTED": "DEFEATED",
    "UNMENTIONED": "HYPOTHETICAL",
}

STATUS_SECTIONS = ("裁判结果", "法院观点")

GENERIC_TERMS = {
    "原告",
    "被告",
    "上诉人",
    "被上诉人",
    "请求",
    "诉求",
    "法院",
    "本院",
    "依法",
    "判令",
    "判决",
    "一审",
    "二审",
    "其他",
    "其余",
    "本案",
}

ACTION_FAMILY_MAP = {
    "支付": "monetary",
    "赔偿": "monetary",
    "返还": "monetary",
    "承担": "monetary",
    "确认": "declaratory",
    "解除": "declaratory",
    "撤销": "declaratory",
    "优先受偿": "declaratory",
    "准予": "declaratory",
    "解散": "declaratory",
    "交付": "conduct",
    "继续履行": "conduct",
    "排除妨害": "conduct",
    "办理": "conduct",
    "恢复原状": "conduct",
    "停止侵权": "conduct",
    "抚养": "conduct",
    "分割": "conduct",
    "迁葬": "conduct",
}

_AMOUNT_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?(?:万|亿)?元|[零一二三四五六七八九十百千万亿两]+(?:\.[零一二三四五六七八九十百千万亿两]+)?(?:万|亿)?元)"
)

_TEXT_SPLIT_RE = re.compile(r"[\n\r]+")
_SENTENCE_END_RE = re.compile(r"[。；;!?！？]")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class StatusRuleBundle:
    """Frozen heuristic and model thresholds used by Step 06A status inference.

    Attributes:
        sections: Judgment sections eligible for evidence retrieval.
        window_sentence_max: Maximum number of sentences per evidence window.
        retrieval_top_k: Hybrid retriever candidate count before reranking.
        rerank_keep_k: Number of candidates kept after cross-encoder reranking.
        bm25_weight: Weight assigned to sparse retrieval scores.
        dense_weight: Weight assigned to dense retrieval scores.
        zeroshot_labels: Canonical-to-natural-language label mapping.
        zeroshot_hypothesis_template: Hypothesis template for zero-shot NLI.
        zeroshot_min_confidence: Minimum top-label confidence for auto-labeling.
        zeroshot_min_margin: Minimum top-two margin for auto-labeling.
        section_priority: Preferred ordering among candidate evidence sections.
        appeal_request_patterns: Regex patterns that indicate appeal requests.
        original_trial_recital_patterns: Patterns for prior-instance recitals.
        appeal_reject_patterns: Patterns indicating appeal rejection.
        reject_other_patterns: Patterns indicating generalized rejection.
        partial_support_patterns: Patterns indicating partial support only.
        direct_reject_patterns: Patterns indicating direct rejection.
        direct_accept_patterns: Patterns indicating direct support.
        review_reason_priority: Priority ordering for manual-review reasons.
    """

    sections: tuple[str, ...]
    window_sentence_max: int
    retrieval_top_k: int
    rerank_keep_k: int
    bm25_weight: float
    dense_weight: float
    zeroshot_labels: dict[str, str]
    zeroshot_hypothesis_template: str
    zeroshot_min_confidence: float
    zeroshot_min_margin: float
    section_priority: dict[str, int]
    appeal_request_patterns: tuple[str, ...]
    original_trial_recital_patterns: tuple[str, ...]
    appeal_reject_patterns: tuple[str, ...]
    reject_other_patterns: tuple[str, ...]
    partial_support_patterns: tuple[str, ...]
    direct_reject_patterns: tuple[str, ...]
    direct_accept_patterns: tuple[str, ...]
    review_reason_priority: dict[str, int]


@dataclass(frozen=True)
class EvidenceWindow:
    """One candidate evidence span retrieved from the judgment text.

    Attributes:
        section: Source judgment section.
        text: Normalized window text.
        start: Absolute start offset in the section text.
        end: Absolute end offset in the section text.
        sentence_count: Number of merged sentences in the window.
    """

    section: str
    text: str
    start: int
    end: int
    sentence_count: int


@dataclass
class RankedEvidence:
    """Evidence window plus retrieval, rerank, and fusion scores.

    Attributes:
        window: Underlying evidence window.
        bm25_score: Sparse retrieval score.
        dense_score: Dense retrieval score.
        combined_score: Hybrid retrieval score before reranking.
        rerank_score: Optional cross-encoder rerank score.
    """

    window: EvidenceWindow
    bm25_score: float
    dense_score: float
    combined_score: float
    rerank_score: float | None = None


@dataclass
class StatusPrediction:
    """Predicted status row together with its uncertainty flag.

    Attributes:
        row: Serialized status row.
        uncertain: Whether the row should still be treated as uncertain.
    """

    row: dict[str, Any]
    uncertain: bool = False


@dataclass
class EvidenceIndex:
    """Dense and sparse retrieval structures built from one case document.

    Attributes:
        windows: Evidence windows extracted from the case.
        tokenized_windows: Sparse-retrieval tokens for each window.
        embeddings: Dense embeddings aligned with ``windows``.
        bm25: Sparse retrieval backend over ``tokenized_windows``.
    """

    windows: list[EvidenceWindow]
    tokenized_windows: list[list[str]]
    embeddings: np.ndarray
    bm25: BM25Okapi


def normalize_space(text: str) -> str:
    """Collapse repeated whitespace while preserving token order.

    Args:
        text: Raw text span.

    Returns:
        Whitespace-normalized text.
    """

    return _SPACE_RE.sub(" ", str(text or "")).strip()


def normalize_compact(text: str) -> str:
    """Remove all whitespace from one text span after normalization.

    Args:
        text: Raw text span.

    Returns:
        Fully compacted text without whitespace.
    """

    return "".join(normalize_space(text).split())


def load_status_rule_bundle(path: str | Path) -> StatusRuleBundle:
    """Load the frozen Step 06A rule bundle from disk.

    Args:
        path: JSON file containing the frozen rule bundle.

    Returns:
        Parsed status-inference rule bundle.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    return StatusRuleBundle(
        sections=tuple(data.get("sections", STATUS_SECTIONS)),
        window_sentence_max=int(data.get("window_sentence_max", 3)),
        retrieval_top_k=int(data.get("retrieval_top_k", 8)),
        rerank_keep_k=int(data.get("rerank_keep_k", 3)),
        bm25_weight=float(data.get("bm25_weight", 0.4)),
        dense_weight=float(data.get("dense_weight", 0.6)),
        zeroshot_labels=dict(data["zeroshot_labels"]),
        zeroshot_hypothesis_template=str(data["zeroshot_hypothesis_template"]),
        zeroshot_min_confidence=float(data.get("zeroshot_min_confidence", 0.60)),
        zeroshot_min_margin=float(data.get("zeroshot_min_margin", 0.15)),
        section_priority={
            str(key): int(value)
            for key, value in data.get("section_priority", {}).items()
        },
        appeal_request_patterns=tuple(data.get("appeal_request_patterns", [])),
        original_trial_recital_patterns=tuple(
            data.get("original_trial_recital_patterns", [])
        ),
        appeal_reject_patterns=tuple(data.get("appeal_reject_patterns", [])),
        reject_other_patterns=tuple(data.get("reject_other_patterns", [])),
        partial_support_patterns=tuple(data.get("partial_support_patterns", [])),
        direct_reject_patterns=tuple(data.get("direct_reject_patterns", [])),
        direct_accept_patterns=tuple(data.get("direct_accept_patterns", [])),
        review_reason_priority={
            str(key): int(value)
            for key, value in data.get("review_reason_priority", {}).items()
        },
    )


def action_family(action: str) -> str:
    """Map one action label to a coarse action family.

    Args:
        action: Canonical action label.

    Returns:
        Coarse family label such as ``monetary`` or ``declaratory``.
    """

    return ACTION_FAMILY_MAP.get(str(action or ""), "other")


def canonical_to_eval(status: str) -> str:
    """Convert one canonical status label into the evaluation label space.

    Args:
        status: Canonical status label in the inference space.

    Returns:
        Evaluation-space label used by downstream experiments.
    """

    return STATUS_EVAL_MAP[str(status)]


def build_action_aliases(ontology: dict[str, Any]) -> dict[str, list[str]]:
    """Build sorted action aliases from the ontology payload.

    Args:
        ontology: Claim ontology payload with action aliases.

    Returns:
        Canonical action to sorted alias-list mapping.
    """

    return _build_alias_map(ontology.get("actions") or {})


def build_liability_aliases(ontology: dict[str, Any]) -> dict[str, list[str]]:
    """Build sorted liability aliases from the ontology payload.

    Args:
        ontology: Claim ontology payload with liability aliases.

    Returns:
        Canonical liability to sorted alias-list mapping.
    """

    return _build_alias_map(ontology.get("liability") or {})


def _build_alias_map(values: dict[str, Any]) -> dict[str, list[str]]:
    alias_map: dict[str, list[str]] = {}

    for action, aliases in values.items():
        items = [str(action)]

        if isinstance(aliases, list):
            items.extend(str(item) for item in aliases)

        alias_map[str(action)] = sorted(set(items), key=lambda item: (-len(item), item))

    return alias_map


def _char_tokens(text: str) -> list[str]:
    compact = normalize_compact(text)
    chars = [ch for ch in compact if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"]

    if not chars:
        return []

    tokens = list(chars)
    tokens.extend("".join(chars[idx : idx + 2]) for idx in range(len(chars) - 1))
    return tokens


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    segments: list[tuple[str, int, int]] = []
    cursor = 0
    normalized = str(text or "")

    for part in _TEXT_SPLIT_RE.split(normalized):
        if not part:
            cursor += 1
            continue

        line_start = normalized.find(part, cursor)
        line_end = line_start + len(part)
        cursor = line_end
        segment_cursor = line_start
        last_end = line_start

        for match in _SENTENCE_END_RE.finditer(part):
            seg_end = line_start + match.end()
            seg_text = normalize_space(normalized[segment_cursor:seg_end])

            if seg_text:
                segments.append((seg_text, segment_cursor, seg_end))

            segment_cursor = seg_end
            last_end = seg_end

        tail = normalize_space(normalized[last_end:line_end])

        if tail:
            segments.append((tail, last_end, line_end))

    return segments


def build_evidence_windows(
    case: dict[str, Any], rules: StatusRuleBundle
) -> list[EvidenceWindow]:
    """Split configured case sections into overlapping evidence windows.

    Args:
        case: Raw case payload containing judgment sections.
        rules: Status inference rule bundle.

    Returns:
        Deduplicated overlapping evidence windows across configured sections.
    """
    content = case.get("content") or {}
    windows: list[EvidenceWindow] = []

    for section in rules.sections:
        text = str(content.get(section, "") or "")
        segments = _split_sentences(text)

        if not segments:
            continue

        for start_idx in range(len(segments)):
            merged_text: list[str] = []
            start = segments[start_idx][1]
            end = segments[start_idx][2]

            for width in range(rules.window_sentence_max):
                idx = start_idx + width

                if idx >= len(segments):
                    break

                seg_text, _, seg_end = segments[idx]
                merged_text.append(seg_text)
                end = seg_end
                window_text = normalize_space(" ".join(merged_text))

                if window_text:
                    windows.append(
                        EvidenceWindow(
                            section=section,
                            text=window_text,
                            start=start,
                            end=end,
                            sentence_count=width + 1,
                        )
                    )

    dedup: dict[tuple[str, int, int], EvidenceWindow] = {}

    for window in windows:
        dedup[(window.section, window.start, window.end)] = window

    return list(dedup.values())


class DenseRetriever:
    """Sentence-transformer encoder used for dense evidence retrieval.

    Args:
        model_path: Local sentence-transformer model path.
        device: Optional device override. Defaults to CUDA when available.
    """

    def __init__(self, model_path: str, device: str | None = None) -> None:
        import torch

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_path, device=resolved_device)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        return np.asarray(
            self.model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )


class CrossEncoderReranker:
    """Cross-encoder reranker for candidate evidence windows.

    Args:
        model_path: Local cross-encoder model path.
    """

    def __init__(self, model_path: str) -> None:
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        kwargs: dict[str, Any] = {"trust_remote_code": True}

        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, **kwargs
        )

        self.device = next(self.model.parameters()).device

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch

        if not pairs:
            return []

        left = [pair[0] for pair in pairs]
        right = [pair[1] for pair in pairs]

        inputs = self.tokenizer(
            left,
            right,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)

        if logits.ndim == 0:
            return [float(logits.detach().cpu().item())]

        return [float(item) for item in logits.detach().cpu().tolist()]


class ZeroShotStatusClassifier:
    """Zero-shot classifier wrapper for fallback status prediction.

    Args:
        model_path: Zero-shot classification model path.
        hypothesis_template: Hypothesis template used by the classifier.
    """

    def __init__(self, model_path: str, hypothesis_template: str) -> None:
        import torch

        kwargs: dict[str, Any] = {"trust_remote_code": True}

        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"

        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_path,
            tokenizer=model_path,
            hypothesis_template=hypothesis_template,
            **kwargs,
        )

    def classify(self, sequence: str, candidate_labels: list[str]) -> dict[str, Any]:
        return self.classifier(
            sequence, candidate_labels=candidate_labels, multi_label=False
        )


def build_evidence_index(
    case: dict[str, Any],
    rules: StatusRuleBundle,
    embedder: DenseRetriever,
) -> EvidenceIndex:
    """Build BM25 and dense-retrieval indexes over case evidence windows.

    Args:
        case: Raw case payload.
        rules: Status inference rule bundle.
        embedder: Dense retrieval backend.

    Returns:
        Evidence index containing sparse and dense retrieval structures.
    """
    windows = build_evidence_windows(case, rules)
    tokenized = [_char_tokens(window.text) for window in windows]
    bm25 = BM25Okapi(tokenized if tokenized else [["空"]])
    embeddings = embedder.encode([window.text for window in windows])

    return EvidenceIndex(
        windows=windows,
        tokenized_windows=tokenized,
        embeddings=embeddings,
        bm25=bm25,
    )


def _scale_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if math.isclose(minimum, maximum):
        return [1.0 if maximum > 0 else 0.0 for _ in scores]

    return [(item - minimum) / (maximum - minimum) for item in scores]


def build_status_query(row: dict[str, Any]) -> str:
    """Assemble the retrieval query text for one gold claim row.

    Args:
        row: Gold claim row awaiting status inference.

    Returns:
        Retrieval query text assembled from normalized claim fields.
    """
    parts = [
        str(row.get("claim_text_norm", "") or ""),
        str(row.get("action", "") or ""),
        str(row.get("target", "") or ""),
        str(row.get("amount", "") or ""),
        str(row.get("liability", "") or ""),
        str(row.get("stage_role", "") or ""),
        str(row.get("stage", "") or ""),
    ]

    return normalize_space(" ".join(part for part in parts if part))


def retrieve_candidates(
    row: dict[str, Any],
    index: EvidenceIndex,
    rules: StatusRuleBundle,
    embedder: DenseRetriever,
) -> list[RankedEvidence]:
    """Retrieve top evidence candidates with hybrid BM25 and dense scoring.

    Args:
        row: Gold claim row awaiting status inference.
        index: Evidence retrieval index for the case.
        rules: Status inference rule bundle.
        embedder: Dense retrieval backend used for the query vector.

    Returns:
        Top hybrid-ranked evidence candidates before reranking.
    """
    if not index.windows:
        return []

    query = build_status_query(row)
    token_query = _char_tokens(query)

    bm25_scores = (
        [float(score) for score in index.bm25.get_scores(token_query)]
        if token_query
        else [0.0 for _ in index.windows]
    )

    dense_query = embedder.encode([query])

    dense_scores = (
        (index.embeddings @ dense_query[0]).astype(float).tolist()
        if len(dense_query)
        else [0.0 for _ in index.windows]
    )

    bm25_scaled = _scale_scores(bm25_scores)
    dense_scaled = _scale_scores(dense_scores)

    ranked = [
        RankedEvidence(
            window=window,
            bm25_score=bm25_scores[idx],
            dense_score=dense_scores[idx],
            combined_score=(
                rules.bm25_weight * bm25_scaled[idx]
                + rules.dense_weight * dense_scaled[idx]
            ),
        )
        for idx, window in enumerate(index.windows)
    ]

    ranked.sort(
        key=lambda item: (
            -item.combined_score,
            rules.section_priority.get(item.window.section, 9),
            item.window.start,
        )
    )

    return ranked[: rules.retrieval_top_k]


def rerank_candidates(
    row: dict[str, Any],
    candidates: list[RankedEvidence],
    reranker: CrossEncoderReranker,
    rules: StatusRuleBundle,
) -> list[RankedEvidence]:
    """Rerank hybrid-retrieved evidence candidates with a cross encoder.

    Args:
        row: Gold claim row awaiting status inference.
        candidates: Hybrid retrieval candidates.
        reranker: Cross-encoder reranker.
        rules: Status inference rule bundle.

    Returns:
        Top reranked evidence candidates.
    """
    if not candidates:
        return []
    query = build_status_query(row)
    pairs = [(query, item.window.text) for item in candidates]
    scores = reranker.score_pairs(pairs)
    reranked: list[RankedEvidence] = []

    for item, score in zip(candidates, scores):
        reranked.append(
            RankedEvidence(
                window=item.window,
                bm25_score=item.bm25_score,
                dense_score=item.dense_score,
                combined_score=item.combined_score,
                rerank_score=float(score),
            )
        )

    reranked.sort(
        key=lambda item: (
            -(item.rerank_score or -999.0),
            -item.combined_score,
            rules.section_priority.get(item.window.section, 9),
            item.window.start,
        )
    )

    return reranked[: rules.rerank_keep_k]


def _amount_to_float(value: str) -> float | None:
    text = normalize_space(value)

    if not text or not text.endswith("元"):
        return None

    core = text[:-1]
    multiplier = 1.0

    if core.endswith("万"):
        multiplier = 10000.0
        core = core[:-1]

    elif core.endswith("亿"):
        multiplier = 100000000.0
        core = core[:-1]

    try:
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", core):
            return float(core) * multiplier

        return float(cn2an(core, "smart")) * multiplier

    except Exception:
        return None


def extract_amount_values(text: str) -> list[float]:
    """Extract normalized numeric yuan amounts from free-form Chinese text.

    Args:
        text: Free-form Chinese text.

    Returns:
        Normalized numeric amounts in yuan.
    """
    values: list[float] = []

    for match in _AMOUNT_RE.finditer(str(text or "")):
        amount = _amount_to_float(match.group(1))

        if amount is not None:
            values.append(amount)

    return values


def _target_terms(row: dict[str, Any]) -> list[str]:
    text = normalize_space(
        str(row.get("target", "") or row.get("claim_text_norm", "") or "")
    )

    if not text:
        return []

    terms = re.split(r"[，,、（）()；;。:/：\s]", text)
    cleaned: list[str] = []

    for term in terms:
        compact = normalize_compact(term)

        if len(compact) < 2:
            continue

        if compact in GENERIC_TERMS:
            continue

        cleaned.append(compact)

        if len(compact) >= 6:
            for size in (4, 6, 8):
                if len(compact) > size:
                    cleaned.append(compact[-size:])

    return sorted(set(cleaned), key=lambda item: (-len(item), item))


def _action_aliases(
    row: dict[str, Any], action_alias_map: dict[str, list[str]]
) -> list[str]:
    action = str(row.get("action", "") or "")

    if not action:
        return []

    return action_alias_map.get(action, [action])


def _liability_aliases(
    row: dict[str, Any], liability_alias_map: dict[str, list[str]]
) -> list[str]:
    liability = str(row.get("liability", "") or "")

    if not liability:
        return []

    return liability_alias_map.get(liability, [liability])


def _fuzzy_match(left: str, right: str) -> int:
    from rapidfuzz import fuzz

    return int(fuzz.partial_ratio(normalize_compact(left), normalize_compact(right)))


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _is_reject_text(text: str, rules: StatusRuleBundle) -> bool:
    return _contains_any(text, rules.direct_reject_patterns)


def _is_support_text(text: str, rules: StatusRuleBundle) -> bool:
    return _contains_any(text, rules.direct_accept_patterns)


def _alias_in_evidence(alias: str, evidence_text: str, evidence_compact: str) -> bool:
    compact_alias = normalize_compact(alias)

    if not compact_alias:
        return False

    if compact_alias == "确认":
        return bool(re.search(r"确认(?!书|函|单)", evidence_text))

    return compact_alias in evidence_compact


def _match_signals(
    row: dict[str, Any],
    evidence_text: str,
    action_alias_map: dict[str, list[str]],
    liability_alias_map: dict[str, list[str]],
    rules: StatusRuleBundle,
) -> dict[str, Any]:
    evidence_compact = normalize_compact(evidence_text)
    action_aliases = _action_aliases(row, action_alias_map)
    liability_aliases = _liability_aliases(row, liability_alias_map)
    target_terms = _target_terms(row)

    target_terms.extend(
        [
            normalize_compact(term)
            for term in liability_aliases
            if len(normalize_compact(term)) >= 2
        ]
    )

    target_terms = sorted(
        {term for term in target_terms if term},
        key=lambda item: (-len(item), item),
    )

    claim_amounts = extract_amount_values(str(row.get("claim_text_raw", "") or ""))
    evidence_amounts = extract_amount_values(evidence_text)
    amount_hit = False

    if claim_amounts and evidence_amounts:
        amount_hit = any(
            math.isclose(left, right, rel_tol=0.02)
            for left in claim_amounts
            for right in evidence_amounts
        )

    fuzzy_target = max(
        (_fuzzy_match(term, evidence_text) for term in target_terms[:6]), default=0
    )

    return {
        "action_hit": bool(action_aliases)
        and any(
            _alias_in_evidence(alias, evidence_text, evidence_compact)
            for alias in action_aliases
        ),
        "liability_hit": bool(liability_aliases)
        and any(
            _alias_in_evidence(alias, evidence_text, evidence_compact)
            for alias in liability_aliases
        ),
        "target_hit": any(term in evidence_compact for term in target_terms[:6]),
        "amount_hit": amount_hit,
        "fuzzy_claim": _fuzzy_match(
            str(row.get("claim_text_norm", "") or row.get("claim_text_raw", "") or ""),
            evidence_text,
        ),
        "fuzzy_target": fuzzy_target,
        "reject_text": _is_reject_text(evidence_text, rules),
        "support_text": _is_support_text(evidence_text, rules),
        "partial_text": _contains_any(evidence_text, rules.partial_support_patterns),
    }


def detect_partial_support(
    row: dict[str, Any],
    evidence_text: str,
    rules: StatusRuleBundle,
    action_alias_map: dict[str, list[str]] | None = None,
    liability_alias_map: dict[str, list[str]] | None = None,
) -> bool:
    """Detect whether evidence only partially supports a requested claim.

    Args:
        row: Gold claim row under evaluation.
        evidence_text: Candidate evidence text.
        rules: Status inference rule bundle.
        action_alias_map: Optional action alias mapping.
        liability_alias_map: Optional liability alias mapping.

    Returns:
        True when the evidence appears to support only part of the claim.
    """
    if _is_request_recital_only(evidence_text, rules):
        return False

    if _is_issue_statement_only(evidence_text, rules):
        return False

    claim_amounts = extract_amount_values(str(row.get("claim_text_raw", "") or ""))
    evidence_amounts = extract_amount_values(evidence_text)

    if claim_amounts and evidence_amounts:
        claim_amount = max(claim_amounts)
        evidence_amount = max(evidence_amounts)

        if (
            claim_amount > 0
            and evidence_amount > 0
            and not math.isclose(claim_amount, evidence_amount, rel_tol=0.02)
        ):
            if action_alias_map is None or liability_alias_map is None:
                return True

            signals = _match_signals(
                row, evidence_text, action_alias_map, liability_alias_map, rules
            )

            if (
                signals["target_hit"]
                or signals["liability_hit"]
                or signals["fuzzy_target"] >= 70
                or signals["fuzzy_claim"] >= 70
            ):
                return True

        if any(
            not math.isclose(claim_amount, item, rel_tol=0.02)
            for item in evidence_amounts
        ) and (
            _contains_any(evidence_text, rules.partial_support_patterns)
            or _contains_any(evidence_text, rules.reject_other_patterns)
        ):
            if action_alias_map is None or liability_alias_map is None:
                return True

            signals = _match_signals(
                row, evidence_text, action_alias_map, liability_alias_map, rules
            )

            if (
                signals["target_hit"]
                or signals["liability_hit"]
                or signals["fuzzy_target"] >= 70
                or signals["fuzzy_claim"] >= 70
            ):
                return True

    if _contains_any(evidence_text, rules.partial_support_patterns):
        if action_alias_map is not None and liability_alias_map is not None:
            signals = _match_signals(
                row, evidence_text, action_alias_map, liability_alias_map, rules
            )

            if (signals["action_hit"] or signals["liability_hit"]) and (
                signals["amount_hit"]
                or signals["target_hit"]
                or signals["fuzzy_target"] >= 88
            ):
                return False

        return True

    if re.search(r"不再予以判决|不再处理|已无继续处理必要|已无处理必要", evidence_text):
        return True

    if (
        _has_explicit_support_disposition(evidence_text)
        and _is_reject_text(evidence_text, rules)
        and not _contains_any(evidence_text, rules.reject_other_patterns)
        and action_alias_map is not None
        and liability_alias_map is not None
    ):
        signals = _match_signals(
            row, evidence_text, action_alias_map, liability_alias_map, rules
        )

        if (
            signals["amount_hit"]
            or signals["target_hit"]
            or signals["liability_hit"]
            or signals["fuzzy_target"] >= 70
            or signals["fuzzy_claim"] >= 75
        ):
            return True

    return False


def _is_compound_claim(row: dict[str, Any]) -> bool:
    claim_text = str(row.get("claim_text_raw", "") or "")

    if len(extract_amount_values(claim_text)) >= 2:
        return True

    compound_markers = [
        "并",
        "及",
        "以及",
        "其中",
        "利息",
        "违约金",
        "分别",
        "查阅",
        "复印",
        "拍照",
    ]

    if any(marker in claim_text for marker in compound_markers):
        return True

    if claim_text.count("、") >= 1 and len(claim_text) >= 20:
        return True

    if claim_text.count("；") >= 1:
        return True

    return False


def _has_amount_mismatch(row: dict[str, Any], evidence_text: str) -> bool:
    claim_amounts = extract_amount_values(str(row.get("claim_text_raw", "") or ""))
    evidence_amounts = extract_amount_values(evidence_text)

    if not claim_amounts or not evidence_amounts:
        return False

    return not any(
        math.isclose(left, right, rel_tol=0.02)
        for left in claim_amounts
        for right in evidence_amounts
    )


def _has_global_reject_text(evidence_text: str, rules: StatusRuleBundle) -> bool:
    if _contains_any(evidence_text, rules.reject_other_patterns):
        return True

    return bool(
        re.search(
            r"(?:诉讼请求|诉请|上诉请求|上诉理由).{0,12}(?:不予支持|予以驳回|应予驳回|不能成立|不应支持)",
            evidence_text,
        )
    )


def _claim_seeks_negative_declaration(text: str) -> bool:
    return bool(re.search(r"无效|违法|撤销|解除|不发生效力|不存在|终止", text))


def _claim_seeks_positive_declaration(text: str) -> bool:
    return bool(
        re.search(r"有效|成立|继续履行|准予|应予确认|予以确认|合法", text)
    ) and not _claim_seeks_negative_declaration(text)


def _evidence_supports_negative_declaration(text: str) -> bool:
    return bool(
        re.search(r"无效|违法|撤销|解除|不发生效力|不存在|终止|予以撤销|应予撤销", text)
    )


def _evidence_supports_positive_declaration(text: str) -> bool:
    return bool(
        re.search(
            r"合法有效|依法成立|合同成立|应当履行|按照约定履行|应按合同约定履行|按照合同约定履行|应当按照合同约定履行|继续履行|予以维持|维持原判|应予确认|予以确认|有效",
            text,
        )
    )


def _has_declaratory_polarity_conflict(row: dict[str, Any], evidence_text: str) -> bool:
    action = str(row.get("action", ""))

    if action not in {"确认", "撤销", "解除", "准予", "解散"}:
        return False

    claim_text = str(
        row.get("claim_text_raw", "") or row.get("claim_text_norm", "") or ""
    )

    claim_negative = _claim_seeks_negative_declaration(claim_text)
    claim_positive = _claim_seeks_positive_declaration(claim_text)
    evidence_negative = _evidence_supports_negative_declaration(evidence_text)
    evidence_positive = _evidence_supports_positive_declaration(evidence_text)

    if claim_negative and evidence_positive and not evidence_negative:
        return True

    if claim_positive and evidence_negative and not evidence_positive:
        return True

    return False


def _claim_seeks_negative_obligation(text: str) -> bool:
    return bool(re.search(r"无需|无须|不承担|不支付|不返还|不退还|不负|免于", text))


def _evidence_supports_negative_obligation(text: str) -> bool:
    return bool(
        re.search(r"不应承担|不承担|无需|无须|不支付|不返还|不退还|不负|免于", text)
    )


def _has_obligation_polarity_conflict(
    row: dict[str, Any], evidence_text: str, signals: dict[str, Any]
) -> bool:
    action = str(row.get("action", ""))

    if action not in {
        "支付",
        "返还",
        "承担",
        "赔偿",
        "交付",
        "继续履行",
        "排除妨害",
        "办理",
        "恢复原状",
        "停止侵权",
        "抚养",
        "分割",
        "迁葬",
    }:
        return False

    claim_text = str(
        row.get("claim_text_raw", "") or row.get("claim_text_norm", "") or ""
    )

    claim_negative = _claim_seeks_negative_obligation(claim_text)
    evidence_negative = _evidence_supports_negative_obligation(evidence_text)

    has_alignment = (
        signals["amount_hit"]
        or signals["target_hit"]
        or signals["fuzzy_target"] >= 70
        or signals["fuzzy_claim"] >= 80
    )

    if (
        claim_negative
        and not evidence_negative
        and has_alignment
        and (signals["action_hit"] or signals["liability_hit"])
    ):
        return True

    if (
        (not claim_negative)
        and evidence_negative
        and has_alignment
        and (signals["action_hit"] or signals["liability_hit"])
    ):
        return True

    return False


def _has_explicit_support_disposition(text: str) -> bool:
    return bool(
        re.search(
            r"本院.{0,8}予以支持|应予支持|于法有据.{0,8}予以支持|予以确认|应予确认|本院予以确认|本院予以纠正",
            text,
        )
    )


def _has_numbered_disposition(text: str) -> bool:
    return bool(re.search(r"(?:^|[；;。\n])\s*[一二三四五六七八九十\d]+[、.]", text))


def _has_negative_reasoning_cue(text: str) -> bool:
    return bool(
        re.search(
            r"未能.{0,8}证实|未举证证明|证据不足|缺乏依据|无法认定|不予认定|不予采信|未提供充分证据|无事实依据",
            text,
        )
    )


def _is_request_recital_only(text: str, rules: StatusRuleBundle) -> bool:
    if not re.search(
        r"诉至本院|请求依法|起诉称|诉称|上诉称|原告诉至本院|原告.*诉请|原告.*请求|原告主张",
        text,
    ):
        return False

    has_decision_marker = (
        _has_explicit_support_disposition(text)
        or _is_reject_text(text, rules)
        or _contains_any(text, rules.partial_support_patterns)
        or _has_numbered_disposition(text)
        or bool(re.search(r"判决[:：]|判令|于本判决生效", text))
    )

    return not has_decision_marker


def _is_issue_statement_only(text: str, rules: StatusRuleBundle) -> bool:
    if not re.search(
        r"争议焦点|焦点为|争议在于|核心在于|关于.{0,24}问题|是否应当", text
    ):
        return False

    has_decision_marker = (
        _has_explicit_support_disposition(text)
        or _is_reject_text(text, rules)
        or _contains_any(text, rules.partial_support_patterns)
        or _has_numbered_disposition(text)
        or bool(
            re.search(
                r"判决[:：]|判令|于本判决生效|本院予以采信|本院予以确认|本院予以纠正|由[^。；\n]{0,24}(承担|支付|返还|赔偿|交付)",
                text,
            )
        )
    )

    return not has_decision_marker


def should_autolabel_zeroshot(
    row: dict[str, Any],
    evidence_text: str,
    *,
    status_canonical: str,
    top_score: float,
    margin: float,
    rules: StatusRuleBundle,
    action_alias_map: dict[str, list[str]],
    liability_alias_map: dict[str, list[str]],
) -> bool:
    """Decide whether zero-shot status output is reliable enough to keep.

    Args:
        row: Gold claim row under evaluation.
        evidence_text: Top reranked evidence text.
        status_canonical: Top zero-shot canonical status label.
        top_score: Top zero-shot confidence score.
        margin: Margin between the top two zero-shot labels.
        rules: Status inference rule bundle.
        action_alias_map: Action alias mapping.
        liability_alias_map: Liability alias mapping.

    Returns:
        True when the zero-shot prediction is reliable enough for auto-labeling.
    """
    if top_score < rules.zeroshot_min_confidence or margin < rules.zeroshot_min_margin:
        return False

    support_match = direct_support_match(
        row, evidence_text, action_alias_map, liability_alias_map, rules
    )

    reject_match = direct_reject_match(
        row, evidence_text, action_alias_map, liability_alias_map, rules
    )

    partial_match = detect_partial_support(
        row, evidence_text, rules, action_alias_map, liability_alias_map
    )

    compound_claim = _is_compound_claim(row)
    amount_mismatch = _has_amount_mismatch(row, evidence_text)
    support_text = _is_support_text(evidence_text, rules)
    reject_text = _is_reject_text(evidence_text, rules)
    global_reject = _has_global_reject_text(evidence_text, rules)

    signals = _match_signals(
        row, evidence_text, action_alias_map, liability_alias_map, rules
    )

    if _is_request_recital_only(evidence_text, rules):
        return False

    if _is_issue_statement_only(evidence_text, rules):
        return False

    if _has_declaratory_polarity_conflict(row, evidence_text):
        return False

    if _has_obligation_polarity_conflict(row, evidence_text, signals):
        return False

    if status_canonical == "ACCEPTED":
        if (
            str(row.get("action", "")) in {"确认", "撤销", "解除", "准予", "解散"}
            and not support_match
        ):
            return False

        if support_match:
            return True

        if reject_text or partial_match or amount_mismatch or compound_claim:
            return False

        return (
            top_score >= max(rules.zeroshot_min_confidence, 0.70)
            and support_text
            and (signals["fuzzy_claim"] >= 72 or signals["fuzzy_target"] >= 72)
        )

    if status_canonical == "REJECTED":
        if compound_claim and not global_reject and signals["fuzzy_claim"] < 85:
            return False

        if reject_match:
            return True

        if support_match:
            return False

        return (
            top_score >= max(rules.zeroshot_min_confidence, 0.68)
            and reject_text
            and (
                global_reject
                or signals["action_hit"]
                or signals["liability_hit"]
                or signals["target_hit"]
                or signals["fuzzy_claim"] >= 65
                or signals["fuzzy_target"] >= 70
            )
        )

    if status_canonical == "UNMENTIONED":
        if partial_match:
            return True

        if support_match or reject_match:
            return False

        partial_like = bool(
            re.search(
                r"另行主张|暂不予处理|酌定|多诉部分|部分.*不予支持|部分.*予以驳回",
                evidence_text,
            )
        )

        return top_score >= max(rules.zeroshot_min_confidence, 0.70) and partial_like

    return False


def direct_support_match(
    row: dict[str, Any],
    evidence_text: str,
    action_alias_map: dict[str, list[str]],
    liability_alias_map: dict[str, list[str]],
    rules: StatusRuleBundle,
) -> bool:
    """Return whether evidence directly supports the claim as written.

    Args:
        row: Gold claim row under evaluation.
        evidence_text: Candidate evidence text.
        action_alias_map: Action alias mapping.
        liability_alias_map: Liability alias mapping.
        rules: Status inference rule bundle.

    Returns:
        True when the evidence directly supports the claim as written.
    """
    signals = _match_signals(
        row, evidence_text, action_alias_map, liability_alias_map, rules
    )

    if _is_request_recital_only(evidence_text, rules):
        return False

    if _is_issue_statement_only(evidence_text, rules):
        return False

    if _has_declaratory_polarity_conflict(row, evidence_text):
        return False

    if _has_obligation_polarity_conflict(row, evidence_text, signals):
        return False

    if signals["reject_text"] and not (
        (
            _contains_any(evidence_text, rules.reject_other_patterns)
            or signals["partial_text"]
        )
        and (
            signals["amount_hit"]
            or signals["target_hit"]
            or signals["fuzzy_target"] >= 82
        )
    ):
        return False

    if _has_amount_mismatch(row, evidence_text):
        return False

    if not (signals["action_hit"] or signals["liability_hit"]):
        return False

    if (
        _has_negative_reasoning_cue(evidence_text)
        and not _has_explicit_support_disposition(evidence_text)
        and not _has_numbered_disposition(evidence_text)
    ):
        return False

    declaratory_action = str(row.get("action", "")) in {
        "确认",
        "解除",
        "撤销",
        "准予",
        "解散",
    }

    declaratory_support = (
        _has_explicit_support_disposition(evidence_text)
        or _has_numbered_disposition(evidence_text)
        or (
            str(row.get("action", "")) == "确认"
            and signals["action_hit"]
            and signals["fuzzy_target"] >= 60
            and re.search(
                r"各占|份额|共同享有|归.+所有|归.+共有|有效|无效|予以确认|应予确认",
                evidence_text,
            )
        )
    )

    if declaratory_action and not declaratory_support:
        return False

    if signals["amount_hit"]:
        return True

    if signals["target_hit"] and (signals["action_hit"] or signals["liability_hit"]):
        return True

    if (
        signals["support_text"]
        and signals["action_hit"]
        and signals["fuzzy_target"] >= 70
    ):
        return True

    if (
        str(row.get("action", "")) == "确认"
        and signals["action_hit"]
        and signals["fuzzy_target"] >= 60
        and re.search(
            r"各占|份额|共同享有|归.+所有|归.+共有|有效|无效|予以确认|应予确认",
            evidence_text,
        )
    ):
        return True

    if (
        str(row.get("action", "")) in {"确认", "解除", "撤销", "准予", "解散"}
        and _has_explicit_support_disposition(evidence_text)
        and signals["fuzzy_claim"] >= 72
    ):
        return True

    if signals["fuzzy_target"] >= 82 and (
        signals["action_hit"] or signals["liability_hit"]
    ):
        return True

    if signals["fuzzy_claim"] >= 88 and (
        signals["amount_hit"]
        or signals["target_hit"]
        or signals["fuzzy_target"] >= 82
        or signals["liability_hit"]
    ):
        return True

    return False


def direct_reject_match(
    row: dict[str, Any],
    evidence_text: str,
    action_alias_map: dict[str, list[str]],
    liability_alias_map: dict[str, list[str]],
    rules: StatusRuleBundle,
) -> bool:
    """Return whether evidence directly rejects the claim as written.

    Args:
        row: Gold claim row under evaluation.
        evidence_text: Candidate evidence text.
        action_alias_map: Action alias mapping.
        liability_alias_map: Liability alias mapping.
        rules: Status inference rule bundle.

    Returns:
        True when the evidence directly rejects the claim as written.
    """
    signals = _match_signals(
        row, evidence_text, action_alias_map, liability_alias_map, rules
    )

    if _is_request_recital_only(evidence_text, rules):
        return False

    if _is_issue_statement_only(evidence_text, rules):
        return False

    if _has_declaratory_polarity_conflict(row, evidence_text) and (
        signals["target_hit"]
        or signals["fuzzy_target"] >= 45
        or signals["fuzzy_claim"] >= 25
    ):
        return True

    if _has_obligation_polarity_conflict(row, evidence_text, signals):
        return True

    if not signals["reject_text"]:
        return False

    if (
        _has_explicit_support_disposition(evidence_text)
        and not _contains_any(evidence_text, rules.reject_other_patterns)
        and not _has_global_reject_text(evidence_text, rules)
    ):
        return False

    if _contains_any(evidence_text, rules.reject_other_patterns) and (
        signals["amount_hit"]
        or signals["target_hit"]
        or signals["fuzzy_target"] >= 82
        or signals["fuzzy_claim"] >= 88
    ):
        return False

    if signals["amount_hit"]:
        return True

    if signals["target_hit"]:
        return True

    if signals["fuzzy_target"] >= 80:
        return True

    if str(row.get("action", "")) in {"确认", "解除", "撤销", "准予", "解散"} and (
        signals["fuzzy_target"] >= 60 or signals["fuzzy_claim"] >= 35
    ):
        return True

    return signals["fuzzy_claim"] >= 82


def _row_start_offset(row: dict[str, Any]) -> int:
    source_span = row.get("source_span") or {}

    try:
        return max(0, int(source_span.get("start", 0)))

    except (TypeError, ValueError):
        return 0


def is_probable_appeal_claim(
    case: dict[str, Any], row: dict[str, Any], rules: StatusRuleBundle
) -> bool:
    """Heuristically detect whether a second-instance claim is an appeal request.

    Args:
        case: Raw case payload.
        row: Gold claim row under evaluation.
        rules: Status inference rule bundle.

    Returns:
        True when the row is likely an appeal request rather than a recital.
    """
    if str(row.get("stage", "")) != "二审":
        return False

    plaintiff_text = str((case.get("content") or {}).get("原告诉称", "") or "")

    if not plaintiff_text:
        return True

    start = _row_start_offset(row)
    prefix = plaintiff_text[:start] if start else plaintiff_text[:160]
    nearby = prefix[-200:]
    has_appeal_cue = _contains_any(nearby, rules.appeal_request_patterns)

    has_original_trial_cue = _contains_any(
        nearby, rules.original_trial_recital_patterns
    )

    if has_original_trial_cue and not has_appeal_cue:
        return False

    if has_appeal_cue:
        return True

    global_prefix = plaintiff_text[: min(len(plaintiff_text), max(start, 200))]

    if _contains_any(
        global_prefix, rules.original_trial_recital_patterns
    ) and not _contains_any(global_prefix, rules.appeal_request_patterns):
        return False

    return True


def build_status_row(
    row: dict[str, Any],
    *,
    status_canonical: str,
    status_source: str,
    decision_mode: str,
    evidence: RankedEvidence | None,
    confidence: float | None,
    review_reason: str = "",
) -> dict[str, Any]:
    """Build one finalized automatic status row.

    Args:
        row: Gold claim row under evaluation.
        status_canonical: Canonical status label.
        status_source: Source of the label, e.g. ``rule`` or ``zeroshot``.
        decision_mode: Fine-grained decision-path label.
        evidence: Optional supporting evidence candidate.
        confidence: Optional confidence score.
        review_reason: Optional manual-review rationale.

    Returns:
        Serialized automatic status row.

    Raises:
        ValueError: If ``status_canonical`` is outside the canonical label set.
    """
    if status_canonical not in STATUS_CANONICAL:
        raise ValueError(f"invalid status_canonical={status_canonical}")

    evidence_spans: list[dict[str, Any]] = []
    evidence_text = ""
    evidence_section = ""

    if evidence is not None:
        evidence_spans = [
            {
                "section": evidence.window.section,
                "start": evidence.window.start,
                "end": evidence.window.end,
            }
        ]

        evidence_text = evidence.window.text
        evidence_section = evidence.window.section

    return {
        "uid": str(row["uid"]),
        "claim_id": str(row["claim_id"]),
        "split": str(row["split"]),
        "hard_case": int(row["hard_case"]),
        "stage": str(row["stage"]),
        "sample_cause": str(row.get("sample_cause", "")),
        "claim_text_raw": str(row["claim_text_raw"]),
        "claim_text_norm": str(row.get("claim_text_norm", "")),
        "action": str(row.get("action", "")),
        "target": str(row.get("target", "")),
        "amount": str(row.get("amount", "")),
        "liability": str(row.get("liability", "")),
        "stage_role": str(row.get("stage_role", "")),
        "claim_source": str(row.get("claim_source", "")),
        "status_canonical": status_canonical,
        "status_eval": canonical_to_eval(status_canonical),
        "status_source": status_source,
        "decision_mode": decision_mode,
        "evidence_section": evidence_section,
        "evidence_spans": evidence_spans,
        "evidence_text": evidence_text,
        "confidence": None if confidence is None else float(confidence),
        "review_reason": str(review_reason or ""),
    }


def build_uncertain_row(
    row: dict[str, Any],
    *,
    candidates: list[RankedEvidence],
    review_reason: str,
) -> dict[str, Any]:
    """Build one manual-review row with serialized candidate evidence.

    Args:
        row: Gold claim row under evaluation.
        candidates: Candidate evidence windows retained for review.
        review_reason: Primary reason for manual review.

    Returns:
        Serialized manual-review row.
    """
    case_candidates = []

    for item in candidates:
        case_candidates.append(
            {
                "section": item.window.section,
                "start": item.window.start,
                "end": item.window.end,
                "text": item.window.text,
                "bm25_score": float(item.bm25_score),
                "dense_score": float(item.dense_score),
                "combined_score": float(item.combined_score),
                "rerank_score": None
                if item.rerank_score is None
                else float(item.rerank_score),
            }
        )

    return {
        "uid": str(row["uid"]),
        "claim_id": str(row["claim_id"]),
        "split": str(row["split"]),
        "hard_case": int(row["hard_case"]),
        "stage": str(row["stage"]),
        "sample_cause": str(row.get("sample_cause", "")),
        "claim_text_raw": str(row["claim_text_raw"]),
        "claim_text_norm": str(row.get("claim_text_norm", "")),
        "action": str(row.get("action", "")),
        "target": str(row.get("target", "")),
        "amount": str(row.get("amount", "")),
        "liability": str(row.get("liability", "")),
        "stage_role": str(row.get("stage_role", "")),
        "claim_source": str(row.get("claim_source", "")),
        "review_reasons": [str(review_reason)],
        "candidate_evidence": case_candidates,
        "needs_manual_review": True,
    }


def infer_case_statuses(
    case: dict[str, Any],
    claim_rows: list[dict[str, Any]],
    *,
    rules: StatusRuleBundle,
    action_alias_map: dict[str, list[str]],
    liability_alias_map: dict[str, list[str]],
    embedder: DenseRetriever,
    reranker: CrossEncoderReranker,
    zeroshot: ZeroShotStatusClassifier,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Infer statuses for all gold claims belonging to a single case.

    Args:
        case: Raw case payload.
        claim_rows: Gold claim rows belonging to the case.
        rules: Status inference rule bundle.
        action_alias_map: Action alias mapping.
        liability_alias_map: Liability alias mapping.
        embedder: Dense retrieval backend.
        reranker: Cross-encoder reranker.
        zeroshot: Zero-shot fallback classifier.

    Returns:
        A pair of automatic status rows and uncertain manual-review rows.
    """
    if not claim_rows:
        return [], []

    judgment_text = str((case.get("content") or {}).get("裁判结果", "") or "")
    appeal_reject = _contains_any(judgment_text, rules.appeal_reject_patterns)
    reject_other = _contains_any(judgment_text, rules.reject_other_patterns)
    index = build_evidence_index(case, rules, embedder)
    auto_rows: list[dict[str, Any]] = []
    uncertain_rows: list[dict[str, Any]] = []

    for row in sorted(claim_rows, key=lambda item: int(item.get("claim_index", 0))):
        candidates = rerank_candidates(
            row,
            retrieve_candidates(row, index, rules, embedder),
            reranker,
            rules,
        )

        top = candidates[0] if candidates else None
        top_text = top.window.text if top is not None else ""
        stage_role = str(row.get("stage_role", ""))

        support_candidate = next(
            (
                item
                for item in candidates
                if direct_support_match(
                    row,
                    item.window.text,
                    action_alias_map,
                    liability_alias_map,
                    rules,
                )
            ),
            None,
        )

        partial_candidate = next(
            (
                item
                for item in candidates
                if detect_partial_support(
                    row,
                    item.window.text,
                    rules,
                    action_alias_map,
                    liability_alias_map,
                )
            ),
            None,
        )

        reject_candidate = next(
            (
                item
                for item in candidates
                if direct_reject_match(
                    row,
                    item.window.text,
                    action_alias_map,
                    liability_alias_map,
                    rules,
                )
            ),
            None,
        )

        appeal_candidate = next(
            (
                item
                for item in candidates
                if _contains_any(item.window.text, rules.appeal_reject_patterns)
            ),
            None,
        )

        reject_other_candidate = next(
            (
                item
                for item in candidates
                if _contains_any(item.window.text, rules.reject_other_patterns)
            ),
            None,
        )

        if (
            stage_role == "appeal_substantive"
            and appeal_reject
            and appeal_candidate is not None
            and support_candidate is None
            and is_probable_appeal_claim(case, row, rules)
        ):
            auto_rows.append(
                build_status_row(
                    row,
                    status_canonical="REJECTED",
                    status_source="rule",
                    decision_mode="appeal_disposition_rule",
                    evidence=appeal_candidate,
                    confidence=1.0,
                )
            )

            continue

        if support_candidate is not None:
            if detect_partial_support(
                row,
                support_candidate.window.text,
                rules,
                action_alias_map,
                liability_alias_map,
            ):
                auto_rows.append(
                    build_status_row(
                        row,
                        status_canonical="UNMENTIONED",
                        status_source="rule",
                        decision_mode="partial_support_rule",
                        evidence=support_candidate,
                        confidence=0.95,
                    )
                )

                continue

            auto_rows.append(
                build_status_row(
                    row,
                    status_canonical="ACCEPTED",
                    status_source="rule",
                    decision_mode="direct_support_rule",
                    evidence=support_candidate,
                    confidence=0.98,
                )
            )

            continue

        if partial_candidate is not None:
            auto_rows.append(
                build_status_row(
                    row,
                    status_canonical="UNMENTIONED",
                    status_source="rule",
                    decision_mode="partial_support_rule",
                    evidence=partial_candidate,
                    confidence=0.95,
                )
            )

            continue

        if reject_candidate is not None:
            auto_rows.append(
                build_status_row(
                    row,
                    status_canonical="REJECTED",
                    status_source="rule",
                    decision_mode="direct_reject_rule",
                    evidence=reject_candidate,
                    confidence=0.98,
                )
            )

            continue

        if (
            reject_other
            and stage_role == "first_instance_substantive"
            and reject_other_candidate is not None
            and support_candidate is None
        ):
            auto_rows.append(
                build_status_row(
                    row,
                    status_canonical="REJECTED",
                    status_source="rule",
                    decision_mode="general_reject_rule",
                    evidence=reject_other_candidate,
                    confidence=0.96,
                )
            )

            continue

        if not candidates:
            uncertain_rows.append(
                build_uncertain_row(
                    row,
                    candidates=[],
                    review_reason="no_evidence_candidate",
                )
            )

            continue

        candidate_labels = [
            rules.zeroshot_labels["ACCEPTED"],
            rules.zeroshot_labels["REJECTED"],
            rules.zeroshot_labels["UNMENTIONED"],
        ]

        sequence = f"诉求：{row['claim_text_raw']}\n裁判文本：{top.window.text}"
        result = zeroshot.classify(sequence, candidate_labels)
        labels = [str(item) for item in result.get("labels", [])]
        scores = [float(item) for item in result.get("scores", [])]

        if not labels or not scores:
            uncertain_rows.append(
                build_uncertain_row(
                    row,
                    candidates=candidates,
                    review_reason="zeroshot_empty_result",
                )
            )

            continue

        top_label = labels[0]
        top_score = scores[0]
        margin = top_score - (scores[1] if len(scores) > 1 else 0.0)
        inverse_map = {value: key for key, value in rules.zeroshot_labels.items()}
        status_canonical = inverse_map.get(top_label)

        if status_canonical is None:
            uncertain_rows.append(
                build_uncertain_row(
                    row,
                    candidates=candidates,
                    review_reason="zeroshot_unknown_label",
                )
            )

            continue

        if should_autolabel_zeroshot(
            row,
            top.window.text,
            status_canonical=status_canonical,
            top_score=top_score,
            margin=margin,
            rules=rules,
            action_alias_map=action_alias_map,
            liability_alias_map=liability_alias_map,
        ):
            auto_rows.append(
                build_status_row(
                    row,
                    status_canonical=status_canonical,
                    status_source="zeroshot",
                    decision_mode="zeroshot_top1",
                    evidence=top,
                    confidence=top_score,
                )
            )

            continue

        uncertain_rows.append(
            build_uncertain_row(
                row,
                candidates=candidates,
                review_reason="low_confidence",
            )
        )

    return auto_rows, uncertain_rows


def build_status_summary(
    *,
    auto_rows: list[dict[str, Any]],
    uncertain_rows: list[dict[str, Any]],
    total_claims: int,
    mode: str,
) -> dict[str, Any]:
    """Aggregate summary statistics for a status-inference export.

    Args:
        auto_rows: Automatically labeled status rows.
        uncertain_rows: Manual-review rows.
        total_claims: Total claim count processed in the run.
        mode: Export mode identifier.

    Returns:
        Aggregate summary payload for reporting and auditing.
    """
    by_source: dict[str, int] = {}
    by_canonical: dict[str, int] = {}
    by_decision_mode: dict[str, int] = {}
    uncertain_reasons: dict[str, int] = {}

    for row in auto_rows:
        by_source[row["status_source"]] = by_source.get(row["status_source"], 0) + 1

        by_canonical[row["status_canonical"]] = (
            by_canonical.get(row["status_canonical"], 0) + 1
        )

        by_decision_mode[row["decision_mode"]] = (
            by_decision_mode.get(row["decision_mode"], 0) + 1
        )

    for row in uncertain_rows:
        reasons = row.get("review_reasons", []) or []

        for reason in reasons:
            uncertain_reasons[str(reason)] = uncertain_reasons.get(str(reason), 0) + 1

    return {
        "mode": mode,
        "total_claims": total_claims,
        "auto_labeled_count": len(auto_rows),
        "uncertain_count": len(uncertain_rows),
        "auto_label_rate": 0.0 if total_claims == 0 else len(auto_rows) / total_claims,
        "status_source_counts": dict(sorted(by_source.items())),
        "status_canonical_counts": dict(sorted(by_canonical.items())),
        "decision_mode_counts": dict(sorted(by_decision_mode.items())),
        "uncertain_reason_counts": dict(sorted(uncertain_reasons.items())),
    }
