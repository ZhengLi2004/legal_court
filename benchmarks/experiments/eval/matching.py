"""Claim deduplication and one-to-one matching helpers for Step 07."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

GOLD_CLAIM_ID_PREFIX = "GOLD_"


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def _normalize_compact(text: str) -> str:
    return "".join(str(text or "").split())


class TextEncoder(Protocol):
    """Minimal encoder interface required by the Step 07 matcher.

    The matcher only depends on batched text-to-vector conversion and assumes
    the returned matrix is 2D and row-aligned with the input texts.

    Attributes:
        None. This protocol only constrains the ``encode`` method.
    """

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into a 2D float array.

        Args:
            texts: Claim texts to encode in batch order.

        Returns:
            A 2D float array whose rows align with ``texts``.
        """


class SentenceTransformerTextEncoder:
    """Thin wrapper around sentence-transformers for local matching.

    Args:
        model_path: Local sentence-transformers model path.
        device: Optional device override forwarded to the backend model.
    """

    def __init__(self, model_path: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_path, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return np.asarray(vectors, dtype=np.float32)


@dataclass(frozen=True)
class MatchingConfig:
    """Frozen configuration for claim deduplication and one-to-one matching.

    Attributes:
        dedup_similarity_threshold: Cosine threshold above which prediction
            rows are clustered as semantic duplicates.
        match_similarity_threshold: Minimum semantic similarity required for a
            feasible gold/prediction match.
        semantic_weight: Weight assigned to semantic distance in the Hungarian
            assignment cost.
        length_penalty_weight: Weight assigned to claim-length mismatch.
        length_mode: Reserved selector for length normalization strategy.
        text_field: Claim-row field used as matcher input text.
        unmatched_cost: Cost assigned to leaving one item unmatched.
        infeasible_cost: Sentinel cost for impossible assignments.
    """

    dedup_similarity_threshold: float = 0.82
    match_similarity_threshold: float = 0.67
    semantic_weight: float = 0.85
    length_penalty_weight: float = 0.15
    length_mode: str = "char_count"
    text_field: str = "claim_text_norm"
    unmatched_cost: float = 1.0
    infeasible_cost: float = 1_000_000.0


@dataclass(frozen=True)
class DedupCluster:
    """One semantic duplicate cluster among prediction claims.

    Attributes:
        cluster_id: Stable cluster identifier scoped to one case and method.
        representative_claim_id: Claim chosen as the cluster representative.
        member_claim_ids: All member claim ids in the duplicate cluster.
        representative_row: Representative prediction row.
        member_rows: Raw member rows that collapsed into the cluster.
        centroid_similarity: Similarity between the representative embedding and
            the cluster centroid.
    """

    cluster_id: str
    representative_claim_id: str
    member_claim_ids: tuple[str, ...]
    representative_row: dict[str, Any]
    member_rows: tuple[dict[str, Any], ...]
    centroid_similarity: float


@dataclass(frozen=True)
class MatchPair:
    """One matched gold/prediction claim pair with scoring metadata.

    Attributes:
        uid: Case uid shared by the matched rows.
        method_name: Method that produced the prediction row.
        gold_row: Matched gold claim row.
        prediction_row: Matched prediction claim row.
        semantic_similarity: Embedding cosine similarity for the pair.
        length_gap: Relative claim-length mismatch penalty term.
        cost: Final assignment cost used by the Hungarian solver.
        matching_mode: ``semantic`` or ``claim_id_direct``.
        direct_claim_id: Matched claim id for direct seeded matching.
    """

    uid: str
    method_name: str
    gold_row: dict[str, Any]
    prediction_row: dict[str, Any]
    semantic_similarity: float
    length_gap: float
    cost: float
    matching_mode: str = "semantic"
    direct_claim_id: str = ""


@dataclass(frozen=True)
class CaseMatchBundle:
    """All matching outputs for one method on one case under one scenario.

    Attributes:
        uid: Case uid.
        method_name: Method name under evaluation.
        scenario_name: Matching perturbation scenario name.
        gold_rows: Gold claim rows for the case.
        prediction_rows_raw: Original prediction rows before deduplication.
        prediction_rows_deduped: Prediction rows after duplicate collapse.
        dedup_clusters: Duplicate-cluster metadata for predictions.
        matched_pairs: One-to-one matched claim pairs.
        unmatched_gold_rows: Gold rows left unmatched after assignment.
        unmatched_prediction_rows: Deduplicated prediction rows left unmatched.
        matching_mode: ``semantic`` or ``claim_id_direct``.
    """

    uid: str
    method_name: str
    scenario_name: str
    gold_rows: tuple[dict[str, Any], ...]
    prediction_rows_raw: tuple[dict[str, Any], ...]
    prediction_rows_deduped: tuple[dict[str, Any], ...]
    dedup_clusters: tuple[DedupCluster, ...]
    matched_pairs: tuple[MatchPair, ...]
    unmatched_gold_rows: tuple[dict[str, Any], ...]
    unmatched_prediction_rows: tuple[dict[str, Any], ...]
    matching_mode: str = "semantic"


def _copy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return dict(row)


def _case_uid(row: Mapping[str, Any]) -> str:
    return str(row.get("uid", "") or "")


def _claim_id(row: Mapping[str, Any]) -> str:
    return str(row.get("claim_id", "") or "")


def _is_gold_claim_id(value: str) -> bool:
    return str(value or "").startswith(GOLD_CLAIM_ID_PREFIX)


def _should_use_claim_id_direct_matching(
    *,
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if not gold_rows:
        return False

    if not all(_is_gold_claim_id(_claim_id(row)) for row in gold_rows):
        return False

    return all(_is_gold_claim_id(_claim_id(row)) for row in prediction_rows)


def claim_text(row: Mapping[str, Any], config: MatchingConfig) -> str:
    """Read and normalize the configured text field from one claim row.

    Args:
        row: Claim row from either gold or prediction data.
        config: Matching configuration that selects the text field.

    Returns:
        A whitespace-normalized claim text string.
    """

    value = row.get(config.text_field, "")

    if value is None:
        return ""

    return _normalize_space(str(value))


def require_claim_text(row: Mapping[str, Any], config: MatchingConfig) -> str:
    """Return non-empty normalized claim text or raise a descriptive error.

    Args:
        row: Claim row from either gold or prediction data.
        config: Matching configuration that selects the text field.

    Returns:
        A non-empty normalized claim text.

    Raises:
        ValueError: If the configured text field is empty after normalization.
    """

    text = claim_text(row, config)

    if text:
        return text

    raise ValueError(
        f"Missing non-empty `{config.text_field}` for uid={_case_uid(row)} claim_id={_claim_id(row)}"
    )


def _text_length(text: str, config: MatchingConfig) -> int:
    del config
    return len(_normalize_compact(text))


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0.0] = 1.0
    b_norm[b_norm == 0.0] = 1.0
    sim = (a / a_norm) @ (b / b_norm).T
    return np.clip(sim, -1.0, 1.0).astype(np.float32)


def _pairwise_cosine(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)

    vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    return _cosine_similarity_matrix(vector, matrix)[0]


def _encode_texts(
    texts: Sequence[str],
    *,
    encoder: TextEncoder,
    config: MatchingConfig,
    embedding_cache: dict[tuple[str, str], np.ndarray] | None = None,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    cache = embedding_cache if embedding_cache is not None else {}
    cache_keys = [(config.text_field, text) for text in texts]
    missing_texts = [text for key, text in zip(cache_keys, texts) if key not in cache]

    if missing_texts:
        encoded = np.asarray(encoder.encode(missing_texts), dtype=np.float32)

        if encoded.ndim != 2:
            raise ValueError("encoder.encode must return a 2D array")

        for text, vector in zip(missing_texts, encoded):
            cache[(config.text_field, text)] = np.asarray(vector, dtype=np.float32)

    return np.asarray([cache[key] for key in cache_keys], dtype=np.float32)


def _prepare_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        copied = _copy_row(row)

        if not _claim_id(copied):
            copied["claim_id"] = f"AUTO_PRED_{idx:03d}"

        prepared.append(copied)

    return prepared


def _validate_case_universe(
    rows: Sequence[Mapping[str, Any]],
    *,
    case_uids: Sequence[str],
) -> None:
    case_uid_set = set(case_uids)

    for row in rows:
        uid = _case_uid(row)

        if uid not in case_uid_set:
            raise ValueError(f"uid={uid} is not present in the provided case universe")


def _group_rows_by_uid(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        grouped.setdefault(_case_uid(row), []).append(_copy_row(row))

    return grouped


def _ensure_unique_case_claim_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    uid: str,
    row_kind: str,
) -> None:
    seen: set[str] = set()

    for row in rows:
        claim_id = _claim_id(row)

        if claim_id in seen:
            raise ValueError(f"Duplicate {row_kind} claim_id in uid={uid}: {claim_id}")

        seen.add(claim_id)


def deduplicate_prediction_rows(
    *,
    uid: str,
    method_name: str,
    rows: Sequence[Mapping[str, Any]],
    encoder: TextEncoder,
    config: MatchingConfig,
    embedding_cache: dict[tuple[str, str], np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], list[DedupCluster]]:
    """Collapse semantically duplicate prediction rows before matching.

    Args:
        uid: Case uid shared by the prediction rows.
        method_name: Method that produced the rows.
        rows: Raw prediction rows for one case.
        encoder: Text encoder used for duplicate clustering.
        config: Matching configuration controlling duplicate thresholds.
        embedding_cache: Optional reusable text-embedding cache.

    Returns:
        A pair of deduplicated prediction rows and duplicate-cluster metadata.
    """

    prepared_rows = _prepare_prediction_rows(rows)
    _ensure_unique_case_claim_ids(prepared_rows, uid=uid, row_kind="prediction")

    if not prepared_rows:
        return [], []

    texts = [require_claim_text(row, config) for row in prepared_rows]

    embeddings = _encode_texts(
        texts,
        encoder=encoder,
        config=config,
        embedding_cache=embedding_cache,
    )

    if len(prepared_rows) == 1:
        cluster = DedupCluster(
            cluster_id=f"{uid}:{method_name}:cluster_001",
            representative_claim_id=_claim_id(prepared_rows[0]),
            member_claim_ids=(_claim_id(prepared_rows[0]),),
            representative_row=_copy_row(prepared_rows[0]),
            member_rows=(_copy_row(prepared_rows[0]),),
            centroid_similarity=1.0,
        )

        return [_copy_row(prepared_rows[0])], [cluster]

    similarity = _cosine_similarity_matrix(embeddings, embeddings)
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")

    labels = fcluster(
        linkage_matrix,
        t=max(0.0, 1.0 - config.dedup_similarity_threshold),
        criterion="distance",
    )

    cluster_indices: dict[int, list[int]] = {}

    for idx, label in enumerate(labels):
        cluster_indices.setdefault(int(label), []).append(idx)

    deduped_rows: list[dict[str, Any]] = []
    clusters: list[DedupCluster] = []

    for cluster_number, label in enumerate(sorted(cluster_indices), start=1):
        member_idx = cluster_indices[label]
        member_rows = [prepared_rows[idx] for idx in member_idx]
        member_vectors = embeddings[member_idx]
        centroid = member_vectors.mean(axis=0)
        centroid_similarity = _pairwise_cosine(centroid, member_vectors)
        representative_pos = int(np.argmax(centroid_similarity))
        representative_idx = member_idx[representative_pos]
        representative_row = prepared_rows[representative_idx]
        representative_similarity = float(centroid_similarity[representative_pos])
        deduped_rows.append(_copy_row(representative_row))

        clusters.append(
            DedupCluster(
                cluster_id=f"{uid}:{method_name}:cluster_{cluster_number:03d}",
                representative_claim_id=_claim_id(representative_row),
                member_claim_ids=tuple(_claim_id(row) for row in member_rows),
                representative_row=_copy_row(representative_row),
                member_rows=tuple(_copy_row(row) for row in member_rows),
                centroid_similarity=representative_similarity,
            )
        )

    return deduped_rows, clusters


def _text_gap_for_rows(
    gold_row: Mapping[str, Any],
    prediction_row: Mapping[str, Any],
    config: MatchingConfig,
) -> float:
    gold_text = require_claim_text(gold_row, config)
    pred_text = require_claim_text(prediction_row, config)
    gold_length = _text_length(gold_text, config)
    pred_length = _text_length(pred_text, config)
    denom = max(gold_length, pred_length, 1)
    return float(abs(gold_length - pred_length) / denom)


def _match_case_claims_by_claim_id(
    *,
    uid: str,
    method_name: str,
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows_raw: Sequence[Mapping[str, Any]],
    config: MatchingConfig,
    scenario_name: str,
) -> CaseMatchBundle:
    gold_rows_copied = [_copy_row(row) for row in gold_rows]
    prediction_rows_copied = [_copy_row(row) for row in prediction_rows_raw]
    gold_by_id = {_claim_id(row): row for row in gold_rows_copied}
    prediction_by_id = {_claim_id(row): row for row in prediction_rows_copied}
    matched_ids = sorted(set(gold_by_id.keys()) & set(prediction_by_id.keys()))
    matched_pairs: list[MatchPair] = []

    for claim_id in matched_ids:
        gold_row = gold_by_id[claim_id]
        prediction_row = prediction_by_id[claim_id]
        gold_text = require_claim_text(gold_row, config)
        pred_text = require_claim_text(prediction_row, config)

        matched_pairs.append(
            MatchPair(
                uid=uid,
                method_name=method_name,
                gold_row=_copy_row(gold_row),
                prediction_row=_copy_row(prediction_row),
                semantic_similarity=1.0
                if _normalize_space(gold_text) == _normalize_space(pred_text)
                else 0.0,
                length_gap=_text_gap_for_rows(gold_row, prediction_row, config),
                cost=0.0,
                matching_mode="claim_id_direct",
                direct_claim_id=claim_id,
            )
        )

    unmatched_gold_rows = tuple(
        _copy_row(row)
        for claim_id, row in gold_by_id.items()
        if claim_id not in matched_ids
    )

    unmatched_prediction_rows = tuple(
        _copy_row(row)
        for claim_id, row in prediction_by_id.items()
        if claim_id not in matched_ids
    )

    return CaseMatchBundle(
        uid=uid,
        method_name=method_name,
        scenario_name=scenario_name,
        gold_rows=tuple(_copy_row(row) for row in gold_rows_copied),
        prediction_rows_raw=tuple(_copy_row(row) for row in prediction_rows_copied),
        prediction_rows_deduped=tuple(_copy_row(row) for row in prediction_rows_copied),
        dedup_clusters=(),
        matched_pairs=tuple(matched_pairs),
        unmatched_gold_rows=unmatched_gold_rows,
        unmatched_prediction_rows=unmatched_prediction_rows,
        matching_mode="claim_id_direct",
    )


def match_case_claims(
    *,
    uid: str,
    method_name: str,
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    encoder: TextEncoder,
    config: MatchingConfig,
    scenario_name: str = "base",
    embedding_cache: dict[tuple[str, str], np.ndarray] | None = None,
) -> CaseMatchBundle:
    """Match one case's gold claims against one method's prediction claims.

    Args:
        uid: Case uid shared by the rows.
        method_name: Method that produced the predictions.
        gold_rows: Gold claim rows for one case.
        prediction_rows: Prediction claim rows for the same case.
        encoder: Text encoder used for semantic similarity scoring.
        config: Matching configuration controlling thresholds and costs.
        scenario_name: Matching protocol scenario name.
        embedding_cache: Optional shared embedding cache.

    Returns:
        A case-level bundle containing deduplication, matching, and unmatched
        row diagnostics for the method/case pair.
    """

    gold_rows_copied = [_copy_row(row) for row in gold_rows]
    prediction_rows_raw = _prepare_prediction_rows(prediction_rows)
    _ensure_unique_case_claim_ids(gold_rows_copied, uid=uid, row_kind="gold")
    _ensure_unique_case_claim_ids(prediction_rows_raw, uid=uid, row_kind="prediction")

    if _should_use_claim_id_direct_matching(
        gold_rows=gold_rows_copied,
        prediction_rows=prediction_rows_raw,
    ):
        return _match_case_claims_by_claim_id(
            uid=uid,
            method_name=method_name,
            gold_rows=gold_rows_copied,
            prediction_rows_raw=prediction_rows_raw,
            config=config,
            scenario_name=scenario_name,
        )

    prediction_rows_deduped, dedup_clusters = deduplicate_prediction_rows(
        uid=uid,
        method_name=method_name,
        rows=prediction_rows_raw,
        encoder=encoder,
        config=config,
        embedding_cache=embedding_cache,
    )

    if not gold_rows_copied and not prediction_rows_deduped:
        return CaseMatchBundle(
            uid=uid,
            method_name=method_name,
            scenario_name=scenario_name,
            gold_rows=(),
            prediction_rows_raw=tuple(prediction_rows_raw),
            prediction_rows_deduped=(),
            dedup_clusters=tuple(dedup_clusters),
            matched_pairs=(),
            unmatched_gold_rows=(),
            unmatched_prediction_rows=(),
        )

    gold_texts = [require_claim_text(row, config) for row in gold_rows_copied]

    prediction_texts = [
        require_claim_text(row, config) for row in prediction_rows_deduped
    ]

    gold_embeddings = _encode_texts(
        gold_texts,
        encoder=encoder,
        config=config,
        embedding_cache=embedding_cache,
    )

    prediction_embeddings = _encode_texts(
        prediction_texts,
        encoder=encoder,
        config=config,
        embedding_cache=embedding_cache,
    )

    similarity = _cosine_similarity_matrix(gold_embeddings, prediction_embeddings)

    gold_lengths = np.asarray(
        [_text_length(text, config) for text in gold_texts], dtype=np.float32
    )

    prediction_lengths = np.asarray(
        [_text_length(text, config) for text in prediction_texts], dtype=np.float32
    )

    if gold_lengths.size and prediction_lengths.size:
        denom = np.maximum(gold_lengths[:, None], prediction_lengths[None, :])
        denom[denom == 0.0] = 1.0
        length_gap = np.abs(gold_lengths[:, None] - prediction_lengths[None, :]) / denom

    else:
        length_gap = np.zeros_like(similarity, dtype=np.float32)

    pair_cost = (
        config.semantic_weight * (1.0 - similarity)
        + config.length_penalty_weight * length_gap
    ).astype(np.float32)

    infeasible_mask = similarity < config.match_similarity_threshold
    pair_cost[infeasible_mask] = config.infeasible_cost
    gold_count = len(gold_rows_copied)
    prediction_count = len(prediction_rows_deduped)
    total_size = gold_count + prediction_count

    assignment_cost = np.full(
        (total_size, total_size), config.infeasible_cost, dtype=np.float32
    )

    if gold_count and prediction_count:
        assignment_cost[:gold_count, :prediction_count] = pair_cost

    for gold_idx in range(gold_count):
        assignment_cost[gold_idx, prediction_count + gold_idx] = config.unmatched_cost

    for pred_idx in range(prediction_count):
        assignment_cost[gold_count + pred_idx, pred_idx] = config.unmatched_cost

    if gold_count and prediction_count:
        assignment_cost[gold_count:, prediction_count:] = 0.0

    elif gold_count:
        assignment_cost[gold_count:, prediction_count:] = 0.0

    elif prediction_count:
        assignment_cost[gold_count:, prediction_count:] = 0.0

    row_ind, col_ind = linear_sum_assignment(assignment_cost)
    matched_pairs: list[MatchPair] = []
    matched_gold: set[int] = set()
    matched_predictions: set[int] = set()

    for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
        if row_idx < gold_count and col_idx < prediction_count:
            cost = float(pair_cost[row_idx, col_idx])

            if cost >= config.infeasible_cost:
                continue

            matched_gold.add(row_idx)
            matched_predictions.add(col_idx)

            matched_pairs.append(
                MatchPair(
                    uid=uid,
                    method_name=method_name,
                    gold_row=_copy_row(gold_rows_copied[row_idx]),
                    prediction_row=_copy_row(prediction_rows_deduped[col_idx]),
                    semantic_similarity=float(similarity[row_idx, col_idx]),
                    length_gap=float(length_gap[row_idx, col_idx]),
                    cost=cost,
                )
            )

    unmatched_gold_rows = tuple(
        _copy_row(row)
        for idx, row in enumerate(gold_rows_copied)
        if idx not in matched_gold
    )

    unmatched_prediction_rows = tuple(
        _copy_row(row)
        for idx, row in enumerate(prediction_rows_deduped)
        if idx not in matched_predictions
    )

    return CaseMatchBundle(
        uid=uid,
        method_name=method_name,
        scenario_name=scenario_name,
        gold_rows=tuple(_copy_row(row) for row in gold_rows_copied),
        prediction_rows_raw=tuple(_copy_row(row) for row in prediction_rows_raw),
        prediction_rows_deduped=tuple(
            _copy_row(row) for row in prediction_rows_deduped
        ),
        dedup_clusters=tuple(dedup_clusters),
        matched_pairs=tuple(matched_pairs),
        unmatched_gold_rows=unmatched_gold_rows,
        unmatched_prediction_rows=unmatched_prediction_rows,
        matching_mode="semantic",
    )


def build_method_case_matches(
    *,
    case_uids: Sequence[str],
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    method_name: str,
    encoder: TextEncoder,
    config: MatchingConfig,
    scenario_name: str = "base",
    embedding_cache: dict[tuple[str, str], np.ndarray] | None = None,
) -> list[CaseMatchBundle]:
    """Build per-case matching bundles for one method over a case universe.

    Args:
        case_uids: Ordered case universe to evaluate.
        gold_rows: Gold claim rows across the case universe.
        prediction_rows: Prediction rows for one method across the same cases.
        method_name: Method name associated with ``prediction_rows``.
        encoder: Text encoder used for semantic similarity scoring.
        config: Matching configuration controlling thresholds and costs.
        scenario_name: Matching perturbation scenario name.
        embedding_cache: Optional shared embedding cache reused across cases.

    Returns:
        Ordered case-level matching bundles aligned with ``case_uids``.
    """

    normalized_case_uids = tuple(str(uid) for uid in case_uids)
    _validate_case_universe(gold_rows, case_uids=normalized_case_uids)
    _validate_case_universe(prediction_rows, case_uids=normalized_case_uids)
    grouped_gold = _group_rows_by_uid(gold_rows)
    grouped_predictions = _group_rows_by_uid(prediction_rows)
    cache = embedding_cache if embedding_cache is not None else {}
    bundles: list[CaseMatchBundle] = []

    for uid in normalized_case_uids:
        bundle = match_case_claims(
            uid=uid,
            method_name=method_name,
            gold_rows=grouped_gold.get(uid, []),
            prediction_rows=grouped_predictions.get(uid, []),
            encoder=encoder,
            config=config,
            scenario_name=scenario_name,
            embedding_cache=cache,
        )

        bundles.append(bundle)

    return bundles


__all__ = [
    "CaseMatchBundle",
    "DedupCluster",
    "MatchPair",
    "MatchingConfig",
    "SentenceTransformerTextEncoder",
    "TextEncoder",
    "build_method_case_matches",
    "claim_text",
    "deduplicate_prediction_rows",
    "match_case_claims",
    "require_claim_text",
]
