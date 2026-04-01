"""Generate mixed ES+synthetic dataset for `law_jaccard_min_similarity` tuning.

Round-9 target:
- Parameter: `retrieval.law_jaccard_min_similarity`
- Runtime gate: `LegalGMemory.retrieve_cases_by_law_codes` uses `sim > threshold`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv(override=True)
LAW_NAME_PATTERN = re.compile(r"《([^》]{2,80})》")

ARTICLE_PATTERN = re.compile(
    r"第[一二三四五六七八九十百千万零两0-9]{1,12}条(?:第[一二三四五六七八九十百千万零两0-9]{1,8}款)?"
)

GENERIC_PROCEDURAL_LAWS = [
    "中华人民共和国民事诉讼法 第六十四条",
    "最高人民法院关于民事诉讼证据的若干规定 第九十条",
    "中华人民共和国民事诉讼法 第一百七十条",
]

DOMAIN_LAW_SPECS: List[Dict[str, Any]] = [
    {
        "domain": "合同违约责任",
        "laws": [
            "中华人民共和国民法典 第五百七十七条",
            "中华人民共和国民法典 第五百八十四条",
            "中华人民共和国民法典 第五百八十五条",
            "中华人民共和国民法典 第五百六十三条",
        ],
    },
    {
        "domain": "租赁合同纠纷",
        "laws": [
            "中华人民共和国民法典 第七百零三条",
            "中华人民共和国民法典 第七百零九条",
            "中华人民共和国民法典 第七百一十条",
            "中华人民共和国民法典 第七百一十一条",
        ],
    },
    {
        "domain": "借款合同纠纷",
        "laws": [
            "中华人民共和国民法典 第六百六十七条",
            "中华人民共和国民法典 第六百七十六条",
            "中华人民共和国民法典 第六百八十条",
            "中华人民共和国民法典 第六百八十二条",
        ],
    },
    {
        "domain": "建设工程纠纷",
        "laws": [
            "民法典合同编通则司法解释 第二十条",
            "建设工程施工合同司法解释（一） 第二十六条",
            "中华人民共和国民法典 第七百九十三条",
            "中华人民共和国民法典 第八百零六条",
        ],
    },
    {
        "domain": "公司治理与代表权",
        "laws": [
            "中华人民共和国公司法 第十六条",
            "中华人民共和国公司法 第四十三条",
            "中华人民共和国公司法 第四十六条",
            "中华人民共和国民法典 第一百七十一条",
        ],
    },
]


@dataclass(frozen=True)
class LawSetDoc:
    """Case-like law set unit for pair construction.

    Attributes:
        doc_id: Source document identifier.
        source: High-level origin label such as ``es`` or ``synthetic``.
        source_type: ES source partition when the document comes from ES.
        laws: Canonicalized law-token set extracted from the document.
    """

    doc_id: str
    source: str
    source_type: str
    laws: Set[str]


@dataclass(frozen=True)
class PairSample:
    """One binary threshold sample.

    Attributes:
        sample_id: Stable sample identifier.
        seed: Random seed used to build the sample.
        split: Dataset split label.
        label: Gold binary label for the pair.
        pair_type: Pair construction recipe name.
        source: High-level source label such as ``es`` or ``synthetic``.
        query_laws: Canonicalized query-side law tokens.
        candidate_laws: Canonicalized candidate-side law tokens.
        jaccard: Precomputed Jaccard overlap score.
        overlap_count: Size of the intersection between the two law sets.
        union_count: Size of the union between the two law sets.
        negative_type: Optional negative subtype annotation.
        boundary_hint: Whether the sample lies near the intended threshold.
    """

    sample_id: str
    seed: int
    split: str
    label: int
    pair_type: str
    source: str
    query_laws: List[str]
    candidate_laws: List[str]
    jaccard: float
    overlap_count: int
    union_count: int
    negative_type: Optional[str]
    boundary_hint: bool


def _normalize_text(text: str) -> str:
    raw = str(text or "").replace("\n", " ").strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw


def _normalize_law_name(name: str) -> str:
    cleaned = _normalize_text(name)
    cleaned = cleaned.replace("中华人民共和国", "中华人民共和国")
    cleaned = (
        cleaned.replace("民法典", "中华人民共和国民法典")
        if cleaned == "民法典"
        else cleaned
    )
    return cleaned


def _normalize_law_token(name: str, article: Optional[str]) -> str:
    law_name = _normalize_law_name(name)

    if article:
        article_text = _normalize_text(article)
        return f"{law_name} {article_text}".strip()

    return law_name


def _extract_law_tokens_from_citation(text: str) -> Set[str]:
    tokens: Set[str] = set()
    raw = _normalize_text(text)

    if not raw:
        return tokens

    law_matches = list(LAW_NAME_PATTERN.finditer(raw))

    if not law_matches:
        return tokens

    for idx, match in enumerate(law_matches):
        law_name = match.group(1)
        start = match.end()
        end = law_matches[idx + 1].start() if idx + 1 < len(law_matches) else len(raw)
        segment = raw[start:end]
        articles = ARTICLE_PATTERN.findall(segment)

        if not articles:
            token = _normalize_law_token(law_name, None)

            if len(token) >= 6:
                tokens.add(token)

            continue

        for article in articles[:4]:
            token = _normalize_law_token(law_name, article)

            if len(token) >= 8:
                tokens.add(token)

    return tokens


def _extract_doc_law_set(source: Dict[str, Any]) -> Set[str]:
    citations = source.get("citations")
    laws: Set[str] = set()

    if isinstance(citations, list):
        for entry in citations:
            laws.update(_extract_law_tokens_from_citation(str(entry)))

    elif isinstance(citations, str):
        laws.update(_extract_law_tokens_from_citation(citations))

    if len(laws) < 2:
        fallback_text = " ".join(
            [
                str(source.get("focus", "")),
                str(source.get("facts", "")),
                str(source.get("analysis", "")),
            ]
        )

        laws.update(_extract_law_tokens_from_citation(fallback_text))

    if len(laws) > 6:
        laws = set(sorted(laws)[:6])

    return laws


def _fetch_es_docs(
    *,
    es: Elasticsearch,
    source_type: str,
    count: int,
    seed: int,
) -> List[LawSetDoc]:
    if count <= 0:
        return []

    body = {
        "size": count,
        "query": {
            "function_score": {
                "query": {"term": {"source_type": source_type}},
                "random_score": {"seed": int(seed), "field": "_seq_no"},
            }
        },
        "_source": ["source_type", "focus", "facts", "analysis", "citations"],
    }

    response = es.search(index="rag_legal_cases", body=body, request_timeout=30)
    hits = response.get("hits", {}).get("hits", [])
    docs: List[LawSetDoc] = []

    for hit in hits:
        source = hit.get("_source", {})
        laws = _extract_doc_law_set(source)

        if len(laws) < 2:
            continue

        docs.append(
            LawSetDoc(
                doc_id=str(hit.get("_id", "")),
                source="es",
                source_type=str(source.get("source_type", source_type)),
                laws=laws,
            )
        )

    return docs


def _build_synthetic_doc_pool(seed: int, per_domain: int = 40) -> List[LawSetDoc]:
    rng = random.Random(seed)
    docs: List[LawSetDoc] = []
    cursor = 0

    for spec in DOMAIN_LAW_SPECS:
        domain_laws = list(spec["laws"])

        for _ in range(per_domain):
            cursor += 1
            size = rng.randint(2, 4)
            selected = set(rng.sample(domain_laws, k=min(size, len(domain_laws))))

            if rng.random() < 0.45:
                selected.add(rng.choice(GENERIC_PROCEDURAL_LAWS))

            docs.append(
                LawSetDoc(
                    doc_id=f"SYN_{seed}_{cursor:04d}",
                    source="synthetic",
                    source_type=str(spec["domain"]),
                    laws=selected,
                )
            )

    return docs


def _jaccard(a: Set[str], b: Set[str]) -> float:
    union = a.union(b)

    if not union:
        return 0.0

    return len(a.intersection(b)) / len(union)


def _as_sorted_list(values: Set[str]) -> List[str]:
    return sorted(values)


def _build_pair_row(
    *,
    sample_id: str,
    seed: int,
    split: str,
    label: int,
    pair_type: str,
    source: str,
    query_laws: Set[str],
    candidate_laws: Set[str],
    negative_type: Optional[str],
    boundary_hint: bool,
) -> Dict[str, Any]:
    inter = query_laws.intersection(candidate_laws)
    union = query_laws.union(candidate_laws)
    sim = _jaccard(query_laws, candidate_laws)

    return {
        "sample_id": sample_id,
        "seed": seed,
        "split": split,
        "label": int(label),
        "auto_label": int(label),
        "pair_type": pair_type,
        "source": source,
        "query_laws": _as_sorted_list(query_laws),
        "candidate_laws": _as_sorted_list(candidate_laws),
        "jaccard": sim,
        "overlap_count": len(inter),
        "union_count": len(union),
        "negative_type": negative_type,
        "boundary_hint": bool(boundary_hint),
        "review_flag": bool(boundary_hint),
    }


def _build_query_from_doc(base_laws: Set[str], rng: random.Random) -> Set[str]:
    laws = list(base_laws)
    q_size = rng.randint(2, min(4, len(laws))) if len(laws) >= 2 else len(laws)
    query = set(rng.sample(laws, k=max(1, q_size)))

    if rng.random() < 0.35:
        query.add(rng.choice(GENERIC_PROCEDURAL_LAWS))

    return query


def _make_positive_candidate(query: Set[str], rng: random.Random) -> Set[str]:
    candidate = set(query)

    if len(candidate) > 2 and rng.random() < 0.35:
        candidate.remove(rng.choice(list(candidate)))

    if rng.random() < 0.30:
        extra_pool = []

        for spec in DOMAIN_LAW_SPECS:
            extra_pool.extend(spec["laws"])

        candidate.add(rng.choice(extra_pool + GENERIC_PROCEDURAL_LAWS))

    return set(sorted(candidate))


def _make_negative_candidate(
    query: Set[str],
    rng: random.Random,
    mode: str,
) -> Tuple[Set[str], str]:
    all_domain_laws: List[str] = []

    for spec in DOMAIN_LAW_SPECS:
        all_domain_laws.extend(spec["laws"])

    if mode == "hard_generic_overlap":
        candidate = {rng.choice(GENERIC_PROCEDURAL_LAWS)}
        pool = [law for law in all_domain_laws if law not in query]
        candidate.update(rng.sample(pool, k=min(2, len(pool))))
        return candidate, "hard_generic_overlap"

    if mode == "cross_domain_overlap":
        overlap = set(rng.sample(list(query), k=1)) if query else set()
        pool = [law for law in all_domain_laws if law not in query]
        candidate = set(overlap)
        candidate.update(rng.sample(pool, k=min(3, len(pool))))
        return candidate, "cross_domain_overlap"

    pool = [law for law in all_domain_laws if law not in query]
    candidate = set(rng.sample(pool, k=min(3, len(pool))))
    return candidate, "no_overlap"


def _key_cases() -> List[Dict[str, Any]]:
    return [
        {
            "case_id": "KC_LJACCARD_POS_001",
            "category": "positive",
            "label": 1,
            "description": "高重合法条集，应当召回。",
            "query_laws": [
                "中华人民共和国民法典 第五百七十七条",
                "中华人民共和国民法典 第五百八十四条",
                "中华人民共和国民事诉讼法 第六十四条",
            ],
            "candidate_laws": [
                "中华人民共和国民法典 第五百七十七条",
                "中华人民共和国民法典 第五百八十四条",
                "中华人民共和国民事诉讼法 第六十四条",
            ],
        },
        {
            "case_id": "KC_LJACCARD_POS_002",
            "category": "positive",
            "label": 1,
            "description": "中等重合法条集，应保持召回。",
            "query_laws": [
                "中华人民共和国民法典 第七百零九条",
                "中华人民共和国民法典 第七百一十条",
                "中华人民共和国民事诉讼法 第六十四条",
            ],
            "candidate_laws": [
                "中华人民共和国民法典 第七百零九条",
                "中华人民共和国民事诉讼法 第六十四条",
                "中华人民共和国民法典 第五百六十三条",
            ],
        },
        {
            "case_id": "KC_LJACCARD_NEG_003",
            "category": "negative",
            "label": 0,
            "description": "仅共享程序性法条，属于误召噪声。",
            "query_laws": [
                "中华人民共和国民法典 第五百七十七条",
                "中华人民共和国民法典 第五百八十五条",
                "中华人民共和国民事诉讼法 第六十四条",
            ],
            "candidate_laws": [
                "中华人民共和国公司法 第十六条",
                "中华人民共和国公司法 第四十六条",
                "中华人民共和国民事诉讼法 第六十四条",
            ],
        },
        {
            "case_id": "KC_LJACCARD_NEG_004",
            "category": "negative",
            "label": 0,
            "description": "完全无重合，必须过滤。",
            "query_laws": [
                "中华人民共和国民法典 第六百六十七条",
                "中华人民共和国民法典 第六百八十条",
            ],
            "candidate_laws": [
                "中华人民共和国公司法 第十六条",
                "中华人民共和国公司法 第四十三条",
            ],
        },
        {
            "case_id": "KC_LJACCARD_BOUNDARY_005",
            "category": "boundary",
            "label": 0,
            "description": "边界等值样例（验证 sim==threshold 时不通过）。",
            "query_laws": [
                "中华人民共和国民法典 第五百七十七条",
                "中华人民共和国民法典 第五百八十四条",
                "中华人民共和国民法典 第五百八十五条",
            ],
            "candidate_laws": [
                "中华人民共和国民法典 第五百七十七条",
                "中华人民共和国公司法 第十六条",
                "中华人民共和国公司法 第四十三条",
            ],
        },
    ]


def generate_law_jaccard_dataset(
    *,
    seed: int,
    size: int = 600,
    valid_ratio: float = 0.2,
    positive_ratio: float = 0.45,
    es_host: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one mixed ES+synthetic Round-9 dataset.

    Args:
        seed: Random seed controlling ES sampling and synthetic generation.
        size: Total row count to generate.
        valid_ratio: Validation split ratio.
        positive_ratio: Share of positive rows in the final dataset.
        es_host: Optional Elasticsearch host override.

    Returns:
        Dataset payload containing rows, key cases, and summary statistics.

    Raises:
        RuntimeError: If Elasticsearch is unreachable or no law-set documents
            can be sampled.
        ValueError: If the constructed dataset size is inconsistent.
    """
    rng = random.Random(seed)
    resolved_host = es_host or os.getenv("ES_HOST", "http://127.0.0.1:9200")
    es = Elasticsearch(resolved_host, request_timeout=30)

    if not es.ping():
        raise RuntimeError(f"Elasticsearch not reachable at {resolved_host}")

    positive_total = int(round(size * positive_ratio))
    positive_total = min(max(1, positive_total), size - 1)
    negative_total = size - positive_total
    es_pool: List[LawSetDoc] = []

    source_cfg = [
        ("book", max(40, int(size * 0.20)), seed * 17 + 1),
        ("cases", max(30, int(size * 0.15)), seed * 17 + 3),
        ("judge", max(40, int(size * 0.20)), seed * 17 + 5),
    ]

    for source_type, count, random_seed in source_cfg:
        es_pool.extend(
            _fetch_es_docs(
                es=es,
                source_type=source_type,
                count=count,
                seed=random_seed,
            )
        )

    if not es_pool:
        raise RuntimeError(
            "No ES law-set documents were extracted from rag_legal_cases."
        )

    synthetic_pool = _build_synthetic_doc_pool(seed=seed + 1000, per_domain=38)
    combined_pool = es_pool + synthetic_pool
    rows: List[Dict[str, Any]] = []
    cursor = 0

    def next_split() -> str:
        return "valid" if rng.random() < valid_ratio else "train"

    def next_id() -> str:
        nonlocal cursor
        cursor += 1
        return f"LJACCARD_{seed}_{cursor:04d}"

    es_pos_total = int(round(positive_total * 0.65))
    syn_pos_total = positive_total - es_pos_total

    for _ in range(es_pos_total):
        base = rng.choice(es_pool)
        query = _build_query_from_doc(base.laws, rng)
        cand = _make_positive_candidate(query, rng)

        row = _build_pair_row(
            sample_id=next_id(),
            seed=seed,
            split=next_split(),
            label=1,
            pair_type="positive_es",
            source="es",
            query_laws=query,
            candidate_laws=cand,
            negative_type=None,
            boundary_hint=(0.16 <= _jaccard(query, cand) <= 0.26),
        )

        rows.append(row)

    for _ in range(syn_pos_total):
        base = rng.choice(synthetic_pool)
        query = _build_query_from_doc(base.laws, rng)
        cand = _make_positive_candidate(query, rng)

        row = _build_pair_row(
            sample_id=next_id(),
            seed=seed,
            split=next_split(),
            label=1,
            pair_type="positive_synthetic",
            source="synthetic",
            query_laws=query,
            candidate_laws=cand,
            negative_type=None,
            boundary_hint=(0.16 <= _jaccard(query, cand) <= 0.26),
        )

        rows.append(row)

    hard_neg_total = int(round(negative_total * 0.45))
    cross_neg_total = int(round(negative_total * 0.35))
    no_overlap_total = negative_total - hard_neg_total - cross_neg_total

    for _ in range(hard_neg_total):
        base = rng.choice(combined_pool)
        query = _build_query_from_doc(base.laws, rng)
        cand, ntype = _make_negative_candidate(query, rng, mode="hard_generic_overlap")

        row = _build_pair_row(
            sample_id=next_id(),
            seed=seed,
            split=next_split(),
            label=0,
            pair_type="negative_hard_generic",
            source=base.source,
            query_laws=query,
            candidate_laws=cand,
            negative_type=ntype,
            boundary_hint=(0.08 <= _jaccard(query, cand) <= 0.20),
        )

        rows.append(row)

    for _ in range(cross_neg_total):
        base = rng.choice(combined_pool)
        query = _build_query_from_doc(base.laws, rng)
        cand, ntype = _make_negative_candidate(query, rng, mode="cross_domain_overlap")

        row = _build_pair_row(
            sample_id=next_id(),
            seed=seed,
            split=next_split(),
            label=0,
            pair_type="negative_cross_domain",
            source=base.source,
            query_laws=query,
            candidate_laws=cand,
            negative_type=ntype,
            boundary_hint=(0.08 <= _jaccard(query, cand) <= 0.20),
        )

        rows.append(row)

    for _ in range(no_overlap_total):
        base = rng.choice(combined_pool)
        query = _build_query_from_doc(base.laws, rng)
        cand, ntype = _make_negative_candidate(query, rng, mode="no_overlap")

        row = _build_pair_row(
            sample_id=next_id(),
            seed=seed,
            split=next_split(),
            label=0,
            pair_type="negative_no_overlap",
            source=base.source,
            query_laws=query,
            candidate_laws=cand,
            negative_type=ntype,
            boundary_hint=False,
        )

        rows.append(row)

    rng.shuffle(rows)

    if len(rows) != size:
        raise ValueError(f"Dataset size mismatch: expected {size}, got {len(rows)}")

    boundary_count = sum(1 for row in rows if row["boundary_hint"])
    pos_count = sum(1 for row in rows if int(row["label"]) == 1)

    hard_neg = sum(
        1 for row in rows if row.get("negative_type") == "hard_generic_overlap"
    )

    cross_neg = sum(
        1 for row in rows if row.get("negative_type") == "cross_domain_overlap"
    )

    no_overlap_neg = sum(1 for row in rows if row.get("negative_type") == "no_overlap")
    es_source_count = sum(1 for row in rows if row["source"] == "es")

    return {
        "seed": seed,
        "rows": rows,
        "key_cases": _key_cases(),
        "summary": {
            "size": size,
            "positive_count": pos_count,
            "negative_count": size - pos_count,
            "positive_ratio": pos_count / max(1, size),
            "boundary_ratio": boundary_count / max(1, size),
            "es_source_ratio": es_source_count / max(1, size),
            "hard_generic_negative_ratio": hard_neg / max(1, size),
            "cross_domain_negative_ratio": cross_neg / max(1, size),
            "no_overlap_negative_ratio": no_overlap_neg / max(1, size),
            "es_pool_size": len(es_pool),
            "synthetic_pool_size": len(synthetic_pool),
            "es_host": resolved_host,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for law-Jaccard dataset generation.

    Returns:
        Parsed command-line namespace for dataset sizes and output paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=102)
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--positive-ratio", type=float, default=0.45)
    parser.add_argument("--es-host", type=str, default=None)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/law_jaccard_seed102.jsonl"),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/law_jaccard_seed102.summary.json"),
    )

    parser.add_argument(
        "--key-cases-output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/law_jaccard_key_cases_seed102.json"),
    )

    return parser.parse_args()


def main() -> None:
    """Generate the ES-backed law-set overlap dataset.

    Raises:
        RuntimeError: If Elasticsearch is unreachable or no law-set documents
            can be sampled.
        ValueError: If the generated dataset size is inconsistent.
    """
    args = parse_args()

    payload = generate_law_jaccard_dataset(
        seed=args.seed,
        size=args.size,
        valid_ratio=args.valid_ratio,
        positive_ratio=args.positive_ratio,
        es_host=args.es_host,
    )

    _write_jsonl(args.output, payload["rows"])
    _write_json(args.summary_output, payload["summary"])
    _write_json(args.key_cases_output, payload["key_cases"])
    print("[law-jaccard-dataset] done")
    print(f"seed={args.seed}")
    print(f"size={payload['summary']['size']}")
    print(f"positive_ratio={payload['summary']['positive_ratio']:.4f}")
    print(f"es_source_ratio={payload['summary']['es_source_ratio']:.4f}")
    print(f"boundary_ratio={payload['summary']['boundary_ratio']:.4f}")
    print(f"es_pool_size={payload['summary']['es_pool_size']}")
    print(f"output={args.output}")
    print(f"summary={args.summary_output}")
    print(f"key_cases={args.key_cases_output}")


if __name__ == "__main__":
    main()
