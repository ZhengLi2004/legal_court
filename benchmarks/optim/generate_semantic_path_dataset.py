"""Generate mixed ES+synthetic dataset for semantic-path threshold tuning.

Target parameter:
- `retrieval.semantic_min_similarity`

Runtime path alignment:
- query text: current case context
- candidate text: historical case context
- score: cosine similarity of runtime embedding model
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv(override=True)
SOURCE_TYPE_BOOK = "book"
SOURCE_TYPE_CASES = "cases"
SOURCE_TYPE_JUDGE = "judge"

POSITIVE_QUERY_TEMPLATES = [
    "当前案件围绕“{topic}”，对方主张事实链不完整，需要检索高度相似历史事实背景。",
    "本案争议核心是“{topic}”，请优先匹配可直接复用的历史事实认定路径。",
    "针对“{topic}”事实冲突，需要召回事实结构相近的既往案例。",
]

SYNTHETIC_POSITIVE_PAIRS = [
    (
        "2013年2月1日，原告与被告签订厂房租赁合同，约定厂房用于生产、办公、存货。",
        "原被告于2013年签署厂房租赁协议，约定承租用途为生产经营及办公仓储。",
    ),
    (
        "对方未按期支付租金后，原告曾多次书面催告并保留送达凭证。",
        "被告逾期欠租，出租方连续发出催告通知并形成完整送达记录。",
    ),
    (
        "被告主张已搬离但其工商登记地址与持续经营记录仍指向涉案厂房。",
        "承租方虽称已迁出，工商登记与经营流水仍显示其持续占用涉案场地。",
    ),
    (
        "原告提交聊天记录、对账单和邮件往来，形成连续证据链证明违约事实。",
        "出租方以微信记录、往来邮件和对账材料相互印证被告违约过程。",
    ),
]

SYNTHETIC_HARD_NEGATIVE_PAIRS = [
    (
        "本案争议在租赁合同中违约金是否过高以及催告程序是否完备。",
        "争议焦点在建设工程结算金额核减与签证手续是否真实。",
    ),
    (
        "被告抗辩未实际使用厂房，不应承担双倍占用费。",
        "劳动争议中员工主张加班工资，关键在考勤记录真实性。",
    ),
    (
        "原告主张解除通知已送达并据此要求腾退。",
        "公司担保效力争议需审查股东会决议与代表权限范围。",
    ),
]

OOD_NEGATIVE_PAIRS = [
    (
        "请给出一份四周减脂训练计划。",
        "推荐三款适合入门的家用咖啡机。",
    ),
    (
        "如何系统学习摄影构图并输出作品集？",
        "请写一段关于春天夜色的短诗。",
    ),
    (
        "短视频账号从0到1的运营节奏是什么？",
        "请安排一次周末亲子出行路线。",
    ),
]


@dataclass(frozen=True)
class ESDoc:
    """Minimal ES case payload for semantic pair generation."""

    doc_id: str
    source_type: str
    case_title: str
    focus: str
    facts: str
    analysis: str


def _normalize_text(text: str) -> str:
    raw = str(text or "").replace("\n", " ").strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw


def _first_sentence(text: str, max_len: int = 100) -> str:
    cleaned = _normalize_text(text)

    if not cleaned:
        return ""

    splitter = re.split(r"[。；!?！？]", cleaned)
    candidate = splitter[0].strip() if splitter else cleaned

    if not candidate:
        candidate = cleaned

    return candidate[:max_len]


def _extract_topic(doc: ESDoc) -> str:
    for text in (doc.focus, doc.facts, doc.analysis, doc.case_title):
        s = _first_sentence(text, max_len=72)

        if s:
            return s

    return "合同履行事实争议"


def _build_case_context(doc: ESDoc) -> str:
    parts = []

    for text in (doc.facts, doc.focus, doc.analysis):
        sentence = _first_sentence(text, max_len=180)

        if sentence:
            parts.append(sentence)

    if not parts:
        parts = [_normalize_text(doc.case_title) or "案件事实争议"]

    return _normalize_text("。".join(parts))


def _build_positive_query(topic: str, rng: random.Random) -> str:
    return _normalize_text(rng.choice(POSITIVE_QUERY_TEMPLATES).format(topic=topic))


def _build_row(
    *,
    sample_id: str,
    seed: int,
    split: str,
    label: int,
    source: str,
    query_context: str,
    candidate_context: str,
    pair_type: str,
    negative_type: Optional[str] = None,
    boundary_hint: bool = False,
    source_type: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "seed": seed,
        "split": split,
        "label": int(label),
        "auto_label": int(label),
        "source": source,
        "source_type": source_type,
        "doc_id": doc_id,
        "query_context": _normalize_text(query_context),
        "candidate_context": _normalize_text(candidate_context),
        "pair_type": pair_type,
        "negative_type": negative_type,
        "boundary_hint": bool(boundary_hint),
        "review_flag": bool(boundary_hint),
    }


def _fetch_docs_by_source_type(
    *,
    es: Elasticsearch,
    source_type: str,
    count: int,
    seed: int,
) -> List[ESDoc]:
    body = {
        "size": count,
        "query": {
            "function_score": {
                "query": {"term": {"source_type": source_type}},
                "random_score": {"seed": int(seed), "field": "_seq_no"},
            }
        },
        "_source": ["source_type", "case_title", "focus", "facts", "analysis"],
    }

    response = es.search(index="rag_legal_cases", body=body, request_timeout=30)
    hits = response.get("hits", {}).get("hits", [])
    docs: List[ESDoc] = []

    for hit in hits:
        source = hit.get("_source", {})
        facts = _normalize_text(source.get("facts", ""))
        focus = _normalize_text(source.get("focus", ""))
        analysis = _normalize_text(source.get("analysis", ""))

        if not facts and not focus and not analysis:
            continue

        docs.append(
            ESDoc(
                doc_id=str(hit.get("_id", "")),
                source_type=str(source.get("source_type", source_type)),
                case_title=_normalize_text(source.get("case_title", "")),
                focus=focus,
                facts=facts,
                analysis=analysis,
            )
        )

    return docs


def _build_key_cases() -> List[Dict[str, Any]]:
    return [
        {
            "case_id": "KC_SEM_POS_001",
            "category": "positive",
            "label": 1,
            "description": "租赁争议事实链同构，应召回。",
            "query_context": "原告主张被告欠租并持续占用厂房，已多次催告仍未腾退。",
            "candidate_context": "被告在租赁到期后继续占有厂房，出租方反复催告并请求腾退及占用费。",
        },
        {
            "case_id": "KC_SEM_POS_002",
            "category": "positive",
            "label": 1,
            "description": "证据链争议事实高度相关，应召回。",
            "query_context": "原告提交聊天记录和对账单证明对方违约，但被告主张证据断裂。",
            "candidate_context": "出租方提交微信记录与对账凭证证明违约事实，对方抗辩证据链不完整。",
        },
        {
            "case_id": "KC_SEM_NEG_003",
            "category": "negative",
            "label": 0,
            "description": "跨案由高词面重叠，语义不应召回。",
            "query_context": "租赁合同中欠租与腾退争议，核心在持续占用事实。",
            "candidate_context": "建设工程结算争议围绕签证单和工程量核减，不涉及租赁占用事实。",
        },
        {
            "case_id": "KC_SEM_NEG_004",
            "category": "negative",
            "label": 0,
            "description": "完全域外文本，不应召回。",
            "query_context": "如何系统学习摄影构图并完成作品集？",
            "candidate_context": "推荐三条周末亲子出行路线。",
        },
    ]


def generate_semantic_path_dataset(
    *,
    seed: int,
    size: int = 600,
    valid_ratio: float = 0.2,
    positive_ratio: float = 0.45,
    es_host: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one mixed ES+synthetic dataset for semantic-path threshold tuning."""
    rng = random.Random(seed)
    resolved_host = es_host or os.getenv("ES_HOST", "http://127.0.0.1:9200")
    es = Elasticsearch(resolved_host, request_timeout=30)

    if not es.ping():
        raise RuntimeError(f"Elasticsearch not reachable at {resolved_host}")

    positive_total = int(round(size * positive_ratio))
    positive_total = min(max(1, positive_total), size - 1)
    negative_total = size - positive_total
    es_positive_total = int(round(positive_total * 0.7))
    synthetic_positive_total = positive_total - es_positive_total
    neg_cross_total = int(round(negative_total * 0.45))
    neg_hard_total = int(round(negative_total * 0.35))
    neg_ood_total = negative_total - neg_cross_total - neg_hard_total

    source_cfg = [
        (SOURCE_TYPE_BOOK, max(100, int(size * 0.22)), seed * 13 + 1),
        (SOURCE_TYPE_CASES, max(80, int(size * 0.18)), seed * 13 + 3),
        (SOURCE_TYPE_JUDGE, max(120, int(size * 0.25)), seed * 13 + 5),
    ]

    es_docs: List[ESDoc] = []

    for source_type, quota, random_seed in source_cfg:
        docs = _fetch_docs_by_source_type(
            es=es,
            source_type=source_type,
            count=quota,
            seed=random_seed,
        )

        es_docs.extend(docs)

    if len(es_docs) < 20:
        raise RuntimeError(
            "Insufficient ES documents for semantic-path dataset generation."
        )

    rows: List[Dict[str, Any]] = []
    cursor = 0

    def next_split() -> str:
        return "valid" if rng.random() < valid_ratio else "train"

    def next_id() -> str:
        nonlocal cursor
        cursor += 1
        return f"SEMPATH_{seed}_{cursor:04d}"

    for _ in range(es_positive_total):
        doc = rng.choice(es_docs)
        topic = _extract_topic(doc)
        query_text = _build_positive_query(topic, rng)
        candidate_context = _build_case_context(doc)

        rows.append(
            _build_row(
                sample_id=next_id(),
                seed=seed,
                split=next_split(),
                label=1,
                source="es_positive",
                source_type=doc.source_type,
                doc_id=doc.doc_id,
                query_context=query_text,
                candidate_context=candidate_context,
                pair_type="positive_es",
                boundary_hint=(rng.random() < 0.20),
            )
        )

    for _ in range(synthetic_positive_total):
        q_text, c_text = rng.choice(SYNTHETIC_POSITIVE_PAIRS)

        rows.append(
            _build_row(
                sample_id=next_id(),
                seed=seed,
                split=next_split(),
                label=1,
                source="synthetic_positive",
                query_context=q_text,
                candidate_context=c_text,
                pair_type="positive_synthetic",
                boundary_hint=(rng.random() < 0.18),
            )
        )

    for _ in range(neg_cross_total):
        left = rng.choice(es_docs)
        right = rng.choice(es_docs)
        retry = 0

        while right.source_type == left.source_type and retry < 5:
            right = rng.choice(es_docs)
            retry += 1

        query_text = _build_positive_query(_extract_topic(left), rng)
        candidate_context = _build_case_context(right)

        rows.append(
            _build_row(
                sample_id=next_id(),
                seed=seed,
                split=next_split(),
                label=0,
                source="es_negative",
                source_type=right.source_type,
                doc_id=right.doc_id,
                query_context=query_text,
                candidate_context=candidate_context,
                pair_type="negative_cross_doc",
                negative_type="cross_doc",
                boundary_hint=(rng.random() < 0.30),
            )
        )

    for _ in range(neg_hard_total):
        q_text, c_text = rng.choice(SYNTHETIC_HARD_NEGATIVE_PAIRS)

        rows.append(
            _build_row(
                sample_id=next_id(),
                seed=seed,
                split=next_split(),
                label=0,
                source="synthetic_negative",
                query_context=q_text,
                candidate_context=c_text,
                pair_type="negative_hard",
                negative_type="hard_semantic",
                boundary_hint=True,
            )
        )

    for _ in range(neg_ood_total):
        q_text, c_text = rng.choice(OOD_NEGATIVE_PAIRS)

        rows.append(
            _build_row(
                sample_id=next_id(),
                seed=seed,
                split=next_split(),
                label=0,
                source="synthetic_negative",
                query_context=q_text,
                candidate_context=c_text,
                pair_type="negative_ood",
                negative_type="ood",
                boundary_hint=False,
            )
        )

    rng.shuffle(rows)

    if len(rows) != size:
        raise ValueError(f"Dataset size mismatch: expected {size}, got {len(rows)}")

    pos_count = sum(1 for row in rows if int(row["label"]) == 1)
    boundary_count = sum(1 for row in rows if bool(row["boundary_hint"]))
    cross_neg = sum(1 for row in rows if row.get("negative_type") == "cross_doc")
    hard_neg = sum(1 for row in rows if row.get("negative_type") == "hard_semantic")
    ood_neg = sum(1 for row in rows if row.get("negative_type") == "ood")

    return {
        "seed": seed,
        "rows": rows,
        "key_cases": _build_key_cases(),
        "summary": {
            "size": size,
            "positive_count": pos_count,
            "negative_count": size - pos_count,
            "positive_ratio": pos_count / max(1, size),
            "boundary_ratio": boundary_count / max(1, size),
            "cross_doc_negative_ratio": cross_neg / max(1, size),
            "hard_negative_ratio": hard_neg / max(1, size),
            "ood_negative_ratio": ood_neg / max(1, size),
            "es_doc_pool_size": len(es_docs),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--positive-ratio", type=float, default=0.45)
    parser.add_argument("--es-host", type=str, default=None)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/semantic_path_seed112.jsonl"),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/semantic_path_seed112.summary.json"),
    )

    parser.add_argument(
        "--key-cases-output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/semantic_path_key_cases_seed112.json"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = generate_semantic_path_dataset(
        seed=args.seed,
        size=args.size,
        valid_ratio=args.valid_ratio,
        positive_ratio=args.positive_ratio,
        es_host=args.es_host,
    )

    _write_jsonl(args.output, payload["rows"])
    _write_json(args.summary_output, payload["summary"])
    _write_json(args.key_cases_output, payload["key_cases"])
    print("[semantic-path-dataset] done")
    print(f"seed={args.seed}")
    print(f"size={payload['summary']['size']}")
    print(f"positive_ratio={payload['summary']['positive_ratio']:.4f}")
    print(f"boundary_ratio={payload['summary']['boundary_ratio']:.4f}")
    print(f"es_doc_pool_size={payload['summary']['es_doc_pool_size']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
