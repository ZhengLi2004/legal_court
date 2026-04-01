"""Generate ES-backed + synthetic dataset for `fact_worker_threshold` tuning.

Round-5 target:
- Parameter: `worker_threshold.fact_worker_threshold`
- Runtime gate: `FactWorker` checks `max_score >= threshold`
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

OOD_NEGATIVE_INTENTS = [
    "推荐一份健身增肌训练计划并安排每周饮食。",
    "如何在旅行中节约酒店和交通成本，请给出详细建议。",
    "请生成一份家庭烘焙蛋糕食谱并说明步骤。",
    "如何提升公开演讲技巧和临场表达能力。",
    "短视频账号增长运营策略有哪些。",
    "如何搭建个人博客并做SEO优化。",
    "请写一首关于春天的现代诗。",
    "电影剪辑软件推荐以及入门学习路线。",
    "数据库索引优化的一般方法是什么。",
    "手机摄影构图与后期调色技巧。",
]

LAW_CENTRIC_NEGATIVE_TEMPLATES = [
    "请解释{law_hint}的构成要件与法律后果，不需要检索具体案件事实。",
    "请概括{law_hint}在司法适用中的理论争议，仅需法理分析。",
    "围绕{law_hint}给出条文释义与学理观点，不涉及个案事实比对。",
    "请总结{law_hint}的法条体系位置与规范目的，无需案例检索。",
]

ABSTRACT_NEGATIVE_INTENTS = [
    "请给出辩论攻防的通用策略模板，不限定具体事实或争议焦点。",
    "在不依赖案件事实的前提下，总结法庭陈述的表达结构。",
    "请提供通用庭审沟通技巧，不需要历史案例事实支持。",
    "请输出一套抽象诉讼策略框架，不限定案由与证据链。",
]

SYNTHETIC_POSITIVE_TOPICS = [
    "出租人未有效催告即解除合同后主张双倍占用费的事实认定",
    "借款本金返还与逾期利息争议中的实际还款事实认定",
    "交通事故挂靠车辆责任承担中的车辆控制关系认定",
    "建设工程价款拖欠纠纷中的工程量与结算事实认定",
    "买卖合同质量异议中的验收流程与瑕疵通知事实认定",
    "劳动争议中加班事实与工时记录真实性认定",
]

POSITIVE_INTENT_TEMPLATES = [
    "请检索与“{topic}”相关的历史案例事实认定模式，并关注关键证据链。",
    "围绕“{topic}”检索可比案例，提取事实争点与法院认定路径。",
    "针对“{topic}”查找历史案例中支持或反驳该事实主张的证据结构。",
    "请结合“{topic}”检索相似案例，重点关注事实成立条件与反证要点。",
]

NEGATIVE_LAW_HINTS = [
    "民法典第五百七十七条违约责任",
    "民法典第五百六十三条合同解除",
    "民法典第一百五十三条民事法律行为效力",
    "民法典第六百八十条借款合同利息规则",
    "民诉法第六十四条举证责任分配",
]


@dataclass(frozen=True)
class ESDoc:
    """Minimal ES case document fields used for intent generation.

    Attributes:
        doc_id: Elasticsearch document identifier.
        source_type: ES source partition such as ``book`` or ``judge``.
        case_title: Case title text.
        focus: Focus-section text.
        facts: Fact-section text.
        analysis: Analysis-section text.
    """

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


def _first_sentence(text: str, max_len: int = 72) -> str:
    cleaned = _normalize_text(text)

    if not cleaned:
        return ""

    splitter = re.split(r"[。；!?！？]", cleaned)
    candidate = splitter[0].strip() if splitter else cleaned

    if not candidate:
        candidate = cleaned

    return candidate[:max_len]


def _extract_topic(doc: ESDoc) -> str:
    for candidate in (doc.focus, doc.facts, doc.analysis, doc.case_title):
        topic = _first_sentence(candidate, max_len=72)

        if topic:
            return topic

    return "合同履行与争议事实认定"


def _build_positive_intent(topic: str, rng: random.Random) -> str:
    template = rng.choice(POSITIVE_INTENT_TEMPLATES)
    return template.format(topic=topic)


def _topic_from_intent(intent: str) -> str:
    cleaned = _normalize_text(intent)
    cleaned = re.sub(r"[“”\"']", "", cleaned)
    return cleaned[:48] if cleaned else "案件事实争议"


def build_queries(intent: str, rng: random.Random) -> List[str]:
    """Build 2-3 deterministic query variants for one intent.

    Args:
        intent: Source retrieval intent.
        rng: Random generator controlling optional query expansion.

    Returns:
        Deduplicated query variants aligned with runtime query decomposition.
    """
    topic = _topic_from_intent(intent)

    candidates = [
        _normalize_text(intent),
        f"{topic} 的事实争点与证据链认定案例",
        f"围绕 {topic} 的裁判事实基础与抗辩要点",
    ]

    if rng.random() < 0.45:
        candidates.append(f"{topic} 相关法院事实认定路径与证据标准")

    dedup: List[str] = []
    seen = set()

    for query in candidates:
        if not query or query in seen:
            continue

        seen.add(query)
        dedup.append(query)

    return dedup[:3]


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
        "_source": ["case_title", "focus", "facts", "analysis", "source_type"],
    }

    response = es.search(index="rag_legal_cases", body=body, request_timeout=30)
    hits = response.get("hits", {}).get("hits", [])
    docs: List[ESDoc] = []

    for hit in hits:
        source = hit.get("_source", {})

        docs.append(
            ESDoc(
                doc_id=str(hit.get("_id", "")),
                source_type=str(source.get("source_type", source_type)),
                case_title=_normalize_text(source.get("case_title", "")),
                focus=_normalize_text(source.get("focus", "")),
                facts=_normalize_text(source.get("facts", "")),
                analysis=_normalize_text(source.get("analysis", "")),
            )
        )

    return docs


def _build_row(
    *,
    sample_id: str,
    seed: int,
    label: int,
    split: str,
    intent: str,
    queries: List[str],
    source: str,
    source_type: Optional[str] = None,
    doc_id: Optional[str] = None,
    negative_type: Optional[str] = None,
    boundary_hint: bool = False,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "seed": seed,
        "split": split,
        "label": int(label),
        "auto_label": int(label),
        "intent": intent,
        "queries": queries,
        "source": source,
        "source_type": source_type,
        "doc_id": doc_id,
        "negative_type": negative_type,
        "boundary_hint": bool(boundary_hint),
        "review_flag": bool(boundary_hint),
    }


def _key_cases() -> List[Dict[str, Any]]:
    rng = random.Random(123)

    rows = [
        {
            "case_id": "KC_FWORKER_POS_001",
            "category": "positive",
            "label": 1,
            "description": "事实导向租赁争议意图，期望 FOUND。",
            "intent": "在租赁合同纠纷中，原告未有效催告即解除合同后主张双倍占用费，需检索事实认定路径和证据链。",
        },
        {
            "case_id": "KC_FWORKER_POS_002",
            "category": "positive",
            "label": 1,
            "description": "借款纠纷事实认定意图，期望 FOUND。",
            "intent": "原告主张返还借款本金并请求逾期利息，被告抗辩已部分清偿，请检索可比事实认定案例。",
        },
        {
            "case_id": "KC_FWORKER_NEG_003",
            "category": "negative",
            "label": 0,
            "description": "纯法条解释意图，期望 NOT_FOUND。",
            "intent": "请解释民法典第五百七十七条的构成要件与规范目的，不需要检索案件事实。",
        },
        {
            "case_id": "KC_FWORKER_NEG_004",
            "category": "negative",
            "label": 0,
            "description": "非法律任务意图，期望 NOT_FOUND。",
            "intent": "如何提升公开演讲技巧并优化表达结构。",
        },
        {
            "case_id": "KC_FWORKER_NEG_005",
            "category": "negative",
            "label": 0,
            "description": "抽象策略意图缺少事实锚点，期望 NOT_FOUND。",
            "intent": "请给出通用诉讼策略框架，不限定具体事实争点或证据对象。",
        },
    ]

    for row in rows:
        row["queries"] = build_queries(row["intent"], rng)

    return rows


def generate_fact_worker_es_dataset(
    *,
    seed: int,
    size: int = 600,
    valid_ratio: float = 0.2,
    positive_ratio: float = 0.45,
    es_host: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one mixed ES+synthetic dataset for fact-worker threshold tuning.

    Args:
        seed: Random seed controlling ES sampling and synthetic negatives.
        size: Total row count to generate.
        valid_ratio: Validation split ratio.
        positive_ratio: Share of positive rows in the final dataset.
        es_host: Optional Elasticsearch host override.

    Returns:
        Dataset payload containing rows, key cases, and summary statistics.

    Raises:
        RuntimeError: If Elasticsearch is unreachable or no positive ES
            documents can be sampled.
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
    es_positive_total = int(round(positive_total * 0.8))
    synthetic_positive_total = positive_total - es_positive_total
    negative_law_total = int(round(negative_total * 0.45))
    negative_ood_total = int(round(negative_total * 0.35))
    negative_abstract_total = negative_total - negative_law_total - negative_ood_total
    book_quota = int(round(es_positive_total * 0.50))
    cases_quota = int(round(es_positive_total * 0.35))
    judge_quota = es_positive_total - book_quota - cases_quota

    source_cfg = [
        (SOURCE_TYPE_BOOK, book_quota, seed * 11 + 1),
        (SOURCE_TYPE_CASES, cases_quota, seed * 11 + 3),
        (SOURCE_TYPE_JUDGE, judge_quota, seed * 11 + 5),
    ]

    es_positive_docs: List[ESDoc] = []

    for source_type, quota, random_seed in source_cfg:
        if quota <= 0:
            continue

        docs = _fetch_docs_by_source_type(
            es=es,
            source_type=source_type,
            count=max(quota, 1),
            seed=random_seed,
        )

        es_positive_docs.extend(docs[:quota])

    if not es_positive_docs:
        raise RuntimeError("No ES documents sampled for positive dataset construction.")

    rows: List[Dict[str, Any]] = []
    cursor = 0

    def next_split() -> str:
        return "valid" if rng.random() < valid_ratio else "train"

    def next_sample_id() -> str:
        nonlocal cursor
        cursor += 1
        return f"FWORKER_{seed}_{cursor:04d}"

    for i in range(es_positive_total):
        doc = es_positive_docs[i % len(es_positive_docs)]
        topic = _extract_topic(doc)
        intent = _build_positive_intent(topic, rng)
        queries = build_queries(intent, rng)

        rows.append(
            _build_row(
                sample_id=next_sample_id(),
                seed=seed,
                label=1,
                split=next_split(),
                intent=intent,
                queries=queries,
                source="es_positive",
                source_type=doc.source_type,
                doc_id=doc.doc_id,
                boundary_hint=(rng.random() < 0.25),
            )
        )

    for _ in range(synthetic_positive_total):
        topic = rng.choice(SYNTHETIC_POSITIVE_TOPICS)
        intent = _build_positive_intent(topic, rng)
        queries = build_queries(intent, rng)

        rows.append(
            _build_row(
                sample_id=next_sample_id(),
                seed=seed,
                label=1,
                split=next_split(),
                intent=intent,
                queries=queries,
                source="synthetic_positive",
                boundary_hint=(rng.random() < 0.20),
            )
        )

    for _ in range(negative_law_total):
        law_hint = rng.choice(NEGATIVE_LAW_HINTS)
        intent = rng.choice(LAW_CENTRIC_NEGATIVE_TEMPLATES).format(law_hint=law_hint)
        queries = build_queries(intent, rng)

        rows.append(
            _build_row(
                sample_id=next_sample_id(),
                seed=seed,
                label=0,
                split=next_split(),
                intent=intent,
                queries=queries,
                source="synthetic_negative",
                negative_type="law_centric",
                boundary_hint=True,
            )
        )

    for _ in range(negative_ood_total):
        intent = rng.choice(OOD_NEGATIVE_INTENTS)
        queries = build_queries(intent, rng)

        rows.append(
            _build_row(
                sample_id=next_sample_id(),
                seed=seed,
                label=0,
                split=next_split(),
                intent=intent,
                queries=queries,
                source="synthetic_negative",
                negative_type="ood_non_case",
                boundary_hint=False,
            )
        )

    for _ in range(negative_abstract_total):
        intent = rng.choice(ABSTRACT_NEGATIVE_INTENTS)
        queries = build_queries(intent, rng)

        rows.append(
            _build_row(
                sample_id=next_sample_id(),
                seed=seed,
                label=0,
                split=next_split(),
                intent=intent,
                queries=queries,
                source="synthetic_negative",
                negative_type="abstract_strategy",
                boundary_hint=True,
            )
        )

    rng.shuffle(rows)

    if len(rows) != size:
        raise ValueError(f"Dataset size mismatch: expected {size}, got {len(rows)}")

    boundary_count = sum(1 for row in rows if row["boundary_hint"])
    pos_count = sum(1 for row in rows if row["label"] == 1)
    es_count = sum(1 for row in rows if row["source"] == "es_positive")
    law_neg = sum(1 for row in rows if row.get("negative_type") == "law_centric")
    ood_neg = sum(1 for row in rows if row.get("negative_type") == "ood_non_case")

    abstract_neg = sum(
        1 for row in rows if row.get("negative_type") == "abstract_strategy"
    )

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
            "es_source_ratio": es_count / max(1, size),
            "law_centric_negative_ratio": law_neg / max(1, size),
            "ood_negative_ratio": ood_neg / max(1, size),
            "abstract_negative_ratio": abstract_neg / max(1, size),
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
    """Parse CLI arguments for fact-worker retrieval dataset generation.

    Returns:
        Parsed command-line namespace for dataset sizes and output paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--positive-ratio", type=float, default=0.45)
    parser.add_argument("--es-host", type=str, default=None)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/fact_worker_seed82.jsonl"),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/fact_worker_seed82.summary.json"),
    )

    parser.add_argument(
        "--key-cases-output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/fact_worker_key_cases.seed82.json"),
    )

    return parser.parse_args()


def main() -> None:
    """Generate the ES-backed fact-worker retrieval dataset.

    Raises:
        RuntimeError: If Elasticsearch is unreachable or source documents are
            unavailable.
        ValueError: If the generated dataset size is inconsistent.
    """
    args = parse_args()

    payload = generate_fact_worker_es_dataset(
        seed=args.seed,
        size=args.size,
        valid_ratio=args.valid_ratio,
        positive_ratio=args.positive_ratio,
        es_host=args.es_host,
    )

    _write_jsonl(args.output, payload["rows"])
    _write_json(args.summary_output, payload["summary"])
    _write_json(args.key_cases_output, payload["key_cases"])
    print("[fact-worker-dataset] done")
    print(f"seed={args.seed}")
    print(f"size={payload['summary']['size']}")
    print(f"positive_ratio={payload['summary']['positive_ratio']:.4f}")
    print(f"es_source_ratio={payload['summary']['es_source_ratio']:.4f}")
    print(f"boundary_ratio={payload['summary']['boundary_ratio']:.4f}")
    print(f"output={args.output}")
    print(f"summary={args.summary_output}")
    print(f"key_cases={args.key_cases_output}")


if __name__ == "__main__":
    main()
