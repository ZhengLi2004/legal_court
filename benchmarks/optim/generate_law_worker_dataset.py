"""Generate synthetic dataset for `law_worker_threshold` tuning.

Round-6 target:
- Parameter: `worker_threshold.law_worker_threshold`
- Runtime gate: `LawWorker` checks `max_score >= threshold`
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class LawDoc:
    """Synthetic law article record for retrieval simulation."""

    doc_id: str
    law_name: str
    article_id: str
    content: str
    domain: str


LAW_DOMAIN_SPECS: List[Dict[str, Any]] = [
    {
        "domain": "合同违约责任",
        "law_name": "中华人民共和国民法典",
        "article_prefix": "第五百七十七条",
        "core_terms": ["违约责任", "继续履行", "损害赔偿", "可得利益", "违约金调整"],
        "rules": [
            "当事人一方不履行合同义务或者履行不符合约定的，应当承担违约责任。",
            "违约损失赔偿范围包括合同履行后可得利益，但不得超过订立合同时可预见范围。",
            "当事人约定违约金的，人民法院可以根据实际损失审查是否明显过高或过低。",
        ],
    },
    {
        "domain": "合同解除与催告程序",
        "law_name": "中华人民共和国民法典",
        "article_prefix": "第五百六十三条",
        "core_terms": ["法定解除", "催告", "合理期限", "解除通知", "解除生效"],
        "rules": [
            "一方迟延履行主要债务，经催告后在合理期限内仍未履行的，对方可以解除合同。",
            "解除权行使应以通知方式作出，通知到达对方时解除发生效力。",
            "解除权的行使应符合诚实信用原则，不得滥用解除权损害交易安全。",
        ],
    },
    {
        "domain": "租赁合同与占用费",
        "law_name": "中华人民共和国民法典",
        "article_prefix": "第七百零九条",
        "core_terms": ["租赁合同", "租金支付", "腾退义务", "占有使用", "双倍占用费"],
        "rules": [
            "承租人应当按照约定支付租金并在租赁期限届满后返还租赁物。",
            "租赁物返还义务与实际占有状态有关，应结合经营活动与控制关系综合认定。",
            "主张高额占用费应有合同依据并符合比例原则，避免惩罚性失衡。",
        ],
    },
    {
        "domain": "借款合同与利息规则",
        "law_name": "中华人民共和国民法典",
        "article_prefix": "第六百八十条",
        "core_terms": ["借款合同", "利息约定", "逾期利息", "实际还款", "资金占用损失"],
        "rules": [
            "借款利息约定不得违反国家有关利率保护规则。",
            "借款人逾期还款的，应按照约定或法定规则承担逾期利息责任。",
            "已还款事实影响本金与利息计算，法院应结合流水与对账材料审查。",
        ],
    },
    {
        "domain": "举证责任与证明标准",
        "law_name": "中华人民共和国民事诉讼法",
        "article_prefix": "第六十四条",
        "core_terms": [
            "举证责任",
            "谁主张谁举证",
            "证明责任分配",
            "证据链",
            "高度盖然性",
        ],
        "rules": [
            "当事人对自己提出的主张，有责任提供证据。",
            "人民法院应结合证据的真实性、合法性、关联性进行综合判断。",
            "对于关键待证事实，证据不足的一方承担不利后果。",
        ],
    },
    {
        "domain": "证据规则与电子数据",
        "law_name": "最高人民法院关于民事诉讼证据的若干规定",
        "article_prefix": "第九十条",
        "core_terms": ["电子数据", "证据真实性", "证据保全", "形成时间", "完整性校验"],
        "rules": [
            "电子数据作为证据应当审查形成方式、存储介质和提取过程。",
            "对电子证据真实性有异议的，人民法院可结合校验信息与来源主体认定。",
            "证据保全过程合规性直接影响电子数据证明力。",
        ],
    },
    {
        "domain": "劳动合同与加班工资",
        "law_name": "中华人民共和国劳动合同法",
        "article_prefix": "第三十一条",
        "core_terms": ["劳动报酬", "加班工资", "考勤记录", "工时制度", "用人单位举证"],
        "rules": [
            "用人单位应按照国家规定支付劳动者加班工资。",
            "涉及加班事实争议的，考勤记录和排班记录是核心证据来源。",
            "实行特殊工时制度的，应当依法审批并明确适用边界。",
        ],
    },
    {
        "domain": "建设工程价款与结算",
        "law_name": "最高人民法院关于审理建设工程施工合同纠纷案件适用法律问题的解释（一）",
        "article_prefix": "第二十六条",
        "core_terms": ["工程价款", "工程量确认", "结算协议", "签证资料", "实际履行"],
        "rules": [
            "工程价款结算应以合同约定、工程签证和结算资料为基础。",
            "发包人对工程量提出异议的，应当在合理期限内提出并说明依据。",
            "未完成结算不当然否定工程价款请求，法院可结合履行事实认定。",
        ],
    },
    {
        "domain": "侵权责任与过错认定",
        "law_name": "中华人民共和国民法典",
        "article_prefix": "第一千一百六十五条",
        "core_terms": ["侵权责任", "过错", "损害后果", "因果关系", "过失相抵"],
        "rules": [
            "行为人因过错侵害他人民事权益造成损害的，应承担侵权责任。",
            "侵权责任成立应审查行为、损害后果、因果关系和过错要素。",
            "受害人对损害发生有过错的，可减轻侵权人责任。",
        ],
    },
    {
        "domain": "公司决议与代表权",
        "law_name": "中华人民共和国公司法",
        "article_prefix": "第十六条",
        "core_terms": ["公司决议", "代表权限", "内部授权", "对外效力", "善意相对人"],
        "rules": [
            "公司对外担保或重大处分事项通常应履行公司内部决议程序。",
            "法定代表人或授权代表越权行为的效力，应结合相对人善意与否判断。",
            "内部治理瑕疵不当然否定对外法律行为效力，应区分内部责任与外部效力。",
        ],
    },
]

POSITIVE_INTENT_TEMPLATES = [
    "请检索与“{topic}”直接相关的法律规定，梳理构成要件、适用边界和法律后果。",
    "围绕“{topic}”需要适用哪些法条，请明确认定标准与抗辩空间。",
    "针对“{topic}”提炼可直接用于庭审的法律依据，重点解释要件匹配路径。",
    "请从法条层面分析“{topic}”，并指出支持与限制该主张的规范条件。",
]

FACT_NEGATIVE_TEMPLATES = [
    "请按时间线整理以下案件事实并标注证据出处：{topic}，不要补充法条检索。",
    "请仅输出本案事实争议点与证据链缺口：{topic}，无需检索法律规定。",
    "围绕{topic}提炼事实矛盾与证人证言冲突，不需要法条说明。",
]

STRATEGY_NEGATIVE_TEMPLATES = [
    "请给出{topic}场景下的庭审发问策略，不需要新增法律检索。",
    "请制定{topic}争点的攻防顺序与表达结构，不要求引用法条。",
    "围绕{topic}设计对质提纲，仅输出论证节奏与回应策略。",
]

OOD_NEGATIVE_INTENTS = [
    "请设计一份四周减脂训练计划，并附上饮食安排。",
    "推荐三条适合周末亲子出行的城市路线。",
    "请给出短视频账号从0到1的运营节奏。",
    "如何系统学习摄影构图并完成入门作品集。",
    "请写一段关于春天和城市夜色的短诗。",
    "如何搭建个人知识管理系统并制定复盘流程。",
    "请比较三种家用咖啡机并给出购买建议。",
    "请提供英语口语训练的每日打卡计划。",
]

LAW_PRESENT_NO_RETRIEVAL_TEMPLATES = [
    "已确定适用《{law_name}》{article_id}，请将其改写为庭审口头表达，不要再检索法条。",
    "请基于既有法条结论（{law_name}{article_id}）写一段结案陈词，不需要新增法条。",
    "围绕已给定规则“{topic}”输出法官说理摘要，明确不进行法条检索。",
    "请把现有法律依据整理成PPT提纲（含{law_name}{article_id}），无需检索新的规定。",
]


def _normalize_text(text: str) -> str:
    raw = str(text or "").replace("\n", " ").strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw


def _extract_topic(spec: Dict[str, Any], rng: random.Random) -> str:
    term_a, term_b = rng.sample(spec["core_terms"], 2)
    return f"{spec['domain']}中的{term_a}与{term_b}争议"


def _build_article_content(spec: Dict[str, Any], idx: int, rng: random.Random) -> str:
    rule = spec["rules"][idx % len(spec["rules"])]
    term_a, term_b = rng.sample(spec["core_terms"], 2)

    return _normalize_text(
        f"{rule} 在{spec['domain']}中，司法审查通常围绕{term_a}与{term_b}展开，并结合具体履行过程进行要件匹配。"
    )


def build_synthetic_law_corpus(seed: int, per_domain: int = 12) -> List[LawDoc]:
    """Build synthetic law corpus used by Round-6 retrieval simulation."""
    rng = random.Random(seed)
    docs: List[LawDoc] = []
    cursor = 0

    for spec in LAW_DOMAIN_SPECS:
        for i in range(per_domain):
            cursor += 1
            article_id = f"{spec['article_prefix']}（模拟{str(i + 1).zfill(2)}）"
            content = _build_article_content(spec, i, rng)

            docs.append(
                LawDoc(
                    doc_id=f"LAW_{seed}_{str(cursor).zfill(4)}",
                    law_name=spec["law_name"],
                    article_id=article_id,
                    content=content,
                    domain=spec["domain"],
                )
            )

    return docs


def _build_queries(intent: str, topic: str, mode: str, rng: random.Random) -> List[str]:
    if mode == "positive":
        candidates = [
            intent,
            f"{topic} 的法律定义、构成要件与适用边界",
            f"{topic} 的法律后果与可行抗辩路径",
        ]

        if rng.random() < 0.4:
            candidates.append(f"{topic} 对应的法条依据与裁判认定标准")

    elif mode == "law_present_no_retrieval":
        candidates = [
            intent,
            f"{topic} 的口头表达与说理结构优化（不新增法条）",
            f"{topic} 的结案陈词组织方式（限定既有法条）",
        ]

    elif mode == "fact_only":
        candidates = [
            intent,
            f"{topic} 的事实时间线与证据对应关系",
            f"{topic} 的证据矛盾点与补强方向",
        ]

    elif mode == "strategy_only":
        candidates = [
            intent,
            f"{topic} 的庭审攻防顺序与提问策略",
            f"{topic} 的法庭表达结构与回应节奏",
        ]

    else:
        candidates = [
            intent,
            f"{intent} 的执行步骤",
            f"{intent} 的注意事项",
        ]

    dedup: List[str] = []
    seen = set()

    for query in candidates:
        normalized = _normalize_text(query)

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        dedup.append(normalized)

    return dedup[:3]


def _build_row(
    *,
    sample_id: str,
    seed: int,
    split: str,
    label: int,
    intent: str,
    queries: List[str],
    source: str,
    topic: str,
    negative_type: str | None = None,
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
        "topic": topic,
        "negative_type": negative_type,
        "boundary_hint": bool(boundary_hint),
        "review_flag": bool(boundary_hint),
    }


def _build_key_cases(rng: random.Random) -> List[Dict[str, Any]]:
    rows = [
        {
            "case_id": "KC_LWORKER_POS_001",
            "category": "positive",
            "label": 1,
            "description": "典型法条检索意图，期望 FOUND。",
            "topic": "违约金是否过高的司法调整标准",
            "intent": "请检索违约金是否过高的法律依据，说明构成要件与调整边界。",
            "mode": "positive",
        },
        {
            "case_id": "KC_LWORKER_POS_002",
            "category": "positive",
            "label": 1,
            "description": "解除程序与催告义务的法条检索意图，期望 FOUND。",
            "topic": "解除权行使是否必须先行催告",
            "intent": "围绕解除权行使中的催告程序，检索法律规定并提炼适用条件。",
            "mode": "positive",
        },
        {
            "case_id": "KC_LWORKER_NEG_003",
            "category": "negative",
            "label": 0,
            "description": "纯事实整理意图，期望 NOT_FOUND。",
            "topic": "租赁合同履行时间线",
            "intent": "请按时间线梳理租赁合同签订、催告和搬离事实，不需要法条检索。",
            "mode": "fact_only",
        },
        {
            "case_id": "KC_LWORKER_NEG_004",
            "category": "negative",
            "label": 0,
            "description": "给定法条后的写作改写意图，期望 NOT_FOUND。",
            "topic": "违约责任口头表达优化",
            "intent": "已确定适用《中华人民共和国民法典》第五百七十七条，请改写为庭审口头陈述，不要再检索法条。",
            "mode": "law_present_no_retrieval",
        },
        {
            "case_id": "KC_LWORKER_NEG_005",
            "category": "negative",
            "label": 0,
            "description": "非法律任务意图，期望 NOT_FOUND。",
            "topic": "周末出行路线推荐",
            "intent": "推荐三条适合周末亲子出行的城市路线。",
            "mode": "ood_non_legal",
        },
    ]

    for row in rows:
        row["queries"] = _build_queries(
            intent=row["intent"],
            topic=row["topic"],
            mode=row["mode"],
            rng=rng,
        )

    return rows


def generate_law_worker_dataset(
    *,
    seed: int,
    size: int = 600,
    valid_ratio: float = 0.2,
    positive_ratio: float = 0.45,
) -> Dict[str, Any]:
    """Generate one synthetic dataset for Round-6 tuning."""
    rng = random.Random(seed)
    corpus = build_synthetic_law_corpus(seed=seed, per_domain=12)
    positive_total = int(round(size * positive_ratio))
    positive_total = max(1, min(size - 1, positive_total))
    negative_total = size - positive_total
    neg_fact_total = int(round(negative_total * 0.35))
    neg_strategy_total = int(round(negative_total * 0.20))
    neg_ood_total = int(round(negative_total * 0.20))
    neg_hard_total = (
        negative_total - neg_fact_total - neg_strategy_total - neg_ood_total
    )
    rows: List[Dict[str, Any]] = []
    sample_idx = 0

    def next_sample_id() -> str:
        nonlocal sample_idx
        sample_idx += 1
        return f"LWORKER_{seed}_{str(sample_idx).zfill(4)}"

    def next_split() -> str:
        return "valid" if rng.random() < valid_ratio else "train"

    positive_boundary_total = int(round(positive_total * 0.22))
    positive_boundary_pick = set(
        rng.sample(range(positive_total), k=max(1, positive_boundary_total))
    )
    domain_pool = LAW_DOMAIN_SPECS.copy()

    for i in range(positive_total):
        spec = domain_pool[i % len(domain_pool)]
        topic = _extract_topic(spec, rng)
        intent = rng.choice(POSITIVE_INTENT_TEMPLATES).format(topic=topic)

        rows.append(
            _build_row(
                sample_id=next_sample_id(),
                seed=seed,
                split=next_split(),
                label=1,
                intent=intent,
                queries=_build_queries(
                    intent=intent, topic=topic, mode="positive", rng=rng
                ),
                source="synthetic_positive",
                topic=topic,
                boundary_hint=i in positive_boundary_pick,
            )
        )

    for _ in range(neg_fact_total):
        spec = rng.choice(LAW_DOMAIN_SPECS)
        topic = _extract_topic(spec, rng)
        intent = rng.choice(FACT_NEGATIVE_TEMPLATES).format(topic=topic)

        rows.append(
            _build_row(
                sample_id=next_sample_id(),
                seed=seed,
                split=next_split(),
                label=0,
                intent=intent,
                queries=_build_queries(
                    intent=intent, topic=topic, mode="fact_only", rng=rng
                ),
                source="synthetic_negative",
                topic=topic,
                negative_type="fact_only",
                boundary_hint=False,
            )
        )

    for _ in range(neg_strategy_total):
        spec = rng.choice(LAW_DOMAIN_SPECS)
        topic = _extract_topic(spec, rng)
        intent = rng.choice(STRATEGY_NEGATIVE_TEMPLATES).format(topic=topic)

        rows.append(
            _build_row(
                sample_id=next_sample_id(),
                seed=seed,
                split=next_split(),
                label=0,
                intent=intent,
                queries=_build_queries(
                    intent=intent, topic=topic, mode="strategy_only", rng=rng
                ),
                source="synthetic_negative",
                topic=topic,
                negative_type="strategy_only",
                boundary_hint=False,
            )
        )

    for i in range(neg_ood_total):
        intent = rng.choice(OOD_NEGATIVE_INTENTS)
        topic = "非法律任务"

        rows.append(
            _build_row(
                sample_id=next_sample_id(),
                seed=seed,
                split=next_split(),
                label=0,
                intent=intent,
                queries=_build_queries(
                    intent=intent, topic=topic, mode="ood_non_legal", rng=rng
                ),
                source="synthetic_negative",
                topic=topic,
                negative_type="ood_non_legal",
                boundary_hint=i < max(1, neg_ood_total // 5),
            )
        )

    hard_boundary_total = max(1, int(round(neg_hard_total * 0.7)))
    for i in range(neg_hard_total):
        spec = rng.choice(LAW_DOMAIN_SPECS)
        topic = _extract_topic(spec, rng)
        article_id = f"{spec['article_prefix']}（示例）"

        intent = rng.choice(LAW_PRESENT_NO_RETRIEVAL_TEMPLATES).format(
            law_name=spec["law_name"], article_id=article_id, topic=topic
        )

        rows.append(
            _build_row(
                sample_id=next_sample_id(),
                seed=seed,
                split=next_split(),
                label=0,
                intent=intent,
                queries=_build_queries(
                    intent=intent, topic=topic, mode="law_present_no_retrieval", rng=rng
                ),
                source="synthetic_negative",
                topic=topic,
                negative_type="law_present_no_retrieval",
                boundary_hint=i < hard_boundary_total,
            )
        )

    rng.shuffle(rows)

    if len(rows) != size:
        raise RuntimeError(f"Dataset size mismatch: expected {size}, got {len(rows)}")

    key_cases = _build_key_cases(rng)

    for case in key_cases:
        case["seed"] = seed

    summary = {
        "size": len(rows),
        "positive_count": sum(1 for row in rows if int(row["label"]) == 1),
        "negative_count": sum(1 for row in rows if int(row["label"]) == 0),
        "positive_ratio": round(
            sum(1 for row in rows if int(row["label"]) == 1) / len(rows), 4
        ),
        "boundary_ratio": round(
            sum(1 for row in rows if row["boundary_hint"]) / len(rows), 4
        ),
        "hard_negative_ratio": round(
            sum(
                1
                for row in rows
                if row.get("negative_type") == "law_present_no_retrieval"
            )
            / len(rows),
            4,
        ),
        "ood_negative_ratio": round(
            sum(1 for row in rows if row.get("negative_type") == "ood_non_legal")
            / len(rows),
            4,
        ),
        "corpus_size": len(corpus),
    }

    return {
        "rows": rows,
        "summary": summary,
        "key_cases": key_cases,
        "law_corpus": [doc.__dict__ for doc in corpus],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=92)
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--positive-ratio", type=float, default=0.45)

    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("benchmarks/optim/artifacts/law_worker_dataset_seed92.jsonl"),
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            "benchmarks/optim/artifacts/law_worker_dataset_seed92.summary.json"
        ),
    )

    parser.add_argument(
        "--corpus-json",
        type=Path,
        default=Path("benchmarks/optim/artifacts/law_worker_corpus_seed92.json"),
    )

    parser.add_argument(
        "--key-cases-json",
        type=Path,
        default=Path("benchmarks/optim/artifacts/law_worker_key_cases_seed92.json"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = generate_law_worker_dataset(
        seed=args.seed,
        size=args.size,
        valid_ratio=args.valid_ratio,
        positive_ratio=args.positive_ratio,
    )

    _write_jsonl(args.out_jsonl, payload["rows"])
    _write_json(args.summary_json, payload["summary"])
    _write_json(args.corpus_json, payload["law_corpus"])
    _write_json(args.key_cases_json, payload["key_cases"])

    print(
        "[law-worker-dataset] done "
        f"seed={args.seed} size={payload['summary']['size']} corpus={payload['summary']['corpus_size']}"
    )


if __name__ == "__main__":
    main()
