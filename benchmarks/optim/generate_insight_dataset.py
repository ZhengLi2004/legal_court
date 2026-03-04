"""Generate synthetic AIGC-style insight datasets for threshold tuning.

All samples are self-constructed and follow actionable strategy language style:
- Trigger condition
- Tactical action
- Proof/legal goal
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

SIDE_P = "PLAINTIFF"
SIDE_D = "DEFENDANT"
SIDE_C = "COMMON"


@dataclass(frozen=True)
class InsightScenario:
    """One dispute scenario used to synthesize strategy insights."""

    issue_id: str
    domain: str
    context_facts: List[str]
    plaintiff_trigger: List[str]
    plaintiff_action: List[str]
    plaintiff_goal: List[str]
    defendant_trigger: List[str]
    defendant_action: List[str]
    defendant_goal: List[str]
    common_trigger: List[str]
    common_action: List[str]
    common_goal: List[str]


SCENARIOS: List[InsightScenario] = [
    InsightScenario(
        issue_id="lease_liquidated_damage",
        domain="lease",
        context_facts=[
            "争议聚焦于违约金标准是否过高及是否应当调减。",
            "双方均围绕实际损失举证完整性展开攻防。",
            "合同条款明确约定了逾期违约责任。",
        ],
        plaintiff_trigger=[
            "对方主张违约金过高",
            "被告请求法院按LPR下调违约金",
            "对方强调约定违约金明显超过实际损失",
        ],
        plaintiff_action=[
            "主动举证资金占用成本与履约机会损失",
            "同步提交催告记录与履约受阻证据",
            "逐项量化逾期期间的损失构成并固定证据链",
        ],
        plaintiff_goal=[
            "证明约定标准与损失规模具有对应关系以维持条款效力",
            "支撑违约金标准未显著过高的法律结论",
            "增强法院对约定违约责任可执行性的确信",
        ],
        defendant_trigger=[
            "原告主张按高比例违约金全额支持",
            "对方仅以合同文字主张违约金不应调整",
            "原告回避实际损失举证",
        ],
        defendant_action=[
            "要求原告具体举证实际损失并核验计算口径",
            "提交同类交易资金成本对照材料",
            "强调原告减损义务履行不足并提出比例调减方案",
        ],
        defendant_goal=[
            "证明违约金与实际损失明显不匹配以促成调减",
            "削弱高额违约金请求的合理性基础",
            "将裁判重心引导至实际损失与公平衡量",
        ],
        common_trigger=[
            "双方围绕违约金是否过高产生对抗",
            "案件进入违约金调整审查阶段",
        ],
        common_action=[
            "优先构建可核验的损失计算明细并同步提交原始凭证",
            "将履约过程、催告记录与损失变化做时间线化呈现",
        ],
        common_goal=[
            "避免法院因事实不清而降低说理可信度",
            "提升损失与责任条款之间的可解释性",
        ],
    ),
    InsightScenario(
        issue_id="lease_occupancy_and_operation",
        domain="lease",
        context_facts=[
            "对方以未实际使用或已搬离抗辩租金支付义务。",
            "争议焦点在于是否形成持续占用并经营事实。",
            "双方对工商登记与业务流水证据证明力存在分歧。",
        ],
        plaintiff_trigger=[
            "对方以未实际使用厂房抗辩租金责任",
            "被告主张已搬离不再承担占用费用",
            "对方否认持续经营事实",
        ],
        plaintiff_action=[
            "重点调取工商登记地址变更记录与业务流水",
            "串联门禁记录、用电数据与交易往来证据",
            "建立占用期间的连续经营事实时间线",
        ],
        plaintiff_goal=[
            "证明其持续占用并经营从而支撑租金与占用费请求",
            "削弱未实际使用抗辩的事实基础",
            "形成可闭环的占用控制证据链",
        ],
        defendant_trigger=[
            "原告以登记地址直接推定持续占用",
            "对方以间接经营信息主张全部期间占用",
            "原告将零散业务记录解释为持续经营",
        ],
        defendant_action=[
            "提交搬离通知、交接清单与场地移交证据",
            "区分登记信息与实际控制状态的时间差",
            "对原告证据中无法对应具体期间的部分提出排除意见",
        ],
        defendant_goal=[
            "证明占用期间与责任期间并不一致以限制给付范围",
            "动摇持续经营推定的证明链稳定性",
            "降低占用费与租金请求的支持区间",
        ],
        common_trigger=[
            "双方围绕实际占用状态发生冲突",
            "案件需判断登记信息与实际控制状态的一致性",
        ],
        common_action=[
            "同步核验客观运行数据与场地交接材料",
            "将证据按期间切片映射到具体占用主张",
        ],
        common_goal=[
            "避免以单一证据替代完整占用事实认定",
            "提升占用责任裁判的期间精度",
        ],
    ),
    InsightScenario(
        issue_id="electronic_evidence_chain",
        domain="evidence",
        context_facts=[
            "双方围绕电子数据是否被篡改发生争议。",
            "证据来源、提取过程与存储完整性是核心审查点。",
            "对方主张聊天记录截取片段不能反映完整事实。",
        ],
        plaintiff_trigger=[
            "对方质疑电子证据来源不明",
            "被告主张聊天记录存在截取和编辑风险",
            "对方攻击证据链条连续性",
        ],
        plaintiff_action=[
            "补充提交原始载体提取记录与哈希校验信息",
            "说明取证流程并固定关键节点见证材料",
            "将电子记录与线下交易痕迹做交叉印证",
        ],
        plaintiff_goal=[
            "证明电子数据形成与流转过程完整可信",
            "提升电子证据可采性并稳定证明力",
            "阻断对方对证据完整性的系统性攻击",
        ],
        defendant_trigger=[
            "原告仅提交截屏而未提供原始载体",
            "对方以单一电子记录推导核心事实",
            "原告未说明数据提取与保全过程",
        ],
        defendant_action=[
            "要求核验原始设备与提取日志",
            "重点攻击时间戳连续性与元数据一致性",
            "提出电子证据与客观行为不一致的反证材料",
        ],
        defendant_goal=[
            "削弱电子证据可采性并降低其证明强度",
            "阻断对方以不完整电子记录建立关键事实",
            "将争议引回举证责任与证明标准审查",
        ],
        common_trigger=[
            "案件核心证据类型为电子数据",
            "双方均依赖电子记录主张关键事实",
        ],
        common_action=[
            "优先梳理取证链路并补齐原始数据校验信息",
            "建立电子证据与客观外部证据的对应关系",
        ],
        common_goal=[
            "降低电子证据因链条缺口被否定的风险",
            "提升事实认定的可复核性",
        ],
    ),
    InsightScenario(
        issue_id="contract_termination_service",
        domain="contract",
        context_facts=[
            "一方主张解除通知送达有效，另一方主张程序瑕疵。",
            "争议焦点包括送达对象、送达地址与授权范围。",
            "双方对解除生效时间存在明显分歧。",
        ],
        plaintiff_trigger=[
            "对方以送达程序瑕疵否定解除效力",
            "被告主张签收人无授权",
            "对方攻击解除通知生效时间",
        ],
        plaintiff_action=[
            "补充提交签收链路、送达回证与授权材料",
            "对照合同约定说明解除条件已成就",
            "将送达过程与后续履行行为做一致性比对",
        ],
        plaintiff_goal=[
            "证明解除通知有效到达并明确生效节点",
            "稳固合同解除后的责任起算基础",
            "削弱程序抗辩对实体责任的阻断效果",
        ],
        defendant_trigger=[
            "原告仅以形式签收主张解除完成",
            "对方忽略送达对象授权边界",
            "原告未证明解除前已完成必要催告",
        ],
        defendant_action=[
            "要求审查签收权限与送达对象关系",
            "提交未实际收到或迟延收到的证据材料",
            "突出解除流程与合同约定存在不一致之处",
        ],
        defendant_goal=[
            "动摇解除通知生效结论并延后责任起算",
            "降低解除后给付请求的支持范围",
            "强化程序正当性在解除审查中的地位",
        ],
        common_trigger=[
            "案件进入解除程序合法性审查",
            "双方争执解除通知何时生效",
        ],
        common_action=[
            "将送达事实、授权事实与催告事实分层举证",
            "按时间顺序重建解除行为全流程",
        ],
        common_goal=[
            "提高解除效力判断的可验证性",
            "避免程序事实与实体责任脱节",
        ],
    ),
    InsightScenario(
        issue_id="labor_overtime_evidence",
        domain="labor",
        context_facts=[
            "劳动者主张长期加班但用人单位否认审批流程。",
            "争议集中在考勤记录与工作指令的对应性。",
            "双方对综合工时制适用条件存在分歧。",
        ],
        plaintiff_trigger=[
            "单位以无审批流否认加班事实",
            "被告主张实行综合工时制无需加班费",
            "对方质疑打卡记录真实性",
        ],
        plaintiff_action=[
            "联合提交考勤、任务分配与沟通记录",
            "核验综合工时制审批文件的适用范围",
            "按节假日与工作日分别计算加班工时",
        ],
        plaintiff_goal=[
            "证明实际延时用工持续发生并应支付加班费",
            "限制单位以制度标签替代举证责任",
            "提升加班工时认定的精确度",
        ],
        defendant_trigger=[
            "员工仅提交零散打卡记录主张高额加班费",
            "对方将弹性工时等同于法定加班",
            "原告忽略岗位排班与审批制度",
        ],
        defendant_action=[
            "提交排班制度、审批流与调休执行记录",
            "逐项核对应计工时与实际岗位属性",
            "识别并剔除重复或异常打卡片段",
        ],
        defendant_goal=[
            "压缩无法核验工时对应的加班请求",
            "证明部分工时不属于法定加班范围",
            "降低加班工资计算基数和区间",
        ],
        common_trigger=[
            "加班争议进入证据真实性与关联性审查",
            "双方围绕工时数据口径产生对抗",
        ],
        common_action=[
            "建立工时记录、工作安排与报酬发放的映射关系",
            "将争议工时按证据强度分层展示",
        ],
        common_goal=[
            "减少因工时口径混乱导致的裁判偏差",
            "提升加班主张的可审计性",
        ],
    ),
    InsightScenario(
        issue_id="loan_delivery_and_intent",
        domain="loan",
        context_facts=[
            "出借人缺少完整转账凭证但主张借贷关系成立。",
            "双方围绕借款合意与实际交付是否同时成立争执。",
            "聊天记录与借条的证明力成为关键。",
        ],
        plaintiff_trigger=[
            "对方以缺少银行流水否认借款交付",
            "被告主张款项性质属于投资而非借款",
            "对方攻击聊天记录片段化",
        ],
        plaintiff_action=[
            "整合借条、催收记录与用途确认信息",
            "补充提交第三方转账路径与资金去向说明",
            "将借贷合意与交付事实分步骤举证",
        ],
        plaintiff_goal=[
            "证明借贷合意与实际交付形成闭环",
            "削弱投资关系抗辩的事实基础",
            "提高借款债务成立的证明强度",
        ],
        defendant_trigger=[
            "原告仅凭聊天记录主张借贷成立",
            "对方无法提交完整资金流证明",
            "原告混同借款与合作往来",
        ],
        defendant_action=[
            "要求原告还原完整资金路径与账户对应关系",
            "提交合作背景和收益分配证据反证借贷性质",
            "针对借条形成过程提出真实性审查申请",
        ],
        defendant_goal=[
            "阻断借贷关系成立的关键证明链",
            "提升投资或合作性质抗辩的可信度",
            "降低本金与利息请求的可支持性",
        ],
        common_trigger=[
            "借贷争议同时涉及合意与交付双要件",
            "核心证据以聊天记录和借条为主",
        ],
        common_action=[
            "将意思表示证据与资金流证据并行校验",
            "对关键时间点证据进行前后闭环核对",
        ],
        common_goal=[
            "避免单一证据主导导致事实认定偏差",
            "提升借贷性质判断的稳健性",
        ],
    ),
]


WEAK_ACTIONS = [
    "重复抽象表述",
    "仅复述对方观点",
    "单纯依赖口头解释",
    "泛化主张而不落证据",
]


def _sample(rng: random.Random, values: List[str]) -> str:
    return values[rng.randrange(len(values))]


def detect_style(text: str) -> str:
    """Detect style pattern of one insight sentence."""
    normalized = " ".join(text.strip().split())

    if re.match(r"^在.+时，应.+，以.+。$", normalized):
        return "A"

    if re.match(r"^在.+中，若.+，应重点.+，构建.+。$", normalized):
        return "B"

    if re.match(r"^面对.+，不应仅.+，应.+以实现.+。$", normalized):
        return "C"

    return "UNKNOWN"


def assess_insight_quality(text: str) -> Dict[str, Any]:
    """Assess whether one insight is actionable and style-compliant."""
    normalized = " ".join(str(text or "").strip().split())
    style = detect_style(normalized)
    length_ok = 30 <= len(normalized) <= 120
    has_trigger = any(token in normalized for token in ("在", "若", "当", "面对"))
    has_action = "应" in normalized
    has_goal = ("以" in normalized) or ("构建" in normalized)

    quality_pass = (
        length_ok
        and has_trigger
        and has_action
        and has_goal
        and style in {"A", "B", "C"}
    )

    return {
        "insight_style": style,
        "has_trigger": has_trigger,
        "has_action": has_action,
        "has_goal": has_goal,
        "quality_pass": quality_pass,
        "char_len": len(normalized),
    }


def _build_insight_text(
    *,
    scenario: InsightScenario,
    side: str,
    style: str,
    rng: random.Random,
    contradictory: bool = False,
) -> str:
    """Build one style-constrained insight sentence."""
    if side not in {SIDE_P, SIDE_D, SIDE_C}:
        raise ValueError(f"Unsupported side: {side}")

    if style not in {"A", "B", "C"}:
        raise ValueError(f"Unsupported style: {style}")

    if side == SIDE_P:
        trigger = _sample(rng, scenario.plaintiff_trigger)
        action = _sample(rng, scenario.plaintiff_action)
        goal = _sample(rng, scenario.plaintiff_goal)
        opposite_action = _sample(rng, scenario.defendant_action)
        opposite_goal = _sample(rng, scenario.defendant_goal)

    elif side == SIDE_D:
        trigger = _sample(rng, scenario.defendant_trigger)
        action = _sample(rng, scenario.defendant_action)
        goal = _sample(rng, scenario.defendant_goal)
        opposite_action = _sample(rng, scenario.plaintiff_action)
        opposite_goal = _sample(rng, scenario.plaintiff_goal)

    else:
        trigger = _sample(rng, scenario.common_trigger)
        action = _sample(rng, scenario.common_action)
        goal = _sample(rng, scenario.common_goal)

        opposite_action = _sample(
            rng, scenario.plaintiff_action + scenario.defendant_action
        )

        opposite_goal = _sample(rng, scenario.plaintiff_goal + scenario.defendant_goal)

    if contradictory:
        action = opposite_action
        goal = opposite_goal

    domain_label = {
        "lease": "租赁合同纠纷",
        "evidence": "电子证据争议",
        "contract": "合同解除争议",
        "labor": "劳动争议",
        "loan": "借贷纠纷",
    }.get(scenario.domain, "合同争议")

    if style == "A":
        text = f"在{trigger}时，应{action}，以{goal}。"

    elif style == "B":
        text = f"在{domain_label}中，若{trigger}，应重点{action}，构建{goal}。"

    else:
        weak = _sample(rng, WEAK_ACTIONS)
        text = f"面对{trigger}，不应仅{weak}，应{action}以实现{goal}。"

    quality = assess_insight_quality(text)

    if not quality["quality_pass"]:
        raise ValueError(f"Generated low-quality insight: {text}")

    return text


def _build_context(
    *,
    scenario: InsightScenario,
    query_side: str,
    rng: random.Random,
) -> str:
    """Build one synthetic debate context paragraph."""
    fact_1 = _sample(rng, scenario.context_facts)
    fact_2 = _sample(rng, scenario.context_facts)

    if query_side == SIDE_P:
        side_clause = "当前我方角色为原告代理，需要稳固给付请求与证据链完整性。"

    else:
        side_clause = "当前我方角色为被告代理，需要压缩责任区间并强化程序抗辩。"

    return f"{fact_1}{fact_2}{side_clause}"


def _compatible(query_side: str, insight_side: str) -> bool:
    if query_side == SIDE_P:
        return insight_side in {SIDE_P, SIDE_C}

    return insight_side in {SIDE_D, SIDE_C}


def _build_retrieval_record(
    *,
    pair_id: int,
    scenario: InsightScenario,
    context: str,
    query_side: str,
    insight_text: str,
    insight_side: str,
    label: int,
    sample_type: str,
    hard_sample: bool,
    mode: str,
    seed: int,
) -> Dict[str, Any]:
    quality = assess_insight_quality(insight_text)

    return {
        "pair_id": f"INS_RET_{pair_id:04d}",
        "label": int(label),
        "sample_type": sample_type,
        "hard_sample": bool(hard_sample),
        "mode": mode,
        "issue_id": scenario.issue_id,
        "domain": scenario.domain,
        "query_side": query_side,
        "insight_side": insight_side,
        "context": context,
        "insight": insight_text,
        "source": "synthetic_aigc",
        "seed": seed,
        **quality,
    }


def _build_dedup_record(
    *,
    pair_id: int,
    scenario: InsightScenario,
    existing_insight: str,
    new_insight: str,
    existing_side: str,
    new_side: str,
    label: int,
    sample_type: str,
    hard_sample: bool,
    mode: str,
    seed: int,
) -> Dict[str, Any]:
    quality_existing = assess_insight_quality(existing_insight)
    quality_new = assess_insight_quality(new_insight)

    return {
        "pair_id": f"INS_DED_{pair_id:04d}",
        "label": int(label),
        "sample_type": sample_type,
        "hard_sample": bool(hard_sample),
        "mode": mode,
        "issue_id": scenario.issue_id,
        "domain": scenario.domain,
        "existing_side": existing_side,
        "new_side": new_side,
        "existing_insight": existing_insight,
        "new_insight": new_insight,
        "source": "synthetic_aigc",
        "seed": seed,
        "insight_style": quality_new["insight_style"],
        "has_trigger": quality_new["has_trigger"] and quality_existing["has_trigger"],
        "has_action": quality_new["has_action"] and quality_existing["has_action"],
        "has_goal": quality_new["has_goal"] and quality_existing["has_goal"],
        "quality_pass": quality_new["quality_pass"]
        and quality_existing["quality_pass"],
        "char_len": max(quality_new["char_len"], quality_existing["char_len"]),
    }


def _split_rows(
    rows: List[Dict[str, Any]], valid_ratio: float, rng: random.Random
) -> None:
    rng.shuffle(rows)
    valid_size = int(round(len(rows) * valid_ratio))
    valid_size = min(max(valid_size, 1), len(rows) - 1)
    split_point = len(rows) - valid_size

    for idx, row in enumerate(rows):
        row["split"] = "train" if idx < split_point else "valid"


def _generate_retrieval_pairs(
    *,
    rng: random.Random,
    seed: int,
    total_size: int,
    valid_ratio: float,
) -> List[Dict[str, Any]]:
    positive_count = total_size // 2
    negative_count = total_size - positive_count
    positive_easy = positive_count // 2
    positive_hard = positive_count - positive_easy
    negative_hard = int(round(negative_count * 0.6))
    negative_easy = negative_count - negative_hard
    rows: List[Dict[str, Any]] = []
    pair_id = 1
    styles = ["A", "B", "C"]

    for _ in range(positive_easy + positive_hard):
        scenario = rng.choice(SCENARIOS)
        query_side = rng.choice([SIDE_P, SIDE_D])
        context = _build_context(scenario=scenario, query_side=query_side, rng=rng)

        if query_side == SIDE_P:
            insight_side = rng.choice([SIDE_P, SIDE_C])

        else:
            insight_side = rng.choice([SIDE_D, SIDE_C])

        style = rng.choice(styles)
        hard_sample = len(rows) >= positive_easy

        insight_text = _build_insight_text(
            scenario=scenario,
            side=insight_side,
            style=style,
            rng=rng,
            contradictory=False,
        )

        rows.append(
            _build_retrieval_record(
                pair_id=pair_id,
                scenario=scenario,
                context=context,
                query_side=query_side,
                insight_text=insight_text,
                insight_side=insight_side,
                label=1,
                sample_type="positive",
                hard_sample=hard_sample,
                mode="semantic_match",
                seed=seed,
            )
        )

        pair_id += 1

    for _ in range(negative_easy):
        scenario = rng.choice(SCENARIOS)

        other = rng.choice(
            [row for row in SCENARIOS if row.issue_id != scenario.issue_id]
        )

        query_side = rng.choice([SIDE_P, SIDE_D])
        context = _build_context(scenario=scenario, query_side=query_side, rng=rng)
        insight_side = query_side if rng.random() < 0.5 else SIDE_C

        insight_text = _build_insight_text(
            scenario=other,
            side=insight_side,
            style=rng.choice(styles),
            rng=rng,
            contradictory=False,
        )

        rows.append(
            _build_retrieval_record(
                pair_id=pair_id,
                scenario=scenario,
                context=context,
                query_side=query_side,
                insight_text=insight_text,
                insight_side=insight_side,
                label=0,
                sample_type="negative",
                hard_sample=False,
                mode="cross_issue",
                seed=seed,
            )
        )

        pair_id += 1

    for _ in range(negative_hard):
        scenario = rng.choice(SCENARIOS)
        query_side = rng.choice([SIDE_P, SIDE_D])
        context = _build_context(scenario=scenario, query_side=query_side, rng=rng)
        mode = rng.choice(["cross_side", "opposite_goal"])

        if mode == "cross_side":
            insight_side = SIDE_D if query_side == SIDE_P else SIDE_P
            contradictory = False

        else:
            insight_side = query_side
            contradictory = True

        insight_text = _build_insight_text(
            scenario=scenario,
            side=insight_side,
            style=rng.choice(styles),
            rng=rng,
            contradictory=contradictory,
        )

        rows.append(
            _build_retrieval_record(
                pair_id=pair_id,
                scenario=scenario,
                context=context,
                query_side=query_side,
                insight_text=insight_text,
                insight_side=insight_side,
                label=0,
                sample_type="negative",
                hard_sample=True,
                mode=mode,
                seed=seed,
            )
        )

        pair_id += 1

    for row in rows:
        if row["mode"] == "cross_side" and _compatible(
            row["query_side"], row["insight_side"]
        ):
            row["label"] = 1

    _split_rows(rows, valid_ratio, rng)
    return rows


def _generate_dedup_pairs(
    *,
    rng: random.Random,
    seed: int,
    total_size: int,
    valid_ratio: float,
) -> List[Dict[str, Any]]:
    positive_count = total_size // 2
    negative_count = total_size - positive_count
    styles = ["A", "B", "C"]
    rows: List[Dict[str, Any]] = []
    pair_id = 1

    for _ in range(positive_count):
        scenario = rng.choice(SCENARIOS)
        side = rng.choice([SIDE_P, SIDE_D, SIDE_C])

        existing = _build_insight_text(
            scenario=scenario,
            side=side,
            style=rng.choice(styles),
            rng=rng,
            contradictory=False,
        )

        new = _build_insight_text(
            scenario=scenario,
            side=side,
            style=rng.choice(styles),
            rng=rng,
            contradictory=False,
        )

        rows.append(
            _build_dedup_record(
                pair_id=pair_id,
                scenario=scenario,
                existing_insight=existing,
                new_insight=new,
                existing_side=side,
                new_side=side,
                label=1,
                sample_type="positive",
                hard_sample=rng.random() < 0.5,
                mode="paraphrase_match",
                seed=seed,
            )
        )

        pair_id += 1

    for _ in range(negative_count):
        scenario = rng.choice(SCENARIOS)
        mode = rng.choice(["opposite_goal", "cross_side", "cross_issue"])

        if mode == "cross_issue":
            other = rng.choice(
                [row for row in SCENARIOS if row.issue_id != scenario.issue_id]
            )

            side = rng.choice([SIDE_P, SIDE_D, SIDE_C])

            existing = _build_insight_text(
                scenario=scenario,
                side=side,
                style=rng.choice(styles),
                rng=rng,
                contradictory=False,
            )

            new = _build_insight_text(
                scenario=other,
                side=side,
                style=rng.choice(styles),
                rng=rng,
                contradictory=False,
            )

            existing_side = side
            new_side = side

        elif mode == "cross_side":
            side = rng.choice([SIDE_P, SIDE_D])
            opposite = SIDE_D if side == SIDE_P else SIDE_P

            existing = _build_insight_text(
                scenario=scenario,
                side=side,
                style=rng.choice(styles),
                rng=rng,
                contradictory=False,
            )

            new = _build_insight_text(
                scenario=scenario,
                side=opposite,
                style=rng.choice(styles),
                rng=rng,
                contradictory=False,
            )

            existing_side = side
            new_side = opposite

        else:
            side = rng.choice([SIDE_P, SIDE_D, SIDE_C])

            existing = _build_insight_text(
                scenario=scenario,
                side=side,
                style=rng.choice(styles),
                rng=rng,
                contradictory=False,
            )

            new = _build_insight_text(
                scenario=scenario,
                side=side,
                style=rng.choice(styles),
                rng=rng,
                contradictory=True,
            )

            existing_side = side
            new_side = side

        rows.append(
            _build_dedup_record(
                pair_id=pair_id,
                scenario=scenario,
                existing_insight=existing,
                new_insight=new,
                existing_side=existing_side,
                new_side=new_side,
                label=0,
                sample_type="negative",
                hard_sample=(mode != "cross_issue"),
                mode=mode,
                seed=seed,
            )
        )

        pair_id += 1

    _split_rows(rows, valid_ratio, rng)
    return rows


def _build_key_cases() -> List[Dict[str, Any]]:
    """Fixed review cases for retrieval and dedup semantics."""
    return [
        {
            "case_id": "RET_POS_01",
            "category": "retrieval",
            "label": 1,
            "query_side": SIDE_P,
            "insight_side": SIDE_P,
            "context": "争议聚焦于违约金过高，被告请求调减，当前我方角色为原告代理。",
            "candidate": "在对方主张违约金过高时，应主动举证资金占用成本与履约机会损失，以证明约定标准与损失规模具有对应关系以维持条款效力。",
            "description": "原告语境下的违约金举证策略",
        },
        {
            "case_id": "RET_POS_02",
            "category": "retrieval",
            "label": 1,
            "query_side": SIDE_P,
            "insight_side": SIDE_C,
            "context": "对方否认实际占用厂房，双方争执工商登记与经营流水证明力。",
            "candidate": "在租赁合同纠纷中，若对方以未实际使用厂房抗辩租金责任，应重点调取工商登记地址变更记录与业务流水，构建其持续占用并经营的法律事实证据链。",
            "description": "COMMON insight 在原告场景可用",
        },
        {
            "case_id": "RET_NEG_01",
            "category": "retrieval",
            "label": 0,
            "query_side": SIDE_P,
            "insight_side": SIDE_D,
            "context": "争议聚焦于违约金过高，被告请求调减，当前我方角色为原告代理。",
            "candidate": "面对原告仅以合同文字主张违约金不应调整，不应仅重复抽象表述，应要求原告具体举证实际损失并核验计算口径以实现违约金调减目标。",
            "description": "跨侧策略污染（应拒召）",
        },
        {
            "case_id": "RET_NEG_02",
            "category": "retrieval",
            "label": 0,
            "query_side": SIDE_D,
            "insight_side": SIDE_D,
            "context": "劳动争议中对方主张长期加班但证据分散。",
            "candidate": "在借贷纠纷中，若对方以缺少银行流水否认借款交付，应重点整合借条、催收记录与用途确认信息，构建借贷合意与交付闭环。",
            "description": "同侧但跨争点（应拒召）",
        },
        {
            "case_id": "DED_POS_01",
            "category": "dedup",
            "label": 1,
            "query_side": SIDE_P,
            "insight_side": SIDE_P,
            "context": "在对方主张违约金过高时，应主动举证资金占用成本与履约机会损失，以证明约定标准与损失规模具有对应关系以维持条款效力。",
            "candidate": "面对对方主张违约金过高，不应仅复述条款，应同步提交催告记录与履约受阻证据以实现维持约定违约责任的裁判目标。",
            "description": "同策略改写（应合并）",
        },
        {
            "case_id": "DED_NEG_01",
            "category": "dedup",
            "label": 0,
            "query_side": SIDE_P,
            "insight_side": SIDE_P,
            "context": "在对方主张违约金过高时，应主动举证资金占用成本与履约机会损失，以证明约定标准与损失规模具有对应关系以维持条款效力。",
            "candidate": "在对方主张违约金过高时，应要求原告具体举证实际损失并核验计算口径，以证明违约金与损失明显不匹配从而促成调减。",
            "description": "目标相反（不应合并）",
        },
    ]


def _build_summary(
    retrieval_rows: List[Dict[str, Any]],
    dedup_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build one compact summary for generated dataset."""

    def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        pos = sum(1 for row in rows if int(row["label"]) == 1)
        neg = len(rows) - pos
        hard = sum(1 for row in rows if bool(row.get("hard_sample")))
        style_pass = sum(1 for row in rows if bool(row.get("quality_pass")))
        trigger_pass = sum(1 for row in rows if bool(row.get("has_trigger")))
        action_pass = sum(1 for row in rows if bool(row.get("has_action")))
        goal_pass = sum(1 for row in rows if bool(row.get("has_goal")))
        split_train = sum(1 for row in rows if row.get("split") == "train")
        split_valid = len(rows) - split_train

        return {
            "total": len(rows),
            "positive": pos,
            "negative": neg,
            "hard_samples": hard,
            "split": {"train": split_train, "valid": split_valid},
            "style_compliance_rate": (style_pass / len(rows)) if rows else 0.0,
            "has_trigger_rate": (trigger_pass / len(rows)) if rows else 0.0,
            "has_action_rate": (action_pass / len(rows)) if rows else 0.0,
            "has_goal_rate": (goal_pass / len(rows)) if rows else 0.0,
        }

    return {
        "retrieval": summarize(retrieval_rows),
        "dedup": summarize(dedup_rows),
        "overall_style_compliance_rate": (
            (
                sum(
                    1
                    for row in retrieval_rows + dedup_rows
                    if bool(row.get("quality_pass"))
                )
                / (len(retrieval_rows) + len(dedup_rows))
            )
            if (retrieval_rows or dedup_rows)
            else 0.0
        ),
    }


def generate_insight_dataset(
    *,
    seed: int = 52,
    retrieval_size: int = 720,
    dedup_size: int = 480,
    valid_ratio: float = 0.2,
) -> Dict[str, Any]:
    """Generate synthetic retrieval+dedup datasets for insight-threshold tuning."""
    if retrieval_size < 200:
        raise ValueError("retrieval_size must be >= 200")

    if dedup_size < 200:
        raise ValueError("dedup_size must be >= 200")

    if not 0.05 <= valid_ratio <= 0.5:
        raise ValueError("valid_ratio must be in [0.05, 0.5]")

    rng = random.Random(seed)

    retrieval_rows = _generate_retrieval_pairs(
        rng=rng,
        seed=seed,
        total_size=retrieval_size,
        valid_ratio=valid_ratio,
    )

    dedup_rows = _generate_dedup_pairs(
        rng=rng,
        seed=seed,
        total_size=dedup_size,
        valid_ratio=valid_ratio,
    )

    summary = _build_summary(retrieval_rows, dedup_rows)

    return {
        "seed": seed,
        "retrieval_pairs": retrieval_rows,
        "dedup_pairs": dedup_rows,
        "key_cases": _build_key_cases(),
        "summary": summary,
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=52, help="Random seed.")

    parser.add_argument(
        "--retrieval-size",
        type=int,
        default=720,
        help="Pair count for retrieval dataset.",
    )

    parser.add_argument(
        "--dedup-size",
        type=int,
        default=480,
        help="Pair count for dedup dataset.",
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/optim/artifacts"),
        help="Output directory.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = generate_insight_dataset(
        seed=args.seed,
        retrieval_size=args.retrieval_size,
        dedup_size=args.dedup_size,
        valid_ratio=args.valid_ratio,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = output_dir / f"insight_retrieval_seed{args.seed}.jsonl"
    dedup_path = output_dir / f"insight_dedup_seed{args.seed}.jsonl"
    summary_path = output_dir / f"insight_dataset_seed{args.seed}.summary.json"
    key_cases_path = output_dir / "insight_key_cases_seed_base.json"
    _write_jsonl(retrieval_path, payload["retrieval_pairs"])
    _write_jsonl(dedup_path, payload["dedup_pairs"])

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload["summary"], f, ensure_ascii=False, indent=2)

    with key_cases_path.open("w", encoding="utf-8") as f:
        json.dump(payload["key_cases"], f, ensure_ascii=False, indent=2)

    print(
        f"[insight-dataset] retrieval={len(payload['retrieval_pairs'])} -> {retrieval_path}"
    )

    print(f"[insight-dataset] dedup={len(payload['dedup_pairs'])} -> {dedup_path}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
