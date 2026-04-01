"""Generate synthetic AIGC-style projection pairs for threshold tuning.

This script intentionally avoids external data sources. All query/candidate pairs are
programmatically synthesized to mimic debate outputs used in projection retrieval.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class IssueTemplate:
    """One semantic issue profile used to synthesize debate-style arguments.

    Attributes:
        issue_id: Stable template identifier.
        domain: High-level legal domain name.
        plaintiff_points: Plaintiff-side argument variants.
        defendant_points: Defendant-side argument variants.
        fact_clauses: Factual clauses inserted into candidate texts.
        law_clauses: Legal clauses inserted into candidate texts.
        distractor_clauses: Off-topic distractors for hard negatives.
    """

    issue_id: str
    domain: str
    plaintiff_points: List[str]
    defendant_points: List[str]
    fact_clauses: List[str]
    law_clauses: List[str]
    distractor_clauses: List[str]


ISSUE_TEMPLATES: List[IssueTemplate] = [
    IssueTemplate(
        issue_id="lease_termination_notice",
        domain="lease",
        plaintiff_points=[
            "被告长期拖欠租金且逾期超过约定期限，出租人解除合同具备充分依据。",
            "解除通知到达承租人后合同即告解除，后续占有应当支付占用费。",
            "逾期腾退期间的费用应按合同约定标准计算。",
        ],
        defendant_points=[
            "解除前未给予合理催告期限，解除程序存在瑕疵。",
            "双倍占用费具有惩罚性，明显高于实际损失。",
            "解除条件虽有约定，但行使解除权仍需满足正当程序。",
        ],
        fact_clauses=[
            "承租方连续多个结算期未支付租金。",
            "出租方已向工商登记地址送达解除函并留存签收记录。",
            "合同明确约定逾期达到一定天数可触发解除条款。",
        ],
        law_clauses=[
            "依据《民法典》关于合同解除通知到达规则，通知生效即产生解除效力。",
            "依据《民法典》租赁合同章节，合同终止后承租人应返还租赁物。",
            "依据违约责任的一般规则，守约方有权请求合同约定的违约后果。",
        ],
        distractor_clauses=[
            "对方同时争执电费倍率和线损参数，但该争议并不改变解除权基础。",
            "被告援引市场景气下行作为商业风险抗辩。",
            "双方就保证金条款是否默示变更存在分歧。",
        ],
    ),
    IssueTemplate(
        issue_id="lease_double_occupancy_fee",
        domain="lease",
        plaintiff_points=[
            "双倍占用费系合同终止后占有使用的特别费用安排，并非普通违约金。",
            "逾期违约金针对履行期，双倍占用费针对终止后占有期，二者功能不同。",
            "承租方拒不腾退且主观恶意明显，应按约承担占用费。",
        ],
        defendant_points=[
            "双倍占用费与逾期违约金叠加构成重复惩罚，应予调减。",
            "占用费应回归实际市场租金，不应机械适用双倍标准。",
            "原告未证明双倍标准与实际损失之间存在比例关系。",
        ],
        fact_clauses=[
            "合同终止后承租方仍持续占有厂房。",
            "出租方已多次催告腾退但未获配合。",
            "争议主要集中在占用期间费用计算标准。",
        ],
        law_clauses=[
            "依据《民法典》关于违约责任条款，约定内容应优先适用。",
            "依据租赁合同司法解释，逾期腾退可支持占有使用费请求。",
            "法院可在明显不合理时审查约定费用的调整请求。",
        ],
        distractor_clauses=[
            "双方对是否应补缴物业服务费也有争议。",
            "合同中关于免费装修期的记载被反复引用。",
            "一方试图以政策变化解释持续占有行为。",
        ],
    ),
    IssueTemplate(
        issue_id="sale_quality_acceptance",
        domain="sale",
        plaintiff_points=[
            "买受人已完成验收并投入使用，不得事后以轻微瑕疵拒付价款。",
            "质量争议不影响已到期货款的主债务履行。",
            "瑕疵主张应以及时通知和专业检验为前提。",
        ],
        defendant_points=[
            "交付设备存在核心性能缺陷，买受人有权拒付尾款。",
            "卖方未按合同技术参数交货，构成根本违约。",
            "验收记录存在格式化签署，不能当然推定质量合格。",
        ],
        fact_clauses=[
            "设备交付后已连续运行一段期间。",
            "双方对验收报告签字真实性存在争议。",
            "买受人提出的缺陷集中在关键产能指标。",
        ],
        law_clauses=[
            "依据买卖合同规则，瑕疵担保责任以不符合约定质量为前提。",
            "依据举证规则，主张设备不合格一方需提供专业检验依据。",
            "迟延通知可能影响买受人主张瑕疵救济的范围。",
        ],
        distractor_clauses=[
            "双方还争执培训服务是否另行计费。",
            "运输保险条款解释被反复提及。",
            "合同附件中的交货清单编号出现不一致。",
        ],
    ),
    IssueTemplate(
        issue_id="sale_liquidated_damage",
        domain="sale",
        plaintiff_points=[
            "买方逾期付款期间长且金额大，约定违约金不宜轻易调减。",
            "违约金条款已被明确确认，应维持交易稳定预期。",
            "守约方融资成本和机会损失应纳入损失衡量。",
        ],
        defendant_points=[
            "日万分之五违约金显著高于实际损失，应按公平原则调整。",
            "卖方未及时交付完整单据，亦应承担相应过错。",
            "违约金与其他费用并行主张时存在过度赔偿风险。",
        ],
        fact_clauses=[
            "双方对逾期付款事实本身并无实质争议。",
            "争议聚焦在违约金是否过高及其计算区间。",
            "部分付款行为发生在催告后。",
        ],
        law_clauses=[
            "依据《民法典》违约金调整规则，明显过高可酌减。",
            "法院审查时会综合履行情况、损失规模与过错程度。",
            "当事人确认条款不当然排除法定调整权。",
        ],
        distractor_clauses=[
            "双方就发票开具时间也有次要争执。",
            "交货地点变更是否影响运输费用承担被附带讨论。",
            "售后维护期起算点在证据中表述不一致。",
        ],
    ),
    IssueTemplate(
        issue_id="loan_transfer_proof",
        domain="loan",
        plaintiff_points=[
            "虽缺少银行回单，但聊天记录与借条可形成借贷合意闭环。",
            "长期催收记录可反证借款关系持续存在。",
            "被告对资金用途的陈述前后矛盾，削弱其否认借款的可信性。",
        ],
        defendant_points=[
            "原告未提交完整转账凭证，无法证明借款实际交付。",
            "聊天记录存在截取风险，不能单独证明借贷事实。",
            "涉案款项更符合合伙投资性质而非民间借贷。",
        ],
        fact_clauses=[
            "双方在多个时间点就资金归还进行沟通。",
            "资金流向记录与用途说明存在不一致。",
            "借条签署时间与转账时间并非完全重合。",
        ],
        law_clauses=[
            "依据民间借贷审理规则，出借关系应兼顾合意与交付双要件。",
            "电子证据的完整性与来源合法性影响证明力。",
            "若主张投资关系，应承担更高的事实举证责任。",
        ],
        distractor_clauses=[
            "双方另有一份未履行的购销框架协议。",
            "被告主张利息计算方式与行业惯例一致。",
            "原告曾以第三人账户代收部分款项。",
        ],
    ),
    IssueTemplate(
        issue_id="loan_interest_adjust",
        domain="loan",
        plaintiff_points=[
            "借款利率在签约时已经明确，被告应按约支付利息。",
            "逾期期间利息与违约金可在法定边界内并行主张。",
            "被告长期占用资金导致出借人实际损失扩大。",
        ],
        defendant_points=[
            "复合计息导致实际利率显著超限，应调整至法定上限。",
            "原告同时主张高额违约金与罚息构成重复赔偿。",
            "利息起算日与资金实际到达日不一致。",
        ],
        fact_clauses=[
            "借款合同中约定了名义利率和逾期处理条款。",
            "借款人存在分段还款行为。",
            "双方对罚息是否另行约定存在争议。",
        ],
        law_clauses=[
            "民间借贷利率保护规则对可支持区间有明确限制。",
            "超出法定保护上限的利息请求通常不予支持。",
            "法院会审查名义利率与实际利率是否存在规避。",
        ],
        distractor_clauses=[
            "双方就借款用途是否用于经营存在分歧。",
            "担保人签字页是否完整被单独争执。",
            "还款凭证中备注字段存在多种写法。",
        ],
    ),
    IssueTemplate(
        issue_id="labor_overtime_pay",
        domain="labor",
        plaintiff_points=[
            "考勤记录与工作指令相互印证，加班事实成立。",
            "企业长期安排延时工作却未依法支付加班工资。",
            "调休安排未实际落实，不足以替代加班报酬。",
        ],
        defendant_points=[
            "员工主张的加班时间缺少审批流，证据不足。",
            "岗位实行综合工时制，不应按标准工时机械计算。",
            "员工提交的打卡截图存在编辑可能性。",
        ],
        fact_clauses=[
            "争议期间存在节假日值班安排。",
            "双方均提交了部分电子考勤记录。",
            "工资条中未单列加班项目。",
        ],
        law_clauses=[
            "劳动法体系要求用人单位对工时管理负有规范义务。",
            "加班工资请求通常以考勤与工作安排证据综合认定。",
            "综合工时制适用需满足审批与岗位条件。",
        ],
        distractor_clauses=[
            "双方还争执绩效奖金是否计入基数。",
            "离职交接文件的签署时间存在错位。",
            "原告另主张未休年假折现补偿。",
        ],
    ),
    IssueTemplate(
        issue_id="labor_non_compete",
        domain="labor",
        plaintiff_points=[
            "竞业限制协议约定清晰且补偿已按月支付，应认定有效。",
            "员工离职后入职直接竞争企业，已违反竞业义务。",
            "违约金标准与岗位敏感性相匹配。",
        ],
        defendant_points=[
            "企业未持续支付竞业补偿，竞业条款不应继续约束员工。",
            "新岗位与原岗位业务重合度低，不构成竞争关系。",
            "约定违约金明显过高，应按比例调整。",
        ],
        fact_clauses=[
            "员工离职后在短期内入职同产业链公司。",
            "双方对补偿金发放月份有争议。",
            "劳动合同附件载明了保密与竞业条款。",
        ],
        law_clauses=[
            "竞业限制效力依赖于补偿支付和岗位竞争关系认定。",
            "违约金可结合企业商业秘密价值进行审查。",
            "未按约支付补偿可能影响竞业限制的可执行性。",
        ],
        distractor_clauses=[
            "劳动关系终止证明抬头存在格式差异。",
            "员工主张加班费与竞业争议并案处理。",
            "双方就培训服务期条款是否独立有效存在争执。",
        ],
    ),
    IssueTemplate(
        issue_id="tort_medical_causality",
        domain="tort",
        plaintiff_points=[
            "诊疗记录与鉴定意见能够证明损害后果与医疗行为存在因果关系。",
            "医院未尽到充分告知义务，应承担相应赔偿责任。",
            "并发症发生时间与处置延迟高度相关。",
        ],
        defendant_points=[
            "患者原有基础疾病是主要损害原因，医院过错程度有限。",
            "鉴定意见仅表述关联性，未达到高度盖然性标准。",
            "院方已履行常规诊疗流程与风险告知。",
        ],
        fact_clauses=[
            "患者在手术后出现持续性并发症。",
            "病历中存在多次会诊与处置记录。",
            "双方均申请了专业鉴定。",
        ],
        law_clauses=[
            "医疗损害责任需综合过错、因果关系与损害结果认定。",
            "鉴定意见通常是因果判断的重要依据但并非唯一依据。",
            "知情同意义务的履行直接影响责任分配。",
        ],
        distractor_clauses=[
            "双方对住院期间护理费标准存在争议。",
            "发票抬头是否一致被附带讨论。",
            "患者家属签字时间与病程记录存在细微偏差。",
        ],
    ),
    IssueTemplate(
        issue_id="tort_traffic_fault_share",
        domain="tort",
        plaintiff_points=[
            "被告超速且未保持安全车距，应承担主要责任。",
            "事故认定书与现场监控共同指向被告过错更大。",
            "维修损失和停运损失均有票据支撑。",
        ],
        defendant_points=[
            "原告违规变道是事故直接诱因，应承担同等或主要责任。",
            "事故认定书未充分考虑道路施工与视线遮挡因素。",
            "停运损失计算周期明显偏长。",
        ],
        fact_clauses=[
            "事故发生于高峰时段且路段车流密集。",
            "双方车辆均存在不同程度损坏。",
            "交警部门已出具责任认定意见。",
        ],
        law_clauses=[
            "侵权责任分担应结合行为过错与原因力大小。",
            "交通事故赔偿项目需以实际损失和合理期间为基础。",
            "电子监控资料可作为过错认定的重要证据。",
        ],
        distractor_clauses=[
            "双方对车辆残值是否扣减存在分歧。",
            "保险公司垫付范围被附带争论。",
            "原告还主张了精神损害抚慰金。",
        ],
    ),
    IssueTemplate(
        issue_id="company_share_transfer_payment",
        domain="company",
        plaintiff_points=[
            "股权转让价款已到支付节点，受让方应按约付款。",
            "工商变更并非付款的先决条件，双方约定应优先适用。",
            "受让方已实际行使股东权利，不能再否认交易效力。",
        ],
        defendant_points=[
            "目标公司存在重大未披露负债，受让方有权暂缓付款。",
            "交割先决条件未满足，付款义务尚未到期。",
            "转让方保证条款存在重大违背，构成先违约。",
        ],
        fact_clauses=[
            "双方签署了股权转让协议及补充协议。",
            "部分董事会权限已由受让方实际控制。",
            "争议集中在交割条件是否成就。",
        ],
        law_clauses=[
            "股权转让纠纷应优先审查合同约定的交割与付款条件。",
            "信息披露义务违背可能影响价款支付节奏。",
            "先违约抗辩需满足对待给付关系与证据门槛。",
        ],
        distractor_clauses=[
            "双方对审计报告基准日存在技术性分歧。",
            "章程修订备案时间被反复提及。",
            "目标公司分红政策历史记录存在口径差异。",
        ],
    ),
    IssueTemplate(
        issue_id="company_control_right_dispute",
        domain="company",
        plaintiff_points=[
            "被告绕过股东会程序擅自更换管理层，侵害股东知情与表决权。",
            "临时股东会通知程序违法，应确认相关决议无效。",
            "公司印章和账户控制行为加剧治理失衡。",
        ],
        defendant_points=[
            "原告长期不参与治理且阻碍经营，被告采取应急管理措施具有必要性。",
            "决议程序存在瑕疵但未影响实质表决结果。",
            "相关决议属于内部管理事项，不宜当然认定无效。",
        ],
        fact_clauses=[
            "双方围绕董事会与股东会权限边界长期冲突。",
            "争议期间公司发生多次高管更迭。",
            "印章与网银U盾由不同派别分别控制。",
        ],
        law_clauses=[
            "公司决议效力审查需同时考量程序合法性与实体影响。",
            "股东知情权、表决权受侵害时可请求司法救济。",
            "紧急经营管理抗辩不能当然豁免法定程序。",
        ],
        distractor_clauses=[
            "双方还争论办公场地租约续签权限。",
            "员工社保补缴情形被附带提出。",
            "工商备案材料中个别签字存在形式差异。",
        ],
    ),
]


def _sample(rng: random.Random, values: List[str]) -> str:
    return values[rng.randrange(len(values))]


def _render_argument(
    *,
    issue: IssueTemplate,
    side: str,
    style: str,
    hard_noise: bool,
    contradictory: bool,
    rng: random.Random,
) -> str:
    """Render one debate-style paragraph."""
    if side not in {"plaintiff", "defendant"}:
        raise ValueError(f"Unsupported side: {side}")

    side_cn = "原告" if side == "plaintiff" else "被告"

    openers = {
        "opening": [
            f"{side_cn}围绕核心争点提出如下意见：",
            f"{side_cn}认为本案应抓住给付义务与违约后果这一主线：",
            f"{side_cn}在本轮辩论中强调以下要点：",
        ],
        "rebuttal": [
            f"针对对方刚才的陈述，{side_cn}回应如下：",
            f"{side_cn}对对方抗辩逐点反驳如下：",
            f"{side_cn}进一步说明，前述抗辩并不足以否定其请求：",
        ],
        "summary": [
            f"综合现有证据与规则，{side_cn}总结如下：",
            f"{side_cn}最后陈述其法律评价：",
            f"{side_cn}据此请求法庭采纳以下结论：",
        ],
    }

    conclusion_pos = _sample(rng, issue.plaintiff_points)
    conclusion_neg = _sample(rng, issue.defendant_points)

    if contradictory:
        conclusion = conclusion_neg if side == "plaintiff" else conclusion_pos

    else:
        conclusion = conclusion_pos if side == "plaintiff" else conclusion_neg

    fact_row = _sample(rng, issue.fact_clauses)
    law_row = _sample(rng, issue.law_clauses)

    body = [
        _sample(rng, openers[style]),
        fact_row,
        law_row,
        conclusion,
    ]

    if hard_noise:
        body.append(_sample(rng, issue.distractor_clauses))

        if rng.random() < 0.65:
            body.append(
                "对方虽然反复强调词面相近的条款，但未能证明构成要件与本争点一致。"
            )

        if rng.random() < 0.45:
            body.append(
                "该争议与另一笔交易中的结算争点仅存在术语重合，不应被混同解释。"
            )

    return "".join(body)


def _build_positive_pair(
    pair_id: int,
    issue: IssueTemplate,
    difficulty: str,
    rng: random.Random,
) -> Dict[str, Any]:
    hard = difficulty == "hard"

    query = _render_argument(
        issue=issue,
        side=rng.choice(["plaintiff", "defendant"]),
        style=rng.choice(["opening", "rebuttal"]),
        hard_noise=hard,
        contradictory=False,
        rng=rng,
    )

    candidate = _render_argument(
        issue=issue,
        side=rng.choice(["plaintiff", "defendant"]),
        style=rng.choice(["summary", "rebuttal"]),
        hard_noise=hard,
        contradictory=False,
        rng=rng,
    )

    return {
        "pair_id": f"PRJ_{pair_id:04d}",
        "label": 1,
        "difficulty": difficulty,
        "sample_type": "positive",
        "issue_id": issue.issue_id,
        "domain": issue.domain,
        "query": query,
        "candidate": candidate,
        "source": "synthetic_aigc",
        "boundary_hint": hard,
    }


def _build_negative_pair(
    pair_id: int,
    issue: IssueTemplate,
    domain_map: Dict[str, List[IssueTemplate]],
    difficulty: str,
    rng: random.Random,
) -> Dict[str, Any]:
    hard = difficulty == "hard"

    query = _render_argument(
        issue=issue,
        side=rng.choice(["plaintiff", "defendant"]),
        style=rng.choice(["opening", "rebuttal"]),
        hard_noise=hard,
        contradictory=False,
        rng=rng,
    )

    if hard:
        candidate = _render_argument(
            issue=issue,
            side=rng.choice(["plaintiff", "defendant"]),
            style=rng.choice(["summary", "rebuttal"]),
            hard_noise=True,
            contradictory=True,
            rng=rng,
        )

        negative_mode = "same_issue_contradiction"
        candidate_issue_id = issue.issue_id

    else:
        different_domain_issues: List[IssueTemplate] = []

        for domain, rows in domain_map.items():
            if domain != issue.domain:
                different_domain_issues.extend(rows)

        candidate_issue = rng.choice(different_domain_issues)

        candidate = _render_argument(
            issue=candidate_issue,
            side=rng.choice(["plaintiff", "defendant"]),
            style=rng.choice(["summary", "opening"]),
            hard_noise=False,
            contradictory=False,
            rng=rng,
        )

        negative_mode = "cross_domain"
        candidate_issue_id = candidate_issue.issue_id

    return {
        "pair_id": f"PRJ_{pair_id:04d}",
        "label": 0,
        "difficulty": difficulty,
        "sample_type": "negative",
        "negative_mode": negative_mode,
        "issue_id": issue.issue_id,
        "candidate_issue_id": candidate_issue_id,
        "domain": issue.domain,
        "query": query,
        "candidate": candidate,
        "source": "synthetic_aigc",
        "boundary_hint": hard,
    }


def generate_projection_dataset(
    *,
    seed: int = 42,
    total_pairs: int = 600,
    valid_ratio: float = 0.2,
) -> List[Dict[str, Any]]:
    """Generate labeled query/candidate pairs for projection-threshold tuning.

    Args:
        seed: Random seed for deterministic sampling.
        total_pairs: Total number of query/candidate pairs to generate.
        valid_ratio: Fraction of rows placed into the validation split.

    Returns:
        Synthetic projection dataset rows.
    """
    if total_pairs < 100:
        raise ValueError("total_pairs must be >= 100")

    if not 0.05 <= valid_ratio <= 0.5:
        raise ValueError("valid_ratio must be in [0.05, 0.5]")

    rng = random.Random(seed)
    domain_map: Dict[str, List[IssueTemplate]] = {}

    for issue in ISSUE_TEMPLATES:
        domain_map.setdefault(issue.domain, []).append(issue)

    positive_count = total_pairs // 2
    negative_count = total_pairs - positive_count
    pos_hard_count = positive_count // 2
    pos_easy_count = positive_count - pos_hard_count
    neg_hard_count = int(round(negative_count * 0.6))
    neg_easy_count = negative_count - neg_hard_count
    records: List[Dict[str, Any]] = []
    pair_id = 1

    for _ in range(pos_easy_count):
        issue = rng.choice(ISSUE_TEMPLATES)
        records.append(_build_positive_pair(pair_id, issue, "easy", rng))
        pair_id += 1

    for _ in range(pos_hard_count):
        issue = rng.choice(ISSUE_TEMPLATES)
        records.append(_build_positive_pair(pair_id, issue, "hard", rng))
        pair_id += 1

    for _ in range(neg_easy_count):
        issue = rng.choice(ISSUE_TEMPLATES)
        records.append(_build_negative_pair(pair_id, issue, domain_map, "easy", rng))
        pair_id += 1

    for _ in range(neg_hard_count):
        issue = rng.choice(ISSUE_TEMPLATES)
        records.append(_build_negative_pair(pair_id, issue, domain_map, "hard", rng))
        pair_id += 1

    rng.shuffle(records)
    valid_size = int(round(len(records) * valid_ratio))
    valid_size = min(max(valid_size, 1), len(records) - 1)
    split_point = len(records) - valid_size

    for idx, row in enumerate(records):
        row["split"] = "train" if idx < split_point else "valid"
        row["seed"] = seed

    return records


def _build_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": len(records),
        "positive": 0,
        "negative": 0,
        "difficulty": {"easy": 0, "hard": 0},
        "split": {"train": 0, "valid": 0},
        "domains": {},
    }

    for row in records:
        if row["label"] == 1:
            summary["positive"] += 1

        else:
            summary["negative"] += 1

        summary["difficulty"][row["difficulty"]] += 1
        summary["split"][row["split"]] += 1
        domain = row["domain"]
        summary["domains"][domain] = summary["domains"].get(domain, 0) + 1

    return summary


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for projection dataset generation.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/projection_dataset_seed42.jsonl"),
        help="Output JSONL path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=600,
        help="Total pair count (target 300-800).",
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "benchmarks/optim/artifacts/projection_dataset_seed42.summary.json"
        ),
        help="Optional summary output path.",
    )

    return parser.parse_args()


def main() -> None:
    """Generate the projection-threshold tuning dataset."""
    args = parse_args()

    dataset = generate_projection_dataset(
        seed=args.seed,
        total_pairs=args.size,
        valid_ratio=args.valid_ratio,
    )

    _write_jsonl(args.output, dataset)
    summary = _build_summary(dataset)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    with args.summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[projection-dataset] wrote {len(dataset)} pairs to {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
