"""Generate synthetic CLAIM dedup pairs for tuning `dedup.other_threshold`.

All samples are self-constructed and mimic LLM outputs from controller
`VERIFY_AND_DECIDE` action content style.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class ClaimProfile:
    """Structured slots for claim-style legal argument generation."""

    cutoff_date: str
    clause_no: str
    clause_text: str
    cure_days: int
    fee_term: str
    plaintiff_done_notice: bool
    plaintiff_done_cure: bool
    plaintiff_done_direct_terminate: bool


CUTOFF_DATES: Sequence[str] = (
    "2014年5月29日",
    "2015年3月17日",
    "2016年8月12日",
    "2017年11月6日",
    "2018年4月18日",
    "2019年9月25日",
    "2020年6月3日",
    "2021年10月15日",
)

CLAUSE_NOS: Sequence[str] = (
    "第三十五条",
    "第三十二条",
    "第四十条",
    "第二十八条",
)

CLAUSE_TEXTS: Sequence[str] = (
    "逾期30天视同提前终止",
    "逾期15日可解除合同",
    "承租人连续两期欠租可解除",
    "严重违约时出租人可提前解除",
)

FEE_TERMS: Sequence[str] = ("双倍占用费", "惩罚性占用费", "加倍占有使用费")
CURE_DAYS: Sequence[int] = (3, 5, 7, 10, 15)


def _random_profile(rng: random.Random) -> ClaimProfile:
    return ClaimProfile(
        cutoff_date=rng.choice(list(CUTOFF_DATES)),
        clause_no=rng.choice(list(CLAUSE_NOS)),
        clause_text=rng.choice(list(CLAUSE_TEXTS)),
        cure_days=rng.choice(list(CURE_DAYS)),
        fee_term=rng.choice(list(FEE_TERMS)),
        plaintiff_done_notice=rng.random() < 0.5,
        plaintiff_done_cure=rng.random() < 0.5,
        plaintiff_done_direct_terminate=rng.random() < 0.5,
    )


def _procedure_sentence(
    profile: ClaimProfile, plaintiff_side: bool, variant: int
) -> str:
    if plaintiff_side:
        templates = [
            "被告在{date}前已收到原告有效催告，解除程序并无瑕疵，{fee}请求具有正当基础。",
            "截至{date}，原告已完成有效催告并给予合理履行期限，解除权行使符合程序要求，故{fee}主张可获支持。",
            "原告在{date}前完成催告且履行程序完备，被告补救后仍违约，解除合法有效，{fee}并非无依据主张。",
        ]

    else:
        templates = [
            "{date}前原告未有效催告，解除程序存在瑕疵，不得据此主张惩罚性的{fee}。",
            "截至{date}，原告并未完成有效催告，解除权行使程序失当，因此{fee}请求缺乏合法基础。",
            "{date}前原告未履行充分催告义务且径行解除，程序瑕疵导致解除效力存疑，{fee}主张难以成立。",
        ]

    return templates[variant % len(templates)].format(
        date=profile.cutoff_date,
        fee=profile.fee_term,
    )


def _clause_interpret_sentence(
    profile: ClaimProfile, plaintiff_side: bool, variant: int
) -> str:
    if plaintiff_side:
        templates = [
            "原告主张合同{clause_no}“{clause_text}”属于特别约定，故可直接解除且不再适用催告程序。该条款明确赋予解除权，被告已明知逾期后果而仍未履行，程序并无违法，{fee}请求具有法律依据。",
            "依据合同{clause_no}“{clause_text}”，原告认为该约定已构成对催告程序的特别安排。被告持续违约且在通知后未补救，解除效力应予确认，{fee}主张并非惩罚性滥用。",
            "合同{clause_no}“{clause_text}”已就解除条件作出明确约定，原告据此解除并无不当。结合被告长期违约事实，{fee}请求符合约定责任分配规则。",
        ]

    else:
        templates = [
            "原告主张合同{clause_no}“{clause_text}”为特别约定故不适用催告程序不能成立。该条款系约定解除条件，但解除权行使仍应遵循正当程序，原告未给予合理履行期限即径行解除，剥夺被告补救机会，程序瑕疵导致解除效力存疑，{fee}主张缺乏合法基础。",
            "原告援引合同{clause_no}“{clause_text}”主张可免除催告程序，该观点不能成立。该条款仅是解除条件约定，解除权实际行使仍应保障合理履行期限；原告直接解除剥夺被告补救机会，故解除效力存疑，{fee}请求缺乏基础。",
            "就合同{clause_no}“{clause_text}”而言，原告将其解释为可当然免除催告义务并不成立。解除权行使必须遵循程序正当原则，原告未给予被告合理补救期间而径行解除，导致{fee}主张难获支持。",
        ]

    return templates[variant % len(templates)].format(
        clause_no=profile.clause_no,
        clause_text=profile.clause_text,
        fee=profile.fee_term,
    )


def _cure_opportunity_sentence(
    profile: ClaimProfile, plaintiff_side: bool, variant: int
) -> str:
    if plaintiff_side:
        templates = [
            "原告已书面通知并给予被告{days}日履行期限，被告仍未纠正违约，解除程序符合法律要求，{fee}请求应予支持。",
            "就补救机会而言，原告已给予被告{days}日合理期限并完成通知义务，解除权行使不存在程序瑕疵，主张{fee}具有正当性。",
            "被告在收到催告后于{days}日内未履约，原告随后解除合同符合程序要求，{fee}请求不属于过度惩罚。",
        ]

    else:
        templates = [
            "原告未给予被告合理履行期限即解除合同，剥夺补救机会，程序瑕疵直接影响解除效力，{fee}请求不应支持。",
            "从补救机会看，原告并未向被告提供实质性的履行期限却径行解除，程序不当导致{fee}主张缺乏合法性。",
            "原告在未充分保障被告补救权的情况下直接解除合同，难谓程序正当，故{fee}请求缺少依据。",
        ]

    return templates[variant % len(templates)].format(
        days=profile.cure_days,
        fee=profile.fee_term,
    )


def _fee_legality_sentence(
    profile: ClaimProfile, plaintiff_side: bool, variant: int
) -> str:
    if plaintiff_side:
        templates = [
            "{fee}系合同约定的违约后果安排，原告已完成催告与程序性义务，故该请求并非惩罚性扩张。",
            "在原告程序履行完备的前提下，{fee}属于约定责任承担方式，不能仅以金额较高否定其合法性。",
            "鉴于被告持续违约且原告履行催告程序，{fee}请求属于合同责任实现，不构成不当惩罚。",
        ]

    else:
        templates = [
            "{fee}具有明显惩罚性，且原告解除程序存在瑕疵，在程序基础未成立前不得支持该请求。",
            "原告未先完成程序性催告即主张{fee}，属于将惩罚性后果前置，难以获得法律支持。",
            "在解除程序合法性尚存重大疑问时，原告直接主张{fee}缺乏正当基础。",
        ]

    return templates[variant % len(templates)].format(fee=profile.fee_term)


RENDERERS = {
    "procedure": _procedure_sentence,
    "clause": _clause_interpret_sentence,
    "cure": _cure_opportunity_sentence,
    "fee": _fee_legality_sentence,
}


def _apply_noise(text: str, rng: random.Random, level: str) -> str:
    if level == "clean":
        return text

    prefix_pool = (
        "被告进一步指出，",
        "对于该争点，",
        "结合庭审记录可见，",
        "就程序问题而言，",
    )

    middle_pool = ("首先，", "其次，", "进一步看，", "据此，")

    suffix_pool = (
        "该结论与程序正当原则一致。",
        "该观点能够回应对方核心主张。",
        "该判断具有直接裁判意义。",
    )

    noisy = text

    if level in {"mild", "boundary", "heavy"} and rng.random() < 0.58:
        noisy = rng.choice(prefix_pool) + noisy

    if level in {"boundary", "heavy"} and rng.random() < 0.40:
        noisy = noisy.replace("。", "；", 1)

    if level in {"boundary", "heavy"} and rng.random() < 0.50:
        noisy = noisy.replace("因此", rng.choice(middle_pool))

    if level == "heavy" and rng.random() < 0.60:
        noisy = noisy + rng.choice(suffix_pool)

    return noisy


def _style_pass(text: str) -> bool:
    has_party = any(token in text for token in ("原告", "被告", "甲方", "乙方"))

    has_issue = any(
        token in text for token in ("解除", "催告", "程序", "条款", "效力", "占用费")
    )

    has_reason = any(
        token in text
        for token in (
            "因此",
            "故",
            "不能成立",
            "效力存疑",
            "缺乏",
            "合法",
            "不应",
            "难以",
            "应予支持",
            "请求",
        )
    )

    return has_party and has_issue and has_reason


def _high_overlap_negative_pair(
    profile: ClaimProfile, rng: random.Random
) -> tuple[str, str]:
    variant = rng.randrange(0, 4)

    if variant == 0:
        left = f"{profile.cutoff_date}前原告未有效催告，解除程序存在瑕疵，不得据此主张惩罚性的{profile.fee_term}。"
        right = f"{profile.cutoff_date}前原告已有效催告并通知补救期限，解除程序并无瑕疵，可以据此主张{profile.fee_term}。"
        return left, right

    if variant == 1:
        left = f"原告援引合同{profile.clause_no}“{profile.clause_text}”不能当然排除催告程序，未给予合理履行期限即解除，导致解除效力存疑。"
        right = f"原告援引合同{profile.clause_no}“{profile.clause_text}”可以排除催告程序，且已给予合理履行期限后解除，解除效力应予确认。"
        return left, right

    if variant == 2:
        left = f"原告未给予被告{profile.cure_days}日合理履行期限却径行解除，剥夺补救机会，程序瑕疵导致{profile.fee_term}请求缺乏基础。"
        right = f"原告已给予被告{profile.cure_days}日合理履行期限并完成催告，被告仍未补救，解除程序合法有效，{profile.fee_term}请求具有基础。"
        return left, right

    left = f"即便合同{profile.clause_no}约定“{profile.clause_text}”，解除权行使仍应遵循正当程序，原告程序不完备，故{profile.fee_term}不应支持。"
    right = f"鉴于合同{profile.clause_no}已约定“{profile.clause_text}”，且原告程序履行完备，解除权行使合法，因此{profile.fee_term}应予支持。"
    return left, right


def _build_pair(
    *,
    pair_id: str,
    profile: ClaimProfile,
    claim_type: str,
    label: int,
    rng: random.Random,
    valid_ratio: float,
) -> Dict[str, Any]:
    left_variant = rng.randrange(0, 3)
    right_variant = (left_variant + rng.randrange(1, 3)) % 3
    renderer = RENDERERS[claim_type]
    left_plaintiff_side = profile.plaintiff_done_notice and profile.plaintiff_done_cure
    existing = renderer(profile, left_plaintiff_side, left_variant)

    if label == 1:
        new = renderer(profile, left_plaintiff_side, right_variant)
        scenario = "semantic_equivalent"
        risk_bucket = "positive"

        noise_tag = (
            "boundary"
            if rng.random() < 0.32
            else rng.choice(("clean", "mild", "heavy"))
        )

    else:
        high_overlap_negative = rng.random() < 0.28

        if high_overlap_negative:
            existing, new = _high_overlap_negative_pair(profile, rng)
            scenario = "opposite_conclusion_high_overlap"
            risk_bucket = "contradictory_high_overlap"

        else:
            new = renderer(profile, not left_plaintiff_side, right_variant)
            scenario = "opposite_conclusion"
            risk_bucket = "negative"

        noise_tag = "boundary" if rng.random() < 0.42 else rng.choice(("clean", "mild"))

    existing = _apply_noise(existing, rng, rng.choice(("clean", "mild")))
    new = _apply_noise(new, rng, noise_tag)
    split = "valid" if rng.random() < valid_ratio else "train"

    return {
        "pair_id": pair_id,
        "seed": int(rng.random() * 10_000_000),
        "claim_type": claim_type,
        "scenario": scenario,
        "noise_tag": noise_tag,
        "risk_bucket": risk_bucket,
        "label": int(label),
        "split": split,
        "existing_claim": existing,
        "new_claim": new,
        "style_pass": bool(_style_pass(existing) and _style_pass(new)),
    }


def _key_cases() -> List[Dict[str, Any]]:
    return [
        {
            "case_id": "KC_CLAIM_POS_001",
            "category": "positive",
            "label": 1,
            "description": "同一程序瑕疵主张的长短改写，期望合并。",
            "existing_claim": "2014年5月29日前原告未有效催告，解除程序存在瑕疵，不得据此主张惩罚性的双倍占用费。",
            "new_claim": "在2014年5月29日前，原告并未完成有效催告，解除权行使程序存在明显瑕疵，因此其主张惩罚性双倍占用费缺乏基础。",
        },
        {
            "case_id": "KC_CLAIM_POS_002",
            "category": "positive",
            "label": 1,
            "description": "同一条款解释主张的多分句改写，期望合并。",
            "existing_claim": "原告主张合同第三十五条'逾期30天视同提前终止'为特别约定故不适用催告程序不能成立。该条款系约定解除条件，但解除权行使仍应遵循正当程序，原告未给予合理履行期限即径行解除，剥夺被告补救机会，程序瑕疵导致解除效力存疑，双倍占用费主张缺乏合法基础。",
            "new_claim": "原告依据合同第三十五条“逾期30天视同提前终止”主张可免除催告程序，该观点不能成立。该条款仅约定解除条件，解除权的行使仍须遵循正当程序；原告未给予合理履行期限即直接解除，剥夺被告补救机会，故解除效力存疑，双倍占用费请求缺乏合法依据。",
        },
        {
            "case_id": "KC_CLAIM_NEG_003",
            "category": "negative",
            "label": 0,
            "description": "高重叠但结论相反，期望不合并。",
            "existing_claim": "2014年5月29日前原告未有效催告，解除程序存在瑕疵，不得据此主张惩罚性的双倍占用费。",
            "new_claim": "2014年5月29日前原告已有效催告，解除程序不存在瑕疵，可以据此主张双倍占用费。",
        },
        {
            "case_id": "KC_CLAIM_NEG_004",
            "category": "negative",
            "label": 0,
            "description": "同争点但核心法律结论反转，期望不合并。",
            "existing_claim": "原告主张合同第三十五条“逾期30天视同提前终止”不能排除催告程序，解除权行使仍应遵循正当程序。",
            "new_claim": "原告主张合同第三十五条“逾期30天视同提前终止”已排除催告程序，故可直接解除并当然主张双倍占用费。",
        },
        {
            "case_id": "KC_CLAIM_NEG_005",
            "category": "negative",
            "label": 0,
            "description": "补救机会判断相反，期望不合并。",
            "existing_claim": "原告未给予被告合理履行期限即径行解除，剥夺补救机会，程序瑕疵影响解除效力。",
            "new_claim": "原告已给予被告合理履行期限并完成催告，被告仍未补救，解除程序合法有效。",
        },
    ]


def generate_claim_dedup_dataset(
    *,
    seed: int,
    size: int = 600,
    valid_ratio: float = 0.2,
    positive_ratio: float = 0.45,
) -> Dict[str, Any]:
    """Build one synthetic CLAIM dedup dataset payload."""
    rng = random.Random(seed)
    claim_types = list(RENDERERS.keys())
    rows: List[Dict[str, Any]] = []

    for i in range(size):
        profile = _random_profile(rng)
        claim_type = rng.choice(claim_types)
        label = 1 if rng.random() < positive_ratio else 0

        rows.append(
            _build_pair(
                pair_id=f"CLAIM_PAIR_{seed}_{i:04d}",
                profile=profile,
                claim_type=claim_type,
                label=label,
                rng=rng,
                valid_ratio=valid_ratio,
            )
        )

    boundary_count = sum(1 for row in rows if row["noise_tag"] == "boundary")
    style_pass_count = sum(1 for row in rows if row["style_pass"])

    high_overlap_neg = sum(
        1 for row in rows if row["risk_bucket"] == "contradictory_high_overlap"
    )

    pos_count = sum(1 for row in rows if int(row["label"]) == 1)

    return {
        "seed": seed,
        "pairs": rows,
        "summary": {
            "size": size,
            "positive_count": pos_count,
            "negative_count": size - pos_count,
            "positive_ratio": pos_count / max(1, size),
            "boundary_ratio": boundary_count / max(1, size),
            "style_compliance_rate": style_pass_count / max(1, size),
            "high_overlap_negative_ratio": high_overlap_neg / max(1, size),
        },
        "key_cases": _key_cases(),
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
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--valid-ratio", type=float, default=0.2)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/claim_dedup_seed72.jsonl"),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/claim_dedup_seed72.summary.json"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = generate_claim_dedup_dataset(
        seed=args.seed,
        size=args.size,
        valid_ratio=args.valid_ratio,
    )

    _write_jsonl(args.output, payload["pairs"])
    _write_json(args.summary_output, payload["summary"])
    print("[claim-dedup-dataset] done")
    print(f"seed={args.seed}")
    print(f"size={payload['summary']['size']}")
    print(f"positive_ratio={payload['summary']['positive_ratio']:.4f}")
    print(f"boundary_ratio={payload['summary']['boundary_ratio']:.4f}")
    print(f"style_compliance_rate={payload['summary']['style_compliance_rate']:.4f}")

    print(
        "high_overlap_negative_ratio="
        f"{payload['summary']['high_overlap_negative_ratio']:.4f}"
    )

    print(f"output={args.output}")
    print(f"summary={args.summary_output}")


if __name__ == "__main__":
    main()
