"""Generate synthetic FACT dedup pairs for tuning `dedup.fact_threshold`.

All samples are self-constructed and follow legal fact-extraction sentence style.
No ES/database sampling is used.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class LeaseProfile:
    """Structured lease-fact slots used to render fact sentences."""

    sign_date: str
    start_date: str
    end_date: str
    lessor: str
    lessee: str
    location: str
    building: str
    use_text: str
    monthly_rent: int
    deposit: int


SIGN_DATES: Sequence[str] = (
    "2013年2月1日",
    "2014年5月9日",
    "2015年8月18日",
    "2016年11月3日",
    "2017年6月20日",
    "2018年4月12日",
    "2019年9月6日",
    "2020年1月15日",
    "2021年7月30日",
    "2022年3月10日",
)

LESSORS: Sequence[str] = (
    "天津泰东实业有限公司",
    "天津鸿拓资产管理有限公司",
    "天津联港工业发展有限公司",
    "天津新裕物业管理有限公司",
)

LESSEES: Sequence[str] = (
    "天津恒启机械制造有限公司",
    "天津凌川包装科技有限公司",
    "天津跃成物流有限公司",
    "天津尚维电子有限公司",
)

LOCATIONS: Sequence[str] = (
    "天津开发区第十三大街116号(出口加工区内)",
    "天津开发区第十一大街88号",
    "天津滨海新区海通路56号",
    "天津经济技术开发区渤海路102号",
)

BUILDINGS: Sequence[str] = (
    "泰东工业厂房1栋、2栋",
    "联港工业园A区3栋",
    "渤海产业园B区5栋、6栋",
    "海通标准化厂房2栋",
)

USE_TEXTS: Sequence[str] = (
    "生产、办公、存货",
    "生产、仓储、物流周转",
    "研发、办公、产品测试",
    "生产、展示、仓储",
)

MONTHLY_RENTS: Sequence[int] = (86000, 92000, 98000, 108000, 126000, 138000)
DEPOSITS: Sequence[int] = (120000, 180000, 200000, 260000, 300000)

ACTION_TOKENS: Sequence[str] = (
    "签订",
    "订立",
    "签署",
    "约定",
    "出租",
    "承租",
    "支付",
    "交付",
)


def _month_forward(date_text: str, months: int) -> str:
    year = int(date_text[:4])
    month_start = date_text.find("年") + 1
    month_end = date_text.find("月")
    day_end = date_text.find("日")
    month = int(date_text[month_start:month_end])
    day = int(date_text[month_end + 1 : day_end])
    month += months
    year += (month - 1) // 12
    month = ((month - 1) % 12) + 1
    return f"{year}年{month}月{day}日"


def _random_profile(rng: random.Random) -> LeaseProfile:
    sign_date = rng.choice(list(SIGN_DATES))
    start_date = sign_date
    end_date = _month_forward(sign_date, rng.choice([24, 30, 36, 48]))
    lessor = rng.choice(list(LESSORS))
    lessee = rng.choice(list(LESSEES))

    while lessor == lessee:
        lessee = rng.choice(list(LESSEES))

    return LeaseProfile(
        sign_date=sign_date,
        start_date=start_date,
        end_date=end_date,
        lessor=lessor,
        lessee=lessee,
        location=rng.choice(list(LOCATIONS)),
        building=rng.choice(list(BUILDINGS)),
        use_text=rng.choice(list(USE_TEXTS)),
        monthly_rent=rng.choice(list(MONTHLY_RENTS)),
        deposit=rng.choice(list(DEPOSITS)),
    )


def _render_sign(profile: LeaseProfile, variant: int) -> str:
    templates = [
        "{date}，原告(甲方){lessor}与被告(乙方){lessee}签订厂房租赁合同。",
        "原告(甲方){lessor}与被告(乙方){lessee}于{date}订立厂房租赁合同。",
        "{date}，甲方{lessor}与乙方{lessee}签署厂房租赁合同并约定租赁关系。",
    ]

    return templates[variant % len(templates)].format(
        date=profile.sign_date,
        lessor=profile.lessor,
        lessee=profile.lessee,
    )


def _render_subject(profile: LeaseProfile, variant: int) -> str:
    templates = [
        "合同约定甲方将{location}{building}出租给乙方。",
        "双方在合同中明确：甲方将位于{location}的{building}出租给乙方使用。",
        "租赁标的为{location}{building}，由甲方出租、乙方承租。",
    ]

    return templates[variant % len(templates)].format(
        location=profile.location,
        building=profile.building,
    )


def _render_usage(profile: LeaseProfile, variant: int) -> str:
    templates = [
        "乙方承租厂房的用途为{use_text}。",
        "合同明确乙方承租用途包括{use_text}。",
        "关于使用目的，双方约定乙方可用于{use_text}。",
    ]

    return templates[variant % len(templates)].format(use_text=profile.use_text)


def _render_rent(profile: LeaseProfile, variant: int) -> str:
    templates = [
        "合同约定月租金为人民币{rent}元，按月支付。",
        "双方约定乙方每月应向甲方支付租金{rent}元。",
        "租金标准确定为每月{rent}元，支付周期为月结。",
    ]

    return templates[variant % len(templates)].format(rent=profile.monthly_rent)


def _render_term(profile: LeaseProfile, variant: int) -> str:
    templates = [
        "租赁期限自{start}起至{end}止。",
        "合同约定的租期为{start}至{end}。",
        "双方确认租赁期间为{start}至{end}。",
    ]

    return templates[variant % len(templates)].format(
        start=profile.start_date,
        end=profile.end_date,
    )


def _render_deposit(profile: LeaseProfile, variant: int) -> str:
    templates = [
        "乙方应向甲方支付保证金人民币{deposit}元。",
        "合同约定履约保证金金额为{deposit}元，由乙方支付。",
        "关于保证金，双方约定乙方需缴纳{deposit}元。",
    ]

    return templates[variant % len(templates)].format(deposit=profile.deposit)


FACT_RENDERERS = {
    "sign": _render_sign,
    "subject": _render_subject,
    "usage": _render_usage,
    "rent": _render_rent,
    "term": _render_term,
    "deposit": _render_deposit,
}


def _apply_noise(text: str, rng: random.Random, level: str) -> str:
    if level == "clean":
        return text

    prefix_pool = ["经审理查明，", "另查明，", "合同文本载明，", "证据显示，"]

    suffix_pool = [
        "该事实见租赁合同条款。",
        "双方对该条款无实质争议。",
        "该事项有书面约定。",
    ]

    noisy = text

    if level in {"mild", "boundary", "heavy"} and rng.random() < 0.55:
        noisy = rng.choice(prefix_pool) + noisy

    if level in {"boundary", "heavy"} and rng.random() < 0.45:
        noisy = noisy.replace("，", "、", 1)

    if level == "heavy" and rng.random() < 0.65:
        noisy = noisy + rng.choice(suffix_pool)

    if level == "boundary" and rng.random() < 0.55:
        noisy = noisy.replace("合同约定", "双方在合同中约定")

    return noisy


def _mutate_for_negative(
    profile: LeaseProfile, fact_type: str, rng: random.Random
) -> tuple[LeaseProfile, str]:
    if fact_type == "sign":
        if rng.random() < 0.5:
            changed = replace(
                profile,
                sign_date=rng.choice([d for d in SIGN_DATES if d != profile.sign_date]),
            )

            return changed, "date_conflict"

        changed = replace(
            profile,
            lessee=rng.choice([x for x in LESSEES if x != profile.lessee]),
        )

        return changed, "party_conflict"

    if fact_type == "subject":
        if rng.random() < 0.5:
            changed = replace(
                profile,
                building=rng.choice([b for b in BUILDINGS if b != profile.building]),
            )

            return changed, "building_conflict"

        changed = replace(
            profile,
            location=rng.choice([x for x in LOCATIONS if x != profile.location]),
        )

        return changed, "location_conflict"

    if fact_type == "usage":
        changed = replace(
            profile,
            use_text=rng.choice([u for u in USE_TEXTS if u != profile.use_text]),
        )

        return changed, "usage_conflict"

    if fact_type == "rent":
        changed = replace(
            profile,
            monthly_rent=rng.choice(
                [x for x in MONTHLY_RENTS if x != profile.monthly_rent]
            ),
        )

        return changed, "amount_conflict"

    if fact_type == "term":
        new_end = _month_forward(profile.start_date, rng.choice([12, 18, 60]))
        changed = replace(profile, end_date=new_end)
        return changed, "term_conflict"

    changed = replace(
        profile,
        deposit=rng.choice([x for x in DEPOSITS if x != profile.deposit]),
    )

    return changed, "deposit_conflict"


def _style_flags(text: str) -> Dict[str, bool]:
    has_date = ("年" in text and "月" in text and "日" in text) or ("期限" in text)
    has_party = any(token in text for token in ("甲方", "乙方", "原告", "被告"))
    has_action = any(token in text for token in ACTION_TOKENS)

    has_lease_object = any(
        token in text for token in ("厂房", "租赁", "租金", "保证金", "用途")
    )

    has_fact_clause = any(
        token in text for token in ("合同", "约定", "双方", "乙方", "甲方")
    )

    return {
        "has_date": has_date,
        "has_party": has_party,
        "has_action": has_action,
        "has_lease_object": has_lease_object,
        "style_pass": has_action
        and has_lease_object
        and (has_party or has_date or has_fact_clause),
    }


def _build_pair(
    *,
    pair_id: str,
    profile: LeaseProfile,
    fact_type: str,
    label: int,
    rng: random.Random,
    valid_ratio: float,
) -> Dict[str, Any]:
    left_variant = rng.randrange(0, 3)
    right_variant = (left_variant + rng.randrange(1, 3)) % 3
    renderer = FACT_RENDERERS[fact_type]
    base_left = renderer(profile, left_variant)

    if label == 1:
        scenario = "semantic_equivalent"
        profile_right = profile

        noise_tag = (
            "boundary"
            if rng.random() < 0.32
            else rng.choice(["clean", "mild", "heavy"])
        )

    else:
        profile_right, scenario = _mutate_for_negative(profile, fact_type, rng)
        noise_tag = "boundary" if rng.random() < 0.38 else rng.choice(["clean", "mild"])

    base_right = renderer(profile_right, right_variant)
    left_text = _apply_noise(base_left, rng, rng.choice(["clean", "mild"]))
    right_text = _apply_noise(base_right, rng, noise_tag)
    split = "valid" if rng.random() < valid_ratio else "train"
    flags = _style_flags(left_text)
    right_flags = _style_flags(right_text)

    return {
        "pair_id": pair_id,
        "seed": int(rng.random() * 10_000_000),
        "fact_type": fact_type,
        "scenario": scenario,
        "noise_tag": noise_tag,
        "label": int(label),
        "split": split,
        "existing_fact": left_text,
        "new_fact": right_text,
        "style_pass": bool(flags["style_pass"] and right_flags["style_pass"]),
    }


def _key_cases() -> List[Dict[str, Any]]:
    return [
        {
            "case_id": "KC_POS_SIGN_001",
            "category": "positive",
            "label": 1,
            "description": "同一签约事实改写，期望合并。",
            "existing_fact": "2013年2月1日，原告(甲方)与被告(乙方)签订厂房租赁合同。",
            "new_fact": "2013年2月1日，甲方与乙方签署厂房租赁合同并确认租赁关系。",
        },
        {
            "case_id": "KC_POS_SUBJECT_002",
            "category": "positive",
            "label": 1,
            "description": "同一租赁标的不同句式，期望合并。",
            "existing_fact": "合同约定甲方将天津开发区第十三大街116号(出口加工区内)泰东工业厂房1栋、2栋出租给乙方。",
            "new_fact": "双方在合同中明确：甲方将位于天津开发区第十三大街116号(出口加工区内)的泰东工业厂房1栋、2栋出租给乙方使用。",
        },
        {
            "case_id": "KC_NEG_USAGE_003",
            "category": "negative",
            "label": 0,
            "description": "用途槽位冲突，期望不合并。",
            "existing_fact": "乙方承租厂房的用途为生产、办公、存货。",
            "new_fact": "乙方承租厂房的用途为仓储、展示、培训。",
        },
        {
            "case_id": "KC_NEG_RENT_004",
            "category": "negative",
            "label": 0,
            "description": "租金槽位冲突，期望不合并。",
            "existing_fact": "合同约定月租金为人民币98000元，按月支付。",
            "new_fact": "合同约定月租金为人民币128000元，按月支付。",
        },
        {
            "case_id": "KC_NEG_BUILDING_005",
            "category": "negative",
            "label": 0,
            "description": "栋号槽位冲突，期望不合并。",
            "existing_fact": "合同约定甲方将天津开发区第十三大街116号(出口加工区内)泰东工业厂房1栋、2栋出租给乙方。",
            "new_fact": "合同约定甲方将天津开发区第十三大街116号(出口加工区内)泰东工业厂房3栋、4栋出租给乙方。",
        },
    ]


def generate_fact_dedup_dataset(
    *,
    seed: int,
    size: int = 500,
    valid_ratio: float = 0.2,
    positive_ratio: float = 0.45,
) -> Dict[str, Any]:
    """Build one synthetic fact-dedup dataset payload."""
    rng = random.Random(seed)
    fact_types = list(FACT_RENDERERS.keys())
    pairs: List[Dict[str, Any]] = []

    for i in range(size):
        profile = _random_profile(rng)
        fact_type = rng.choice(fact_types)
        label = 1 if rng.random() < positive_ratio else 0

        pair = _build_pair(
            pair_id=f"FACT_PAIR_{seed}_{i:04d}",
            profile=profile,
            fact_type=fact_type,
            label=label,
            rng=rng,
            valid_ratio=valid_ratio,
        )

        pairs.append(pair)

    boundary_count = sum(1 for row in pairs if row["noise_tag"] == "boundary")
    style_pass_count = sum(1 for row in pairs if row["style_pass"])
    pos_count = sum(1 for row in pairs if int(row["label"]) == 1)

    return {
        "seed": seed,
        "pairs": pairs,
        "summary": {
            "size": size,
            "positive_count": pos_count,
            "negative_count": size - pos_count,
            "positive_ratio": pos_count / max(1, size),
            "boundary_ratio": boundary_count / max(1, size),
            "style_compliance_rate": style_pass_count / max(1, size),
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
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--valid-ratio", type=float, default=0.2)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/fact_dedup_seed62.jsonl"),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmarks/optim/artifacts/fact_dedup_seed62.summary.json"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = generate_fact_dedup_dataset(
        seed=args.seed,
        size=args.size,
        valid_ratio=args.valid_ratio,
    )

    _write_jsonl(args.output, payload["pairs"])
    _write_json(args.summary_output, payload["summary"])
    print("[fact-dedup-dataset] done")
    print(f"seed={args.seed}")
    print(f"size={payload['summary']['size']}")
    print(f"positive_ratio={payload['summary']['positive_ratio']:.4f}")
    print(f"boundary_ratio={payload['summary']['boundary_ratio']:.4f}")
    print(f"style_compliance_rate={payload['summary']['style_compliance_rate']:.4f}")
    print(f"output={args.output}")
    print(f"summary={args.summary_output}")


if __name__ == "__main__":
    main()
