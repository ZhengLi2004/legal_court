"""Feature extraction helpers shared by Step 05 data scripts."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np

CONTENT_SECTION_KEYS = ("原告诉称", "被告辩称", "审理查明", "法院观点", "裁判结果")

_CLAIM_PATTERNS = (
    re.compile(r"[（(]\s*\d+\s*[）)]"),
    re.compile(r"(?:^|[\n；;。])\s*\d+\s*[、.．]"),
)


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []

    return [value]


def count_claims_from_text(text: str) -> int:
    """Estimate the number of诉讼请求 items from claim text."""
    stripped = (text or "").strip()

    if not stripped:
        return 0

    match_count = max(len(pattern.findall(stripped)) for pattern in _CLAIM_PATTERNS)

    if match_count == 0:
        fallback = stripped.count("诉讼请求")

        if fallback == 0:
            fallback = stripped.count("请求")

        match_count = fallback

    return max(1, int(match_count))


def get_case_features(case: dict[str, Any]) -> dict[str, Any]:
    """Extract Step 05 stratification features from one case record."""
    meta_info = case.get("metaInfo")
    content = case.get("content")
    extra_info = case.get("extraInfo")

    if not isinstance(meta_info, dict):
        raise ValueError("Field `metaInfo` must be a JSON object")

    if not isinstance(content, dict):
        raise ValueError("Field `content` must be a JSON object")

    if not isinstance(extra_info, dict):
        raise ValueError("Field `extraInfo` must be a JSON object")

    uid = str(meta_info.get("uid", "")).strip()

    if not uid:
        raise ValueError("Field `metaInfo.uid` is required")

    sample_cause = str(extra_info.get("sample_cause", "")).strip()

    if not sample_cause:
        raise ValueError(f"Field `extraInfo.sample_cause` is required for uid={uid}")

    cause_list = _ensure_list(meta_info.get("案由"))
    person_list = _ensure_list(meta_info.get("人物信息"))
    law_list = _ensure_list(meta_info.get("法律条款"))
    claim_text = str(content.get("原告诉称", "") or "")
    claim_count = count_claims_from_text(claim_text)
    full_text = "".join(str(content.get(key, "") or "") for key in CONTENT_SECTION_KEYS)

    return {
        "uid": uid,
        "sample_cause": sample_cause,
        "cause_count": len(cause_list),
        "person_count": len(person_list),
        "law_count": len(law_list),
        "claim_count": claim_count,
        "text_len": len(full_text),
        "content_sections": {
            key: len(str(content.get(key, "") or "")) for key in CONTENT_SECTION_KEYS
        },
    }


def q_value(
    values: Iterable[float], q: float, *, as_int_ceil: bool = True
) -> float | int:
    """Compute quantile value with optional ceil-to-int conversion."""
    arr = np.asarray(list(values), dtype=float)

    if arr.size == 0:
        raise ValueError("Cannot compute quantile for empty values")

    value = float(np.quantile(arr, q))

    if as_int_ceil:
        return int(math.ceil(value))

    return value
