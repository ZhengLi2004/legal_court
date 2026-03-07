"""Claim extraction helpers for Step 06 Gold Claim construction."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Protocol

from cn2an import cn2an
from rapidfuzz import fuzz

PRIMARY_SECTION = "原告诉称"


class SyntaxSplitter(Protocol):
    """Protocol for optional HanLP-backed local splitting."""

    def split(self, text: str, action_aliases: list[str]) -> list[tuple[str, int, int]]:
        """Split one composite claim string into claim spans."""


@dataclass(frozen=True)
class RequestBlockResult:
    """Store one located request block inside plaintiff text."""

    text: str
    source: str
    cue: str
    start: int
    end: int
    body_start: int
    body_end: int

    @property
    def body_text(self) -> str:
        return self.text[self.body_start - self.start : self.body_end - self.start]


@dataclass(frozen=True)
class SourceSpan:
    """Absolute span in `content.原告诉称`."""

    section: str
    start: int
    end: int


@dataclass
class ClaimItem:
    """One finalized claim row before serialization."""

    claim_text_raw: str
    claim_text_norm: str
    source_span: SourceSpan
    request_block_source: str
    boundary_source: str
    claim_source: str
    action: str
    target: str
    amount: str
    liability: str
    stage_role: str
    claim_flags: list[str] = field(default_factory=list)


@dataclass
class RuleExtractionResult:
    """Store rule-path extraction result for one case."""

    request_block: RequestBlockResult | None
    claims: list[ClaimItem]
    trigger_reasons: list[str]
    auto_candidates: list[str]

    @property
    def needs_expert_review(self) -> bool:
        return bool(self.trigger_reasons)


@dataclass
class CaseClaimExtractionResult:
    """Final extraction result for one case."""

    uid: str
    split: str
    hard_case: int
    sample_cause: str
    stage: str
    claims: list[ClaimItem]
    request_block: RequestBlockResult | None
    auto_candidates: list[str]
    review_reasons: list[str]
    needs_manual_review: bool
    claim_source: str


@dataclass(frozen=True)
class RuleBundle:
    """Frozen extraction rules compiled from JSON."""

    first_instance_label_cues: tuple[str, ...]
    first_instance_inline_cues: tuple[str, ...]
    first_instance_long_tail_cues: tuple[str, ...]
    appeal_label_cues: tuple[str, ...]
    appeal_inline_cues: tuple[str, ...]
    section_stop_cues: tuple[str, ...]
    procedural_fee_keywords: tuple[str, ...]
    meta_appeal_patterns: tuple[str, ...]
    split_connectors: tuple[str, ...]
    petition_prefixes: tuple[str, ...]
    actor_prefix_patterns: tuple[str, ...]
    negative_request_prefix_patterns: tuple[str, ...]
    label_follow_prefixes: tuple[str, ...]
    evidence_like_keywords: tuple[str, ...]
    obvious_request_markers: tuple[str, ...]
    argument_like_patterns: tuple[str, ...]
    no_action_max_chars: int
    short_claim_chars: int


@dataclass(frozen=True)
class NumberMarker:
    """One numbered-item marker inside request text."""

    level: int
    start: int
    end: int
    kind: str
    marker_text: str


@dataclass
class NumberNode:
    """One node in the numbered request tree."""

    marker: NumberMarker
    body_end: int | None = None
    children: list["NumberNode"] = field(default_factory=list)


_NUMBER_PATTERNS: tuple[tuple[str, int, re.Pattern[str]], ...] = (
    ("cn_top", 1, re.compile(r"(?:^|[\n；;。:：])\s*([一二三四五六七八九十]+)[、.]")),
    ("arabic_top", 1, re.compile(r"(?:^|[\n；;。:：])\s*(\d+)[、.．]")),
    (
        "cn_paren",
        2,
        re.compile(r"(?:^|[\n；;。:：])\s*[（(]([一二三四五六七八九十]+)[)）]"),
    ),
    ("arabic_paren", 3, re.compile(r"(?:^|[\n；;。:：])\s*[（(](\d+)[)）]")),
)

_AMOUNT_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?(?:万|亿)?元|[零一二三四五六七八九十百千万亿两]+(?:\.[零一二三四五六七八九十百千万亿两]+)?(?:万|亿)?元)"
)

_MULTI_SPACE_RE = re.compile(r"\s+")


def load_rule_bundle(path: str | Path) -> RuleBundle:
    """Load frozen Step 06 extraction rules from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    return RuleBundle(
        first_instance_label_cues=tuple(data["first_instance"]["label_cues"]),
        first_instance_inline_cues=tuple(data["first_instance"]["inline_cues"]),
        first_instance_long_tail_cues=tuple(
            data["first_instance"].get("long_tail_cues", [])
        ),
        appeal_label_cues=tuple(data["appeal"]["label_cues"]),
        appeal_inline_cues=tuple(data["appeal"]["inline_cues"]),
        section_stop_cues=tuple(data["section_stop_cues"]),
        procedural_fee_keywords=tuple(data["procedural_fee_keywords"]),
        meta_appeal_patterns=tuple(data["meta_appeal_patterns"]),
        split_connectors=tuple(data["split_connectors"]),
        petition_prefixes=tuple(data["petition_prefixes"]),
        actor_prefix_patterns=tuple(data["actor_prefix_patterns"]),
        negative_request_prefix_patterns=tuple(
            data.get("negative_request_prefix_patterns", [])
        ),
        label_follow_prefixes=tuple(data.get("label_follow_prefixes", [])),
        evidence_like_keywords=tuple(data.get("evidence_like_keywords", [])),
        obvious_request_markers=tuple(data.get("obvious_request_markers", [])),
        argument_like_patterns=tuple(data.get("argument_like_patterns", [])),
        no_action_max_chars=int(data.get("no_action_max_chars", 120)),
        short_claim_chars=int(data.get("short_claim_chars", 4)),
    )


def load_claim_ontology(path: str | Path) -> dict[str, Any]:
    """Load frozen claim ontology JSON."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


class HanLPTokenizerSplitter:
    """Lightweight HanLP tokenizer-backed splitter.

    The implementation intentionally uses the tokenizer model only. It avoids
    hard-coding cache paths and relies on `HANLP_HOME` / `HF_HOME` when set by
    the caller environment.
    """

    def __init__(self) -> None:
        self._tokenizer: Any | None = None

    def _load(self) -> Any:
        if self._tokenizer is None:
            import hanlp

            self._tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

        return self._tokenizer

    @staticmethod
    def _align_tokens(text: str, tokens: list[str]) -> list[tuple[str, int, int]]:
        aligned: list[tuple[str, int, int]] = []
        cursor = 0

        for token in tokens:
            idx = text.find(token, cursor)

            if idx < 0:
                continue

            end = idx + len(token)
            aligned.append((token, idx, end))
            cursor = end

        return aligned

    def split(self, text: str, action_aliases: list[str]) -> list[tuple[str, int, int]]:
        tokenizer = self._load()
        tokens = tokenizer(text)

        if not isinstance(tokens, list):
            return []

        aligned = self._align_tokens(text, [str(token) for token in tokens])

        if not aligned:
            return []

        connector_tokens = {
            "并",
            "并且",
            "且",
            "及",
            "并向",
            "并支付",
            "并赔偿",
            "并返还",
            "并确认",
            "并解除",
            "并承担",
        }

        spans: list[tuple[int, int]] = []
        last_start = 0

        for idx, (token, start, end) in enumerate(aligned):
            if token not in connector_tokens:
                continue

            left = text[last_start:start].strip(" ，,；;。:\n\t")
            right = text[end:].strip(" ，,；;。:\n\t")

            if not left or not right:
                continue

            if not any(alias in left for alias in action_aliases):
                continue

            if not any(alias in right for alias in action_aliases):
                continue

            spans.append((last_start, start))
            last_start = end

        if not spans:
            return []

        spans.append((last_start, len(text)))
        results: list[tuple[str, int, int]] = []

        for start, end in spans:
            segment = text[start:end].strip(" ，,；;。:\n\t")

            if not segment:
                continue

            seg_start = text.find(segment, start, end)

            if seg_start < 0:
                continue

            results.append((segment, seg_start, seg_start + len(segment)))

        return results


def normalize_space(text: str) -> str:
    """Collapse whitespace while keeping original punctuation."""

    return _MULTI_SPACE_RE.sub("", text or "").strip()


def _stage_key(stage: str) -> str:
    return "appeal" if str(stage).strip() == "二审" else "first_instance"


def _all_action_aliases(ontology: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    actions = ontology.get("actions", {})

    for variants in actions.values():
        aliases.extend(str(item) for item in variants)

    return sorted(set(aliases), key=len, reverse=True)


def _iter_cue_hits(text: str, cues: tuple[str, ...]) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []

    for cue in cues:
        cursor = 0

        while True:
            idx = text.find(cue, cursor)

            if idx < 0:
                break

            hits.append((idx, cue))
            cursor = idx + len(cue)

    return sorted(hits, key=lambda item: (item[0], len(item[1])))


def _starts_with_number_marker(text: str) -> bool:
    stripped = text.lstrip(" \n\t")

    return bool(
        re.match(
            r"(?:[一二三四五六七八九十]+[、.]|\d+[、.．]|[（(][一二三四五六七八九十]+[)）]|[（(]\d+[)）])",
            stripped,
        )
    )


def _has_negative_request_prefix(text: str, start: int, rules: RuleBundle) -> bool:
    window = normalize_space(text[max(0, start - 24) : start])

    return any(
        re.search(pattern, window) for pattern in rules.negative_request_prefix_patterns
    )


def _has_obvious_request_marker(text: str, rules: RuleBundle) -> bool:
    normalized = normalize_space(text)

    return any(
        re.search(pattern, normalized) for pattern in rules.obvious_request_markers
    )


def _has_stage_obvious_request_marker(text: str, stage: str, rules: RuleBundle) -> bool:
    normalized = normalize_space(text)

    if "判如诉请" in normalized or "判如所请" in normalized:
        return False

    if _stage_key(stage) == "appeal":
        patterns = (
            r"提起上诉.*请求[:：]",
            r"向本院提出上诉.*请求[:：]",
            r"向本院提起上诉.*请求[:：]",
            r"上诉称.*请求",
            r"请求二审",
            r"请求撤销(?:原判|原审判决|一审判决)",
            r"请求改判",
        )

        return any(re.search(pattern, normalized) for pattern in patterns)

    patterns = (
        r"现起诉要求",
        r"现起诉诉请",
        r"起诉要求",
        r"故诉至法院请求判令",
        r"诉至法院要求",
        r"诉至法院，请求",
        r"请求法院判令",
        r"请求人民法院(?:依法)?",
        r"要求判令",
        r"诉请法院判令",
        r"诉请判令",
        r"现我请求",
        r"为此，我请求",
        r"特向法院提起诉讼",
        r"诉诸法律",
        r"请求[:：]",
        r"请求判令",
    )

    return any(re.search(pattern, normalized) for pattern in patterns)


def _valid_label_follow_text(
    text: str, start: int, cue: str, rules: RuleBundle
) -> bool:
    tail = text[start + len(cue) :].lstrip(" \n\t")

    if not tail:
        return False

    if tail[0] in "：:，,":
        return True

    return any(tail.startswith(prefix) for prefix in rules.label_follow_prefixes)


def _valid_inline_follow_text(
    text: str, start: int, cue: str, rules: RuleBundle
) -> bool:
    strict_cues = {
        "故诉至法院",
        "诉至法院",
        "现诉至法院要求",
        "诉至法院要求",
        "现起诉",
        "现起诉诉请",
        "现我请求",
        "为此，我请求",
        "为此请求",
        "请求人民法院",
        "请求人民法院依法",
        "要求判令",
        "特向法院提起诉讼",
        "诉诸法律",
        "诉诸法律，要求",
        "诉诸法律,要求",
        "请求判决",
        "请求:",
        "请求：",
        "要求被告",
    }

    if cue not in strict_cues:
        return True

    tail = text[start + len(cue) :].lstrip(" \n\t")

    if not tail:
        return False

    if tail[0] in "：:，,":
        tail = tail[1:].lstrip(" \n\t")

    if not tail:
        return False

    if _starts_with_number_marker(tail):
        return True

    if cue.endswith("判令") and re.match(
        r"^[\u4e00-\u9fffA-Za-z0-9×xX某、，,（）()]{1,20}(支付|给付|赔偿|返还|承担|继续履行|确认|解除|停止|拆除|恢复|分割|交付|过户|办理|补缴)",
        tail,
    ):
        return True

    return any(tail.startswith(prefix) for prefix in rules.label_follow_prefixes)


def _find_request_candidate(
    text: str,
    *,
    cues: tuple[str, ...],
    source: str,
    rules: RuleBundle,
) -> list[tuple[int, str, str]]:
    matches: list[tuple[int, str, str]] = []

    for idx, cue in _iter_cue_hits(text, cues):
        if _has_negative_request_prefix(text, idx, rules):
            continue

        if source == "label_cue" and not _valid_label_follow_text(
            text, idx, cue, rules
        ):
            continue

        if source == "inline_cue" and not _valid_inline_follow_text(
            text, idx, cue, rules
        ):
            continue

        matches.append((idx, source, cue))

    return matches


def _clip_numbered_request_end(body_text: str) -> int | None:
    if not _starts_with_number_marker(body_text):
        return None

    marker_count = len(_collect_markers(body_text))

    if marker_count < 2:
        return None

    for match in re.finditer(r"[。；;]", body_text):
        tail = body_text[match.end() :].lstrip(" \n\t")

        if not tail:
            return match.end()

        if _starts_with_number_marker(tail):
            continue

        return match.end()

    return None


def _find_request_end(text: str, *, start: int, stage: str, rules: RuleBundle) -> int:
    end_candidates = [
        idx
        for stop in rules.section_stop_cues
        for idx in [text.find(stop, start)]
        if idx >= 0
    ]

    dynamic_patterns = [
        r"[\u4e00-\u9fffA-Za-z0-9×xX某、，,]{1,30}辩称[:：，,]",
        r"[\u4e00-\u9fffA-Za-z0-9×xX某、，,]{1,30}为证明.{0,20}(诉讼主张|诉讼请求|主张)",
        r"[\u4e00-\u9fffA-Za-z0-9×xX某、，,]{1,30}向本院提交.{0,20}证据",
        r"其余各案被告在举证期间内均未",
    ]

    if _stage_key(stage) == "appeal":
        dynamic_patterns.append(r"[\u4e00-\u9fffA-Za-z0-9×xX某、，,]{1,30}上诉称[:：]")

    tail = text[start:]

    for pattern in dynamic_patterns:
        match = re.search(pattern, tail)

        if match:
            end_candidates.append(start + match.start())

    return min(end_candidates) if end_candidates else len(text)


def extract_request_block(
    plaintiff_text: str,
    *,
    stage: str,
    rules: RuleBundle,
) -> RequestBlockResult | None:
    """Locate the request block inside `content.原告诉称`."""

    text = plaintiff_text or ""
    stage_key = _stage_key(stage)

    if stage_key == "appeal":
        label_cues = rules.appeal_label_cues
        inline_cues = rules.appeal_inline_cues

    else:
        label_cues = rules.first_instance_label_cues
        inline_cues = rules.first_instance_inline_cues

    candidates = _find_request_candidate(
        text, cues=label_cues, source="label_cue", rules=rules
    )

    candidates.extend(
        _find_request_candidate(
            text, cues=inline_cues, source="inline_cue", rules=rules
        )
    )

    if not candidates:
        return None

    start, source, cue = min(candidates, key=lambda item: item[0])
    end = _find_request_end(text, start=start + len(cue), stage=stage, rules=rules)
    body_start = start

    if source == "label_cue":
        body_start = start + len(cue)

        while body_start < end and text[body_start] in "：: \n\t":
            body_start += 1

        while body_start < end and text[body_start] in "，,":
            body_start += 1

    elif cue in {"请求:", "请求："}:
        body_start = start + len(cue)

        while body_start < end and text[body_start] in "：:，, \n\t":
            body_start += 1

    clipped_end = _clip_numbered_request_end(text[body_start:end])

    if clipped_end is not None:
        end = body_start + clipped_end

    return RequestBlockResult(
        text=text[start:end],
        source=source,
        cue=cue,
        start=start,
        end=end,
        body_start=body_start,
        body_end=end,
    )


def _collect_markers(text: str) -> list[NumberMarker]:
    markers: list[NumberMarker] = []
    seen: set[tuple[int, int]] = set()

    for kind, level, pattern in _NUMBER_PATTERNS:
        for match in pattern.finditer(text):
            start = match.start(1)
            marker_text = match.group(0)
            marker_end = match.end(0)
            key = (start, marker_end)

            if key in seen:
                continue

            seen.add(key)

            markers.append(
                NumberMarker(
                    level=level,
                    start=start,
                    end=marker_end,
                    kind=kind,
                    marker_text=marker_text,
                )
            )

    return sorted(markers, key=lambda item: (item.start, item.level))


def _build_number_tree(text: str) -> tuple[list[NumberNode], bool]:
    markers = _collect_markers(text)

    if not markers:
        return [], False

    roots: list[NumberNode] = []
    stack: list[NumberNode] = []
    conflict = False

    for marker in markers:
        while stack and marker.level <= stack[-1].marker.level:
            node = stack.pop()

            if node.body_end is None:
                node.body_end = marker.start

        if stack:
            parent = stack[-1]

            if parent.body_end is None:
                parent.body_end = marker.start

            if marker.start <= parent.marker.end:
                conflict = True

            parent.children.append(NumberNode(marker=marker))
            stack.append(parent.children[-1])

        else:
            node = NumberNode(marker=marker)
            roots.append(node)
            stack.append(node)

    for node in stack:
        if node.body_end is None:
            node.body_end = len(text)

    return roots, conflict


def _body_text_from_node(text: str, node: NumberNode) -> tuple[str, int, int]:
    start = node.marker.end
    end = node.body_end if node.body_end is not None else len(text)
    segment = text[start:end].strip(" ：:\n\t；;，,")

    if not segment:
        return "", start, end

    seg_start = text.find(segment, start, end)

    if seg_start < 0:
        seg_start = start

    return segment, seg_start, seg_start + len(segment)


def _has_nested_numbering(nodes: list[NumberNode]) -> bool:
    for node in nodes:
        if node.children or _has_nested_numbering(node.children):
            return True

    return False


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        ordered.append(value)

    return ordered


def _contains_substantive_action(text: str, ontology: dict[str, Any]) -> bool:
    normalized = normalize_space(text)
    return any(alias in normalized for alias in _all_action_aliases(ontology))


def _is_procedural_fee_claim(text: str, rules: RuleBundle) -> bool:
    normalized = normalize_space(text)
    return any(keyword in normalized for keyword in rules.procedural_fee_keywords)


def _has_mixed_substantive_fee_components(text: str) -> bool:
    normalized = normalize_space(text)

    substantive_markers = (
        "赔偿",
        "赔付",
        "损失",
        "医疗费",
        "护理费",
        "交通费",
        "伙食补助",
        "误工费",
        "残疾赔偿金",
        "抚慰金",
        "补助费",
        "支付",
        "返还",
        "违约金",
        "货款",
        "工程款",
        "借款",
        "价款",
    )

    return any(marker in normalized for marker in substantive_markers)


def _is_evidence_like_text(text: str, rules: RuleBundle) -> bool:
    normalized = normalize_space(text)

    invoice_substantive_patterns = (
        r"确认.+发票违法",
        r"确认.+发票无效",
        r"(重新给付发票|给付发票|开具发票|补开发票)",
    )

    if any(re.search(pattern, normalized) for pattern in invoice_substantive_patterns):
        return False

    if any(keyword in normalized for keyword in rules.evidence_like_keywords):
        return True

    return bool(
        re.search(
            r"(复印件|结婚证|身份证|户口薄|户口本|收据|发票|证明|报告|笔录|清单|照片|通知书).{0,8}(一份|一张|一组|一册|数张|复印件)",
            normalized,
        )
    )


def _is_argument_like_text(text: str, rules: RuleBundle) -> bool:
    normalized = normalize_space(text)

    return any(
        re.search(pattern, normalized) for pattern in rules.argument_like_patterns
    )


def _is_auxiliary_request(text: str) -> bool:
    normalized = normalize_space(text)

    if normalized in {
        "特向法院提起诉讼",
        "故诉至法院",
        "请求二审法院改判一审判决不当的部分",
    }:
        return True

    if re.match(
        r"^(其余计算至|截止到|截止至|截至|暂计算至).{0,60}(之日止|行为止|付清之日止)$",
        normalized,
    ):
        return True

    if normalized.startswith("暂计") and bool(_extract_first_amount(normalized)):
        return True

    if any(marker in normalized for marker in ("致歉内容", "法院依法审核", "致歉版面")):
        return True

    if re.search(r"请求驳回原告.{0,20}诉讼请求", normalized):
        return True

    if re.search(r"驳回.+申请", normalized):
        return True

    if re.search(r"^改判原判第[一二三四五六七八九十0-9]+项为", normalized):
        return True

    if normalized.startswith("总计") and bool(_extract_first_amount(normalized)):
        return True

    if normalized.startswith(("请求依法判决", "依法维护")):
        return True

    if "判如诉请" in normalized or "判如所请" in normalized:
        return True

    substantive_markers = (
        "支付",
        "给付",
        "赔偿",
        "返还",
        "归",
        "确认",
        "解除",
        "抚养",
        "补办",
        "办理",
        "记载",
        "过户",
        "交付",
        "停止",
        "排除",
        "迁回",
        "安葬",
        "继续履行",
        "继续承包",
        "补缴",
        "缴纳",
        "承担",
        "负担",
        "享有",
    )

    if (
        normalized.startswith(("原告", "被告", "二原告", "其余各案被告"))
        and not _extract_first_amount(normalized)
        and not any(marker in normalized for marker in substantive_markers)
    ):
        return True

    if normalized.startswith(("、", "（", "(")) and not _extract_first_amount(
        normalized
    ):
        return True

    return any(
        marker in normalized
        for marker in (
            "鉴定",
            "见证",
            "造价预算",
            "预算",
            "核实",
            "测量",
            "公开本案",
            "法律责任",
        )
    )


def _is_meta_appeal_only(
    text: str, rules: RuleBundle, ontology: dict[str, Any]
) -> bool:
    normalized = normalize_space(text)

    has_meta = any(
        re.search(pattern, normalized) for pattern in rules.meta_appeal_patterns
    )

    if not has_meta:
        return False

    stripped = _trim_meta_appeal_prefix(normalized, rules)

    if stripped == normalized:
        return True

    return not _contains_substantive_action(stripped, ontology)


def _is_pure_meta_appeal_claim(text: str, rules: RuleBundle) -> bool:
    normalized = normalize_space(text)

    pieces = [
        piece.strip() for piece in re.split(r"[，,；;]", normalized) if piece.strip()
    ]

    if not pieces:
        return False

    return all(
        any(re.search(pattern, piece) for pattern in rules.meta_appeal_patterns)
        for piece in pieces
    )


def _trim_meta_appeal_prefix(text: str, rules: RuleBundle) -> str:
    normalized = normalize_space(text)
    pieces = re.split(r"[，,；;]", normalized)
    substantive: list[str] = []

    for piece in pieces:
        part = piece.strip()

        if not part:
            continue

        if any(re.search(pattern, part) for pattern in rules.meta_appeal_patterns):
            continue

        substantive.append(part)

    return "，".join(substantive).strip() or normalized


def _has_recoverable_meta_tail(text: str) -> bool:
    normalized = normalize_space(text)

    patterns = (
        r"改判由.{0,40}(赔偿|支付|返还|承担)",
        r"改判被上诉人(返还|支付)",
        r"改判.{0,20}不存在劳动关系",
        r"无需向.{0,20}支付",
    )

    return any(re.search(pattern, normalized) for pattern in patterns)


def _prune_non_substantive_clauses(
    text: str,
    *,
    stage: str,
    rules: RuleBundle,
    ontology: dict[str, Any],
) -> str:
    normalized = normalize_space(text)

    if not normalized:
        return ""

    pieces = [
        piece.strip() for piece in re.split(r"[，,；;。]", normalized) if piece.strip()
    ]

    if len(pieces) <= 1:
        return normalized

    kept: list[str] = []

    for piece in pieces:
        if _is_procedural_fee_claim(piece, rules):
            continue

        if _is_auxiliary_request(piece):
            continue

        if _is_argument_like_text(piece, rules) and not _contains_substantive_action(
            piece, ontology
        ):
            continue

        if _stage_key(stage) == "appeal" and _is_pure_meta_appeal_claim(piece, rules):
            continue

        kept.append(piece)

    return "，".join(kept).strip() if kept else ""


def _split_by_semicolon(text: str) -> list[tuple[str, int, int]]:
    parts: list[tuple[str, int, int]] = []
    cursor = 0

    for raw in re.split(r"([；;])", text):
        if raw in {"；", ";"}:
            cursor += len(raw)
            continue

        if not raw:
            continue

        segment = raw.strip(" ，,；;。:\n\t")

        if not segment:
            cursor += len(raw)
            continue

        start = text.find(segment, cursor)

        if start < 0:
            start = cursor

        end = start + len(segment)
        parts.append((segment, start, end))
        cursor = end

    return parts


def _normalize_amount_text(amount: str) -> str:
    if not amount:
        return ""

    normalized = normalize_space(amount)

    if re.search(r"^[零一二三四五六七八九十百千万亿两]+(?:万|亿)?元$", normalized):
        number_text = normalized[:-1]
        unit = "元"
        multiplier = 1

        if number_text.endswith("万"):
            number_text = number_text[:-1]
            multiplier = 10000

        elif number_text.endswith("亿"):
            number_text = number_text[:-1]
            multiplier = 100000000

        try:
            return f"{int(cn2an(number_text, 'smart')) * multiplier}{unit}"

        except Exception:
            return normalized

    return normalized


def _extract_first_amount(text: str) -> str:
    match = _AMOUNT_RE.search(text)

    if not match:
        return ""

    return _normalize_amount_text(match.group(1))


def _canonical_by_alias(text: str, mapping: dict[str, list[str]]) -> tuple[str, str]:
    normalized = normalize_space(text)
    hits: list[tuple[int, int, str, str]] = []

    for canonical, aliases in mapping.items():
        for alias in aliases:
            idx = normalized.find(alias)

            if idx >= 0:
                hits.append((idx, -len(alias), canonical, alias))

    if not hits:
        return "", ""

    _, _, canonical, alias = min(hits)
    return canonical, alias


def _strip_prefixes(text: str, rules: RuleBundle) -> str:
    stripped = normalize_space(text)
    for prefix in rules.petition_prefixes:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break

    stripped = re.sub(r"^(判令|请求|依法|改判为|改判|撤销|确认|解除)", "", stripped)
    return stripped.strip(" ，,；;。:\n\t")


def _extract_target(
    text: str,
    *,
    action_alias: str,
    amount: str,
    liability_alias: str,
    rules: RuleBundle,
) -> str:
    target = _strip_prefixes(text, rules)
    if action_alias:
        target = target.replace(action_alias, "", 1)

    if amount:
        target = target.replace(amount, "", 1)

    if liability_alias:
        target = target.replace(liability_alias, "", 1)

    for pattern in rules.actor_prefix_patterns:
        target = re.sub(pattern, "", target)

    target = target.strip(" ，,；;。:\n\t")
    return target or normalize_space(text)


def _build_claim_norm(
    action: str, target: str, amount: str, liability: str, fallback: str
) -> str:
    parts = [part for part in [action, target, amount, liability] if part]

    if parts:
        return normalize_space("".join(parts))

    return normalize_space(fallback)


def _infer_missing_action(
    normalized: str, amount: str, liability: str
) -> tuple[str, str]:
    if liability:
        return "承担", "承担"

    if "离婚" in normalized:
        return "准予", "离婚"

    if "抚养" in normalized:
        return "抚养", "抚养"

    if any(marker in normalized for marker in ("迁回", "安葬", "重新安葬")):
        return "迁葬", "迁回" if "迁回" in normalized else "安葬"

    if any(marker in normalized for marker in ("补缴", "缴纳")):
        return "支付", "补缴" if "补缴" in normalized else "缴纳"

    if "继续承包" in normalized:
        return "继续履行", "继续承包"

    if (
        "无效" in normalized
        or re.search(r"归[^，,；;。]{0,30}所有", normalized)
        or re.search(r"归(原告|被告|上诉人|被上诉人)", normalized)
        or re.search(r"由(原告|被告|上诉人|被上诉人).{0,20}享有", normalized)
        or "事实劳动关系" in normalized
        or "不存在劳动关系" in normalized
        or "无股东资格" in normalized
    ):
        return "确认", "确认"

    if any(marker in normalized for marker in ("删除", "销毁")):
        return "停止侵权", "删除" if "删除" in normalized else "销毁"

    if any(
        marker in normalized
        for marker in ("拆除", "排除妨碍", "排除妨害", "让出", "搬出", "迁出")
    ):
        return "排除妨害", "拆除" if "拆除" in normalized else "排除妨害"

    if any(marker in normalized for marker in ("解散", "重新审理", "发回重审")):
        return "", ""

    if any(marker in normalized for marker in ("腾空交还", "交还给原告")):
        return "交付", "交付"

    if any(marker in normalized for marker in ("公布", "查阅", "复制")):
        return "交付", "公布"

    if amount and any(
        marker in normalized
        for marker in (
            "工资",
            "款",
            "价款",
            "租金",
            "违约金",
            "利息",
            "补偿",
            "工程款",
            "货款",
            "借款",
            "奖金",
            "年终奖",
            "成交奖",
            "费用",
            "费",
        )
    ):
        return "支付", "支付"

    if amount and "损失" in normalized:
        return "赔偿", "赔偿"
    return "", ""


def _claim_from_segment(
    *,
    text: str,
    abs_start: int,
    abs_end: int,
    request_block_source: str,
    boundary_source: str,
    claim_source: str,
    stage: str,
    rules: RuleBundle,
    ontology: dict[str, Any],
    extra_flags: list[str] | None = None,
    allow_evidence_like: bool = False,
    allow_mixed_fee_claim: bool = False,
) -> ClaimItem | None:
    normalized = normalize_space(text)

    if not normalized:
        return None

    argument_like_input = _is_argument_like_text(normalized, rules)

    if _stage_key(stage) == "appeal":
        normalized = _trim_meta_appeal_prefix(normalized, rules)

        if _is_meta_appeal_only(normalized, rules, ontology):
            return None

    if allow_mixed_fee_claim and _has_mixed_substantive_fee_components(normalized):
        normalized = re.sub(r"[，,]总计[0-9]+(?:\\.[0-9]+)?元?$", "", normalized).strip(
            " ，,；;。"
        )

    else:
        normalized = _prune_non_substantive_clauses(
            normalized, stage=stage, rules=rules, ontology=ontology
        )

    if not normalized:
        return None

    if _is_procedural_fee_claim(normalized, rules):
        if not (
            allow_mixed_fee_claim and _has_mixed_substantive_fee_components(normalized)
        ):
            return None

    if _is_evidence_like_text(normalized, rules) and not allow_evidence_like:
        return None

    if (
        argument_like_input
        and not _contains_substantive_action(normalized, ontology)
        and not _extract_first_amount(normalized)
    ):
        return None

    if _is_argument_like_text(normalized, rules) and not _contains_substantive_action(
        normalized, ontology
    ):
        return None

    if _is_auxiliary_request(normalized):
        if not (
            allow_mixed_fee_claim and _has_mixed_substantive_fee_components(normalized)
        ):
            return None

    action, action_alias = _canonical_by_alias(normalized, ontology.get("actions", {}))

    liability, liability_alias = _canonical_by_alias(
        normalized, ontology.get("liability", {})
    )

    if liability and "承担" in normalized and "责任" in normalized and action == "支付":
        action = "承担"
        action_alias = "承担"

    amount = _extract_first_amount(normalized)

    if not action:
        inferred_action, inferred_alias = _infer_missing_action(
            normalized, amount, liability
        )

        action = inferred_action or action
        action_alias = inferred_alias or action_alias

    if not action and not amount and len(normalized) <= 6:
        return None

    if not action and len(normalized) > rules.no_action_max_chars:
        return None

    target = _extract_target(
        normalized,
        action_alias=action_alias,
        amount=amount,
        liability_alias=liability_alias,
        rules=rules,
    )

    stage_role = (
        "appeal_substantive"
        if _stage_key(stage) == "appeal"
        else "first_instance_substantive"
    )

    flags = list(extra_flags or [])

    if len(normalized) < rules.short_claim_chars:
        flags.append("claim_too_short")

    if not action:
        flags.append("action_missing")

    claim_norm = _build_claim_norm(action, target, amount, liability, normalized)

    return ClaimItem(
        claim_text_raw=normalized,
        claim_text_norm=claim_norm,
        source_span=SourceSpan(section=PRIMARY_SECTION, start=abs_start, end=abs_end),
        request_block_source=request_block_source,
        boundary_source=boundary_source,
        claim_source=claim_source,
        action=action,
        target=target,
        amount=amount,
        liability=liability,
        stage_role=stage_role,
        claim_flags=_ordered_unique(flags),
    )


def _deduplicate_claims(claims: list[ClaimItem]) -> list[ClaimItem]:
    finalized: list[ClaimItem] = []

    for item in claims:
        duplicate_idx = None

        for idx, existing in enumerate(finalized):
            if item.claim_text_norm == existing.claim_text_norm:
                duplicate_idx = idx
                break

            if fuzz.ratio(item.claim_text_norm, existing.claim_text_norm) >= 96:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            finalized.append(item)
            continue

        existing = finalized[duplicate_idx]

        if (item.source_span.end - item.source_span.start) > (
            existing.source_span.end - existing.source_span.start
        ):
            finalized[duplicate_idx] = item

    return finalized


def _extract_rule_candidates(
    request_block: RequestBlockResult,
    *,
    stage: str,
    rules: RuleBundle,
    ontology: dict[str, Any],
    syntax_splitter: SyntaxSplitter | None,
) -> tuple[list[ClaimItem], list[str], list[str]]:
    body_text = request_block.body_text

    if not normalize_space(body_text):
        return [], [], ["request_block_empty"]

    roots, conflict = _build_number_tree(body_text)
    claims: list[ClaimItem] = []
    raw_candidates: list[str] = []
    trigger_reasons: list[str] = []
    nested_numbering = _has_nested_numbering(roots)

    if conflict or nested_numbering:
        trigger_reasons.append("number_tree_conflict")

    if roots:
        nodes_to_visit = roots if not nested_numbering else roots

        def visit(nodes: list[NumberNode], *, recurse: bool) -> None:
            for node in nodes:
                segment, rel_start, rel_end = _body_text_from_node(body_text, node)

                if segment:
                    raw_candidates.append(segment)

                    claim = _claim_from_segment(
                        text=segment,
                        abs_start=request_block.body_start + rel_start,
                        abs_end=request_block.body_start + rel_end,
                        request_block_source=request_block.source,
                        boundary_source="number_tree",
                        claim_source="rule",
                        stage=stage,
                        rules=rules,
                        ontology=ontology,
                    )

                    if claim is not None:
                        claims.append(claim)

                if recurse and node.children:
                    visit(node.children, recurse=True)

        visit(nodes_to_visit, recurse=not nested_numbering)

    else:
        for segment, rel_start, rel_end in _split_by_semicolon(body_text):
            raw_candidates.append(segment)

            claim = _claim_from_segment(
                text=segment,
                abs_start=request_block.body_start + rel_start,
                abs_end=request_block.body_start + rel_end,
                request_block_source=request_block.source,
                boundary_source="single_span",
                claim_source="rule",
                stage=stage,
                rules=rules,
                ontology=ontology,
            )

            if claim is not None:
                claims.append(claim)

    if not claims and syntax_splitter is not None:
        syntax_claims = syntax_splitter.split(body_text, _all_action_aliases(ontology))

        if syntax_claims:
            raw_candidates.extend(segment for segment, _, _ in syntax_claims)

            for segment, rel_start, rel_end in syntax_claims:
                claim = _claim_from_segment(
                    text=segment,
                    abs_start=request_block.body_start + rel_start,
                    abs_end=request_block.body_start + rel_end,
                    request_block_source=request_block.source,
                    boundary_source="syntax_split",
                    claim_source="rule",
                    stage=stage,
                    rules=rules,
                    ontology=ontology,
                )

                if claim is not None:
                    claims.append(claim)

        elif re.search(r"并|且|及", body_text):
            trigger_reasons.append("syntax_split_ambiguous")

    claims = _deduplicate_claims(claims)

    if not claims:
        if roots and all(
            _is_procedural_fee_claim(candidate, rules) for candidate in raw_candidates
        ):
            trigger_reasons.append("procedural_only")

        elif (
            _stage_key(stage) == "appeal"
            and body_text
            and _is_meta_appeal_only(body_text, rules, ontology)
        ):
            trigger_reasons.append("meta_appeal_only")

        elif not trigger_reasons:
            trigger_reasons.append("no_substantive_claim")

    return claims, _ordered_unique(raw_candidates), _ordered_unique(trigger_reasons)


def extract_rule_claims(
    case: dict[str, Any],
    *,
    rules: RuleBundle,
    ontology: dict[str, Any],
    syntax_splitter: SyntaxSplitter | None,
) -> RuleExtractionResult:
    """Extract claims with rules and local HanLP splitting only."""

    content = case.get("content") or {}
    extra = case.get("extraInfo") or {}
    plaintiff_text = str(content.get(PRIMARY_SECTION, "") or "")
    stage = str(extra.get("stage", "") or "")
    request_block = extract_request_block(plaintiff_text, stage=stage, rules=rules)

    if request_block is None:
        return RuleExtractionResult(
            request_block=None,
            claims=[],
            trigger_reasons=["request_block_missing"],
            auto_candidates=[],
        )

    claims, auto_candidates, trigger_reasons = _extract_rule_candidates(
        request_block,
        stage=stage,
        rules=rules,
        ontology=ontology,
        syntax_splitter=syntax_splitter,
    )

    return RuleExtractionResult(
        request_block=request_block,
        claims=claims,
        trigger_reasons=trigger_reasons,
        auto_candidates=auto_candidates,
    )


def build_claim_from_expert_span(
    case: dict[str, Any],
    *,
    start: int,
    end: int,
    rules: RuleBundle,
    ontology: dict[str, Any],
    adjudication_reason: str = "",
) -> ClaimItem:
    """Build one normalized claim from an expert-selected span in `原告诉称`."""

    content = case.get("content") or {}
    extra = case.get("extraInfo") or {}
    plaintiff_text = str(content.get(PRIMARY_SECTION, "") or "")

    if start < 0 or end <= start or end > len(plaintiff_text):
        raise ValueError(
            f"Expert span out of range: start={start}, end={end}, text_len={len(plaintiff_text)}"
        )

    request_block = extract_request_block(
        plaintiff_text, stage=str(extra.get("stage", "") or ""), rules=rules
    )

    request_block_source = "expert_full_text"

    if (
        request_block is not None
        and start >= request_block.body_start
        and end <= request_block.body_end
    ):
        request_block_source = request_block.source

    raw_text = normalize_space(plaintiff_text[start:end])

    if not raw_text:
        raise ValueError("Expert span resolves to empty text")

    claim = _claim_from_segment(
        text=raw_text,
        abs_start=start,
        abs_end=end,
        request_block_source=request_block_source,
        boundary_source="expert_span_select",
        claim_source="expert_adjudication",
        stage=str(extra.get("stage", "") or ""),
        rules=rules,
        ontology=ontology,
        extra_flags=["from_expert_adjudication"],
    )

    if claim is None and _contains_substantive_action(raw_text, ontology):
        claim = _claim_from_segment(
            text=raw_text,
            abs_start=start,
            abs_end=end,
            request_block_source=request_block_source,
            boundary_source="expert_span_select",
            claim_source="expert_adjudication",
            stage=str(extra.get("stage", "") or ""),
            rules=rules,
            ontology=ontology,
            extra_flags=["from_expert_adjudication"],
            allow_evidence_like=True,
            allow_mixed_fee_claim=True,
        )

    if claim is None:
        reason_suffix = f" ({adjudication_reason})" if adjudication_reason else ""

        raise ValueError(
            f"Expert-selected span did not normalize into a valid substantive claim: {raw_text}{reason_suffix}"
        )

    return claim


def claim_item_from_row(row: dict[str, Any]) -> ClaimItem:
    """Reconstruct one ClaimItem from a serialized claim row."""

    span = row["source_span"]

    return ClaimItem(
        claim_text_raw=str(row["claim_text_raw"]),
        claim_text_norm=str(row["claim_text_norm"]),
        source_span=SourceSpan(
            section=str(span["section"]),
            start=int(span["start"]),
            end=int(span["end"]),
        ),
        request_block_source=str(row["request_block_source"]),
        boundary_source=str(row["boundary_source"]),
        claim_source=str(row["claim_source"]),
        action=str(row["action"]),
        target=str(row["target"]),
        amount=str(row["amount"]),
        liability=str(row["liability"]),
        stage_role=str(row["stage_role"]),
        claim_flags=[str(flag) for flag in row.get("claim_flags", [])],
    )


def merge_claim_result(
    case: dict[str, Any],
    *,
    split_meta: dict[str, Any],
    rule_result: RuleExtractionResult,
    adjudicated_claims: list[ClaimItem] | None = None,
) -> CaseClaimExtractionResult:
    """Combine metadata with rule-path or expert-adjudicated result."""

    meta = case.get("metaInfo") or {}
    extra = case.get("extraInfo") or {}
    final_claims = list(adjudicated_claims or rule_result.claims)
    claim_source = "expert_adjudication" if adjudicated_claims else "rule"
    needs_manual_review = False
    review_reasons: list[str] = []

    if rule_result.trigger_reasons and not adjudicated_claims:
        needs_manual_review = True
        review_reasons = list(rule_result.trigger_reasons)
        claim_source = "uncertain"
        final_claims = []

    return CaseClaimExtractionResult(
        uid=str(meta.get("uid", "") or ""),
        split=str(split_meta["split"]),
        hard_case=int(split_meta["hard_case"]),
        sample_cause=str(extra.get("sample_cause", "") or ""),
        stage=str(extra.get("stage", "") or ""),
        claims=final_claims,
        request_block=rule_result.request_block,
        auto_candidates=list(rule_result.auto_candidates),
        review_reasons=review_reasons,
        needs_manual_review=needs_manual_review,
        claim_source=claim_source,
    )


def serialize_claim_rows(
    results: list[CaseClaimExtractionResult],
) -> list[dict[str, Any]]:
    """Flatten case-level results into one row per claim."""

    rows: list[dict[str, Any]] = []

    for result in results:
        for idx, claim in enumerate(result.claims, start=1):
            rows.append(
                {
                    "uid": result.uid,
                    "split": result.split,
                    "hard_case": result.hard_case,
                    "sample_cause": result.sample_cause,
                    "stage": result.stage,
                    "claim_id": f"GOLD_{result.uid}_{idx:03d}",
                    "claim_index": idx,
                    "claim_text_raw": claim.claim_text_raw,
                    "claim_text_norm": claim.claim_text_norm,
                    "source_span": {
                        "section": claim.source_span.section,
                        "start": claim.source_span.start,
                        "end": claim.source_span.end,
                    },
                    "request_block_source": claim.request_block_source,
                    "boundary_source": claim.boundary_source,
                    "claim_source": claim.claim_source,
                    "action": claim.action,
                    "target": claim.target,
                    "amount": claim.amount,
                    "liability": claim.liability,
                    "stage_role": claim.stage_role,
                    "claim_flags": list(claim.claim_flags),
                }
            )

    return rows


def build_uncertain_rows(
    results: list[CaseClaimExtractionResult], case_map: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Serialize uncertain cases for manual review."""

    rows: list[dict[str, Any]] = []

    for result in results:
        if not result.needs_manual_review:
            continue

        case = case_map[result.uid]
        plaintiff_text = str((case.get("content") or {}).get(PRIMARY_SECTION, "") or "")

        request_text = (
            result.request_block.body_text if result.request_block is not None else ""
        )

        request_source = (
            result.request_block.source
            if result.request_block is not None
            else "missing"
        )

        rows.append(
            {
                "uid": result.uid,
                "split": result.split,
                "hard_case": result.hard_case,
                "sample_cause": result.sample_cause,
                "stage": result.stage,
                "plaintiff_text": plaintiff_text,
                "request_block_text": request_text,
                "request_block_source": request_source,
                "auto_candidates": list(result.auto_candidates),
                "review_reasons": list(result.review_reasons),
                "needs_manual_review": True,
            }
        )

    return rows


def build_summary(
    results: list[CaseClaimExtractionResult],
    claim_rows: list[dict[str, Any]],
    uncertain_rows: list[dict[str, Any]],
    expert_audit_rows: list[dict[str, Any]],
    *,
    total_cases: int,
    rules: RuleBundle,
) -> dict[str, Any]:
    """Build high-level extraction summary for Step 06 reporting."""

    claims_per_case: dict[str, int] = {}

    for row in claim_rows:
        claims_per_case[row["uid"]] = claims_per_case.get(row["uid"], 0) + 1

    action_missing_count = sum(
        "action_missing" in row.get("claim_flags", []) for row in claim_rows
    )

    evidence_like_claim_count = sum(
        _is_evidence_like_text(row["claim_text_raw"], rules) for row in claim_rows
    )

    argument_like_claim_count = sum(
        _is_argument_like_text(row["claim_text_raw"], rules) for row in claim_rows
    )

    pure_meta_appeal_count = sum(
        row["stage"] == "二审"
        and _is_pure_meta_appeal_claim(row["claim_text_raw"], rules)
        for row in claim_rows
    )

    recoverable_meta_shell_case_count = sum(
        "no_substantive_claim" in row.get("review_reasons", [])
        and _has_recoverable_meta_tail(row.get("request_block_text", ""))
        for row in uncertain_rows
    )

    first_instance_long_tail_cue_recovered_count = sum(
        result.stage == "一审"
        and result.request_block is not None
        and result.request_block.cue in rules.first_instance_long_tail_cues
        and bool(result.claims)
        for result in results
    )

    request_block_missing_obvious_cue_count = sum(
        "request_block_missing" in row.get("review_reasons", [])
        and _has_stage_obvious_request_marker(
            row.get("plaintiff_text", ""), row.get("stage", ""), rules
        )
        for row in uncertain_rows
    )

    return {
        "total_cases": total_cases,
        "claim_count": len(claim_rows),
        "case_with_claims": len(claims_per_case),
        "uncertain_case_count": len(uncertain_rows),
        "expert_queue_case_count": len(uncertain_rows),
        "expert_adjudicated_case_count": len(
            {str(row.get("uid", "")) for row in expert_audit_rows}
        ),
        "expert_adjudicated_claim_count": sum(
            int(row.get("accepted_claim_count", 0)) for row in expert_audit_rows
        ),
        "median_claims_per_case": median(claims_per_case.values())
        if claims_per_case
        else 0,
        "action_missing_count": action_missing_count,
        "action_missing_rate": (action_missing_count / len(claim_rows))
        if claim_rows
        else 0.0,
        "evidence_like_claim_count": evidence_like_claim_count,
        "argument_like_claim_count": argument_like_claim_count,
        "pure_meta_appeal_count": pure_meta_appeal_count,
        "recoverable_meta_shell_case_count": recoverable_meta_shell_case_count,
        "first_instance_long_tail_cue_recovered_count": first_instance_long_tail_cue_recovered_count,
        "request_block_missing_obvious_cue_count": request_block_missing_obvious_cue_count,
        "claim_source_counts": {
            source: sum(1 for row in claim_rows if row["claim_source"] == source)
            for source in sorted({row["claim_source"] for row in claim_rows})
        },
        "boundary_source_counts": {
            source: sum(1 for row in claim_rows if row["boundary_source"] == source)
            for source in sorted({row["boundary_source"] for row in claim_rows})
        },
        "uncertain_reason_counts": {
            reason: sum(
                reason in row.get("review_reasons", []) for row in uncertain_rows
            )
            for reason in sorted(
                {
                    reason
                    for row in uncertain_rows
                    for reason in row.get("review_reasons", [])
                }
            )
        },
    }


def build_reason_count_rows(
    uncertain_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build flat reason counts CSV rows."""

    counts: dict[str, int] = {}

    for row in uncertain_rows:
        for reason in row.get("review_reasons", []):
            counts[reason] = counts.get(reason, 0) + 1

    return [{"reason": reason, "count": counts[reason]} for reason in sorted(counts)]


def configure_hanlp_env(
    *, hanlp_home: str | None, hf_home: str | None, hanlp_verbose: str | None
) -> None:
    """Apply optional HanLP/HF cache overrides before importing HanLP."""

    if hanlp_home:
        os.environ["HANLP_HOME"] = hanlp_home

    if hf_home:
        os.environ["HF_HOME"] = hf_home

    if hanlp_verbose is not None:
        os.environ["HANLP_VERBOSE"] = hanlp_verbose
