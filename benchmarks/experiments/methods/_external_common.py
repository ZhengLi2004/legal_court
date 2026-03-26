"""Experiment-side helpers for external Claim 1 baselines.

These runners intentionally do not instantiate `LegalSystem` or `DebateEngine`.
They only reuse low-level infrastructure clients from `mas.infrastructure`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from metagpt.schema import Message

from benchmarks.experiments.methods.base import (
    _run_coro_sync,
    apply_seed,
    prepare_case_for_engine,
    validate_method_result,
)
from mas.infrastructure.embedding import EmbeddingFunc
from mas.infrastructure.fact_es_tool import FactEsTool
from mas.infrastructure.law_es_tool import LawEsTool
from mas.infrastructure.llm import GPTChat, get_price
from mas.infrastructure.settings_provider import (
    build_llm_config_view,
    build_system_config,
)
from prompts.common_prompts import (
    GRAPH_READING_GUIDE,
    JUDGE_DIRECT_VERDICT_PROMPT,
    SYSTEM_PROMPT_DIRECT_JUDGE,
)

ALLOWED_STATUS = ("VALIDATED", "DEFEATED", "HYPOTHETICAL")

_JUDGE_STATUS_MAP = {
    "ACCEPTED": "VALIDATED",
    "REJECTED": "DEFEATED",
    "UNMENTIONED": "HYPOTHETICAL",
}

FIXED_CASE_TOP_K = 3
FIXED_LAW_TOP_K = 3

_DIRECT_STATUS_SCHEMA = {
    "name": "external_claim_statuses",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": sorted(_JUDGE_STATUS_MAP.keys()),
                        },
                    },
                    "required": ["claim_id", "status"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    },
}


def _json_block(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _trim_text(text: str, *, limit: int = 1200) -> str:
    normalized = str(text or "").strip()

    if len(normalized) <= limit:
        return normalized

    return normalized[:limit] + " ..."


def _compact_case_hit(hit: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(hit.get("_source", {}) or {})

    return {
        "score": float(hit.get("_score", 0.0) or 0.0),
        "case_title": str(source.get("case_title", "") or "").strip(),
        "analysis": _trim_text(str(source.get("analysis", "") or "")),
        "fact_finding": _trim_text(str(source.get("fact_finding", "") or "")),
    }


def _compact_law_hit(hit: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(hit.get("_source", {}) or {})

    return {
        "score": float(hit.get("_score", 0.0) or 0.0),
        "law_name": str(source.get("law_name", "") or "").strip(),
        "article_id": str(source.get("article_id", "") or "").strip(),
        "content": _trim_text(str(source.get("content", "") or "")),
    }


def _dedupe_law_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for hit in hits:
        key = (
            str(hit.get("law_name", "") or "").strip(),
            str(hit.get("article_id", "") or "").strip(),
        )

        current = best_by_key.get(key)

        if current is None or float(hit.get("score", 0.0) or 0.0) > float(
            current.get("score", 0.0) or 0.0
        ):
            best_by_key[key] = dict(hit)

    return sorted(
        best_by_key.values(),
        key=lambda item: float(item.get("score", 0.0) or 0.0),
        reverse=True,
    )[:FIXED_LAW_TOP_K]


def _build_claim_inventory(
    root_claim_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "claim_id": str(row.get("claim_id", "") or "").strip(),
            "claim_text": str(
                row.get("claim_text_norm", "") or row.get("claim_text_raw", "") or ""
            ).strip(),
        }
        for row in root_claim_rows
    ]


def _validate_status_rows(
    *,
    rows: Any,
    expected_claim_ids: list[str],
) -> list[dict[str, str]]:
    if not isinstance(rows, list):
        raise ValueError("External adjudication response `claims` must be a list.")

    expected_set = set(expected_claim_ids)
    seen_ids: set[str] = set()
    normalized_rows: list[dict[str, str]] = []

    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Each adjudication row must be a JSON object.")

        claim_id = str(row.get("claim_id", "") or "").strip()
        status = str(row.get("status", "") or "").strip().upper()

        if not claim_id:
            raise ValueError("Adjudication row missing non-empty `claim_id`.")

        if claim_id in seen_ids:
            raise ValueError(f"Duplicate adjudication claim_id: {claim_id}")

        if claim_id not in expected_set:
            raise ValueError(f"Unexpected adjudication claim_id: {claim_id}")

        if status not in _JUDGE_STATUS_MAP:
            raise ValueError(f"Invalid adjudication status: {status}")

        seen_ids.add(claim_id)

        normalized_rows.append(
            {"claim_id": claim_id, "status": _JUDGE_STATUS_MAP[status]}
        )

    missing_ids = sorted(expected_set - seen_ids)

    if missing_ids:
        raise ValueError(f"Missing adjudication claim_ids: {missing_ids}")

    return sorted(
        normalized_rows,
        key=lambda item: expected_claim_ids.index(item["claim_id"]),
    )


def _build_claim_result_rows(
    *,
    case_uid_value: str,
    root_claim_rows: list[dict[str, Any]],
    adjudication_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    claim_rows = [dict(row) for row in root_claim_rows]

    status_lookup = {
        str(row["claim_id"]): str(row["status"]) for row in adjudication_rows
    }

    status_rows = [
        {
            "uid": case_uid_value,
            "claim_id": str(row.get("claim_id", "") or "").strip(),
            "status_eval": status_lookup[str(row.get("claim_id", "") or "").strip()],
        }
        for row in claim_rows
    ]

    return claim_rows, status_rows


def _build_result(
    *,
    method_name: str,
    case_uid_value: str,
    claim_rows: list[dict[str, Any]],
    status_rows: list[dict[str, str]],
    trace: dict[str, Any],
    warnings: list[str],
    started_at: float,
    start_completion: int,
    start_prompt: int,
) -> dict[str, Any]:
    end_completion, end_prompt = get_price()

    result = {
        "method": method_name,
        "case_uid": case_uid_value,
        "claims": claim_rows,
        "status": status_rows,
        "trace": trace,
        "token_usage": {
            "prompt_tokens": max(0, int(end_prompt) - int(start_prompt)),
            "completion_tokens": max(0, int(end_completion) - int(start_completion)),
            "total_tokens": max(0, int(end_prompt) - int(start_prompt))
            + max(0, int(end_completion) - int(start_completion)),
        },
        "latency": round(time.perf_counter() - started_at, 6),
        "warnings": warnings,
    }

    validate_method_result(result)
    return result


def _build_adjudicator_prompt(
    *,
    case_uid_value: str,
    cause: str,
    fact_finding: str,
    claims: list[dict[str, str]],
    retrieved_case_hits: list[dict[str, Any]] | None = None,
    retrieved_law_hits: list[dict[str, Any]] | None = None,
    transcript: list[str] | None = None,
) -> str:
    issue_list = "\n".join(
        [
            f"{idx + 1}. [{row['claim_id']}] {row['claim_text']}"
            for idx, row in enumerate(claims)
        ]
    )

    evidence_context = _json_block(
        {
            "case_uid": case_uid_value,
            "cause": cause,
            "fact_finding": fact_finding,
            "retrieved_cases": retrieved_case_hits or [],
            "retrieved_laws": retrieved_law_hits or [],
            "debate_transcript": transcript or [],
        }
    )

    prompt = JUDGE_DIRECT_VERDICT_PROMPT.format(
        issue_list=issue_list,
        graph_context=evidence_context,
    )

    return (
        f"{prompt}\n\n"
        "【附加约束】\n"
        "1. 当前输入中的 claim_id 已固定，你必须逐条覆盖，且不得新增或改写 claim_id。\n"
        "2. JSON 中状态标签必须使用 ACCEPTED、REJECTED、UNMENTIONED。"
    )


def _adjudicate_claim_statuses(
    *,
    llm: GPTChat,
    prompt: str,
    expected_claim_ids: list[str],
) -> list[dict[str, str]]:
    response = llm.ask_json_schema(
        messages=[
            Message(role="system", content=SYSTEM_PROMPT_DIRECT_JUDGE),
            Message(role="user", content=prompt),
        ],
        schema=_DIRECT_STATUS_SCHEMA,
        temperature=0.0,
    )

    if not isinstance(response, dict):
        raise ValueError("External adjudication response must be a JSON object.")

    return _validate_status_rows(
        rows=response.get("claims"),
        expected_claim_ids=expected_claim_ids,
    )


def _run_case_and_law_retrieval(
    *,
    cfg: Any,
    fact_finding: str,
    cause: str,
    root_claim_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    embedding = EmbeddingFunc(cfg.path.embedding_model_path)
    fact_tool = FactEsTool(cfg.es.host, embedding)
    law_tool = LawEsTool(cfg.es.host, embedding)

    async def _search() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        case_hits_raw = await fact_tool.search_cases_raw(
            fact_finding, top_k=FIXED_CASE_TOP_K
        )

        law_hits_raw: list[dict[str, Any]] = []

        for row in root_claim_rows:
            claim_text = str(
                row.get("claim_text_norm", "") or row.get("claim_text_raw", "") or ""
            ).strip()

            query_text = f"{cause} {claim_text}".strip()

            law_hits_raw.extend(
                await law_tool.search_laws_raw(query_text, top_k=FIXED_LAW_TOP_K)
            )

        await fact_tool.close()
        await law_tool.close()
        return case_hits_raw, law_hits_raw

    case_hits_raw, law_hits_raw = _run_coro_sync(_search())
    compact_case_hits = [_compact_case_hit(hit) for hit in case_hits_raw]
    compact_law_hits = _dedupe_law_hits([_compact_law_hit(hit) for hit in law_hits_raw])
    return compact_case_hits, compact_law_hits


def _chat_text(
    *,
    llm: GPTChat,
    system_prompt: str,
    prompt: str,
    max_tokens: int = 512,
) -> str:
    raw = llm(
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=prompt),
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        raise_on_error=True,
    )

    answer = str(raw or "").strip()

    if not answer:
        raise ValueError("External debate turn returned empty content.")

    return answer


def _build_debate_turn_prompt(
    *,
    side_label: str,
    fact_finding: str,
    claims: list[dict[str, str]],
    transcript: list[str],
) -> str:
    role_name = "原告" if side_label == "Plaintiff Advocate" else "被告"

    return (
        f"【角色】\n你是【{role_name}】的辩护律师。\n\n"
        "【当前案情】\n"
        f"{fact_finding}\n\n"
        "【核心诉求】\n"
        f"{_json_block(claims)}\n\n"
        "【已有攻防记录】\n"
        f"{_json_block(transcript)}\n\n"
        f"{GRAPH_READING_GUIDE}\n\n"
        "【任务】\n"
        "请站在本方立场，给出一段简洁、可执行、可复核的论证：\n"
        "1. 指出本方最值得强调的1-2个点。\n"
        "2. 指出对方最薄弱的1个点。\n"
        "3. 不要输出 JSON，不要下最终裁决。"
    )


def prepare_external_method_context(
    *,
    case: Mapping[str, Any],
    storage_root_dir: str | None,
    seed: int | None,
) -> tuple[
    Any, GPTChat, dict[str, Any], list[dict[str, Any]], list[str], float, int, int
]:
    """Prepare shared external-baseline execution context."""
    apply_seed(seed)
    prepared_case, warnings = prepare_case_for_engine(case)
    cfg = build_system_config(storage_root_dir=storage_root_dir)

    llm = GPTChat(
        defaults=build_llm_config_view(cfg),
        model_name=cfg.llm.model_name,
    )

    started_at = time.perf_counter()
    start_completion, start_prompt = get_price()

    root_claim_rows = [
        dict(row) for row in prepared_case.get("experiment_root_claim_rows") or []
    ]

    return (
        cfg,
        llm,
        prepared_case,
        root_claim_rows,
        warnings,
        started_at,
        start_completion,
        start_prompt,
    )


def write_trace_json(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
