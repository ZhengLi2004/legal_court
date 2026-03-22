"""Shared Step 09 method-runner helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import threading
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from mas.core.engine import DebateEngine
from mas.core.graph import NodeStatus, NodeType
from mas.infrastructure.initializer import CaseInitializer
from mas.infrastructure.legal_system_factory import build_legal_system
from mas.infrastructure.llm import GPTChat, get_price
from mas.infrastructure.settings_provider import (
    build_llm_config_view,
    build_system_config,
)

MethodRunner = Callable[..., dict[str, Any]]
ALLOWED_STATUS = {"VALIDATED", "DEFEATED", "HYPOTHETICAL"}


def normalize_space(text: str) -> str:
    """Collapse repeated whitespace for stable claim text."""
    return " ".join(str(text or "").split())


def case_uid(case: Mapping[str, Any]) -> str:
    """Return one stable case uid across experiment data shapes."""
    direct = str(case.get("uid", "") or case.get("case_uid", "") or "").strip()

    if direct:
        return direct

    meta = case.get("metaInfo") or {}
    meta_uid = str(meta.get("uid", "") or "").strip()

    if meta_uid:
        return meta_uid

    extra = case.get("extraInfo") or {}
    line_no = str(extra.get("line_no", "") or "").strip()

    if line_no:
        return f"line_{line_no}"

    payload = json.dumps(case, ensure_ascii=False, sort_keys=True)
    return f"case_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def prepare_case_for_engine(
    case: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Normalize benchmark case payload into engine setup shape."""
    warnings: list[str] = []
    payload = deepcopy(dict(case))
    content = payload.get("content") or {}
    extra = payload.get("extraInfo") or {}
    meta = payload.get("metaInfo") or {}
    uid = case_uid(case)

    fact_finding = str(
        payload.get("fact_finding", "")
        or content.get("审理查明", "")
        or content.get("原告诉称", "")
        or ""
    ).strip()

    cause = str(
        payload.get("cause", "")
        or extra.get("sample_cause", "")
        or meta.get("cause", "")
        or "未知案由"
    ).strip()

    title = str(payload.get("title", "") or meta.get("caseName", "") or "").strip()

    if not fact_finding:
        warnings.append("fact_finding_missing")

    payload["uid"] = uid
    payload["case_id"] = uid
    payload["title"] = title
    payload["fact_finding"] = fact_finding
    payload["cause"] = [cause] if cause else ["未知案由"]
    return payload, warnings


def apply_seed(seed: int | None) -> None:
    """Apply deterministic seed to standard local RNGs."""
    if seed is None:
        return

    seed_int = int(seed)
    random.seed(seed_int)

    try:
        import numpy as np

        np.random.seed(seed_int % (2**32))

    except Exception:
        pass


def build_method_config(
    *,
    skip_validate_step: bool = False,
    disable_recall_worker: bool = False,
    disable_initial_insights: bool = False,
    budget: Mapping[str, Any] | None = None,
    retrieval_config: Mapping[str, Any] | None = None,
) -> tuple[Any, list[str]]:
    """Create one per-run mutable config with experiment overrides."""
    cfg = build_system_config()
    warnings: list[str] = []
    cfg.experiment.skip_validate_step = bool(skip_validate_step)
    cfg.experiment.disable_recall_worker = bool(disable_recall_worker)
    cfg.experiment.disable_initial_insights = bool(disable_initial_insights)

    if retrieval_config:
        for key, value in retrieval_config.items():
            if not hasattr(cfg.retrieval, key):
                raise ValueError(f"Unknown retrieval_config key: {key}")

            setattr(cfg.retrieval, key, value)

    if budget:
        unused_budget_keys: list[str] = []

        for key, value in budget.items():
            if key == "max_turns":
                cfg.convergence.max_turns = int(value)

            else:
                unused_budget_keys.append(str(key))

        if unused_budget_keys:
            warnings.append(
                "unused_budget_keys:" + ",".join(sorted(set(unused_budget_keys)))
            )

    return cfg, warnings


def _run_coro_sync(coro: Any) -> Any:
    """Run one coroutine from sync code, even under an active event loop."""
    try:
        running_loop = asyncio.get_running_loop()

    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        result_box: dict[str, Any] = {}
        error_box: dict[str, BaseException] = {}

        def _thread_main() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result_box["value"] = loop.run_until_complete(coro)

            except BaseException as exc:  # pragma: no cover - defensive bridge
                error_box["error"] = exc

            finally:
                loop.close()

        thread = threading.Thread(target=_thread_main, daemon=True)
        thread.start()
        thread.join()

        if "error" in error_box:
            raise error_box["error"]

        return result_box.get("value")

    return asyncio.run(coro)


def _extract_claim_rows(
    *,
    graph: Any,
    case_uid_value: str,
    root_claims_status: Mapping[str, Any],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract root-claim predictions and status rows from one engine graph."""
    claim_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    normalized_status = {
        str(claim_id): (
            status.value if hasattr(status, "value") else str(status or "HYPOTHETICAL")
        )
        for claim_id, status in root_claims_status.items()
    }

    root_nodes: list[tuple[str, str]] = []

    if graph is None or not hasattr(graph, "graph"):
        return claim_rows, status_rows

    for node_id, data in graph.graph.nodes(data=True):
        metadata = data.get("metadata", {})

        if metadata.get("is_root_claim", False):
            root_nodes.append(
                (str(node_id), str(data.get("content", "") or "").strip())
            )

    for node_id, content in sorted(root_nodes, key=lambda item: item[0]):
        status_eval = normalized_status.get(node_id, NodeStatus.HYPOTHETICAL.value)

        if node_id not in normalized_status:
            warnings.append(f"missing_root_claim_status:{node_id}")

        claim_rows.append(
            {
                "uid": case_uid_value,
                "claim_id": node_id,
                "claim_text_raw": content,
                "claim_text_norm": normalize_space(content),
            }
        )

        status_rows.append(
            {
                "uid": case_uid_value,
                "claim_id": node_id,
                "status_eval": str(status_eval),
            }
        )

    return claim_rows, status_rows


def _build_graph_stats_from_graph(graph: Any) -> dict[str, int]:
    """Build lightweight graph stats without DebateEngine snapshot helpers."""
    if graph is None or not hasattr(graph, "graph"):
        return {"node_count": 0, "edge_count": 0, "claim_nodes": 0}

    claim_nodes = 0

    for _, data in graph.graph.nodes(data=True):
        node_type = str(data.get("type", "") or "")
        metadata = data.get("metadata", {})

        if metadata.get("is_root_claim", False) or node_type == NodeType.CLAIM.value:
            claim_nodes += 1

    return {
        "node_count": int(graph.graph.number_of_nodes()),
        "edge_count": int(graph.graph.number_of_edges()),
        "claim_nodes": int(claim_nodes),
    }


def _execute_structured_single_pass(
    *,
    method_name: str,
    prepared_case: Mapping[str, Any],
    cfg: Any,
    warnings: list[str],
    budget: Mapping[str, Any] | None = None,
    retrieval_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run B1-style single-pass structured adjudication without DebateEngine."""
    case_uid_value = case_uid(prepared_case)
    llm_defaults = build_llm_config_view(cfg)
    initializer_llm = GPTChat(defaults=llm_defaults, model_name=cfg.llm.model_name)
    initializer = CaseInitializer(initializer_llm)
    legal_sys = build_legal_system(cfg)
    fact_finding = str(prepared_case.get("fact_finding", "") or "")
    cause_value = prepared_case.get("cause", ["未知案由"])

    if isinstance(cause_value, list):
        cause = str(cause_value[0] if cause_value else "未知案由")

    else:
        cause = str(cause_value or "未知案由")

    init_res = _run_coro_sync(initializer.initialize(fact_finding, cause))
    graph, _ = legal_sys.new_case(fact_finding)
    fact_count = 0

    for fact_statement in init_res.fact_statements:
        _, is_new = graph.add_node(
            content=fact_statement,
            node_type=NodeType.FACT,
            agent_id="System_Init",
            metadata={"is_objective_fact": True},
        )

        if is_new:
            fact_count += 1

    claim_count = 0

    for claim_statement in init_res.root_claim_actions:
        _, is_new = graph.add_node(
            content=claim_statement,
            node_type=NodeType.CLAIM,
            agent_id="System_Init",
            metadata={"is_root_claim": True},
        )

        if is_new:
            claim_count += 1

    graph.refresh_context(current_step=0)

    transcript = [
        (
            "【Structured Baseline 初始化】\n"
            f"案件案由：{cause}\n"
            f"注入事实节点：{fact_count}\n"
            f"注入根诉求节点：{claim_count}\n"
            "未进入多轮辩论，直接执行一次性 adjudication。"
        )
    ]

    judge = getattr(legal_sys, "judge", None)
    original_extraction_llm = None

    if (
        judge is not None
        and hasattr(judge, "judge_llm")
        and hasattr(judge, "extraction_llm")
    ):
        original_extraction_llm = judge.extraction_llm
        judge.extraction_llm = judge.judge_llm

    try:
        judgment_document = judge.evaluate(context=fact_finding, graph=graph)

        root_claims_status = _run_coro_sync(
            judge.extract_verdict(judgment_document, graph)
        )

    finally:
        if original_extraction_llm is not None:
            judge.extraction_llm = original_extraction_llm

    claim_rows, status_rows = _extract_claim_rows(
        graph=graph,
        case_uid_value=case_uid_value,
        root_claims_status=root_claims_status,
        warnings=warnings,
    )

    return {
        "method": method_name,
        "case_uid": case_uid_value,
        "claims": claim_rows,
        "status": status_rows,
        "trace": {
            "judgment_document": judgment_document,
            "full_transcript": transcript,
            "turn_artifacts": [],
            "graph_stats": _build_graph_stats_from_graph(graph),
            "requested_budget": dict(budget or {}),
            "requested_retrieval_config": dict(retrieval_config or {}),
        },
        "warnings": warnings,
    }


def validate_method_result(result: Mapping[str, Any]) -> None:
    """Validate Step 09 runner output schema."""
    required_top_level = {
        "method",
        "case_uid",
        "claims",
        "status",
        "trace",
        "token_usage",
        "latency",
        "warnings",
    }

    missing = sorted(required_top_level - set(result.keys()))

    if missing:
        raise ValueError(f"Method result missing required fields: {missing}")

    claims = result["claims"]
    status_rows = result["status"]

    if not isinstance(claims, list) or not isinstance(status_rows, list):
        raise ValueError("`claims` and `status` must be lists.")

    claim_ids = set()

    for row in claims:
        if not isinstance(row, Mapping):
            raise ValueError("Each claim row must be a mapping.")

        for key in ("uid", "claim_id", "claim_text_raw", "claim_text_norm"):
            if not str(row.get(key, "") or "").strip():
                raise ValueError(f"Claim row missing non-empty `{key}`.")

        claim_id = str(row["claim_id"])

        if claim_id in claim_ids:
            raise ValueError(f"Duplicate claim_id detected: {claim_id}")

        claim_ids.add(claim_id)

    status_claim_ids = set()

    for row in status_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Each status row must be a mapping.")

        uid = str(row.get("uid", "") or "").strip()
        claim_id = str(row.get("claim_id", "") or "").strip()
        status_eval = str(row.get("status_eval", "") or "").strip()

        if not uid or not claim_id or not status_eval:
            raise ValueError("Status row missing non-empty uid/claim_id/status_eval.")

        if status_eval not in ALLOWED_STATUS:
            raise ValueError(f"Invalid status_eval: {status_eval}")

        status_claim_ids.add(claim_id)

    if status_claim_ids != claim_ids:
        raise ValueError("Claim rows and status rows must cover the same claim_id set.")

    token_usage = result["token_usage"]

    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key not in token_usage:
            raise ValueError(f"token_usage missing `{key}`")

    if not isinstance(result["warnings"], list):
        raise ValueError("`warnings` must be a list.")


def execute_method(
    *,
    method_name: str,
    case: Mapping[str, Any],
    budget: Mapping[str, Any] | None = None,
    seed: int | None = None,
    retrieval_config: Mapping[str, Any] | None = None,
    verbose: bool = False,
    skip_validate_step: bool = False,
    disable_recall_worker: bool = False,
    disable_initial_insights: bool = False,
    direct_adjudication: bool = False,
) -> dict[str, Any]:
    """Run one experiment method on one case and return normalized output."""
    if case is None:
        raise ValueError("`case` must be provided.")

    apply_seed(seed)
    prepared_case, case_warnings = prepare_case_for_engine(case)

    cfg, cfg_warnings = build_method_config(
        skip_validate_step=skip_validate_step,
        disable_recall_worker=disable_recall_worker,
        disable_initial_insights=disable_initial_insights,
        budget=budget,
        retrieval_config=retrieval_config,
    )

    warnings = list(case_warnings) + list(cfg_warnings)
    case_uid_value = case_uid(prepared_case)
    start_completion, start_prompt = get_price()
    start_ts = time.perf_counter()

    if direct_adjudication:
        result = _execute_structured_single_pass(
            method_name=method_name,
            prepared_case=prepared_case,
            cfg=cfg,
            warnings=warnings,
            budget=budget,
            retrieval_config=retrieval_config,
        )

        end_completion, end_prompt = get_price()

        result["token_usage"] = {
            "prompt_tokens": max(0, int(end_prompt) - int(start_prompt)),
            "completion_tokens": max(0, int(end_completion) - int(start_completion)),
            "total_tokens": max(0, int(end_prompt) - int(start_prompt))
            + max(0, int(end_completion) - int(start_completion)),
        }

        result["latency"] = round(time.perf_counter() - start_ts, 6)
        validate_method_result(result)
        return result

    engine = DebateEngine(config=cfg, judge_config={})

    async def _run() -> None:
        await engine.setup(case_data=prepared_case, verbose=verbose)

        if direct_adjudication:
            await engine.adjudicate()
            return

        while not engine.is_finished:
            if engine.is_ready_for_adjudication:
                await engine.adjudicate()

            else:
                await engine.step()

    try:
        _run_coro_sync(_run())
        serial_snapshot = {}

        if hasattr(engine, "get_serializable_snapshot"):
            serial_snapshot = engine.get_serializable_snapshot()

        plain_snapshot = (
            engine.get_snapshot() if hasattr(engine, "get_snapshot") else {}
        )

        claim_rows, status_rows = _extract_claim_rows(
            graph=getattr(engine, "graph", None),
            case_uid_value=case_uid_value,
            root_claims_status=getattr(engine, "root_claims_status", {}),
            warnings=warnings,
        )

        end_completion, end_prompt = get_price()

        result = {
            "method": method_name,
            "case_uid": case_uid_value,
            "claims": claim_rows,
            "status": status_rows,
            "trace": {
                "judgment_document": str(
                    serial_snapshot.get("judgment_document")
                    or plain_snapshot.get("judgment_document", "")
                    or ""
                ),
                "full_transcript": list(
                    serial_snapshot.get("full_transcript")
                    or plain_snapshot.get("full_transcript", [])
                    or []
                ),
                "turn_artifacts": list(
                    getattr(engine, "turn_artifacts", [])
                    if hasattr(engine, "turn_artifacts")
                    else []
                ),
                "graph_stats": dict(serial_snapshot.get("graph_stats", {}) or {}),
                "requested_budget": dict(budget or {}),
                "requested_retrieval_config": dict(retrieval_config or {}),
            },
            "token_usage": {
                "prompt_tokens": max(0, int(end_prompt) - int(start_prompt)),
                "completion_tokens": max(
                    0, int(end_completion) - int(start_completion)
                ),
                "total_tokens": max(0, int(end_prompt) - int(start_prompt))
                + max(0, int(end_completion) - int(start_completion)),
            },
            "latency": round(time.perf_counter() - start_ts, 6),
            "warnings": warnings,
        }

        validate_method_result(result)
        return result

    finally:
        close_resources = getattr(engine, "close_resources", None)

        if callable(close_resources):
            _run_coro_sync(close_resources())


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON artifact to disk."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
