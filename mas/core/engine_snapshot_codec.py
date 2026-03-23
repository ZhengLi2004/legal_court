"""Snapshot codec helpers for DebateEngine."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set

from .engine_convergence import (
    determine_engine_converged,
    normalize_engine_convergence_payload,
)
from .graph import EdgeType, NodeStatus, ShadowGraph


def create_engine_snapshot(engine: Any, round_idx: int, turn: str) -> Dict[str, Any]:
    """Create one persisted round snapshot from engine runtime state.

    Args:
        engine: Debate engine instance.
        round_idx: Round index to persist.
        turn: Turn label (`"plaintiff"` or `"defendant"`).

    Returns:
        Snapshot dict containing graph payload, convergence payload, transcript,
        root-claim status map, and summary stats.
    """
    graph_data = {
        "nodes": [
            {"id": node_id, **{k: engine._serialize_value(v) for k, v in data.items()}}
            for node_id, data in engine.graph.graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                **{k: engine._serialize_value(v) for k, v in data.items()},
            }
            for source, target, data in engine.graph.graph.edges(data=True)
        ],
    }

    serializable_claims_status = {
        key: value.value if hasattr(value, "value") else str(value)
        for key, value in engine.root_claims_status.items()
    }

    graph_stats = build_engine_graph_stats(engine, graph_data)

    convergence_payload = normalize_engine_convergence_payload(
        engine,
        engine.last_step_log.get("convergence", {}),
        fallback_is_converged=bool(engine.is_ready_for_adjudication),
    )

    snapshot = {
        "round_idx": round_idx,
        "turn": turn,
        "current_turn": engine.current_turn.value,
        "timestamp": engine.legal_sys.step_counter if engine.legal_sys else 0,
        "ts_ms": int(time.time() * 1000),
        "graph_data": graph_data,
        "focus_node_ids": build_engine_focus_node_ids(engine),
        "latest_turn_uid": engine.latest_turn_uid,
        "convergence": convergence_payload,
        "transcript": list(engine.transcript),
        "root_claims_status": serializable_claims_status,
        "stats": {
            "node_count": graph_stats["node_count"],
            "edge_count": graph_stats["edge_count"],
            "claim_nodes": graph_stats["claim_nodes"],
            "conflict_edges": graph_stats["conflict_edges"],
        },
        "action_summary": engine.last_step_log.get("action", ""),
        "is_ready_for_adjudication": engine.is_ready_for_adjudication,
        "is_finished": engine.is_finished,
    }

    return snapshot


def count_engine_conflict_edges(engine: Any) -> int:
    """Count CONFLICT edges in the current graph.

    Args:
        engine: Debate engine instance.

    Returns:
        Number of edges whose type resolves to `EdgeType.CONFLICT`.
    """
    if not engine.graph or not engine.graph.graph:
        return 0

    count = 0

    for _, _, data in engine.graph.graph.edges(data=True):
        edge_type = data.get("type")

        if edge_type == EdgeType.CONFLICT or str(edge_type) in {
            "CONFLICT",
            "EdgeType.CONFLICT",
        }:
            count += 1

    return count


def build_engine_graph_data(engine: Any) -> Dict[str, Any]:
    """Build JSON-safe graph payload for transport or API responses.

    Args:
        engine: Debate engine instance.

    Returns:
        Dict with `nodes` and `edges` arrays containing JSON-safe values.
    """
    if not engine.graph:
        return {"nodes": [], "edges": []}

    return {
        "nodes": [
            {"id": node_id, **{k: engine._to_json_safe(v) for k, v in data.items()}}
            for node_id, data in engine.graph.graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                **{k: engine._to_json_safe(v) for k, v in data.items()},
            }
            for source, target, data in engine.graph.graph.edges(data=True)
        ],
    }


def build_engine_focus_node_ids(engine: Any) -> List[str]:
    """Build ordered focus-node identifiers for tactical context rendering.

    Args:
        engine: Debate engine instance.

    Returns:
        Deduplicated focus node id list. Returns empty list on missing graph or
        when focus-node calculation fails.
    """
    if not engine.graph:
        return []

    current_step = 0

    if engine.legal_sys is not None:
        try:
            current_step = int(getattr(engine.legal_sys, "step_counter", 0) or 0)

        except (TypeError, ValueError):
            current_step = 0

    if current_step <= 0:
        current_step = max(0, int(engine.round_idx))

    try:
        focus_nodes = engine.graph._calculate_focus_nodes(current_step)

    except (AttributeError, TypeError, ValueError):
        return []

    ordered: List[str] = []
    seen: Set[str] = set()

    for node_id in focus_nodes:
        node_text = str(node_id).strip()

        if not node_text or node_text in seen:
            continue

        seen.add(node_text)
        ordered.append(node_text)

    return ordered


def count_support_edges_from_graph_data(graph_data: Dict[str, Any]) -> int:
    """Count SUPPORT edges from serialized graph payload.

    Args:
        graph_data: Graph payload containing edge rows.

    Returns:
        Number of edges whose serialized type is support.
    """
    count = 0

    for edge in graph_data.get("edges", []):
        if str(edge.get("type")) in {"SUPPORT", "EdgeType.SUPPORT"}:
            count += 1

    return count


def build_engine_graph_stats(
    engine: Any,
    graph_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """Build unified graph stats payload for snapshots and APIs.

    Args:
        engine: Debate engine instance.
        graph_data: Optional pre-built graph payload. When omitted, the function
            builds graph data from engine state.

    Returns:
        Dict containing node/edge counts and conflict/support derived counts.
    """
    graph_payload = (
        graph_data if graph_data is not None else build_engine_graph_data(engine)
    )

    conflict_count = count_engine_conflict_edges(engine)
    support_count = count_support_edges_from_graph_data(graph_payload)

    return {
        "node_count": len(graph_payload.get("nodes", [])),
        "edge_count": len(graph_payload.get("edges", [])),
        "claim_nodes": engine._count_claim_nodes(),
        "conflict_edges": conflict_count,
        "edge_conflict_count": conflict_count,
        "edge_attack_count": conflict_count,
        "edge_support_count": support_count,
    }


def collect_engine_agent_memories(engine: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Collect serializable controller memories from both teams.

    Args:
        engine: Debate engine instance.

    Returns:
        Dict with `plaintiff` and `defendant` memory item arrays. Missing or
        invalid controller memory sources produce empty arrays.
    """
    plaintiff_memories: List[Dict[str, Any]] = []

    if engine.p_team and engine.p_team.controller:
        try:
            plaintiff_memories = [
                item.model_dump() for item in engine.p_team.controller.get_memories()
            ]

        except (AttributeError, TypeError, ValueError):
            pass

    defendant_memories: List[Dict[str, Any]] = []

    if engine.d_team and engine.d_team.controller:
        try:
            defendant_memories = [
                item.model_dump() for item in engine.d_team.controller.get_memories()
            ]

        except (AttributeError, TypeError, ValueError):
            pass

    return {"plaintiff": plaintiff_memories, "defendant": defendant_memories}


def build_engine_turn_artifact(
    engine: Any,
    turn: str,
    turn_result: Dict[str, Any],
    narrative_text: str,
) -> Dict[str, Any]:
    """Build one JSON-safe turn artifact record.

    Args:
        engine: Debate engine instance.
        turn: Turn side label.
        turn_result: Raw turn execution payload.
        narrative_text: Polished narrative text for this turn.

    Returns:
        Normalized turn artifact dict with stable `turn_uid` and timestamp.
    """
    turn_uid = str(turn_result.get("turn_uid", "")).strip()

    if not turn_uid:
        turn_uid = f"turn_{engine.round_idx}_{turn}_{int(time.time() * 1000)}"

    try:
        ts_ms = int(turn_result.get("ts_ms", int(time.time() * 1000)))

    except (TypeError, ValueError):
        ts_ms = int(time.time() * 1000)

    artifact = {
        "turn_uid": turn_uid,
        "side": turn,
        "round_idx": engine.round_idx,
        "controller_assessment": turn_result.get("controller_assessment", {}),
        "batch_instructions": turn_result.get("batch_instructions", []),
        "worker_reports": turn_result.get("worker_reports_raw", []),
        "decision_raw": turn_result.get("decision_raw", ""),
        "parsed_actions": turn_result.get("parsed_actions", []),
        "execution_logs": turn_result.get("execution_log", ""),
        "retry_history": turn_result.get("retry_history", []),
        "action_cache": turn_result.get("action_cache", []),
        "plan_attempts_used": turn_result.get("plan_attempts_used", 0),
        "push_attempts_used": turn_result.get("push_attempts_used", 0),
        "narrative_raw_sentences": turn_result.get("narrative_raw_sentences", []),
        "narrative_polished": narrative_text or "",
        "action_summary": turn_result.get("summary", ""),
        "ts_ms": ts_ms,
    }

    return engine._to_json_safe(artifact)


def get_engine_turn_artifacts(
    engine: Any,
    turn_uid: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return stored turn artifacts, optionally filtered by turn UID.

    Args:
        engine: Debate engine instance.
        turn_uid: Optional turn UID filter.
        limit: Maximum number of artifact rows to return.

    Returns:
        Tail slice of artifact rows in chronological order.
    """
    rows = list(engine.turn_artifacts)

    if turn_uid:
        rows = [item for item in rows if str(item.get("turn_uid", "")) == turn_uid]

    safe_limit = max(1, int(limit))
    return rows[-safe_limit:]


def restore_engine_snapshot(engine: Any, round_idx: int) -> bool:
    """Restore engine runtime state from one stored round snapshot.

    Args:
        engine: Debate engine instance.
        round_idx: Zero-based snapshot index in `engine.round_snapshots`.

    Returns:
        `True` when restore succeeds; `False` when index is out of range.

    Side Effects:
        Replaces `engine.graph`, updates round/turn/convergence/session fields,
        recalculates `prev_stats`, and updates `engine.last_step_log`.
    """
    if round_idx < 0 or round_idx >= len(engine.round_snapshots):
        return False

    snapshot = engine.round_snapshots[round_idx]
    graph_data = snapshot.get("graph_data", {})
    restored_graph = ShadowGraph()

    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict):
            continue

        node_row = dict(node)
        node_id = node_row.pop("id", None)

        if not node_id:
            continue

        if "type" in node_row:
            node_row["type"] = engine._deserialize_node_type(node_row["type"])

        if "status" in node_row:
            node_row["status"] = engine._deserialize_node_status(node_row["status"])

        restored_graph.graph.add_node(node_id, **node_row)

    for edge in graph_data.get("edges", []):
        if not isinstance(edge, dict):
            continue

        edge_row = dict(edge)
        source = edge_row.pop("source", None)
        target = edge_row.pop("target", None)

        if not source or not target:
            continue

        if "type" in edge_row:
            edge_row["type"] = engine._deserialize_edge_type(edge_row["type"])

        restored_graph.graph.add_edge(source, target, **edge_row)

    engine.graph = restored_graph
    engine.round_idx = int(snapshot.get("round_idx", round_idx))
    engine.transcript = list(snapshot.get("transcript", []))
    engine.is_finished = bool(snapshot.get("is_finished", False))

    engine.latest_turn_uid = str(
        snapshot.get("latest_turn_uid", engine.latest_turn_uid)
    )

    convergence_raw = snapshot.get("convergence", {})
    has_convergence_payload = isinstance(convergence_raw, dict)
    initial_ready_hint = bool(snapshot.get("is_ready_for_adjudication", False))

    convergence_payload = normalize_engine_convergence_payload(
        engine,
        convergence_raw if has_convergence_payload else None,
        fallback_is_converged=initial_ready_hint,
    )

    history = convergence_payload.get("history", [])
    engine.convergence_history = list(history) if isinstance(history, list) else []
    raw_claim_status = snapshot.get("root_claims_status", {})
    restored_claim_status: Dict[str, NodeStatus] = {}

    if isinstance(raw_claim_status, dict):
        for claim_id, status in raw_claim_status.items():
            restored_claim_status[claim_id] = engine._deserialize_node_status(status)

    engine.root_claims_status = restored_claim_status

    if not isinstance(engine.last_step_log, dict):
        engine.last_step_log = {}

    engine.last_step_log["action"] = snapshot.get("action_summary", "")
    engine.last_step_log["round"] = engine.round_idx
    engine.last_step_log["turn"] = snapshot.get("turn", "")

    is_converged_raw = (
        convergence_raw.get("is_converged") if has_convergence_payload else None
    )

    if is_converged_raw is None:
        ready_hint = snapshot.get("is_ready_for_adjudication")

        if ready_hint is None and has_convergence_payload:
            is_converged_raw = determine_engine_converged(
                round_idx=engine.round_idx,
                sma=float(convergence_payload.get("sma", 0.0)),
                epsilon=float(convergence_payload.get("epsilon", 0.0)),
                min_rounds=int(convergence_payload.get("min_rounds", 0)),
            )

        else:
            is_converged_raw = bool(ready_hint)

    convergence_payload["is_converged"] = bool(is_converged_raw)
    engine.is_ready_for_adjudication = bool(is_converged_raw) and not engine.is_finished
    engine.last_step_log["convergence"] = convergence_payload
    turn_name = str(snapshot.get("turn", "")).lower()
    current_turn_type = type(engine.current_turn)
    plaintiff = getattr(current_turn_type, "PLAINTIFF", None)
    defendant = getattr(current_turn_type, "DEFENDANT", None)

    if plaintiff is not None and defendant is not None:
        if turn_name == getattr(plaintiff, "value", "plaintiff"):
            engine.current_turn = defendant

        elif turn_name == getattr(defendant, "value", "defendant"):
            engine.current_turn = plaintiff

        else:
            engine.current_turn = plaintiff

    if engine.legal_sys is not None:
        restored_step = None

        for candidate in (
            snapshot.get("timestamp"),
            snapshot.get("step_counter"),
            snapshot.get("round_idx"),
        ):
            try:
                if candidate is None:
                    continue

                parsed = int(candidate)

                if parsed < 0:
                    continue

                restored_step = parsed
                break

            except (TypeError, ValueError):
                continue

        if restored_step is not None:
            try:
                engine.legal_sys.step_counter = restored_step

            except (AttributeError, TypeError, ValueError):
                pass

    engine.prev_stats = {
        "claim_nodes": engine._count_claim_nodes(),
        "conflict_edges": count_engine_conflict_edges(engine),
    }

    return True


def build_engine_snapshot_state(engine: Any) -> Dict[str, Any]:
    """Build JSON-safe live state snapshot for API responses.

    Args:
        engine: Debate engine instance.

    Returns:
        JSON-safe state payload with graph/convergence/transcript/summary fields.
    """
    graph_data = build_engine_graph_data(engine)
    graph_stats = build_engine_graph_stats(engine, graph_data)
    transcript_rows = list(engine.transcript)

    convergence_payload = normalize_engine_convergence_payload(
        engine,
        engine.last_step_log.get("convergence", {}),
        fallback_is_converged=bool(engine.is_ready_for_adjudication),
    )

    serializable_claims_status = {
        key: engine._serialize_value(value)
        for key, value in engine.root_claims_status.items()
    }

    state = {
        "current_round": engine.round_idx,
        "current_turn": engine.current_turn.value,
        "is_ready_for_adjudication": engine.is_ready_for_adjudication,
        "is_finished": engine.is_finished,
        "winner": engine.winner,
        "root_claims_status": serializable_claims_status,
        "last_log": engine.last_step_log,
        "transcript": transcript_rows,
        "full_transcript": transcript_rows,
        "convergence_history": list(convergence_payload.get("history", [])),
        "convergence": convergence_payload,
        "latest_turn_uid": engine.latest_turn_uid,
        "turn_artifact_count": len(engine.turn_artifacts),
        "graph_data": graph_data,
        "focus_node_ids": build_engine_focus_node_ids(engine),
        "graph_stats": {
            "node_count": graph_stats["node_count"],
            "edge_count": graph_stats["edge_count"],
            "claim_nodes": graph_stats["claim_nodes"],
            "edge_conflict_count": graph_stats["edge_conflict_count"],
            "edge_attack_count": graph_stats["edge_attack_count"],
            "edge_support_count": graph_stats["edge_support_count"],
        },
        "agent_memories": collect_engine_agent_memories(engine),
    }

    return engine._to_json_safe(state)
