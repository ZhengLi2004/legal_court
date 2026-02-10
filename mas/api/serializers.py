"""Serialization helpers for API responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .session_manager import DebateSession


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_graph_from_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    graph_data = snapshot.get("graph_data")

    if isinstance(graph_data, dict):
        nodes = _as_list(graph_data.get("nodes"))
        edges = _as_list(graph_data.get("edges"))
        return {"nodes": nodes, "edges": edges}

    return {"nodes": [], "edges": []}


def _edge_identifier(edge: Dict[str, Any], idx: int) -> str:
    edge_id = edge.get("id")

    if isinstance(edge_id, str) and edge_id:
        return edge_id

    source = edge.get("source", "")
    target = edge.get("target", "")
    edge_type = edge.get("type", "RELATION")
    return f"{source}->{target}:{edge_type}:{idx}"


def snapshot_response(session: DebateSession) -> Dict[str, Any]:
    """Build API snapshot payload with compatibility fields."""
    base = session.engine.get_serializable_snapshot()
    graph_stats = base.get("graph_stats", {})

    payload = {
        **base,
        "session_id": session.session_id,
        "status": session.status,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "metrics": {
            "arguments": int(graph_stats.get("node_count", 0)),
            "attacks": int(graph_stats.get("edge_attack_count", 0)),
            "supports": int(graph_stats.get("edge_support_count", 0)),
        },
    }

    if session.last_error:
        payload["error"] = session.last_error

    return payload


def graph_response(
    session: DebateSession, round_idx: Optional[int] = None
) -> Dict[str, Any]:
    """Return graph payload for current or historical round."""
    if round_idx is None:
        snap = snapshot_response(session)
        graph_data = _safe_graph_from_snapshot(snap)
        current_round = int(snap.get("current_round", 0))

        return {
            "session_id": session.session_id,
            "round_idx": current_round,
            "graph_data": graph_data,
        }

    snapshots = _as_list(getattr(session.engine, "round_snapshots", []))

    if 0 <= round_idx < len(snapshots):
        row = snapshots[round_idx]
        graph_data = _safe_graph_from_snapshot(row)

        return {
            "session_id": session.session_id,
            "round_idx": int(row.get("round_idx", round_idx)),
            "graph_data": graph_data,
        }

    latest = snapshot_response(session)

    return {
        "session_id": session.session_id,
        "round_idx": int(latest.get("current_round", 0)),
        "graph_data": _safe_graph_from_snapshot(latest),
    }


def graph_diff_response(
    session: DebateSession, from_round: int, to_round: int
) -> Dict[str, Any]:
    """Compute graph diff between two rounds."""
    from_graph = graph_response(session, from_round)["graph_data"]
    to_graph = graph_response(session, to_round)["graph_data"]

    from_node_ids = {
        str(node.get("id", "")) for node in _as_list(from_graph.get("nodes"))
    }

    to_node_ids = {str(node.get("id", "")) for node in _as_list(to_graph.get("nodes"))}

    from_edge_ids = {
        _edge_identifier(edge, idx)
        for idx, edge in enumerate(_as_list(from_graph.get("edges")))
    }

    to_edge_ids = {
        _edge_identifier(edge, idx)
        for idx, edge in enumerate(_as_list(to_graph.get("edges")))
    }

    return {
        "session_id": session.session_id,
        "from_round": from_round,
        "to_round": to_round,
        "added_node_ids": sorted(
            [item for item in to_node_ids if item not in from_node_ids]
        ),
        "removed_node_ids": sorted(
            [item for item in from_node_ids if item not in to_node_ids]
        ),
        "added_edge_ids": sorted(
            [item for item in to_edge_ids if item not in from_edge_ids]
        ),
        "removed_edge_ids": sorted(
            [item for item in from_edge_ids if item not in to_edge_ids]
        ),
    }


def memory_response(session: DebateSession) -> Dict[str, Any]:
    """Extract a compact memory payload for frontend display."""
    engine = session.engine
    legal_sys = getattr(engine, "legal_sys", None)
    insight_summaries: List[str] = []
    static_history_count = 0
    dynamic_law_case_count = 0
    task_layer_node_count = 0

    if legal_sys is not None:
        insights_manager = getattr(legal_sys, "insights", None)
        raw_insights = getattr(insights_manager, "insights", [])

        for item in _as_list(raw_insights):
            content = getattr(item, "content", None)

            if isinstance(content, str) and content.strip():
                insight_summaries.append(content.strip())

        static_history_count = len(
            _as_list(getattr(legal_sys, "_static_history_cases", []))
        )

        dynamic_law_case_count = len(
            _as_list(getattr(legal_sys, "_dynamic_law_cases", []))
        )

        memory = getattr(legal_sys, "memory", None)
        task_layer = getattr(memory, "task_layer", None)
        graph = getattr(task_layer, "graph", None)
        node_view = getattr(graph, "nodes", None)

        if node_view is not None:
            try:
                task_layer_node_count = len(node_view)

            except Exception:
                task_layer_node_count = 0

    return {
        "session_id": session.session_id,
        "insight_summaries": insight_summaries,
        "static_history_count": static_history_count,
        "dynamic_law_case_count": dynamic_law_case_count,
        "task_layer": {"node_count": task_layer_node_count},
    }
