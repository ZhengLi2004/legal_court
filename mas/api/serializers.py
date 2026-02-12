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


def _safe_focus_node_ids(snapshot: Dict[str, Any]) -> List[str]:
    rows = snapshot.get("focus_node_ids")

    if not isinstance(rows, list):
        return []

    result: List[str] = []

    for item in rows:
        node_id = str(item).strip()

        if node_id:
            result.append(node_id)

    return result


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
            "focus_node_ids": _safe_focus_node_ids(snap),
        }

    snapshots = _as_list(getattr(session.engine, "round_snapshots", []))

    if 0 <= round_idx < len(snapshots):
        row = snapshots[round_idx]
        graph_data = _safe_graph_from_snapshot(row)

        return {
            "session_id": session.session_id,
            "round_idx": int(row.get("round_idx", round_idx)),
            "graph_data": graph_data,
            "focus_node_ids": _safe_focus_node_ids(row),
        }

    latest = snapshot_response(session)

    return {
        "session_id": session.session_id,
        "round_idx": int(latest.get("current_round", 0)),
        "graph_data": _safe_graph_from_snapshot(latest),
        "focus_node_ids": _safe_focus_node_ids(latest),
    }


def graph_diff_response(
    session: DebateSession, from_round: int, to_round: int
) -> Dict[str, Any]:
    """Compute graph diff between two rounds."""
    from_graph = graph_response(session, from_round)["graph_data"]
    to_graph = graph_response(session, to_round)["graph_data"]
    from_nodes = _as_list(from_graph.get("nodes"))
    to_nodes = _as_list(to_graph.get("nodes"))
    from_node_ids = {str(node.get("id", "")) for node in from_nodes}
    to_node_ids = {str(node.get("id", "")) for node in to_nodes}

    from_edge_ids = {
        _edge_identifier(edge, idx)
        for idx, edge in enumerate(_as_list(from_graph.get("edges")))
    }

    to_edge_ids = {
        _edge_identifier(edge, idx)
        for idx, edge in enumerate(_as_list(to_graph.get("edges")))
    }

    status_changed_node_ids: List[str] = []

    from_node_by_id = {
        str(node.get("id", "")): node
        for node in from_nodes
        if isinstance(node, dict) and node.get("id") is not None
    }

    for node in to_nodes:
        if not isinstance(node, dict):
            continue

        node_id = str(node.get("id", ""))
        prev = from_node_by_id.get(node_id)

        if not isinstance(prev, dict):
            continue

        if str(prev.get("status", "")) != str(node.get("status", "")):
            status_changed_node_ids.append(node_id)

    added_nodes = sorted([item for item in to_node_ids if item not in from_node_ids])
    removed_nodes = sorted([item for item in from_node_ids if item not in to_node_ids])
    added_edges = sorted([item for item in to_edge_ids if item not in from_edge_ids])
    removed_edges = sorted([item for item in from_edge_ids if item not in to_edge_ids])
    status_changed_node_ids = sorted(list(set(status_changed_node_ids)))

    return {
        "session_id": session.session_id,
        "from_round": from_round,
        "to_round": to_round,
        "added_node_ids": added_nodes,
        "removed_node_ids": removed_nodes,
        "added_edge_ids": added_edges,
        "removed_edge_ids": removed_edges,
        "status_changed_node_ids": status_changed_node_ids,
        "changed_node_ids": sorted(
            list(set(added_nodes + removed_nodes + status_changed_node_ids))
        ),
        "changed_edge_ids": sorted(list(set(added_edges + removed_edges))),
    }


def memory_response(session: DebateSession) -> Dict[str, Any]:
    """Extract a compact memory payload for frontend display."""
    engine = session.engine
    legal_sys = getattr(engine, "legal_sys", None)
    insight_summaries: List[str] = []
    insight_items: List[Dict[str, Any]] = []
    representative_case_ids: List[str] = []
    static_history_count = 0
    dynamic_law_case_count = 0
    task_layer_node_count = 0
    task_layer_edge_count = 0
    task_layer_graph: Dict[str, List[Dict[str, Any]]] = {"nodes": [], "edges": []}
    case_snapshots: List[Dict[str, Any]] = []

    if legal_sys is not None:
        insights_manager = getattr(legal_sys, "insights", None)
        raw_insights = getattr(insights_manager, "insights", [])

        for item in _as_list(raw_insights):
            content = getattr(item, "content", None)

            if isinstance(content, str) and content.strip():
                clean_content = content.strip()
                insight_summaries.append(clean_content)
                side_value = getattr(item, "side", "COMMON")
                side_text = getattr(side_value, "value", side_value)
                raw_cases = _as_list(getattr(item, "cases", []))

                cases = sorted(
                    {
                        str(case_id).strip()
                        for case_id in raw_cases
                        if str(case_id).strip()
                    }
                )

                raw_representatives = _as_list(getattr(item, "representatives", []))

                representatives = sorted(
                    {
                        str(case_id).strip()
                        for case_id in raw_representatives
                        if str(case_id).strip()
                    }
                )

                if not representatives:
                    representatives = list(cases)

                representative_case_ids.extend(representatives)

                insight_items.append(
                    {
                        "content": clean_content,
                        "side": str(side_text or "COMMON"),
                        "cases": cases,
                        "representatives": representatives,
                        "case_count": len(cases),
                        "representative_count": len(representatives),
                        "linked_round": int(getattr(engine, "round_idx", 0) or 0),
                    }
                )

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
        edge_view = getattr(graph, "edges", None)

        if node_view is not None:
            try:
                task_layer_node_count = len(node_view)

            except Exception:
                task_layer_node_count = 0

        if edge_view is not None:
            try:
                task_layer_edge_count = len(edge_view)

            except Exception:
                task_layer_edge_count = 0

        if graph is not None:
            try:
                for idx, node in enumerate(graph.nodes()):
                    node_id = str(node).strip() or f"case-{idx}"

                    task_layer_graph["nodes"].append(
                        {
                            "id": node_id,
                            "label": node_id,
                            "kind": "case",
                        }
                    )

                for idx, edge in enumerate(graph.edges()):
                    source = str(edge[0]).strip() if len(edge) > 0 else ""
                    target = str(edge[1]).strip() if len(edge) > 1 else ""

                    edge_type = (
                        str(edge[2]).strip()
                        if len(edge) > 2 and edge[2] not in {None, ""}
                        else "reference"
                    )

                    task_layer_graph["edges"].append(
                        {
                            "id": f"{source}->{target}:{edge_type}:{idx}",
                            "source": source,
                            "target": target,
                            "type": edge_type,
                        }
                    )

            except Exception:
                task_layer_graph = {"nodes": [], "edges": []}

    snapshots = _as_list(getattr(engine, "round_snapshots", []))

    for idx, row in enumerate(snapshots):
        if not isinstance(row, dict):
            continue

        graph_data = row.get("graph_data", {})

        if isinstance(graph_data, dict):
            nodes = _as_list(graph_data.get("nodes"))
            edges = _as_list(graph_data.get("edges"))
            node_count = len(nodes)
            edge_count = len(edges)

        else:
            node_count = 0
            edge_count = 0

        case_snapshots.append(
            {
                "round_idx": int(row.get("round_idx", idx)),
                "turn": str(row.get("turn", "")),
                "ts_ms": int(row.get("ts_ms", row.get("timestamp", 0)) or 0),
                "node_count": node_count,
                "edge_count": edge_count,
            }
        )

    insight_items = sorted(
        insight_items,
        key=lambda item: (
            -int(item.get("case_count", 0)),
            str(item.get("content", "")),
        ),
    )

    return {
        "session_id": session.session_id,
        "insight_summaries": insight_summaries,
        "insight_items": insight_items,
        "representative_case_ids": sorted({item for item in representative_case_ids}),
        "static_history_count": static_history_count,
        "dynamic_law_case_count": dynamic_law_case_count,
        "task_layer": {
            "node_count": task_layer_node_count,
            "edge_count": task_layer_edge_count,
        },
        "task_layer_graph": task_layer_graph,
        "case_snapshots": case_snapshots,
    }
