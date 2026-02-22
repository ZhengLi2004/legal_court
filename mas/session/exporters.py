"""Export helpers for graph and replay bundle serialization."""

from __future__ import annotations

import io
import json
from typing import Any, Callable, Dict, List, Optional

import networkx as nx


def graph_data_for_export(session: Any, round_idx: Optional[int] = None) -> Dict[str, Any]:
    """Resolve graph payload used by export endpoints."""
    snapshots = getattr(session.engine, "round_snapshots", [])

    if isinstance(round_idx, int) and isinstance(snapshots, list):
        if 0 <= round_idx < len(snapshots):
            row = snapshots[round_idx]

            if isinstance(row, dict):
                graph_data = row.get("graph_data", {})

                if isinstance(graph_data, dict):
                    return graph_data

    getter = getattr(session.engine, "get_serializable_snapshot", None)

    if callable(getter):
        payload = getter()

        if isinstance(payload, dict):
            graph_data = payload.get("graph_data", {})

            if isinstance(graph_data, dict):
                return graph_data

    return {"nodes": [], "edges": []}


def export_graph_gexf(
    session: Any,
    round_idx: Optional[int],
    to_json_safe: Callable[[Any], Any],
) -> bytes:
    """Export current or historical graph as GEXF bytes."""
    graph_data = graph_data_for_export(session, round_idx=round_idx)
    graph = nx.DiGraph()

    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict):
            continue

        node_id = str(node.get("id", "")).strip()

        if not node_id:
            continue

        attrs: Dict[str, Any] = {}

        for key, value in node.items():
            if key == "id":
                continue

            if isinstance(value, (str, int, float, bool)) or value is None:
                attrs[str(key)] = value

            else:
                attrs[str(key)] = json.dumps(to_json_safe(value), ensure_ascii=False)

        graph.add_node(node_id, **attrs)

    for edge in graph_data.get("edges", []):
        if not isinstance(edge, dict):
            continue

        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()

        if not source or not target:
            continue

        attrs = {}

        for key, value in edge.items():
            if key in {"source", "target"}:
                continue

            if isinstance(value, (str, int, float, bool)) or value is None:
                attrs[str(key)] = value

            else:
                attrs[str(key)] = json.dumps(to_json_safe(value), ensure_ascii=False)

        graph.add_edge(source, target, **attrs)

    buffer = io.BytesIO()
    nx.write_gexf(graph, buffer, encoding="utf-8")
    return buffer.getvalue()


def build_replay_bundle(
    *,
    session: Any,
    snapshot_index: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    artifacts: List[Dict[str, Any]],
    utc_now_iso: Callable[[], str],
    to_json_safe: Callable[[Any], Any],
) -> Dict[str, Any]:
    """Build a complete replay bundle for offline analysis."""
    getter = getattr(session.engine, "get_serializable_snapshot", None)
    snapshot_payload = getter() if callable(getter) else {}

    if not isinstance(snapshot_payload, dict):
        snapshot_payload = {}

    snapshots = getattr(session.engine, "round_snapshots", [])

    if not isinstance(snapshots, list):
        snapshots = []

    return {
        "session": {
            "session_id": session.session_id,
            "status": session.status,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "failure_simulation": dict(session.failure_simulation),
        },
        "snapshot": snapshot_payload,
        "snapshot_index": snapshot_index,
        "snapshots": to_json_safe(snapshots),
        "events": events,
        "turn_artifacts": to_json_safe(artifacts),
        "metadata": {
            "generated_at": utc_now_iso(),
            "event_count": len(events),
            "artifact_count": len(artifacts),
            "snapshot_count": len(snapshots),
        },
    }

