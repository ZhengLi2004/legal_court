"""Shared helpers for API response serializers."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _as_list(value: Any) -> List[Any]:
    """Return `value` when it is a list, otherwise return an empty list."""
    return value if isinstance(value, list) else []


def _safe_graph_from_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a normalized `graph_data` payload from a snapshot dict."""
    graph_data = snapshot.get("graph_data")

    if isinstance(graph_data, dict):
        nodes = _as_list(graph_data.get("nodes"))
        edges = _as_list(graph_data.get("edges"))
        return {"nodes": nodes, "edges": edges}

    return {"nodes": [], "edges": []}


def _safe_focus_node_ids(snapshot: Dict[str, Any]) -> List[str]:
    """Read and sanitize `focus_node_ids` from a snapshot payload."""
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
    """Build a stable edge identifier for diff computation."""
    edge_id = edge.get("id")

    if isinstance(edge_id, str) and edge_id:
        return edge_id

    source = edge.get("source", "")
    target = edge.get("target", "")
    edge_type = edge.get("type", "RELATION")
    return f"{source}->{target}:{edge_type}:{idx}"


def _enum_value(raw: Any) -> str:
    """Return enum `.value` text when present, otherwise string text."""
    if hasattr(raw, "value"):
        return str(getattr(raw, "value"))

    return str(raw or "")


def _serialize_shadow_graph(shadow_graph: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Serialize one shadow graph object to API `graph_data` shape."""
    graph = getattr(shadow_graph, "graph", None)

    if graph is None:
        return {"nodes": [], "edges": []}

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    try:
        graph_nodes = list(graph.nodes(data=True))

    except (AttributeError, TypeError, ValueError):
        graph_nodes = []

    for idx, row in enumerate(graph_nodes):
        node_id = str(row[0]).strip() if len(row) > 0 else f"node-{idx}"
        data = row[1] if len(row) > 1 and isinstance(row[1], dict) else {}
        content = str(data.get("content", "") or "").strip()
        node_type = _enum_value(data.get("type", "UNKNOWN")) or "UNKNOWN"
        status = _enum_value(data.get("status", ""))
        agent_id = str(data.get("agent_id", "") or "").strip()
        metadata = data.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "label": content or node_id,
                "content": content,
                "status": status,
                "agent_id": agent_id,
                "metadata": metadata,
            }
        )

    try:
        graph_edges = list(graph.edges(data=True))

    except (AttributeError, TypeError, ValueError):
        graph_edges = []

    for idx, row in enumerate(graph_edges):
        source = str(row[0]).strip() if len(row) > 0 else ""
        target = str(row[1]).strip() if len(row) > 1 else ""
        data = row[2] if len(row) > 2 and isinstance(row[2], dict) else {}
        edge_type = _enum_value(data.get("type", "RELATION")) or "RELATION"
        edge_id = str(data.get("id", "") or "").strip()

        if not source or not target:
            continue

        edges.append(
            {
                "id": edge_id or f"{source}->{target}:{edge_type}:{idx}",
                "source": source,
                "target": target,
                "type": edge_type,
            }
        )

    return {"nodes": nodes, "edges": edges}


def _normalize_case_summary(text: Any, max_len: int = 120) -> str:
    """Convert raw case context into compact single-line summary text."""
    value = " ".join(str(text or "").split()).strip()

    if not value:
        return ""

    if len(value) <= max_len:
        return value

    return f"{value[: max_len - 3]}..."


def _extract_case_fields(row: Any) -> Dict[str, str]:
    """Extract `case_id` and best-effort summary text from row/object."""
    if isinstance(row, dict):
        case_id = str(row.get("case_id") or row.get("id") or "").strip()

        summary = _normalize_case_summary(
            row.get("case_context") or row.get("context") or row.get("summary") or ""
        )

        return {"case_id": case_id, "summary": summary}

    case_id = str(
        getattr(row, "case_id", None) or getattr(row, "id", None) or ""
    ).strip()

    summary = _normalize_case_summary(
        getattr(row, "case_context", None)
        or getattr(row, "context", None)
        or getattr(row, "summary", None)
        or ""
    )

    return {"case_id": case_id, "summary": summary}


def _iter_graph_nodes(graph: Any) -> List[Any]:
    """Return graph nodes for networkx-like objects and simple stubs."""
    if graph is None:
        return []

    node_view = getattr(graph, "nodes", None)

    if callable(node_view):
        try:
            return list(node_view())

        except (TypeError, ValueError):
            return []

    if isinstance(node_view, list):
        return list(node_view)

    try:
        return list(node_view) if node_view is not None else []

    except TypeError:
        return []


def _iter_graph_edges(graph: Any) -> List[Any]:
    """Return graph edges for networkx-like objects and simple stubs."""
    if graph is None:
        return []

    edge_view = getattr(graph, "edges", None)

    if callable(edge_view):
        try:
            return list(edge_view(data=True))

        except TypeError:
            try:
                return list(edge_view())

            except (TypeError, ValueError):
                return []

        except (TypeError, ValueError):
            return []

    if isinstance(edge_view, list):
        return list(edge_view)

    try:
        return list(edge_view) if edge_view is not None else []

    except TypeError:
        return []


def _source_rank(source: str) -> int:
    """Return deterministic ordering rank for related-case source tags."""
    order = {
        "representative": 0,
        "insight_case": 1,
        "topology_neighbor": 2,
        "static_retrieved": 3,
        "dynamic_retrieved": 4,
    }

    return order.get(str(source), 99)


def _normalize_insight_text(text: Any) -> str:
    """Normalize one insight text row and drop placeholder-only values."""
    value = " ".join(str(text or "").strip().split())

    if not value:
        return ""

    value = re.sub(
        r"^\s*SIDE\s*[:：]\s*(PLAINTIFF|DEFENDANT|COMMON)\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"^(PLAINTIFF|DEFENDANT|COMMON)\s*[:：\-]\s*",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()

    for prefix in ("CONTENT:", "content:", "Insight:", "insight:", "策略：", "策略:"):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()

    token = value.upper()

    if token in {"PLAINTIFF", "DEFENDANT", "COMMON", "INSIGHT", "策略", "N/A"}:
        return ""

    if len(value) <= 3:
        return ""

    return value
