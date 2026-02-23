"""Graph diff serializer."""

from __future__ import annotations

from typing import Any, Dict, List

from ..session_manager import DebateSession
from ._common import _as_list, _edge_identifier
from .graph import graph_response


def graph_diff_response(
    session: DebateSession, from_round: int, to_round: int
) -> Dict[str, Any]:
    """Compute structural and status differences between two rounds."""
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
