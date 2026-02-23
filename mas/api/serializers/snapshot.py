"""Snapshot response serializer."""

from __future__ import annotations

from typing import Any, Dict

from ..session_manager import DebateSession


def snapshot_response(session: DebateSession) -> Dict[str, Any]:
    """Build session snapshot response with normalized metrics.

    Args:
        session: Runtime debate session.

    Returns:
        Snapshot payload merged from engine state and session metadata.
    """
    base = session.engine.get_serializable_snapshot()
    graph_stats = base.get("graph_stats", {})

    payload = {
        **base,
        "session_id": session.session_id,
        "status": session.status.value,
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
