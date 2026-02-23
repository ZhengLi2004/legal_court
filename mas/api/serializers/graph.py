"""Graph response serializers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..session_manager import DebateSession
from ._common import (
    _as_list,
    _safe_focus_node_ids,
    _safe_graph_from_snapshot,
)
from .snapshot import snapshot_response


def graph_response(
    session: DebateSession, round_idx: Optional[int] = None
) -> Dict[str, Any]:
    """Build graph payload for latest state or one historical round.

    Args:
        session: Runtime debate session.
        round_idx: Optional round snapshot index.

    Returns:
        Payload containing `session_id`, resolved `round_idx`, `graph_data`,
        and `focus_node_ids`.
    """
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
