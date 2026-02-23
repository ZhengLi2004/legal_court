"""Session-related HTTP routes."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Response

from ..http_errors import raise_as_http
from ..request_models import CreateSessionRequest
from ..serializers import graph_diff_response, graph_response, snapshot_response
from ..session_manager import SessionManager


def build_sessions_router(
    manager: SessionManager,
    *,
    default_case_data: Optional[Dict[str, Any]] = None,
) -> APIRouter:
    """Build the session router with injected manager dependencies."""
    router = APIRouter()

    @router.post("/api/v1/sessions")
    async def create_session(body: CreateSessionRequest) -> Dict[str, Any]:
        """Create a debate session and auto-run setup."""
        case_data = body.case_data if body.case_data is not None else default_case_data

        session = await manager.create_session(
            case_data=case_data,
            auto_setup=True,
        )

        return snapshot_response(session)

    @router.get("/api/v1/sessions")
    async def list_sessions() -> Dict[str, Any]:
        """List active in-memory sessions."""
        return {
            "sessions": [snapshot_response(item) for item in manager.list_sessions()]
        }

    @router.get("/api/v1/sessions/{session_id}/snapshot")
    async def get_session_snapshot(session_id: str) -> Dict[str, Any]:
        """Return the latest serializable snapshot for one session."""
        try:
            session = manager.get_session(session_id)
            return snapshot_response(session)

        except KeyError as exc:
            raise raise_as_http(
                exc,
                action="Get session snapshot",
                mappings=((KeyError, 404),),
            ) from exc

    @router.post("/api/v1/sessions/{session_id}/step")
    async def step_session(session_id: str) -> Dict[str, Any]:
        """Advance a session by one debate turn."""
        try:
            session = await manager.step_session(session_id)
            return snapshot_response(session)

        except (KeyError, ValueError) as exc:
            raise raise_as_http(
                exc,
                action="Step session",
                mappings=((KeyError, 404), (ValueError, 409)),
            ) from exc

    @router.post("/api/v1/sessions/{session_id}/adjudicate")
    async def adjudicate_session(session_id: str) -> Dict[str, Any]:
        """Run final adjudication for a session."""
        try:
            session = await manager.adjudicate_session(session_id)
            return snapshot_response(session)

        except KeyError as exc:
            raise raise_as_http(
                exc,
                action="Adjudicate session",
                mappings=((KeyError, 404),),
            ) from exc

    @router.get("/api/v1/sessions/{session_id}/graph")
    async def get_graph(session_id: str) -> Dict[str, Any]:
        """Return graph data for the latest session state."""
        try:
            session = manager.get_session(session_id)
            return graph_response(session)

        except KeyError as exc:
            raise raise_as_http(
                exc,
                action="Get graph",
                mappings=((KeyError, 404),),
            ) from exc

    @router.get("/api/v1/sessions/{session_id}/snapshots/{round_idx}")
    async def get_round_snapshot(session_id: str, round_idx: int) -> Dict[str, Any]:
        """Return graph data for a specific historical round."""
        try:
            session = manager.get_session(session_id)
            return graph_response(session, round_idx)

        except KeyError as exc:
            raise raise_as_http(
                exc,
                action="Get round snapshot",
                mappings=((KeyError, 404),),
            ) from exc

    @router.get("/api/v1/sessions/{session_id}/snapshots")
    async def get_snapshots_index(session_id: str) -> Dict[str, Any]:
        """List metadata for stored round snapshots in a session."""
        try:
            items = manager.get_snapshot_index(session_id)
            return {"items": items, "total": len(items)}

        except KeyError as exc:
            raise raise_as_http(
                exc,
                action="Get snapshots index",
                mappings=((KeyError, 404),),
            ) from exc

    @router.get("/api/v1/sessions/{session_id}/diff")
    async def get_graph_diff(
        session_id: str,
        from_round: int = Query(0, ge=0),
        to_round: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        """Return graph-level diff between two rounds."""
        try:
            session = manager.get_session(session_id)
            return graph_diff_response(session, from_round, to_round)

        except KeyError as exc:
            raise raise_as_http(
                exc,
                action="Get graph diff",
                mappings=((KeyError, 404),),
            ) from exc

    @router.get("/api/v1/sessions/{session_id}/export/graph.gexf")
    async def export_graph_gexf(
        session_id: str,
        round_idx: Optional[int] = Query(default=None, ge=0),
    ) -> Response:
        """Export session graph as a downloadable GEXF file."""
        try:
            payload = manager.export_graph_gexf(
                session_id=session_id,
                round_idx=round_idx,
            )

            filename = (
                f"{session_id}-round-{round_idx}.gexf"
                if round_idx is not None
                else f"{session_id}.gexf"
            )

            return Response(
                content=payload,
                media_type="application/gexf+xml",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        except KeyError as exc:
            raise raise_as_http(
                exc,
                action="Export graph gexf",
                mappings=((KeyError, 404),),
            ) from exc

    return router
