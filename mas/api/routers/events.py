"""Event stream, artifacts, and timeline routes."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, WebSocket
from fastapi.websockets import WebSocketDisconnect
from metagpt.logs import logger

from ..http_errors import raise_as_http
from ..session_manager import SessionManager


def build_events_router(manager: SessionManager) -> APIRouter:
    """Build event and artifact router with injected manager dependency."""
    router = APIRouter()

    @router.get("/api/v1/sessions/{session_id}/events")
    async def get_events(
        session_id: str,
        limit: int = Query(100, ge=1, le=5000),
        from_seq: Optional[int] = Query(default=None, ge=1),
        to_seq: Optional[int] = Query(default=None, ge=1),
    ) -> Dict[str, Any]:
        """Fetch a filtered slice of event history."""
        try:
            events = manager.get_event_history(
                session_id=session_id,
                limit=limit,
                from_seq=from_seq,
                to_seq=to_seq,
            )

            return {"events": events}

        except KeyError as exc:
            raise raise_as_http(
                exc,
                action="Get events",
                mappings=((KeyError, 404),),
            ) from exc

    @router.websocket("/api/v1/sessions/{session_id}/events")
    async def stream_events(session_id: str, websocket: WebSocket) -> None:
        """Stream live session events over WebSocket."""
        await websocket.accept()

        try:
            manager.get_session(session_id)

        except KeyError:
            await websocket.send_json(
                {"event": "session_error", "detail": "Session not found"}
            )

            await websocket.close(code=4404)
            return

        queue = manager.register_event_subscriber(session_id=session_id)
        from_seq: Optional[int] = None
        from_seq_raw = websocket.query_params.get("from_seq")

        if from_seq_raw:
            try:
                parsed = int(from_seq_raw)
                from_seq = parsed if parsed > 0 else None

            except ValueError:
                from_seq = None

        try:
            if from_seq is not None:
                history = manager.get_event_history(
                    session_id=session_id,
                    limit=5000,
                    from_seq=from_seq,
                    to_seq=None,
                )

                for row in history:
                    await websocket.send_json(row)

            while True:
                event_payload = await queue.get()
                await websocket.send_json(event_payload)

        except WebSocketDisconnect:
            return

        finally:
            try:
                manager.unregister_event_subscriber(session_id=session_id, queue=queue)

            except KeyError:
                logger.debug(
                    "[events] Skip unregister for missing session_id={}", session_id
                )

    @router.get("/api/v1/sessions/{session_id}/turns/artifacts")
    async def get_turn_artifacts(
        session_id: str,
        limit: int = Query(50, ge=1, le=5000),
    ) -> Dict[str, Any]:
        """Return latest turn artifacts for a session."""
        try:
            items = manager.get_turn_artifacts(
                session_id=session_id,
                turn_uid=None,
                limit=limit,
            )

            return {"items": items, "total": len(items)}

        except KeyError as exc:
            raise raise_as_http(
                exc,
                action="Get turn artifacts",
                mappings=((KeyError, 404),),
            ) from exc

    @router.get("/api/v1/sessions/{session_id}/teamflow/stream")
    async def get_teamflow_stream(
        session_id: str,
        limit: int = Query(80, ge=1, le=5000),
    ) -> Dict[str, Any]:
        """Return compact teamflow messages for frontend timeline display."""
        try:
            items = manager.get_teamflow_stream(
                session_id=session_id,
                limit=limit,
            )

            return {"items": items, "total": len(items)}

        except KeyError as exc:
            raise raise_as_http(
                exc,
                action="Get teamflow stream",
                mappings=((KeyError, 404),),
            ) from exc

    @router.get("/api/v1/sessions/{session_id}/turns/{turn_uid}/artifacts")
    async def get_single_turn_artifact(
        session_id: str,
        turn_uid: str,
        limit: int = Query(50, ge=1, le=5000),
    ) -> Dict[str, Any]:
        """Return artifacts for one specific turn UID."""
        try:
            items = manager.get_turn_artifacts(
                session_id=session_id,
                turn_uid=turn_uid,
                limit=limit,
            )

            return {"items": items, "total": len(items)}

        except KeyError as exc:
            raise raise_as_http(
                exc,
                action="Get turn artifact",
                mappings=((KeyError, 404),),
            ) from exc

    return router
