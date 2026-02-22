"""FastAPI server exposing a lightweight DebateEngine API."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from .serializers import (
    graph_diff_response,
    graph_response,
    memory_case_graph_response,
    memory_response,
    snapshot_response,
)
from .session_manager import SessionManager


class CreateSessionRequest(BaseModel):
    """Request body for creating a new debate session."""

    model_config = ConfigDict(extra="forbid")
    case_data: Optional[Dict[str, Any]] = Field(default=None)


class SaveFrontendSnapshotRequest(BaseModel):
    """Request body for saving frontend snapshot payloads."""

    model_config = ConfigDict(extra="forbid")
    session_id: str
    label: Optional[str] = Field(default="")
    frontend_state: Optional[Dict[str, Any]] = Field(default=None)


class ImportFrontendSnapshotRequest(BaseModel):
    """Request body for importing a frontend snapshot payload."""

    model_config = ConfigDict(extra="forbid")
    bundle: Dict[str, Any]
    label: Optional[str] = Field(default="")
    frontend_state: Optional[Dict[str, Any]] = Field(default=None)


def create_app(
    engine_factory: Optional[Callable[[], Any]] = None,
    case_rows: Optional[List[Dict[str, Any]]] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        engine_factory: Optional factory for dependency injection in tests.
        case_rows: Optional in-memory fallback case rows for session bootstrap.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(title="Legal Court API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    manager = SessionManager(engine_factory=engine_factory)
    default_case_data = case_rows[0] if case_rows else None

    @app.get("/")
    async def root() -> Dict[str, str]:
        """Return basic service metadata.

        Returns:
            Service name and version information.
        """
        return {"service": "Legal Court API", "version": "0.1.0"}

    @app.get("/favicon.ico", status_code=204)
    async def favicon() -> Response:
        """Return an empty favicon response.

        Returns:
            HTTP 204 response.
        """
        return Response(status_code=204)

    @app.get("/api/v1/health")
    async def health() -> Dict[str, str]:
        """Return a basic liveness probe payload.

        Returns:
            Health status dictionary.
        """
        return {"status": "ok"}

    @app.post("/api/v1/sessions")
    async def create_session(body: CreateSessionRequest) -> Dict[str, Any]:
        """Create a debate session and auto-run setup.

        Args:
            body: Session creation request payload.

        Returns:
            Serialized session snapshot.
        """
        case_data = body.case_data if body.case_data is not None else default_case_data

        try:
            session = await manager.create_session(
                case_data=case_data,
                auto_setup=True,
            )

            return snapshot_response(session)

        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Create session failed: {exc}"
            ) from exc

    @app.post("/api/v1/frontend-snapshots")
    async def save_frontend_snapshot(
        body: SaveFrontendSnapshotRequest,
    ) -> Dict[str, Any]:
        """Persist frontend state alongside a backend snapshot.

        Args:
            body: Frontend snapshot save request payload.

        Returns:
            Metadata of the saved snapshot record.
        """
        try:
            return manager.save_frontend_snapshot(
                session_id=body.session_id,
                label=body.label or "",
                frontend_state=body.frontend_state,
            )

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Save frontend snapshot failed: {exc}",
            ) from exc

    @app.post("/api/v1/frontend-snapshots/import")
    async def import_frontend_snapshot(
        body: ImportFrontendSnapshotRequest,
    ) -> Dict[str, Any]:
        """Import a replay bundle as a frontend snapshot record.

        Args:
            body: Import payload containing bundle and optional metadata.

        Returns:
            Metadata of the imported snapshot record.
        """
        try:
            return manager.import_frontend_snapshot(
                bundle=body.bundle,
                label=body.label or "",
                frontend_state=body.frontend_state,
            )

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Import frontend snapshot failed: {exc}",
            ) from exc

    @app.get("/api/v1/frontend-snapshots")
    async def list_frontend_snapshots(
        limit: int = Query(20, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        """List stored frontend snapshots with pagination.

        Args:
            limit: Page size.
            offset: Starting offset in snapshot history.

        Returns:
            Paginated snapshot metadata payload.
        """
        return manager.list_frontend_snapshots(limit=limit, offset=offset)

    @app.post("/api/v1/frontend-snapshots/{snapshot_id}/load")
    async def load_frontend_snapshot(snapshot_id: str) -> Dict[str, Any]:
        """Restore a saved frontend snapshot into a new runtime session.

        Args:
            snapshot_id: Stored snapshot identifier.

        Returns:
            Restored snapshot payload for frontend bootstrap.
        """
        try:
            return await manager.load_frontend_snapshot(snapshot_id)

        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Load frontend snapshot failed: {exc}",
            ) from exc

    @app.get("/api/v1/sessions")
    async def list_sessions() -> Dict[str, Any]:
        """List active in-memory sessions.

        Returns:
            Collection of session snapshot payloads.
        """
        return {
            "sessions": [snapshot_response(item) for item in manager.list_sessions()]
        }

    @app.get("/api/v1/sessions/{session_id}/snapshot")
    async def get_session_snapshot(session_id: str) -> Dict[str, Any]:
        """Return the latest serializable snapshot for one session.

        Args:
            session_id: Session identifier.

        Returns:
            Session snapshot payload.
        """
        try:
            session = manager.get_session(session_id)
            return snapshot_response(session)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/step")
    async def step_session(session_id: str) -> Dict[str, Any]:
        """Advance a session by one debate turn.

        Args:
            session_id: Session identifier.

        Returns:
            Updated session snapshot payload.
        """
        try:
            session = await manager.step_session(session_id)
            return snapshot_response(session)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Step failed: {exc}") from exc

    @app.post("/api/v1/sessions/{session_id}/adjudicate")
    async def adjudicate_session(session_id: str) -> Dict[str, Any]:
        """Run final adjudication for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Session snapshot after adjudication.
        """
        try:
            session = await manager.adjudicate_session(session_id)
            return snapshot_response(session)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Adjudication failed: {exc}"
            ) from exc

    @app.get("/api/v1/sessions/{session_id}/graph")
    async def get_graph(session_id: str) -> Dict[str, Any]:
        """Return graph data for the latest session state.

        Args:
            session_id: Session identifier.

        Returns:
            Graph payload with nodes, edges, and focus nodes.
        """
        try:
            session = manager.get_session(session_id)
            return graph_response(session)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/snapshots/{round_idx}")
    async def get_round_snapshot(session_id: str, round_idx: int) -> Dict[str, Any]:
        """Return graph data for a specific historical round.

        Args:
            session_id: Session identifier.
            round_idx: Zero-based round index.

        Returns:
            Graph payload for the requested round.
        """
        try:
            session = manager.get_session(session_id)
            return graph_response(session, round_idx)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/snapshots")
    async def get_snapshots_index(session_id: str) -> Dict[str, Any]:
        """List metadata for stored round snapshots in a session.

        Args:
            session_id: Session identifier.

        Returns:
            Snapshot index payload with total count.
        """
        try:
            items = manager.get_snapshot_index(session_id)
            return {"items": items, "total": len(items)}

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/diff")
    async def get_graph_diff(
        session_id: str,
        from_round: int = Query(0, ge=0),
        to_round: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        """Return graph-level diff between two rounds.

        Args:
            session_id: Session identifier.
            from_round: Baseline round index.
            to_round: Target round index.

        Returns:
            Node and edge delta payload between the two rounds.
        """
        try:
            session = manager.get_session(session_id)
            return graph_diff_response(session, from_round, to_round)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/memory")
    async def get_memory(session_id: str) -> Dict[str, Any]:
        """Return memory/insight data for frontend side panels.

        Args:
            session_id: Session identifier.

        Returns:
            Compact memory payload extracted from engine state.
        """
        try:
            session = manager.get_session(session_id)
            return memory_response(session)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/memory/reset-storage")
    async def reset_memory_storage() -> Dict[str, Any]:
        """Delete all persisted long-term-memory files from disk.

        The operation is only allowed when no active runtime sessions exist.
        """
        try:
            storage_dir = await manager.reset_memory_storage()
            return {"status": "ok", "storage_root_dir": storage_dir}

        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Reset storage memory failed: {exc}",
            ) from exc

    @app.get("/api/v1/sessions/{session_id}/memory/cases/{case_id}/graph")
    async def get_memory_case_graph(session_id: str, case_id: str) -> Dict[str, Any]:
        """Return one historical-case argument graph for memory-page inspection.

        Args:
            session_id: Session identifier.
            case_id: Historical case identifier.

        Returns:
            Graph payload for the selected case.
        """
        try:
            session = manager.get_session(session_id)
            return memory_case_graph_response(session, case_id)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/events")
    async def get_events(
        session_id: str,
        limit: int = Query(100, ge=1, le=5000),
        from_seq: Optional[int] = Query(default=None, ge=1),
        to_seq: Optional[int] = Query(default=None, ge=1),
    ) -> Dict[str, Any]:
        """Fetch a filtered slice of event history.

        Args:
            session_id: Session identifier.
            limit: Maximum number of events to return.
            from_seq: Optional inclusive lower sequence bound.
            to_seq: Optional inclusive upper sequence bound.

        Returns:
            Event list payload.
        """
        try:
            events = manager.get_event_history(
                session_id=session_id,
                limit=limit,
                from_seq=from_seq,
                to_seq=to_seq,
            )

            return {"events": events}

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.websocket("/api/v1/sessions/{session_id}/events")
    async def stream_events(session_id: str, websocket: WebSocket) -> None:
        """Stream live session events over WebSocket.

        Args:
            session_id: Session identifier.
            websocket: Accepted client WebSocket connection.
        """
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
                pass

    @app.get("/api/v1/sessions/{session_id}/turns/artifacts")
    async def get_turn_artifacts(
        session_id: str,
        limit: int = Query(50, ge=1, le=5000),
    ) -> Dict[str, Any]:
        """Return latest turn artifacts for a session.

        Args:
            session_id: Session identifier.
            limit: Maximum number of artifacts to return.

        Returns:
            Artifact list payload with total count.
        """
        try:
            items = manager.get_turn_artifacts(
                session_id=session_id,
                turn_uid=None,
                limit=limit,
            )
            return {"items": items, "total": len(items)}

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/teamflow/stream")
    async def get_teamflow_stream(
        session_id: str,
        limit: int = Query(80, ge=1, le=5000),
    ) -> Dict[str, Any]:
        """Return compact teamflow messages for frontend timeline display.

        Args:
            session_id: Session identifier.
            limit: Maximum number of timeline items to return.

        Returns:
            Teamflow message list payload with total count.
        """
        try:
            items = manager.get_teamflow_stream(
                session_id=session_id,
                limit=limit,
            )

            return {"items": items, "total": len(items)}

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/turns/{turn_uid}/artifacts")
    async def get_single_turn_artifact(
        session_id: str,
        turn_uid: str,
        limit: int = Query(50, ge=1, le=5000),
    ) -> Dict[str, Any]:
        """Return artifacts for one specific turn UID.

        Args:
            session_id: Session identifier.
            turn_uid: Turn UID used to filter artifacts.
            limit: Maximum number of artifacts to return.

        Returns:
            Filtered artifact list payload with total count.
        """
        try:
            items = manager.get_turn_artifacts(
                session_id=session_id,
                turn_uid=turn_uid,
                limit=limit,
            )

            return {"items": items, "total": len(items)}

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/export/graph.gexf")
    async def export_graph_gexf(
        session_id: str,
        round_idx: Optional[int] = Query(default=None, ge=0),
    ) -> Response:
        """Export session graph as a downloadable GEXF file.

        Args:
            session_id: Session identifier.
            round_idx: Optional round index; latest graph when omitted.

        Returns:
            HTTP response carrying GEXF bytes as attachment.
        """
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
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app()
