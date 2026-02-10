"""FastAPI server exposing a lightweight DebateEngine API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .serializers import (
    graph_diff_response,
    graph_response,
    memory_response,
    snapshot_response,
)
from .session_manager import SessionManager


class CreateSessionRequest(BaseModel):
    """Request body for creating a new debate session."""

    case_id: Optional[str] = Field(default=None)
    case_uid: Optional[str] = Field(default=None)
    case_data: Optional[Dict[str, Any]] = Field(default=None)
    max_rounds: Optional[int] = Field(default=None, ge=1)


class SetupSessionRequest(BaseModel):
    """Request body for setup endpoint."""

    case_data: Optional[Dict[str, Any]] = Field(default=None)


def _default_case_file() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "sampling"
        / "cleaned_samples.jsonl"
    )


def _load_cases(limit: int = 200) -> List[Dict[str, Any]]:
    path = _default_case_file()

    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()

            if not text:
                continue

            try:
                payload = json.loads(text)

            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict):
                rows.append(payload)

            if len(rows) >= limit:
                break

    return rows


def _case_list_item(row: Dict[str, Any]) -> Dict[str, Any]:
    cause = row.get("cause", [])

    if isinstance(cause, list):
        cause_summary = " / ".join([str(item) for item in cause[:2]])

    else:
        cause_summary = str(cause)

    return {
        "uid": row.get("uid", ""),
        "title": row.get("title", ""),
        "cause_summary": cause_summary,
    }


def create_app(
    engine_factory: Optional[Callable[[], Any]] = None,
    case_rows: Optional[List[Dict[str, Any]]] = None,
) -> FastAPI:
    """Create a configured FastAPI app instance."""
    app = FastAPI(title="Legal Court API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    manager = SessionManager(engine_factory=engine_factory)
    cases = case_rows if case_rows is not None else _load_cases()
    case_index = {str(row.get("uid", "")): row for row in cases if row.get("uid")}

    @app.get("/api/v1/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/cases")
    async def list_cases(
        limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)
    ) -> Dict[str, Any]:
        items = cases[offset : offset + limit]
        return {"items": [_case_list_item(row) for row in items], "total": len(cases)}

    @app.get("/api/v1/cases/{case_uid}")
    async def get_case(case_uid: str) -> Dict[str, Any]:
        case_data = case_index.get(case_uid)

        if case_data is None:
            raise HTTPException(status_code=404, detail="Case not found")

        return {"case": case_data}

    @app.post("/api/v1/sessions")
    async def create_session(body: CreateSessionRequest) -> Dict[str, Any]:
        case_data = body.case_data
        case_lookup_uid = body.case_uid or body.case_id

        if case_data is None and case_lookup_uid:
            case_data = case_index.get(case_lookup_uid)

            if case_data is None:
                raise HTTPException(status_code=404, detail="Case UID not found")

        try:
            session = await manager.create_session(
                case_data=case_data,
                max_rounds=body.max_rounds,
                auto_setup=True,
            )

            return snapshot_response(session)

        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Create session failed: {exc}"
            ) from exc

    @app.get("/api/v1/sessions")
    async def list_sessions() -> Dict[str, Any]:
        return {
            "sessions": [snapshot_response(item) for item in manager.list_sessions()]
        }

    @app.get("/api/v1/sessions/{session_id}")
    async def get_session(session_id: str) -> Dict[str, Any]:
        try:
            session = manager.get_session(session_id)
            return snapshot_response(session)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/snapshot")
    async def get_session_snapshot(session_id: str) -> Dict[str, Any]:
        try:
            session = manager.get_session(session_id)
            return snapshot_response(session)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/setup")
    async def setup_session(
        session_id: str, body: Optional[SetupSessionRequest] = None
    ) -> Dict[str, Any]:
        try:
            payload = body.case_data if body else None
            session = await manager.setup_session(session_id, case_data=payload)
            return snapshot_response(session)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Setup failed: {exc}") from exc

    @app.post("/api/v1/sessions/{session_id}/step")
    async def step_session(session_id: str) -> Dict[str, Any]:
        try:
            session = await manager.step_session(session_id)
            return snapshot_response(session)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Step failed: {exc}") from exc

    @app.post("/api/v1/sessions/{session_id}/adjudicate")
    async def adjudicate_session(session_id: str) -> Dict[str, Any]:
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
        try:
            session = manager.get_session(session_id)
            return graph_response(session)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/snapshots/{round_idx}")
    async def get_round_snapshot(session_id: str, round_idx: int) -> Dict[str, Any]:
        try:
            session = manager.get_session(session_id)
            return graph_response(session, round_idx)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/diff")
    async def get_graph_diff(
        session_id: str,
        from_round: int = Query(0, ge=0),
        to_round: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            session = manager.get_session(session_id)
            return graph_diff_response(session, from_round, to_round)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/memory")
    async def get_memory(session_id: str) -> Dict[str, Any]:
        try:
            session = manager.get_session(session_id)
            return memory_response(session)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/events")
    async def get_events(
        session_id: str,
        limit: int = Query(100, ge=1, le=5000),
        from_seq: Optional[int] = Query(default=None, ge=1),
        to_seq: Optional[int] = Query(default=None, ge=1),
    ) -> Dict[str, Any]:
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

    @app.get("/api/v1/sessions/{session_id}/events/history")
    async def get_events_history(
        session_id: str,
        limit: int = Query(100, ge=1, le=5000),
        from_seq: Optional[int] = Query(default=None, ge=1),
        to_seq: Optional[int] = Query(default=None, ge=1),
    ) -> Dict[str, Any]:
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

    @app.get("/api/v1/sessions/{session_id}/turns/artifacts")
    async def get_turn_artifacts(
        session_id: str,
        limit: int = Query(50, ge=1, le=5000),
    ) -> Dict[str, Any]:
        try:
            items = manager.get_turn_artifacts(
                session_id=session_id,
                turn_uid=None,
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
        try:
            items = manager.get_turn_artifacts(
                session_id=session_id,
                turn_uid=turn_uid,
                limit=limit,
            )

            return {"items": items, "total": len(items)}

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/v1/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> Response:
        try:
            await manager.delete_session(session_id)
            return Response(status_code=204)

        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app()
