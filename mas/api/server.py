"""FastAPI server exposing a lightweight DebateEngine API."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from .routers import (
    build_events_router,
    build_memory_router,
    build_sessions_router,
    build_snapshots_router,
)
from .session_manager import SessionManager


def create_app(
    engine_factory: Optional[Callable[[], Any]] = None,
    case_rows: Optional[List[Dict[str, Any]]] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
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
        """Return basic service metadata."""
        return {"service": "Legal Court API", "version": "0.1.0"}

    @app.get("/favicon.ico", status_code=204)
    async def favicon() -> Response:
        """Return an empty favicon response."""
        return Response(status_code=204)

    @app.get("/api/v1/health")
    async def health() -> Dict[str, str]:
        """Return a basic liveness probe payload."""
        return {"status": "ok"}

    app.include_router(
        build_sessions_router(manager=manager, default_case_data=default_case_data)
    )

    app.include_router(build_snapshots_router(manager=manager))
    app.include_router(build_memory_router(manager=manager))
    app.include_router(build_events_router(manager=manager))
    return app


app = create_app()
