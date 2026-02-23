"""Memory and case-graph routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from ..http_errors import raise_as_http
from ..serializers import memory_case_graph_response, memory_response
from ..session_manager import SessionManager


def build_memory_router(manager: SessionManager) -> APIRouter:
    """Build memory-related API router.

    Args:
        manager: Session manager dependency used by route handlers.

    Returns:
        Configured `APIRouter` instance with memory endpoints.

    Raises:
        Assumption/Unverified: Exceptions raised during request handling are
            converted inside nested handlers rather than by this builder itself.
    """
    router = APIRouter()

    @router.get("/api/v1/sessions/{session_id}/memory")
    async def get_memory(session_id: str) -> Dict[str, Any]:
        """Return memory/insight data for frontend side panels."""
        try:
            session = manager.get_session(session_id)
            return memory_response(session)

        except KeyError as exc:
            raise raise_as_http(
                exc,
                action="Get memory",
                mappings=((KeyError, 404),),
            ) from exc

    @router.post("/api/v1/memory/reset-storage")
    async def reset_memory_storage() -> Dict[str, Any]:
        """Delete all persisted long-term-memory files from disk."""
        try:
            storage_dir = await manager.reset_memory_storage()
            return {"status": "ok", "storage_root_dir": storage_dir}

        except ValueError as exc:
            raise raise_as_http(
                exc,
                action="Reset memory storage",
                mappings=((ValueError, 409),),
            ) from exc

    @router.get("/api/v1/sessions/{session_id}/memory/cases/{case_id}/graph")
    async def get_memory_case_graph(session_id: str, case_id: str) -> Dict[str, Any]:
        """Return one historical-case argument graph for memory-page inspection."""
        try:
            session = manager.get_session(session_id)
            return memory_case_graph_response(session, case_id)

        except KeyError as exc:
            raise raise_as_http(
                exc,
                action="Get memory case graph",
                mappings=((KeyError, 404),),
            ) from exc

    return router
