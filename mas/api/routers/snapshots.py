"""Frontend snapshot persistence routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Query

from ..http_errors import raise_as_http
from ..request_models import ImportFrontendSnapshotRequest, SaveFrontendSnapshotRequest
from ..session_manager import SessionManager


def build_snapshots_router(manager: SessionManager) -> APIRouter:
    """Build frontend-snapshot API router.

    Args:
        manager: Session manager dependency used by route handlers.

    Returns:
        Configured `APIRouter` instance with snapshot persistence endpoints.

    Raises:
        Assumption/Unverified: Exceptions are raised and converted inside nested
            route handlers rather than by this builder function.
    """
    router = APIRouter()

    @router.post("/api/v1/frontend-snapshots")
    async def save_frontend_snapshot(
        body: SaveFrontendSnapshotRequest,
    ) -> Dict[str, Any]:
        """Persist frontend state alongside a backend snapshot."""
        try:
            return manager.save_frontend_snapshot(
                session_id=body.session_id,
                label=body.label or "",
                frontend_state=body.frontend_state,
            )

        except (KeyError, ValueError) as exc:
            raise raise_as_http(
                exc,
                action="Save frontend snapshot",
                mappings=((KeyError, 404), (ValueError, 400)),
            ) from exc

    @router.post("/api/v1/frontend-snapshots/import")
    async def import_frontend_snapshot(
        body: ImportFrontendSnapshotRequest,
    ) -> Dict[str, Any]:
        """Import a replay bundle as a frontend snapshot record."""
        try:
            return manager.import_frontend_snapshot(
                bundle=body.bundle,
                label=body.label or "",
                frontend_state=body.frontend_state,
            )

        except ValueError as exc:
            raise raise_as_http(
                exc,
                action="Import frontend snapshot",
                mappings=((ValueError, 400),),
            ) from exc

    @router.get("/api/v1/frontend-snapshots")
    async def list_frontend_snapshots(
        limit: int = Query(20, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        """List stored frontend snapshots with pagination."""
        return manager.list_frontend_snapshots(limit=limit, offset=offset)

    @router.post("/api/v1/frontend-snapshots/{snapshot_id}/load")
    async def load_frontend_snapshot(snapshot_id: str) -> Dict[str, Any]:
        """Restore a saved frontend snapshot into a new runtime session."""
        try:
            return await manager.load_frontend_snapshot(snapshot_id)

        except (FileNotFoundError, ValueError) as exc:
            raise raise_as_http(
                exc,
                action="Load frontend snapshot",
                mappings=((FileNotFoundError, 404), (ValueError, 400)),
            ) from exc

    return router
