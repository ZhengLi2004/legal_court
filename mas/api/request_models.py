"""Pydantic request models used by API route modules."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class CreateSessionRequest(BaseModel):
    """Request body for creating a new debate session.

    Attributes:
        case_data: Optional serialized case payload used to bootstrap the
            initial session state.
    """

    model_config = ConfigDict(extra="forbid")
    case_data: Optional[Dict[str, Any]] = Field(default=None)


class SaveFrontendSnapshotRequest(BaseModel):
    """Request body for saving frontend snapshot payloads.

    Attributes:
        session_id: Backend session identifier associated with the snapshot.
        label: Optional human-readable snapshot label.
        frontend_state: Serialized frontend state blob to persist.
    """

    model_config = ConfigDict(extra="forbid")
    session_id: str
    label: Optional[str] = Field(default="")
    frontend_state: Optional[Dict[str, Any]] = Field(default=None)


class ImportFrontendSnapshotRequest(BaseModel):
    """Request body for importing a frontend snapshot payload.

    Attributes:
        bundle: Serialized snapshot bundle exported by the frontend.
        label: Optional human-readable snapshot label.
        frontend_state: Optional frontend state blob to import alongside the
            bundle.
    """

    model_config = ConfigDict(extra="forbid")
    bundle: Dict[str, Any]
    label: Optional[str] = Field(default="")
    frontend_state: Optional[Dict[str, Any]] = Field(default=None)
