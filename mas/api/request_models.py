"""Pydantic request models used by API route modules."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


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

