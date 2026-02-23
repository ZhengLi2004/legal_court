"""Session lifecycle service extracted from API manager facade."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mas.session.event_stream import infer_event_source
from mas.session.session_lifecycle import (
    derive_status,
    load_case_from_jsonl,
    reset_memory_storage_dir,
)
from mas.session.session_status import SessionStatus, ensure_allowed_transition


class SessionService:
    """Encapsulate session lifecycle orchestration and runtime state transitions."""

    def __init__(
        self,
        *,
        engine_factory: Callable[[], Any],
        default_case_path: Path,
        sessions: Dict[str, Any],
        session_factory: Callable[[str, Any], Any],
        record_event: Callable[[str, str, str, Optional[Dict[str, Any]]], None],
        utc_now_iso: Callable[[], str],
    ):
        """Create session service dependencies."""
        self._engine_factory = engine_factory
        self._default_case_path = default_case_path
        self._sessions = sessions
        self._session_factory = session_factory
        self._record_event = record_event
        self._utc_now_iso = utc_now_iso

    def list_sessions(self) -> List[Any]:
        """Return all currently active in-memory sessions."""
        return list(self._sessions.values())

    def get_session(self, session_id: str) -> Any:
        """Fetch a session by ID."""
        try:
            return self._sessions[session_id]

        except KeyError as exc:
            raise KeyError(f"Session not found: {session_id}") from exc

    def _validate_status_type(self, session: Any) -> SessionStatus:
        """Return current session status and enforce enum-based state machine."""
        status = getattr(session, "status", None)

        if not isinstance(status, SessionStatus):
            raise TypeError(f"Session status must be SessionStatus, got: {status!r}")

        return status

    def _set_status(self, session: Any, target: SessionStatus) -> None:
        """Apply a validated status transition and refresh timestamp."""
        current = self._validate_status_type(session)
        session.status = ensure_allowed_transition(current, target)
        session.updated_at = self._utc_now_iso()

    def _set_status_unchecked(self, session: Any, target: SessionStatus) -> None:
        """Set status directly for terminal/derived statuses that may skip edges."""
        session.status = target
        session.updated_at = self._utc_now_iso()

    def _emit_warning(self, session_id: str, stage: str, kind: str, message: str) -> None:
        """Emit one warning event envelope for simulated failure modes."""
        self._record_event(
            session_id,
            event="session_warning",
            source="api",
            data={
                "stage": stage,
                "kind": kind,
                "message": message,
            },
        )

    def _set_error_state(self, session_id: str, session: Any, stage: str, exc: Exception) -> None:
        """Apply error status and publish one session_error event."""
        session.last_error = str(exc)
        self._set_status_unchecked(session, SessionStatus.ERROR)

        self._record_event(
            session_id,
            event="session_error",
            source="api",
            data={"stage": stage, "message": session.last_error},
        )

    def _mark_success_status(self, session: Any) -> None:
        """Sync session status with engine-derived status after successful step."""
        session.last_error = ""
        derived = derive_status(session.engine)
        self._set_status_unchecked(session, derived)

    def _resolve_case_payload(
        self,
        *,
        case_data: Optional[Dict[str, Any]],
        case_data_path: Optional[Path],
    ) -> Dict[str, Any]:
        """Resolve case payload from in-memory data or JSONL file."""
        if case_data is not None:
            return case_data

        target_path = case_data_path or self._default_case_path
        return load_case_from_jsonl(Path(target_path))

    async def create_session(
        self,
        case_data: Optional[Dict[str, Any]] = None,
        auto_setup: bool = True,
    ) -> Any:
        """Create a new session and optionally execute setup immediately."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        engine = self._engine_factory()
        session = self._session_factory(session_id, engine)

        engine.set_state_callback(
            lambda event, data, sid=session_id: self._record_event(
                sid,
                event=event,
                source=infer_event_source(event),
                data=data,
            )
        )

        self._sessions[session_id] = session

        if auto_setup:
            await self.setup_session(session_id, case_data=case_data)

        return session

    async def setup_session(
        self,
        session_id: str,
        case_data: Optional[Dict[str, Any]] = None,
        case_data_path: Optional[Path] = None,
    ) -> Any:
        """Initialize engine state for an existing session."""
        session = self.get_session(session_id)

        async with session.lock:
            current = self._validate_status_type(session)

            if (
                current
                in {
                    SessionStatus.SETUP_DONE,
                    SessionStatus.DEBATING,
                    SessionStatus.READY_FOR_ADJUDICATION,
                    SessionStatus.FINISHED,
                }
                and getattr(session.engine, "graph", None) is not None
            ):
                return session

            payload = self._resolve_case_payload(
                case_data=case_data,
                case_data_path=case_data_path,
            )

            self._set_status(session, SessionStatus.SETTING_UP)

            try:
                await session.engine.setup(case_data=payload)
                self._mark_success_status(session)
                return session

            except Exception as exc:
                self._set_error_state(session_id, session, stage="setup", exc=exc)
                raise

    def _ensure_step_allowed(self, session_id: str, session: Any) -> None:
        """Validate one step request against adjudication/finished guards."""
        if getattr(session.engine, "is_ready_for_adjudication", False):
            self._set_status_unchecked(session, derive_status(session.engine))

            self._record_event(
                session_id,
                event="step_blocked",
                source="api",
                data={
                    "stage": "step",
                    "reason": "ready_for_adjudication",
                    "message": "Session already converged and is ready for adjudication.",
                },
            )

            raise ValueError(
                "Session already converged; step is disabled. Please adjudicate."
            )

        if getattr(session.engine, "is_finished", False):
            self._set_status_unchecked(session, SessionStatus.FINISHED)

            self._record_event(
                session_id,
                event="step_blocked",
                source="api",
                data={
                    "stage": "step",
                    "reason": "finished",
                    "message": "Session already finished.",
                },
            )

            raise ValueError("Session already finished; step is disabled.")

    async def step_session(self, session_id: str) -> Any:
        """Advance the debate by one turn for the given session."""
        session = self.get_session(session_id)

        if self._validate_status_type(session) == SessionStatus.CREATED:
            await self.setup_session(session_id)

        async with session.lock:
            self._ensure_step_allowed(session_id, session)

            try:
                if session.failure_simulation.get("es_unavailable", False):
                    self._emit_warning(
                        session_id,
                        stage="step",
                        kind="es_unavailable",
                        message="Simulated ES unavailable; degrade continue",
                    )

                if session.failure_simulation.get("llm_timeout", False):
                    self._emit_warning(
                        session_id,
                        stage="step",
                        kind="llm_timeout",
                        message="Simulated LLM timeout; degrade continue",
                    )

                await session.engine.step()
                self._mark_success_status(session)
                return session

            except Exception as exc:
                self._set_error_state(session_id, session, stage="step", exc=exc)
                raise

    async def adjudicate_session(self, session_id: str) -> Any:
        """Run final adjudication for a session."""
        session = self.get_session(session_id)

        if self._validate_status_type(session) == SessionStatus.CREATED:
            await self.setup_session(session_id)

        async with session.lock:
            if getattr(session.engine, "is_finished", False):
                self._set_status_unchecked(session, SessionStatus.FINISHED)
                return session

            try:
                if session.failure_simulation.get("llm_timeout", False):
                    self._emit_warning(
                        session_id,
                        stage="adjudicate",
                        kind="llm_timeout",
                        message="Simulated LLM timeout flag enabled",
                    )

                await session.engine.adjudicate()
                self._mark_success_status(session)
                return session

            except Exception as exc:
                self._set_error_state(session_id, session, stage="adjudicate", exc=exc)
                raise

    async def reset_memory_storage(self) -> str:
        """Delete all persisted long-term-memory files on disk."""
        if len(self._sessions) > 0:
            raise ValueError(
                "Active sessions exist. Close all sessions before clearing memory storage."
            )

        return reset_memory_storage_dir(os.getenv("MAS_STORAGE_DIR", ""))
