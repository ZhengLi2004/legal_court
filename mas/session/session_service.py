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
            if (
                session.status
                in {
                    "SETUP_DONE",
                    "DEBATING",
                    "READY_FOR_ADJUDICATION",
                    "FINISHED",
                }
                and getattr(session.engine, "graph", None) is not None
            ):
                return session

            payload = case_data

            if payload is None:
                target_path = case_data_path or self._default_case_path
                payload = load_case_from_jsonl(Path(target_path))

            session.status = "SETTING_UP"
            session.updated_at = self._utc_now_iso()

            try:
                await session.engine.setup(case_data=payload)
                session.last_error = ""
                session.status = derive_status(session.engine)
                session.updated_at = self._utc_now_iso()
                return session

            except Exception as exc:
                session.last_error = str(exc)
                session.status = "ERROR"
                session.updated_at = self._utc_now_iso()

                self._record_event(
                    session_id,
                    event="session_error",
                    source="api",
                    data={"stage": "setup", "message": session.last_error},
                )

                raise

    async def step_session(self, session_id: str) -> Any:
        """Advance the debate by one turn for the given session."""
        session = self.get_session(session_id)

        if session.status == "CREATED":
            await self.setup_session(session_id)

        async with session.lock:
            if getattr(session.engine, "is_ready_for_adjudication", False):
                session.status = derive_status(session.engine)
                session.updated_at = self._utc_now_iso()

                self._record_event(
                    session_id,
                    event="step_blocked",
                    source="api",
                    data={
                        "stage": "step",
                        "reason": "ready_for_adjudication",
                        "message": (
                            "Session already converged and is ready for adjudication."
                        ),
                    },
                )

                raise ValueError(
                    "Session already converged; step is disabled. Please adjudicate."
                )

            if getattr(session.engine, "is_finished", False):
                session.status = "FINISHED"
                session.updated_at = self._utc_now_iso()

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

            try:
                if session.failure_simulation.get("es_unavailable", False):
                    self._record_event(
                        session_id,
                        event="session_warning",
                        source="api",
                        data={
                            "stage": "step",
                            "kind": "es_unavailable",
                            "message": "Simulated ES unavailable; degrade continue",
                        },
                    )

                if session.failure_simulation.get("llm_timeout", False):
                    self._record_event(
                        session_id,
                        event="session_warning",
                        source="api",
                        data={
                            "stage": "step",
                            "kind": "llm_timeout",
                            "message": "Simulated LLM timeout; degrade continue",
                        },
                    )

                await session.engine.step()
                session.last_error = ""
                session.status = derive_status(session.engine)
                session.updated_at = self._utc_now_iso()
                return session

            except Exception as exc:
                session.last_error = str(exc)
                session.status = "ERROR"
                session.updated_at = self._utc_now_iso()

                self._record_event(
                    session_id,
                    event="session_error",
                    source="api",
                    data={"stage": "step", "message": session.last_error},
                )

                raise

    async def adjudicate_session(self, session_id: str) -> Any:
        """Run final adjudication for a session."""
        session = self.get_session(session_id)

        if session.status == "CREATED":
            await self.setup_session(session_id)

        async with session.lock:
            if getattr(session.engine, "is_finished", False):
                session.status = "FINISHED"
                return session

            try:
                if session.failure_simulation.get("llm_timeout", False):
                    self._record_event(
                        session_id,
                        event="session_warning",
                        source="api",
                        data={
                            "stage": "adjudicate",
                            "kind": "llm_timeout",
                            "message": "Simulated LLM timeout flag enabled",
                        },
                    )

                await session.engine.adjudicate()
                session.last_error = ""
                session.status = derive_status(session.engine)
                session.updated_at = self._utc_now_iso()
                return session

            except Exception as exc:
                session.last_error = str(exc)
                session.status = "ERROR"
                session.updated_at = self._utc_now_iso()

                self._record_event(
                    session_id,
                    event="session_error",
                    source="api",
                    data={"stage": "adjudicate", "message": session.last_error},
                )

                raise

    async def reset_memory_storage(self) -> str:
        """Delete all persisted long-term-memory files on disk."""
        if len(self._sessions) > 0:
            raise ValueError(
                "Active sessions exist. Close all sessions before clearing memory storage."
            )

        return reset_memory_storage_dir(os.getenv("MAS_STORAGE_DIR", ""))
