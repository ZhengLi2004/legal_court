"""Session lifecycle manager for the lightweight API layer."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def utc_now_iso() -> str:
    """Return an RFC3339-like UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _infer_event_source(event: str) -> str:
    """Infer event source from event name."""
    if event.startswith("team_"):
        return "team"

    if event.startswith("adjudication"):
        return "judge"

    if (
        event.startswith("setup")
        or event.startswith("turn")
        or event
        in {
            "transcript_update",
            "snapshot_saved",
        }
    ):
        return "engine"

    return "engine"


def _default_case_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "sampling"
        / "cleaned_samples.jsonl"
    )


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]

    if isinstance(value, set):
        return [_to_json_safe(v) for v in sorted(value, key=str)]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    scalar = getattr(value, "value", None)

    if isinstance(scalar, (str, int, float, bool)) or scalar is None:
        if scalar is not None:
            return scalar

    return str(value)


def _default_engine_factory() -> Any:
    from mas.config import SystemConfig
    from mas.core.engine import DebateEngine

    return DebateEngine(config=SystemConfig(), judge_config={})


def _load_case_from_jsonl(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = line.strip()

            if not row:
                continue

            payload = json.loads(row)

            if isinstance(payload, dict):
                return payload

    raise ValueError(f"No valid case row found in {path}")


@dataclass
class DebateSession:
    """Runtime session entity for a debate instance."""

    session_id: str
    engine: Any
    status: str = "CREATED"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    last_error: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)
    current_turn_uid: str = ""
    last_turn_uid: str = ""
    next_seq: int = 1
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionManager:
    """Manage all in-memory debate sessions for API endpoints."""

    def __init__(
        self,
        engine_factory: Optional[Callable[[], Any]] = None,
        default_case_path: Optional[Path] = None,
    ):
        self._engine_factory = engine_factory or _default_engine_factory
        self._default_case_path = default_case_path or _default_case_path()
        self._sessions: Dict[str, DebateSession] = {}

    def list_sessions(self) -> List[DebateSession]:
        return list(self._sessions.values())

    def get_session(self, session_id: str) -> DebateSession:
        try:
            return self._sessions[session_id]

        except KeyError as exc:
            raise KeyError(f"Session not found: {session_id}") from exc

    async def create_session(
        self,
        case_data: Optional[Dict[str, Any]] = None,
        max_rounds: Optional[int] = None,
        auto_setup: bool = True,
    ) -> DebateSession:
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        engine = self._engine_factory()

        if isinstance(max_rounds, int) and max_rounds > 0:
            engine.max_rounds = max_rounds

        session = DebateSession(session_id=session_id, engine=engine)

        engine.set_state_callback(
            lambda event, data, sid=session_id: self._record_event(
                sid,
                event=event,
                source=_infer_event_source(event),
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
    ) -> DebateSession:
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
                payload = _load_case_from_jsonl(Path(target_path))

            session.status = "SETTING_UP"
            session.updated_at = utc_now_iso()

            try:
                await session.engine.setup(case_data=payload)
                session.last_error = ""
                session.status = self._derive_status(session.engine)
                session.updated_at = utc_now_iso()
                return session

            except Exception as exc:
                session.last_error = str(exc)
                session.status = "ERROR"
                session.updated_at = utc_now_iso()

                self._record_event(
                    session_id,
                    event="session_error",
                    source="api",
                    data={"stage": "setup", "message": session.last_error},
                )

                raise

    async def step_session(self, session_id: str) -> DebateSession:
        session = self.get_session(session_id)

        if session.status == "CREATED":
            await self.setup_session(session_id)

        async with session.lock:
            if getattr(session.engine, "is_finished", False):
                session.status = "FINISHED"
                return session

            try:
                await session.engine.step()
                session.last_error = ""
                session.status = self._derive_status(session.engine)
                session.updated_at = utc_now_iso()
                return session

            except Exception as exc:
                session.last_error = str(exc)
                session.status = "ERROR"
                session.updated_at = utc_now_iso()

                self._record_event(
                    session_id,
                    event="session_error",
                    source="api",
                    data={"stage": "step", "message": session.last_error},
                )

                raise

    async def adjudicate_session(self, session_id: str) -> DebateSession:
        session = self.get_session(session_id)

        if session.status == "CREATED":
            await self.setup_session(session_id)

        async with session.lock:
            if getattr(session.engine, "is_finished", False):
                session.status = "FINISHED"
                return session

            try:
                await session.engine.adjudicate()
                session.last_error = ""
                session.status = self._derive_status(session.engine)
                session.updated_at = utc_now_iso()
                return session

            except Exception as exc:
                session.last_error = str(exc)
                session.status = "ERROR"
                session.updated_at = utc_now_iso()

                self._record_event(
                    session_id,
                    event="session_error",
                    source="api",
                    data={"stage": "adjudicate", "message": session.last_error},
                )

                raise

    async def delete_session(self, session_id: str) -> None:
        session = self.get_session(session_id)

        async with session.lock:
            try:
                await session.engine.close_resources()

            finally:
                self._sessions.pop(session_id, None)

    def get_event_history(
        self,
        session_id: str,
        limit: int = 100,
        from_seq: Optional[int] = None,
        to_seq: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        events = session.events

        if from_seq is not None:
            events = [item for item in events if int(item.get("seq", 0)) >= from_seq]

        if to_seq is not None:
            events = [item for item in events if int(item.get("seq", 0)) <= to_seq]

        limit_value = max(1, int(limit))
        return events[-limit_value:]

    def get_turn_artifacts(
        self,
        session_id: str,
        turn_uid: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        getter = getattr(session.engine, "get_turn_artifacts", None)

        if callable(getter):
            return getter(turn_uid=turn_uid, limit=limit)

        return []

    def _record_event(
        self, session_id: str, event: str, source: str, data: Optional[Dict[str, Any]]
    ) -> None:
        session = self._sessions.get(session_id)

        if not session:
            return

        payload = _to_json_safe(data or {})
        explicit_turn_uid = str(payload.get("turn_uid", "")).strip()

        if explicit_turn_uid:
            session.current_turn_uid = explicit_turn_uid
            session.last_turn_uid = explicit_turn_uid

        elif event == "turn_start":
            round_part = str(payload.get("round", "na"))
            side_part = str(payload.get("turn", "unknown"))

            session.current_turn_uid = (
                f"turn_{round_part}_{side_part}_{int(time.time() * 1000)}"
            )

            session.last_turn_uid = session.current_turn_uid

        turn_uid = session.current_turn_uid or session.last_turn_uid

        session.events.append(
            {
                "seq": session.next_seq,
                "ts_ms": int(time.time() * 1000),
                "session_id": session_id,
                "turn_uid": turn_uid,
                "event": event,
                "source": source,
                "data": payload,
            }
        )

        if event == "turn_complete":
            session.last_turn_uid = turn_uid

        session.next_seq += 1
        session.updated_at = utc_now_iso()

    def _derive_status(self, engine: Any) -> str:
        if getattr(engine, "is_finished", False):
            return "FINISHED"

        if getattr(engine, "is_ready_for_adjudication", False):
            return "READY_FOR_ADJUDICATION"

        if getattr(engine, "round_idx", 0) > 0:
            return "DEBATING"

        if getattr(engine, "graph", None) is not None:
            return "SETUP_DONE"

        return "CREATED"
