"""Session lifecycle manager for the lightweight API layer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mas.common.serialization import (
    serialize_value_attr,
    to_json_safe,
)
from mas.session.event_service import EventService
from mas.session.event_stream import (
    record_event as record_session_event,
)
from mas.session.session_lifecycle import (
    default_case_path as get_default_case_path,
)
from mas.session.session_lifecycle import (
    default_engine_factory as build_default_engine_factory,
)
from mas.session.session_lifecycle import (
    default_frontend_snapshots_dir as get_default_frontend_snapshots_dir,
)
from mas.session.session_lifecycle import derive_status
from mas.session.session_service import SessionService
from mas.session.session_status import SessionStatus
from mas.session.snapshot_service import SnapshotService


def utc_now_iso() -> str:
    """Return an RFC3339-like UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _to_json_safe(value: Any) -> Any:
    """Recursively convert values into JSON-serializable structures.

    Args:
        value: Arbitrary value, potentially containing enums, sets, or objects.

    Returns:
        JSON-friendly representation using only primitive containers/scalars.
    """
    return to_json_safe(value, scalar_serializer=serialize_value_attr)


@dataclass
class DebateSession:
    """Runtime session entity for a debate instance."""

    session_id: str
    engine: Any
    status: SessionStatus = SessionStatus.CREATED
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    last_error: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)
    current_turn_uid: str = ""
    last_turn_uid: str = ""
    next_seq: int = 1
    event_subscribers: List[asyncio.Queue] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    failure_simulation: Dict[str, bool] = field(
        default_factory=lambda: {"es_unavailable": False, "llm_timeout": False}
    )


class SessionManager:
    """Manage all in-memory debate sessions for API endpoints."""

    def __init__(
        self,
        engine_factory: Optional[Callable[[], Any]] = None,
        default_case_path: Optional[Path] = None,
        frontend_snapshots_dir: Optional[Path] = None,
    ):
        """Initialize the session manager and storage locations.

        Args:
            engine_factory: Optional dependency-injected engine factory.
            default_case_path: Optional fallback case JSONL path.
            frontend_snapshots_dir: Optional snapshot persistence directory.
        """
        self._engine_factory = engine_factory or build_default_engine_factory
        self._default_case_path = default_case_path or get_default_case_path()

        self._frontend_snapshots_dir = (
            frontend_snapshots_dir or get_default_frontend_snapshots_dir()
        )

        self._sessions: Dict[str, DebateSession] = {}

        self._session_service = SessionService(
            engine_factory=self._engine_factory,
            default_case_path=self._default_case_path,
            sessions=self._sessions,
            session_factory=lambda session_id, engine: DebateSession(
                session_id=session_id,
                engine=engine,
            ),
            record_event=self._record_event,
            utc_now_iso=utc_now_iso,
        )

        self._event_service = EventService(get_session=self.get_session)

        self._snapshot_service = SnapshotService(
            get_session=self.get_session,
            create_session=self.create_session,
            get_event_history=self.get_event_history,
            frontend_snapshots_dir=self._frontend_snapshots_dir,
            to_json_safe=_to_json_safe,
            utc_now_iso=utc_now_iso,
            derive_status=derive_status,
        )

    def list_sessions(self) -> List[DebateSession]:
        """Return all currently active in-memory sessions.

        Returns:
            List of `DebateSession` objects.
        """
        return self._session_service.list_sessions()

    def get_session(self, session_id: str) -> DebateSession:
        """Fetch a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Matching `DebateSession`.
        """
        return self._session_service.get_session(session_id)

    async def create_session(
        self,
        case_data: Optional[Dict[str, Any]] = None,
        auto_setup: bool = True,
    ) -> DebateSession:
        """Create a new session and optionally execute setup immediately.

        Args:
            case_data: Optional case payload used by setup.
            auto_setup: Whether to run setup after creation.

        Returns:
            Newly created `DebateSession`.
        """
        return await self._session_service.create_session(
            case_data=case_data,
            auto_setup=auto_setup,
        )

    async def setup_session(
        self,
        session_id: str,
        case_data: Optional[Dict[str, Any]] = None,
        case_data_path: Optional[Path] = None,
    ) -> DebateSession:
        """Initialize engine state for an existing session.

        Args:
            session_id: Session identifier.
            case_data: Optional in-memory case payload.
            case_data_path: Optional path for loading case payload.

        Returns:
            Updated session after setup.
        """
        return await self._session_service.setup_session(
            session_id=session_id,
            case_data=case_data,
            case_data_path=case_data_path,
        )

    async def step_session(self, session_id: str) -> DebateSession:
        """Advance the debate by one turn for the given session.

        Args:
            session_id: Session identifier.

        Returns:
            Updated session after stepping.
        """
        return await self._session_service.step_session(session_id)

    async def adjudicate_session(self, session_id: str) -> DebateSession:
        """Run final adjudication for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Updated session after adjudication.
        """
        return await self._session_service.adjudicate_session(session_id)

    async def reset_memory_storage(self) -> str:
        """Delete all persisted long-term-memory files on disk.

        Returns:
            Absolute storage directory path that has been reset.

        Raises:
            ValueError: When active sessions exist or storage path is invalid.
        """
        return await self._session_service.reset_memory_storage()

    def get_event_history(
        self,
        session_id: str,
        limit: int = 100,
        from_seq: Optional[int] = None,
        to_seq: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return filtered event history for a session.

        Args:
            session_id: Session identifier.
            limit: Maximum number of events to return.
            from_seq: Optional inclusive lower sequence bound.
            to_seq: Optional inclusive upper sequence bound.

        Returns:
            Event envelopes ordered by original sequence.
        """
        return self._event_service.get_event_history(
            session_id=session_id,
            limit=limit,
            from_seq=from_seq,
            to_seq=to_seq,
        )

    def register_event_subscriber(
        self, session_id: str, max_queue_size: int = 200
    ) -> asyncio.Queue:
        """Register a queue subscriber for live event streaming.

        Args:
            session_id: Session identifier.
            max_queue_size: Queue capacity for backpressure handling.

        Returns:
            Newly created asyncio queue.
        """
        return self._event_service.register_event_subscriber(
            session_id,
            max_queue_size=max_queue_size,
        )

    def unregister_event_subscriber(
        self, session_id: str, queue: asyncio.Queue
    ) -> None:
        """Remove a previously registered live-event subscriber queue.

        Args:
            session_id: Session identifier.
            queue: Queue instance to remove.
        """
        self._event_service.unregister_event_subscriber(session_id, queue)

    def get_snapshot_index(self, session_id: str) -> List[Dict[str, Any]]:
        """Build a compact index for available round snapshots.

        Args:
            session_id: Session identifier.

        Returns:
            List of snapshot metadata rows.
        """
        return self._snapshot_service.get_snapshot_index(session_id)

    def get_turn_artifacts(
        self,
        session_id: str,
        turn_uid: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return turn artifacts, optionally filtered by one turn UID.

        Args:
            session_id: Session identifier.
            turn_uid: Optional turn UID filter.
            limit: Maximum number of artifacts to return.

        Returns:
            Artifact rows returned by the engine.
        """
        return self._snapshot_service.get_turn_artifacts(
            session_id=session_id,
            turn_uid=turn_uid,
            limit=limit,
        )

    def get_teamflow_stream(
        self,
        session_id: str,
        limit: int = 80,
    ) -> List[Dict[str, Any]]:
        """Transform artifacts/events into frontend TeamFlow timeline rows.

        Args:
            session_id: Session identifier.
            limit: Maximum number of turns to include.

        Returns:
            Ordered list of turn-level TeamFlow payloads.
        """
        return self._snapshot_service.get_teamflow_stream(
            session_id=session_id, limit=limit
        )

    def export_graph_gexf(
        self, session_id: str, round_idx: Optional[int] = None
    ) -> bytes:
        """Export current or historical graph as GEXF bytes."""
        return self._snapshot_service.export_graph_gexf(
            session_id=session_id,
            round_idx=round_idx,
        )

    def export_replay_bundle(
        self,
        session_id: str,
        include_events_limit: int = 5000,
        include_artifacts_limit: int = 5000,
    ) -> Dict[str, Any]:
        """Export a complete replay bundle for offline analysis."""
        return self._snapshot_service.export_replay_bundle(
            session_id=session_id,
            include_events_limit=include_events_limit,
            include_artifacts_limit=include_artifacts_limit,
        )

    def save_frontend_snapshot(
        self,
        session_id: str,
        label: str = "",
        frontend_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist current session replay plus optional frontend state.

        Args:
            session_id: Source session identifier.
            label: Optional human-readable snapshot label.
            frontend_state: Optional frontend UI state payload.

        Returns:
            Stored snapshot metadata.
        """
        return self._snapshot_service.save_frontend_snapshot(
            session_id=session_id,
            label=label,
            frontend_state=frontend_state,
        )

    def import_frontend_snapshot(
        self,
        bundle: Dict[str, Any],
        label: str = "",
        frontend_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist an externally provided replay bundle as a snapshot record.

        Args:
            bundle: Imported replay bundle payload.
            label: Optional override label for the imported snapshot.
            frontend_state: Optional override frontend state payload.

        Returns:
            Stored snapshot metadata.
        """
        return self._snapshot_service.import_frontend_snapshot(
            bundle=bundle,
            label=label,
            frontend_state=frontend_state,
        )

    def list_frontend_snapshots(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List persisted frontend snapshots with pagination.

        Args:
            limit: Maximum number of rows to return.
            offset: Pagination offset.

        Returns:
            Paginated snapshot list payload.
        """
        return self._snapshot_service.list_frontend_snapshots(
            limit=limit, offset=offset
        )

    async def load_frontend_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Restore a persisted frontend snapshot into a live session.

        Args:
            snapshot_id: Snapshot identifier to restore.

        Returns:
            Frontend bootstrap payload containing restored session details.
        """
        return await self._snapshot_service.load_frontend_snapshot(snapshot_id)

    def _record_event(
        self, session_id: str, event: str, source: str, data: Optional[Dict[str, Any]]
    ) -> None:
        """Append one event envelope and fan it out to live subscribers."""
        session = self._sessions.get(session_id)

        if not session:
            return

        record_session_event(
            session=session,
            session_id=session_id,
            event=event,
            source=source,
            data=data,
            to_json_safe=_to_json_safe,
            utc_now_iso=utc_now_iso,
        )
