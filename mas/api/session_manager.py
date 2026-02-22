"""Session lifecycle manager for the lightweight API layer."""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mas.common.serialization import (
    serialize_value_attr,
    to_json_safe,
)
from mas.session.event_stream import (
    get_event_history as get_session_event_history,
)
from mas.session.event_stream import (
    infer_event_source,
)
from mas.session.event_stream import (
    record_event as record_session_event,
)
from mas.session.event_stream import (
    register_event_subscriber as register_session_event_subscriber,
)
from mas.session.event_stream import (
    unregister_event_subscriber as unregister_session_event_subscriber,
)
from mas.session.exporters import (
    build_replay_bundle,
)
from mas.session.exporters import (
    export_graph_gexf as export_graph_gexf_bytes,
)
from mas.session.replay_restore import (
    apply_normalized_replay_bundle,
    get_recent_loaded_session,
    mark_recent_load,
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
from mas.session.session_lifecycle import (
    derive_status,
    load_case_from_jsonl,
    resolve_memory_storage_dir,
)
from mas.session.snapshot_store import (
    extract_replay_metadata,
    frontend_snapshot_item,
    frontend_snapshot_load_response,
    normalize_import_bundle,
    read_frontend_snapshot_record,
    write_frontend_snapshot_record,
)
from mas.session.snapshot_store import (
    list_frontend_snapshots as list_frontend_snapshot_items,
)
from mas.session.teamflow_stream import build_teamflow_stream


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
    status: str = "CREATED"
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

        self._recent_frontend_snapshot_loads: Dict[str, Dict[str, Any]] = {}
        self._sessions: Dict[str, DebateSession] = {}

    def list_sessions(self) -> List[DebateSession]:
        """Return all currently active in-memory sessions.

        Returns:
            List of `DebateSession` objects.
        """
        return list(self._sessions.values())

    def get_session(self, session_id: str) -> DebateSession:
        """Fetch a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Matching `DebateSession`.
        """
        try:
            return self._sessions[session_id]

        except KeyError as exc:
            raise KeyError(f"Session not found: {session_id}") from exc

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
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        engine = self._engine_factory()
        session = DebateSession(session_id=session_id, engine=engine)

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
    ) -> DebateSession:
        """Initialize engine state for an existing session.

        Args:
            session_id: Session identifier.
            case_data: Optional in-memory case payload.
            case_data_path: Optional path for loading case payload.

        Returns:
            Updated session after setup.
        """
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
            session.updated_at = utc_now_iso()

            try:
                await session.engine.setup(case_data=payload)
                session.last_error = ""
                session.status = derive_status(session.engine)
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
        """Advance the debate by one turn for the given session.

        Args:
            session_id: Session identifier.

        Returns:
            Updated session after stepping.
        """
        session = self.get_session(session_id)

        if session.status == "CREATED":
            await self.setup_session(session_id)

        async with session.lock:
            if getattr(session.engine, "is_ready_for_adjudication", False):
                session.status = derive_status(session.engine)
                session.updated_at = utc_now_iso()

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
                session.updated_at = utc_now_iso()

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
        """Run final adjudication for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Updated session after adjudication.
        """
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

    async def reset_memory_storage(self) -> str:
        """Delete all persisted long-term-memory files on disk.

        Returns:
            Absolute storage directory path that has been reset.

        Raises:
            ValueError: When active sessions exist or storage path is invalid.
        """
        if len(self._sessions) > 0:
            raise ValueError(
                "Active sessions exist. Close all sessions before clearing memory storage."
            )

        storage_dir = resolve_memory_storage_dir(os.getenv("MAS_STORAGE_DIR", ""))

        if storage_dir.exists():
            shutil.rmtree(storage_dir)

        storage_dir.mkdir(parents=True, exist_ok=True)
        return str(storage_dir)

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
        session = self.get_session(session_id)

        return get_session_event_history(
            session.events,
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
        session = self.get_session(session_id)

        return register_session_event_subscriber(
            session.event_subscribers,
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
        session = self.get_session(session_id)
        unregister_session_event_subscriber(session.event_subscribers, queue)

    def get_snapshot_index(self, session_id: str) -> List[Dict[str, Any]]:
        """Build a compact index for available round snapshots.

        Args:
            session_id: Session identifier.

        Returns:
            List of snapshot metadata rows.
        """
        session = self.get_session(session_id)
        snapshots = getattr(session.engine, "round_snapshots", [])

        if not isinstance(snapshots, list):
            return []

        items: List[Dict[str, Any]] = []

        for idx, row in enumerate(snapshots):
            if not isinstance(row, dict):
                continue

            graph_data = row.get("graph_data", {})

            if isinstance(graph_data, dict):
                nodes = graph_data.get("nodes", [])
                edges = graph_data.get("edges", [])
                node_count = len(nodes) if isinstance(nodes, list) else 0
                edge_count = len(edges) if isinstance(edges, list) else 0

            else:
                node_count = 0
                edge_count = 0

            items.append(
                {
                    "round_idx": int(row.get("round_idx", idx)),
                    "turn": str(row.get("turn", "")),
                    "ts_ms": int(row.get("ts_ms", row.get("timestamp", 0)) or 0),
                    "node_count": node_count,
                    "edge_count": edge_count,
                }
            )

        return items

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
        session = self.get_session(session_id)
        getter = getattr(session.engine, "get_turn_artifacts", None)

        if callable(getter):
            return getter(turn_uid=turn_uid, limit=limit)

        return []

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
        self.get_session(session_id)
        safe_limit = max(1, int(limit))

        artifacts = self.get_turn_artifacts(
            session_id=session_id,
            turn_uid=None,
            limit=safe_limit,
        )

        if not isinstance(artifacts, list) or len(artifacts) == 0:
            return []

        events = self.get_event_history(
            session_id=session_id,
            limit=max(240, safe_limit * 12),
        )
        return build_teamflow_stream(
            artifacts=artifacts,
            events=events,
            limit=safe_limit,
            to_json_safe=_to_json_safe,
        )

    def export_graph_gexf(
        self, session_id: str, round_idx: Optional[int] = None
    ) -> bytes:
        """Export current or historical graph as GEXF bytes."""
        session = self.get_session(session_id)

        return export_graph_gexf_bytes(
            session,
            round_idx=round_idx,
            to_json_safe=_to_json_safe,
        )

    def export_replay_bundle(
        self,
        session_id: str,
        include_events_limit: int = 5000,
        include_artifacts_limit: int = 5000,
    ) -> Dict[str, Any]:
        """Export a complete replay bundle for offline analysis."""
        session = self.get_session(session_id)

        events = self.get_event_history(
            session_id=session_id,
            limit=max(1, int(include_events_limit)),
        )

        artifacts = self.get_turn_artifacts(
            session_id=session_id,
            turn_uid=None,
            limit=max(1, int(include_artifacts_limit)),
        )

        return build_replay_bundle(
            session=session,
            snapshot_index=self.get_snapshot_index(session_id),
            events=events,
            artifacts=artifacts,
            utc_now_iso=utc_now_iso,
            to_json_safe=_to_json_safe,
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
        self.get_session(session_id)
        bundle = self.export_replay_bundle(session_id=session_id)
        snapshot_id = f"fs_{uuid.uuid4().hex[:12]}"
        normalized_label = str(label).strip() or f"{session_id}-snapshot"

        record = {
            "snapshot_id": snapshot_id,
            "label": normalized_label,
            "source_session_id": session_id,
            "created_at": utc_now_iso(),
            "frontend_state": _to_json_safe(frontend_state or {}),
            "replay_bundle": _to_json_safe(bundle),
            "metadata": extract_replay_metadata(bundle),
        }

        write_frontend_snapshot_record(self._frontend_snapshots_dir, record)
        return frontend_snapshot_item(record)

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
        normalized_bundle = normalize_import_bundle(bundle)
        session = normalized_bundle.get("session", {})

        if not isinstance(session, dict):
            session = {}

        source_session_id = str(
            session.get("session_id")
            or normalized_bundle.get("session_id")
            or "imported_session"
        ).strip()

        incoming_label = str(bundle.get("label", "")).strip()
        normalized_label = str(label).strip() or incoming_label or "imported-snapshot"
        incoming_frontend_state = bundle.get("frontend_state")
        merged_frontend_state = frontend_state

        if merged_frontend_state is None and isinstance(incoming_frontend_state, dict):
            merged_frontend_state = incoming_frontend_state

        snapshot_id = f"fs_{uuid.uuid4().hex[:12]}"

        record = {
            "snapshot_id": snapshot_id,
            "label": normalized_label,
            "source_session_id": source_session_id,
            "created_at": utc_now_iso(),
            "frontend_state": _to_json_safe(merged_frontend_state or {}),
            "replay_bundle": normalized_bundle,
            "metadata": extract_replay_metadata(normalized_bundle),
        }

        write_frontend_snapshot_record(self._frontend_snapshots_dir, record)
        return frontend_snapshot_item(record)

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
        return list_frontend_snapshot_items(
            self._frontend_snapshots_dir,
            limit=limit,
            offset=offset,
        )

    async def load_frontend_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Restore a persisted frontend snapshot into a live session.

        Args:
            snapshot_id: Snapshot identifier to restore.

        Returns:
            Frontend bootstrap payload containing restored session details.
        """
        record = read_frontend_snapshot_record(
            self._frontend_snapshots_dir, snapshot_id
        )

        recent_session = get_recent_loaded_session(
            snapshot_id=snapshot_id,
            recent_loads=self._recent_frontend_snapshot_loads,
            sessions=self._sessions,
        )

        if recent_session is not None:
            recent_session.updated_at = utc_now_iso()
            return frontend_snapshot_load_response(record, recent_session)

        replay_bundle = record.get("replay_bundle")

        if not isinstance(replay_bundle, dict):
            raise ValueError("Frontend snapshot has invalid replay bundle")

        normalized_bundle = normalize_import_bundle(replay_bundle)
        restored_session = await self.create_session(auto_setup=True)

        apply_normalized_replay_bundle(
            restored_session=restored_session,
            normalized_bundle=normalized_bundle,
            derive_status=derive_status,
            to_json_safe=_to_json_safe,
            utc_now_iso=utc_now_iso,
        )

        mark_recent_load(
            snapshot_id=snapshot_id,
            session_id=restored_session.session_id,
            recent_loads=self._recent_frontend_snapshot_loads,
        )

        return frontend_snapshot_load_response(record, restored_session)

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
