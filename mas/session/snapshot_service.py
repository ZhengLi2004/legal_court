"""Snapshot/export service extracted from API manager facade."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from mas.session.exporters import (
    build_replay_bundle,
)
from mas.session.exporters import (
    export_graph_gexf as export_graph_gexf_bytes,
)
from mas.session.replay_restore import (
    apply_normalized_replay_bundle,
)
from mas.session.session_status import SessionStatus
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


class SnapshotService:
    """Encapsulate snapshot persistence, replay export, and restore flows."""

    def __init__(
        self,
        *,
        get_session: Callable[[str], Any],
        create_session: Callable[..., Awaitable[Any]],
        get_event_history: Callable[..., List[Dict[str, Any]]],
        frontend_snapshots_dir: Path,
        to_json_safe: Callable[[Any], Any],
        utc_now_iso: Callable[[], str],
        derive_status: Callable[[Any], SessionStatus],
    ):
        """Initialize snapshot service with injected dependencies.

        Args:
            get_session: Callable to resolve one runtime session by id.
            create_session: Callable to create a new runtime session.
            get_event_history: Callable to fetch session events.
            frontend_snapshots_dir: Root directory for persisted snapshot files.
            to_json_safe: Serializer for non-JSON-safe values.
            utc_now_iso: Timestamp factory for record creation times.
            derive_status: Callable that derives `SessionStatus` from engine state.
        """
        self._get_session = get_session
        self._create_session = create_session
        self._get_event_history = get_event_history
        self._frontend_snapshots_dir = frontend_snapshots_dir
        self._to_json_safe = to_json_safe
        self._utc_now_iso = utc_now_iso
        self._derive_status = derive_status

    def get_snapshot_index(self, session_id: str) -> List[Dict[str, Any]]:
        """Build compact metadata index for one session's round snapshots.

        Args:
            session_id: Session identifier.

        Returns:
            List of snapshot metadata rows (round index, turn, timestamp, counts).
        """
        session = self._get_session(session_id)
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
        """Return turn artifacts from engine, optionally filtered by turn UID.

        Args:
            session_id: Session identifier.
            turn_uid: Optional turn UID filter.
            limit: Maximum artifact rows to return.

        Returns:
            Artifact rows, or empty list when engine has no compatible getter.
        """
        session = self._get_session(session_id)
        getter = getattr(session.engine, "get_turn_artifacts", None)

        if callable(getter):
            return getter(turn_uid=turn_uid, limit=limit)

        return []

    def get_teamflow_stream(
        self,
        session_id: str,
        limit: int = 80,
    ) -> List[Dict[str, Any]]:
        """Build TeamFlow timeline rows from artifacts and events.

        Args:
            session_id: Session identifier.
            limit: Maximum number of turn rows to return.

        Returns:
            TeamFlow rows in frontend-consumable shape.
        """
        self._get_session(session_id)
        safe_limit = max(1, int(limit))

        artifacts = self.get_turn_artifacts(
            session_id=session_id,
            turn_uid=None,
            limit=safe_limit,
        )

        if not isinstance(artifacts, list) or len(artifacts) == 0:
            return []

        events = self._get_event_history(
            session_id=session_id,
            limit=max(240, safe_limit * 12),
        )

        return build_teamflow_stream(
            artifacts=artifacts,
            events=events,
            limit=safe_limit,
            to_json_safe=self._to_json_safe,
        )

    def export_graph_gexf(
        self, session_id: str, round_idx: Optional[int] = None
    ) -> bytes:
        """Export current or historical graph as GEXF bytes.

        Args:
            session_id: Session identifier.
            round_idx: Optional historical snapshot index.

        Returns:
            GEXF bytes generated from selected graph state.
        """
        session = self._get_session(session_id)

        return export_graph_gexf_bytes(
            session,
            round_idx=round_idx,
            to_json_safe=self._to_json_safe,
        )

    def export_replay_bundle(
        self,
        session_id: str,
        include_events_limit: int = 5000,
        include_artifacts_limit: int = 5000,
    ) -> Dict[str, Any]:
        """Export replay bundle containing snapshots, events, and artifacts.

        Args:
            session_id: Session identifier.
            include_events_limit: Maximum event rows included in export.
            include_artifacts_limit: Maximum artifact rows included in export.

        Returns:
            Replay bundle dict for persistence or offline analysis.
        """
        session = self._get_session(session_id)

        events = self._get_event_history(
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
            utc_now_iso=self._utc_now_iso,
            to_json_safe=self._to_json_safe,
        )

    def save_frontend_snapshot(
        self,
        session_id: str,
        label: str = "",
        frontend_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a snapshot record for an existing session.

        Args:
            session_id: Source session identifier.
            label: Optional human-readable label.
            frontend_state: Optional frontend state payload.

        Returns:
            Persisted snapshot metadata row.
        """
        self._get_session(session_id)
        bundle = self.export_replay_bundle(session_id=session_id)
        snapshot_id = f"fs_{uuid.uuid4().hex[:12]}"
        normalized_label = str(label).strip() or f"{session_id}-snapshot"

        record = {
            "snapshot_id": snapshot_id,
            "label": normalized_label,
            "source_session_id": session_id,
            "created_at": self._utc_now_iso(),
            "frontend_state": self._to_json_safe(frontend_state or {}),
            "replay_bundle": self._to_json_safe(bundle),
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
        """Persist an external replay bundle as snapshot record.

        Args:
            bundle: Imported replay bundle payload.
            label: Optional override label.
            frontend_state: Optional override frontend state payload.

        Returns:
            Persisted snapshot metadata row.

        Raises:
            ValueError: If `bundle` fails normalization checks.
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
            "created_at": self._utc_now_iso(),
            "frontend_state": self._to_json_safe(merged_frontend_state or {}),
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
            limit: Maximum number of rows.
            offset: Pagination offset.

        Returns:
            Dict with paged `items` and `total`.
        """
        return list_frontend_snapshot_items(
            self._frontend_snapshots_dir,
            limit=limit,
            offset=offset,
        )

    async def load_frontend_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Restore one persisted frontend snapshot into a live runtime session.

        Args:
            snapshot_id: Snapshot identifier.

        Returns:
            Frontend bootstrap payload including restored session metadata.

        Raises:
            ValueError: If stored replay bundle is invalid or restore fails.
            FileNotFoundError: If snapshot record does not exist.
        """
        record = read_frontend_snapshot_record(
            self._frontend_snapshots_dir, snapshot_id
        )

        replay_bundle = record.get("replay_bundle")

        if not isinstance(replay_bundle, dict):
            raise ValueError("Frontend snapshot has invalid replay bundle")

        normalized_bundle = normalize_import_bundle(replay_bundle)
        restored_session = await self._create_session(auto_setup=True)

        apply_normalized_replay_bundle(
            restored_session=restored_session,
            normalized_bundle=normalized_bundle,
            derive_status=self._derive_status,
            to_json_safe=self._to_json_safe,
            utc_now_iso=self._utc_now_iso,
        )

        return frontend_snapshot_load_response(record, restored_session)
