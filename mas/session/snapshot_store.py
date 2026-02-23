"""Helpers for frontend snapshot persistence and replay restoration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from metagpt.logs import logger

from mas.common.serialization import as_non_negative_int, to_json_safe


def ensure_frontend_snapshots_dir(frontend_snapshots_dir: Path) -> Path:
    """Ensure the frontend snapshot directory exists on disk."""
    frontend_snapshots_dir.mkdir(parents=True, exist_ok=True)
    return frontend_snapshots_dir


def frontend_snapshot_path(frontend_snapshots_dir: Path, snapshot_id: str) -> Path:
    """Build a sanitized filesystem path for one snapshot ID."""
    safe_id = "".join(
        ch for ch in str(snapshot_id).strip() if ch.isalnum() or ch in {"-", "_"}
    )

    if not safe_id:
        raise ValueError("Invalid frontend snapshot id")

    return ensure_frontend_snapshots_dir(frontend_snapshots_dir) / f"{safe_id}.json"


def extract_replay_metadata(bundle: Dict[str, Any]) -> Dict[str, int]:
    """Derive replay counts from bundle content and metadata fallback."""
    metadata_raw = bundle.get("metadata", {})

    if not isinstance(metadata_raw, dict):
        metadata_raw = {}

    events = bundle.get("events", [])
    artifacts = bundle.get("turn_artifacts", [])
    snapshots = bundle.get("snapshots", [])

    event_count = (
        len(events)
        if isinstance(events, list)
        else as_non_negative_int(metadata_raw.get("event_count"), 0)
    )

    artifact_count = (
        len(artifacts)
        if isinstance(artifacts, list)
        else as_non_negative_int(metadata_raw.get("artifact_count"), 0)
    )

    snapshot_count = (
        len(snapshots)
        if isinstance(snapshots, list)
        else as_non_negative_int(metadata_raw.get("snapshot_count"), 0)
    )

    return {
        "event_count": event_count,
        "artifact_count": artifact_count,
        "snapshot_count": snapshot_count,
    }


def frontend_snapshot_item(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw snapshot record into API list-item shape."""
    metadata = record.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    event_count = as_non_negative_int(metadata.get("event_count"), 0)
    artifact_count = as_non_negative_int(metadata.get("artifact_count"), 0)
    snapshot_count = as_non_negative_int(metadata.get("snapshot_count"), 0)

    return {
        "snapshot_id": str(record.get("snapshot_id", "")),
        "label": str(record.get("label", "")),
        "source_session_id": str(record.get("source_session_id", "")),
        "created_at": str(record.get("created_at", "")),
        "event_count": event_count,
        "artifact_count": artifact_count,
        "snapshot_count": snapshot_count,
        "metadata": {
            "event_count": event_count,
            "artifact_count": artifact_count,
            "snapshot_count": snapshot_count,
        },
    }


def write_frontend_snapshot_record(
    frontend_snapshots_dir: Path,
    record: Dict[str, Any],
) -> None:
    """Persist a frontend snapshot record as JSON."""
    target = frontend_snapshot_path(
        frontend_snapshots_dir,
        str(record.get("snapshot_id", "")),
    )

    with target.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)


def read_frontend_snapshot_record(
    frontend_snapshots_dir: Path,
    snapshot_id: str,
) -> Dict[str, Any]:
    """Load and validate one frontend snapshot record from disk."""
    target = frontend_snapshot_path(frontend_snapshots_dir, snapshot_id)

    if not target.exists():
        raise FileNotFoundError(f"Frontend snapshot not found: {snapshot_id}")

    try:
        with target.open("r", encoding="utf-8") as file:
            payload = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Frontend snapshot payload is invalid JSON: {snapshot_id}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Frontend snapshot payload must be an object: {snapshot_id}")

    if not str(payload.get("snapshot_id", "")).strip():
        payload["snapshot_id"] = str(snapshot_id).strip()

    return payload


def build_snapshot_payload(session: Any) -> Dict[str, Any]:
    """Build the API-facing snapshot payload for a session."""
    base = session.engine.get_serializable_snapshot()

    if not isinstance(base, dict):
        base = {}

    graph_stats = base.get("graph_stats", {})

    if not isinstance(graph_stats, dict):
        graph_stats = {}

    payload = {
        **base,
        "session_id": session.session_id,
        "status": session.status.value,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "metrics": {
            "arguments": int(graph_stats.get("node_count", 0)),
            "attacks": int(graph_stats.get("edge_attack_count", 0)),
            "supports": int(graph_stats.get("edge_support_count", 0)),
        },
    }

    if session.last_error:
        payload["error"] = session.last_error

    return to_json_safe(payload)


def frontend_snapshot_load_response(
    record: Dict[str, Any], session: Any
) -> Dict[str, Any]:
    """Build response payload returned after loading a frontend snapshot."""
    return {
        "snapshot": frontend_snapshot_item(record),
        "frontend_state": to_json_safe(record.get("frontend_state", {})),
        "session": {
            "session_id": session.session_id,
            "status": session.status.value,
            "current_round": int(getattr(session.engine, "round_idx", 0)),
            "updated_at": session.updated_at,
        },
        "snapshot_payload": build_snapshot_payload(session),
    }


def normalize_import_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Validate imported replay bundle against canonical structure."""
    if not isinstance(bundle, dict):
        raise ValueError("Imported bundle must be an object")

    snapshots = bundle.get("snapshots")

    if not isinstance(snapshots, list) or len(snapshots) == 0:
        raise ValueError("Imported bundle missing non-empty snapshots")

    session_row = bundle.get("session")
    snapshot_row = bundle.get("snapshot")
    events_row = bundle.get("events")
    artifacts_row = bundle.get("turn_artifacts")

    if not isinstance(session_row, dict):
        raise ValueError("Imported bundle missing session object")

    if not isinstance(snapshot_row, dict):
        raise ValueError("Imported bundle missing snapshot object")

    if not isinstance(events_row, list):
        raise ValueError("Imported bundle missing events array")

    if not isinstance(artifacts_row, list):
        raise ValueError("Imported bundle missing turn_artifacts array")

    return to_json_safe(bundle)


def list_frontend_snapshots(
    frontend_snapshots_dir: Path,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    """List persisted frontend snapshots with pagination."""
    limit_value = max(1, int(limit))
    offset_value = max(0, int(offset))
    snapshot_dir = ensure_frontend_snapshots_dir(frontend_snapshots_dir)
    items: List[Dict[str, Any]] = []

    files = sorted(
        snapshot_dir.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for path in files:
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)

            if not isinstance(payload, dict):
                continue

            if not str(payload.get("snapshot_id", "")).strip():
                payload["snapshot_id"] = path.stem

            items.append(frontend_snapshot_item(payload))

        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "[snapshot_store] Skip invalid snapshot file {}: {}",
                path,
                exc,
            )

            continue

    total = len(items)
    paged = items[offset_value : offset_value + limit_value]
    return {"items": paged, "total": total}


def normalize_restored_events(session_id: str, events: Any) -> List[Dict[str, Any]]:
    """Normalize replay events into internal event-envelope format."""
    if not isinstance(events, list):
        return []

    rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(events):
        if not isinstance(item, dict):
            continue

        ts_ms = as_non_negative_int(item.get("ts_ms"), int(time.time() * 1000))
        round_raw = item.get("round_idx")
        round_idx: Optional[int]

        if round_raw is None:
            round_idx = None

        else:
            try:
                round_idx = int(round_raw)

            except (TypeError, ValueError):
                round_idx = None

        event_name = str(item.get("event", "")).strip() or "replay_event"
        source = str(item.get("source", "")).strip() or "replay"
        turn_uid = str(item.get("turn_uid", "")).strip()
        data = item.get("data", {})

        if not isinstance(data, dict):
            data = {"value": to_json_safe(data)}

        seq = idx + 1

        rows.append(
            {
                "event_id": f"{session_id}-{seq:06d}",
                "seq": seq,
                "ts_ms": ts_ms,
                "session_id": session_id,
                "turn_uid": turn_uid,
                "round_idx": round_idx,
                "event": event_name,
                "source": source,
                "data": to_json_safe(data),
            }
        )

    return rows


def restore_round_index(bundle: Dict[str, Any], snapshots: Any) -> int:
    """Choose the best snapshot index to restore from replay data."""
    rows = snapshots if isinstance(snapshots, list) else []
    snapshot_count = len(rows)

    if snapshot_count <= 0:
        raise ValueError("No snapshot data available for restore")

    snapshot = bundle.get("snapshot", {})
    target_round: Optional[int] = None
    target_turn_uid = ""

    if isinstance(snapshot, dict):
        current_round = snapshot.get("current_round", snapshot.get("round_idx"))

        try:
            if current_round is not None:
                target_round = int(current_round)

        except (TypeError, ValueError):
            target_round = None

        target_turn_uid = str(snapshot.get("latest_turn_uid", "")).strip()

    if target_turn_uid:
        for idx in range(snapshot_count - 1, -1, -1):
            row = rows[idx]

            if not isinstance(row, dict):
                continue

            row_turn_uid = str(
                row.get("latest_turn_uid", row.get("turn_uid", ""))
            ).strip()

            if row_turn_uid == target_turn_uid:
                return idx

    if target_round is not None:
        matched_idx = None

        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue

            row_round_raw = row.get("round_idx", row.get("current_round"))

            try:
                if row_round_raw is None:
                    continue

                if int(row_round_raw) == target_round:
                    matched_idx = idx

            except (TypeError, ValueError):
                continue

        if matched_idx is not None:
            return matched_idx

        if target_round < 0:
            return 0

        if target_round >= snapshot_count:
            return snapshot_count - 1

        return target_round

    return snapshot_count - 1
