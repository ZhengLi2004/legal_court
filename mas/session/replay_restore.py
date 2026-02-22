"""Helpers for replay-bundle restoration into runtime sessions."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from .snapshot_store import normalize_restored_events, restore_round_index


def get_recent_loaded_session(
    *,
    snapshot_id: str,
    recent_loads: Dict[str, Dict[str, Any]],
    sessions: Dict[str, Any],
    now_ts: Optional[float] = None,
    ttl_seconds: float = 8.0,
) -> Optional[Any]:
    """Return a recently restored session from short-term cache if available."""
    now = time.time() if now_ts is None else float(now_ts)
    cache_entry = recent_loads.get(snapshot_id, {})
    recent_session_id = str(cache_entry.get("session_id", "")).strip()
    recent_loaded_at = float(cache_entry.get("loaded_at", 0.0) or 0.0)

    if not recent_session_id:
        return None

    if (now - recent_loaded_at) > ttl_seconds:
        return None

    return sessions.get(recent_session_id)


def mark_recent_load(
    *,
    snapshot_id: str,
    session_id: str,
    recent_loads: Dict[str, Dict[str, Any]],
    loaded_at: Optional[float] = None,
) -> None:
    """Write latest restore cache entry for one snapshot id."""
    recent_loads[snapshot_id] = {
        "session_id": session_id,
        "loaded_at": time.time() if loaded_at is None else float(loaded_at),
    }


def apply_normalized_replay_bundle(
    *,
    restored_session: Any,
    normalized_bundle: Dict[str, Any],
    derive_status: Callable[[Any], str],
    to_json_safe: Callable[[Any], Any],
    utc_now_iso: Callable[[], str],
) -> None:
    """Apply normalized replay-bundle state to a newly created session."""
    engine = restored_session.engine
    snapshots = normalized_bundle.get("snapshots", [])
    turn_artifacts = normalized_bundle.get("turn_artifacts", [])
    engine.round_snapshots = snapshots if isinstance(snapshots, list) else []

    engine.turn_artifacts = (
        to_json_safe(turn_artifacts) if isinstance(turn_artifacts, list) else []
    )

    raw_snapshot = normalized_bundle.get("snapshot", {})

    if isinstance(raw_snapshot, dict):
        latest_turn_uid = str(raw_snapshot.get("latest_turn_uid", "")).strip()

        if latest_turn_uid:
            engine.latest_turn_uid = latest_turn_uid

    restore_round_idx = restore_round_index(normalized_bundle, engine.round_snapshots)
    restore_snapshot = getattr(engine, "restore_snapshot", None)

    if not callable(restore_snapshot) or not restore_snapshot(restore_round_idx):
        raise ValueError("Failed to restore engine state from frontend snapshot")

    if isinstance(raw_snapshot, dict):
        ready_raw = raw_snapshot.get("is_ready_for_adjudication")

        if ready_raw is not None:
            is_ready = bool(ready_raw) and not bool(
                getattr(engine, "is_finished", False)
            )
            engine.is_ready_for_adjudication = is_ready

            if isinstance(getattr(engine, "last_step_log", None), dict):
                convergence = engine.last_step_log.get("convergence")

                if not isinstance(convergence, dict):
                    convergence = {}

                convergence["is_converged"] = is_ready
                engine.last_step_log["convergence"] = convergence

    restored_events = normalize_restored_events(
        restored_session.session_id,
        normalized_bundle.get("events", []),
    )

    restored_session.events = restored_events
    restored_session.next_seq = len(restored_events) + 1

    if restored_events:
        last_turn_uid = str(restored_events[-1].get("turn_uid", "")).strip()
        restored_session.current_turn_uid = last_turn_uid
        restored_session.last_turn_uid = last_turn_uid

    else:
        latest_turn_uid = str(getattr(engine, "latest_turn_uid", "")).strip()
        restored_session.current_turn_uid = latest_turn_uid
        restored_session.last_turn_uid = latest_turn_uid

    session_meta = normalized_bundle.get("session", {})

    if not isinstance(session_meta, dict):
        session_meta = {}

    failure_simulation = session_meta.get("failure_simulation", {})

    if isinstance(failure_simulation, dict):
        restored_session.failure_simulation = {
            "es_unavailable": bool(failure_simulation.get("es_unavailable", False)),
            "llm_timeout": bool(failure_simulation.get("llm_timeout", False)),
        }

    restored_session.status = derive_status(engine)
    restored_session.last_error = ""
    restored_session.updated_at = utc_now_iso()
