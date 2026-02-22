"""Event stream helpers for session lifecycle and live updates."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional


def infer_event_source(event: str) -> str:
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


def derive_round_idx(event_payload: Dict[str, Any], engine: Any) -> Optional[int]:
    """Infer round index from payload first, then engine state."""
    candidates = [
        event_payload.get("round_idx"),
        event_payload.get("round"),
    ]

    for item in candidates:
        try:
            if item is None:
                continue

            return int(item)

        except (TypeError, ValueError):
            continue

    try:
        engine_round = getattr(engine, "round_idx", None)

        if engine_round is not None:
            return int(engine_round)

    except (TypeError, ValueError):
        return None

    return None


def get_event_history(
    events: List[Dict[str, Any]],
    limit: int = 100,
    from_seq: Optional[int] = None,
    to_seq: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return filtered event history for one session."""
    rows = events

    if from_seq is not None:
        rows = [item for item in rows if int(item.get("seq", 0)) >= from_seq]

    if to_seq is not None:
        rows = [item for item in rows if int(item.get("seq", 0)) <= to_seq]

    limit_value = max(1, int(limit))
    return rows[-limit_value:]


def register_event_subscriber(
    event_subscribers: List[asyncio.Queue],
    max_queue_size: int = 200,
) -> asyncio.Queue:
    """Register a queue subscriber for live event streaming."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(max_queue_size)))
    event_subscribers.append(queue)
    return queue


def unregister_event_subscriber(
    event_subscribers: List[asyncio.Queue],
    queue: asyncio.Queue,
) -> None:
    """Remove a previously registered live-event subscriber queue."""
    try:
        event_subscribers.remove(queue)

    except ValueError:
        return


def record_event(
    *,
    session: Any,
    session_id: str,
    event: str,
    source: str,
    data: Optional[Dict[str, Any]],
    to_json_safe: Callable[[Any], Any],
    utc_now_iso: Callable[[], str],
) -> None:
    """Append one event envelope and fan it out to live subscribers."""
    payload = to_json_safe(data or {})
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
    round_idx = derive_round_idx(payload, session.engine)
    event_id = f"{session_id}-{session.next_seq:06d}"

    envelope = {
        "event_id": event_id,
        "seq": session.next_seq,
        "ts_ms": int(time.time() * 1000),
        "session_id": session_id,
        "turn_uid": turn_uid,
        "round_idx": round_idx,
        "event": event,
        "source": source,
        "data": payload,
    }

    session.events.append(envelope)

    if event == "turn_complete":
        session.last_turn_uid = turn_uid

    stale_queues: List[asyncio.Queue] = []

    for queue in list(session.event_subscribers):
        try:
            queue.put_nowait(envelope)

        except asyncio.QueueFull:
            try:
                queue.get_nowait()
                queue.put_nowait(envelope)

            except Exception:
                stale_queues.append(queue)

        except Exception:
            stale_queues.append(queue)

    if stale_queues:
        session.event_subscribers = [
            item for item in session.event_subscribers if item not in stale_queues
        ]

    session.next_seq += 1
    session.updated_at = utc_now_iso()

