"""Session lifecycle manager for the lightweight API layer."""

from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import networkx as nx


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


def _derive_round_idx(
    event_payload: Dict[str, Any],
    engine: Any,
) -> Optional[int]:
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
    event_subscribers: List[asyncio.Queue] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    demo_keyframes: List[Dict[str, Any]] = field(default_factory=list)

    failure_simulation: Dict[str, bool] = field(
        default_factory=lambda: {"es_unavailable": False, "llm_timeout": False}
    )


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

    def register_event_subscriber(
        self, session_id: str, max_queue_size: int = 200
    ) -> asyncio.Queue:
        session = self.get_session(session_id)
        queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(max_queue_size)))
        session.event_subscribers.append(queue)
        return queue

    def unregister_event_subscriber(
        self, session_id: str, queue: asyncio.Queue
    ) -> None:
        session = self.get_session(session_id)

        try:
            session.event_subscribers.remove(queue)

        except ValueError:
            return

    def get_snapshot_index(self, session_id: str) -> List[Dict[str, Any]]:
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
        session = self.get_session(session_id)
        getter = getattr(session.engine, "get_turn_artifacts", None)

        if callable(getter):
            return getter(turn_uid=turn_uid, limit=limit)

        return []

    def set_failure_simulation(
        self, session_id: str, kind: str, enabled: bool
    ) -> Dict[str, Any]:
        """Enable or disable per-session failure simulation flags."""
        session = self.get_session(session_id)
        key = str(kind).strip().lower()

        if key not in {"es_unavailable", "llm_timeout"}:
            raise ValueError(f"Unsupported failure kind: {kind}")

        session.failure_simulation[key] = bool(enabled)
        session.updated_at = utc_now_iso()

        self._record_event(
            session_id=session_id,
            event="failure_simulation_set",
            source="api",
            data={"kind": key, "enabled": bool(enabled)},
        )

        return {
            "session_id": session_id,
            "failure_simulation": dict(session.failure_simulation),
            "updated_at": session.updated_at,
        }

    def _graph_data_for_export(
        self, session: DebateSession, round_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        snapshots = getattr(session.engine, "round_snapshots", [])

        if isinstance(round_idx, int) and isinstance(snapshots, list):
            if 0 <= round_idx < len(snapshots):
                row = snapshots[round_idx]

                if isinstance(row, dict):
                    graph_data = row.get("graph_data", {})

                    if isinstance(graph_data, dict):
                        return graph_data

        getter = getattr(session.engine, "get_serializable_snapshot", None)

        if callable(getter):
            payload = getter()

            if isinstance(payload, dict):
                graph_data = payload.get("graph_data", {})

                if isinstance(graph_data, dict):
                    return graph_data

        return {"nodes": [], "edges": []}

    def export_graph_gexf(
        self, session_id: str, round_idx: Optional[int] = None
    ) -> bytes:
        """Export current or historical graph as GEXF bytes."""
        session = self.get_session(session_id)
        graph_data = self._graph_data_for_export(session, round_idx=round_idx)
        graph = nx.DiGraph()

        for node in graph_data.get("nodes", []):
            if not isinstance(node, dict):
                continue

            node_id = str(node.get("id", "")).strip()

            if not node_id:
                continue

            attrs: Dict[str, Any] = {}

            for key, value in node.items():
                if key == "id":
                    continue

                if isinstance(value, (str, int, float, bool)) or value is None:
                    attrs[str(key)] = value

                else:
                    attrs[str(key)] = json.dumps(
                        _to_json_safe(value), ensure_ascii=False
                    )

            graph.add_node(node_id, **attrs)

        for edge in graph_data.get("edges", []):
            if not isinstance(edge, dict):
                continue

            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()

            if not source or not target:
                continue

            attrs = {}

            for key, value in edge.items():
                if key in {"source", "target"}:
                    continue

                if isinstance(value, (str, int, float, bool)) or value is None:
                    attrs[str(key)] = value

                else:
                    attrs[str(key)] = json.dumps(
                        _to_json_safe(value), ensure_ascii=False
                    )

            graph.add_edge(source, target, **attrs)

        buffer = io.BytesIO()
        nx.write_gexf(graph, buffer, encoding="utf-8")
        return buffer.getvalue()

    def export_replay_bundle(
        self,
        session_id: str,
        include_events_limit: int = 5000,
        include_artifacts_limit: int = 5000,
    ) -> Dict[str, Any]:
        """Export a complete replay bundle for offline analysis."""
        session = self.get_session(session_id)
        getter = getattr(session.engine, "get_serializable_snapshot", None)
        snapshot_payload = getter() if callable(getter) else {}

        if not isinstance(snapshot_payload, dict):
            snapshot_payload = {}

        snapshots = getattr(session.engine, "round_snapshots", [])

        if not isinstance(snapshots, list):
            snapshots = []

        events = self.get_event_history(
            session_id=session_id,
            limit=max(1, int(include_events_limit)),
        )

        artifacts = self.get_turn_artifacts(
            session_id=session_id,
            turn_uid=None,
            limit=max(1, int(include_artifacts_limit)),
        )

        return {
            "session": {
                "session_id": session.session_id,
                "status": session.status,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "failure_simulation": dict(session.failure_simulation),
            },
            "snapshot": snapshot_payload,
            "snapshot_index": self.get_snapshot_index(session_id),
            "snapshots": _to_json_safe(snapshots),
            "events": events,
            "turn_artifacts": _to_json_safe(artifacts),
            "debug_bundle": self.build_debug_bundle(
                session_id=session_id,
                event_limit=50,
                include_snapshot=True,
                include_artifact=True,
            ),
            "metadata": {
                "generated_at": utc_now_iso(),
                "event_count": len(events),
                "artifact_count": len(artifacts),
                "snapshot_count": len(snapshots),
            },
        }

    def _make_keyframe(
        self,
        session_id: str,
        event: str,
        reason: str,
        round_idx: Optional[int],
        turn_uid: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "session_id": session_id,
            "event": event,
            "reason": reason,
            "round_idx": int(round_idx or 0),
            "turn_uid": turn_uid,
            "ts_ms": int(time.time() * 1000),
        }

        if extra:
            payload["data"] = _to_json_safe(extra)

        return payload

    async def run_demo_session(
        self,
        session_id: str,
        max_steps: int = 20,
        auto_adjudicate: bool = True,
        capture_keyframes: bool = True,
    ) -> Dict[str, Any]:
        """Run a deterministic demo loop and capture keyframes."""
        session = self.get_session(session_id)

        if session.status == "CREATED":
            await self.setup_session(session_id)

        keyframes: List[Dict[str, Any]] = []
        seen_reasons: set[str] = set()
        first_conflict_seen = False
        first_retry_seen = False
        steps_executed = 0
        ended_by = "max_steps"

        if capture_keyframes:
            keyframes.append(
                self._make_keyframe(
                    session_id=session_id,
                    event="demo_start",
                    reason="demo_start",
                    round_idx=getattr(session.engine, "round_idx", 0),
                    turn_uid=session.last_turn_uid,
                )
            )

        for _ in range(max(1, int(max_steps))):
            engine = session.engine

            if getattr(engine, "is_finished", False):
                ended_by = "finished"
                break

            if getattr(engine, "is_ready_for_adjudication", False):
                if capture_keyframes and "ready_for_adjudication" not in seen_reasons:
                    keyframes.append(
                        self._make_keyframe(
                            session_id=session_id,
                            event="adjudication_ready",
                            reason="ready_for_adjudication",
                            round_idx=getattr(engine, "round_idx", 0),
                            turn_uid=session.last_turn_uid,
                        )
                    )

                    seen_reasons.add("ready_for_adjudication")

                if auto_adjudicate:
                    await self.adjudicate_session(session_id)
                    steps_executed += 1
                    ended_by = "adjudicated"

                    if capture_keyframes:
                        keyframes.append(
                            self._make_keyframe(
                                session_id=session_id,
                                event="adjudication_complete",
                                reason="adjudication_complete",
                                round_idx=getattr(session.engine, "round_idx", 0),
                                turn_uid=session.last_turn_uid,
                            )
                        )

                    break

                ended_by = "ready_for_adjudication"
                break

            await self.step_session(session_id)
            steps_executed += 1
            engine = session.engine
            current_round = int(getattr(engine, "round_idx", 0))
            turn_uid = session.last_turn_uid

            if (
                capture_keyframes
                and current_round == 1
                and "first_round" not in seen_reasons
            ):
                keyframes.append(
                    self._make_keyframe(
                        session_id=session_id,
                        event="turn_complete",
                        reason="first_round",
                        round_idx=current_round,
                        turn_uid=turn_uid,
                    )
                )

                seen_reasons.add("first_round")

            snapshot_getter = getattr(engine, "get_serializable_snapshot", None)
            graph_stats = {}

            if callable(snapshot_getter):
                snap = snapshot_getter()

                if isinstance(snap, dict):
                    row = snap.get("graph_stats", {})

                    if isinstance(row, dict):
                        graph_stats = row

            conflict_count = int(graph_stats.get("edge_conflict_count", 0) or 0)

            if capture_keyframes and conflict_count > 0 and not first_conflict_seen:
                keyframes.append(
                    self._make_keyframe(
                        session_id=session_id,
                        event="turn_complete",
                        reason="first_conflict_edge",
                        round_idx=current_round,
                        turn_uid=turn_uid,
                        extra={"conflict_count": conflict_count},
                    )
                )

                first_conflict_seen = True

            artifacts = self.get_turn_artifacts(session_id, limit=1)

            if (
                capture_keyframes
                and artifacts
                and not first_retry_seen
                and isinstance(artifacts[-1], dict)
            ):
                retry_history = artifacts[-1].get("retry_history", [])

                if isinstance(retry_history, list) and len(retry_history) > 0:
                    keyframes.append(
                        self._make_keyframe(
                            session_id=session_id,
                            event="turn_complete",
                            reason="first_retry",
                            round_idx=current_round,
                            turn_uid=turn_uid,
                            extra={"retry_count": len(retry_history)},
                        )
                    )

                    first_retry_seen = True

        session.demo_keyframes = keyframes
        session.updated_at = utc_now_iso()

        return {
            "session": {
                "session_id": session_id,
                "status": session.status,
                "updated_at": session.updated_at,
            },
            "keyframes": keyframes,
            "demo_summary": {
                "steps_executed": steps_executed,
                "ended_by": ended_by,
            },
            "snapshot": getattr(
                session.engine, "get_serializable_snapshot", lambda: {}
            )(),
        }

    def get_demo_keyframes(self, session_id: str) -> List[Dict[str, Any]]:
        """Return stored demo keyframes."""
        session = self.get_session(session_id)
        return _to_json_safe(session.demo_keyframes)

    def build_debug_bundle(
        self,
        session_id: str,
        event_limit: int = 20,
        include_snapshot: bool = True,
        include_artifact: bool = True,
    ) -> Dict[str, Any]:
        """Build a compact debug bundle for issue reporting and replay."""
        session = self.get_session(session_id)

        event_rows = self.get_event_history(
            session_id=session_id,
            limit=max(1, int(event_limit)),
        )

        snapshot: Dict[str, Any] = {}

        if include_snapshot:
            getter = getattr(session.engine, "get_serializable_snapshot", None)

            if callable(getter):
                maybe_snapshot = getter()

                if isinstance(maybe_snapshot, dict):
                    snapshot = maybe_snapshot

        round_idx = snapshot.get("current_round")

        if round_idx is None:
            try:
                round_idx = int(getattr(session.engine, "round_idx", 0))

            except Exception:
                round_idx = 0

        graph_stats = snapshot.get("graph_stats", {})

        if not isinstance(graph_stats, dict):
            graph_stats = {}

        turn_uid = str(
            snapshot.get("latest_turn_uid")
            or session.last_turn_uid
            or session.current_turn_uid
            or ""
        )

        latest_turn_artifact: Optional[Dict[str, Any]] = None

        if include_artifact:
            artifact_rows = self.get_turn_artifacts(
                session_id=session_id,
                turn_uid=turn_uid if turn_uid else None,
                limit=1,
            )

            if artifact_rows:
                latest_turn_artifact = artifact_rows[-1]

        return {
            "session_id": session.session_id,
            "round_idx": int(round_idx or 0),
            "turn_uid": turn_uid,
            "status": session.status,
            "last_error": session.last_error,
            "snapshot_summary": {
                "phase": session.status,
                "node_count": int(graph_stats.get("node_count", 0)),
                "edge_count": int(graph_stats.get("edge_count", 0)),
                "claim_count": int(graph_stats.get("claim_nodes", 0)),
                "conflict_count": int(graph_stats.get("edge_conflict_count", 0)),
            },
            "recent_events": event_rows,
            "latest_turn_artifact": latest_turn_artifact,
            "generated_at": utc_now_iso(),
        }

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

        round_idx = _derive_round_idx(payload, session.engine)
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
