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
    """Return the default bundled case JSONL path.

    Returns:
        Absolute path to `data/sampling/cleaned_samples.jsonl`.
    """
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "sampling"
        / "cleaned_samples.jsonl"
    )


def _default_frontend_snapshots_dir() -> Path:
    """Return the directory used for persisted frontend snapshot files.

    Returns:
        Absolute path to the frontend snapshot storage directory.
    """
    return Path(__file__).resolve().parents[2] / "tmp" / "frontend_snapshots"


def _to_json_safe(value: Any) -> Any:
    """Recursively convert values into JSON-serializable structures.

    Args:
        value: Arbitrary value, potentially containing enums, sets, or objects.

    Returns:
        JSON-friendly representation using only primitive containers/scalars.
    """
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
    """Build a default `DebateEngine` instance for API sessions.

    Returns:
        Ready-to-use debate engine instance.
    """
    from mas.config import SystemConfig
    from mas.core.engine import DebateEngine

    return DebateEngine(config=SystemConfig(), judge_config={})


def _load_case_from_jsonl(path: Path) -> Dict[str, Any]:
    """Load the first valid case row from a JSONL file.

    Args:
        path: JSONL file path to read.

    Returns:
        Parsed case dictionary.
    """
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
        frontend_snapshots_dir: Optional[Path] = None,
    ):
        """Initialize the session manager and storage locations.

        Args:
            engine_factory: Optional dependency-injected engine factory.
            default_case_path: Optional fallback case JSONL path.
            frontend_snapshots_dir: Optional snapshot persistence directory.
        """
        self._engine_factory = engine_factory or _default_engine_factory
        self._default_case_path = default_case_path or _default_case_path()
        self._frontend_snapshots_dir = (
            frontend_snapshots_dir or _default_frontend_snapshots_dir()
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
                session.status = self._derive_status(session.engine)
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
        """Delete a session and release underlying engine resources.

        Args:
            session_id: Session identifier.
        """
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
        """Register a queue subscriber for live event streaming.

        Args:
            session_id: Session identifier.
            max_queue_size: Queue capacity for backpressure handling.

        Returns:
            Newly created asyncio queue.
        """
        session = self.get_session(session_id)
        queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(max_queue_size)))
        session.event_subscribers.append(queue)
        return queue

    def unregister_event_subscriber(
        self, session_id: str, queue: asyncio.Queue
    ) -> None:
        """Remove a previously registered live-event subscriber queue.

        Args:
            session_id: Session identifier.
            queue: Queue instance to remove.
        """
        session = self.get_session(session_id)

        try:
            session.event_subscribers.remove(queue)

        except ValueError:
            return

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

    @staticmethod
    def _stringify_compact(value: Any) -> str:
        """Render nested values into compact multiline display text.

        Args:
            value: Arbitrary structured value.

        Returns:
            Human-readable compact string.
        """
        if isinstance(value, str):
            return value.strip()

        if isinstance(value, (int, float, bool)):
            return str(value)

        if value is None:
            return ""

        if isinstance(value, dict):
            lines: List[str] = []

            for key, item in value.items():
                item_text = SessionManager._stringify_compact(item)

                if not item_text:
                    continue

                item_lines = item_text.splitlines()

                if len(item_lines) == 1:
                    lines.append(f"{key}: {item_lines[0]}")
                else:
                    lines.append(f"{key}:")
                    lines.extend([f"  {line}" for line in item_lines])

            return "\n".join(lines)

        if isinstance(value, list):
            lines: List[str] = []

            for item in value:
                item_text = SessionManager._stringify_compact(item)

                if not item_text:
                    continue

                item_lines = item_text.splitlines()

                if len(item_lines) == 1:
                    lines.append(f"- {item_lines[0]}")
                else:
                    lines.append("-")
                    lines.extend([f"  {line}" for line in item_lines])

            return "\n".join(lines)

        return str(value)

    @staticmethod
    def _build_teamflow_message(
        message_id: str,
        phase: str,
        actor: str,
        role: str,
        title: str,
        content: str,
        ts_ms: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create one normalized TeamFlow timeline message payload.

        Args:
            message_id: Stable message identifier.
            phase: Pipeline phase (ASSESS, INSTRUCT, WORKER, etc.).
            actor: Display name of the message source.
            role: Actor role used by frontend styling.
            title: Short message title.
            content: Message body text.
            ts_ms: Optional event timestamp in milliseconds.
            meta: Optional auxiliary metadata.

        Returns:
            TeamFlow message dictionary ready for serialization.
        """
        payload: Dict[str, Any] = {
            "id": message_id,
            "phase": phase,
            "actor": actor,
            "role": role,
            "title": title,
            "content": content.strip() or "(empty)",
        }

        if isinstance(ts_ms, int) and ts_ms > 0:
            payload["ts_ms"] = ts_ms

        if isinstance(meta, dict) and len(meta) > 0:
            payload["meta"] = _to_json_safe(meta)

        return payload

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

        events_by_turn: Dict[str, List[Dict[str, Any]]] = {}

        for item in events:
            if not isinstance(item, dict):
                continue

            turn_uid = str(item.get("turn_uid", "")).strip()

            if not turn_uid:
                continue

            events_by_turn.setdefault(turn_uid, []).append(item)

        turns: List[Dict[str, Any]] = []

        for artifact_index, item in enumerate(artifacts):
            if not isinstance(item, dict):
                continue

            turn_uid = str(item.get("turn_uid", "")).strip()

            if not turn_uid:
                turn_uid = f"turn_{artifact_index + 1}"

            round_idx = self._as_non_negative_int(
                item.get("round_idx", item.get("round")),
                artifact_index,
            )

            side = str(item.get("side", "unknown")).strip() or "unknown"
            related_events = events_by_turn.get(turn_uid, [])
            artifact_ts = self._as_non_negative_int(item.get("ts_ms"), 0)
            message_seq = 0
            messages: List[Dict[str, Any]] = []

            def push_message(
                phase: str,
                actor: str,
                role: str,
                title: str,
                content: str,
                *,
                ts_override: Optional[int] = None,
                meta: Optional[Dict[str, Any]] = None,
            ) -> None:
                """Append one formatted message to the current turn bucket.

                Args:
                    phase: Pipeline phase label.
                    actor: Message source name.
                    role: Source role used by frontend rendering.
                    title: Short message title.
                    content: Message text body.
                    ts_override: Optional timestamp override.
                    meta: Optional metadata payload.
                """
                nonlocal message_seq
                message_seq += 1

                payload = self._build_teamflow_message(
                    message_id=f"{turn_uid}-{message_seq}",
                    phase=phase,
                    actor=actor,
                    role=role,
                    title=title,
                    content=content,
                    ts_ms=ts_override if ts_override is not None else artifact_ts,
                    meta=meta,
                )

                messages.append(payload)

            assessment = item.get("controller_assessment", {})
            assessment_text = self._stringify_compact(assessment)

            if assessment_text:
                meta = None

                if isinstance(assessment, dict):
                    meta = {"keys": len(assessment)}

                push_message(
                    "ASSESS",
                    "Controller",
                    "controller",
                    "需求评估",
                    assessment_text,
                    meta=meta,
                )

            instructions = item.get("batch_instructions", [])

            if isinstance(instructions, list):
                for idx, instruction_item in enumerate(instructions, start=1):
                    row = instruction_item if isinstance(instruction_item, dict) else {}
                    target = str(row.get("target", "")).strip() or "Worker"

                    content = self._stringify_compact(
                        row.get("instruction", instruction_item)
                    )

                    if not content:
                        continue

                    push_message(
                        "INSTRUCT",
                        "Controller",
                        "controller",
                        f"任务下发 #{idx}",
                        content,
                        meta={"target": target},
                    )

            else:
                instruction_text = self._stringify_compact(instructions)

                if instruction_text:
                    push_message(
                        "INSTRUCT",
                        "Controller",
                        "controller",
                        "任务下发",
                        instruction_text,
                    )

            worker_reports = item.get("worker_reports", [])

            if isinstance(worker_reports, list):
                for idx, report_item in enumerate(worker_reports, start=1):
                    report = report_item if isinstance(report_item, dict) else {}

                    worker = (
                        str(
                            report.get("worker", report.get("target", "Worker"))
                        ).strip()
                        or "Worker"
                    )

                    content = (
                        self._stringify_compact(report.get("content", report_item))
                        or "（无内容）"
                    )

                    meta: Dict[str, Any] = {}
                    status = str(report.get("status", "")).strip()

                    if status:
                        meta["status"] = status

                    max_score = report.get("max_score")

                    if isinstance(max_score, (int, float)):
                        meta["max_score"] = max_score

                    duration_ms = self._as_non_negative_int(
                        report.get("duration_ms"),
                        -1,
                    )

                    if duration_ms >= 0:
                        meta["duration_ms"] = duration_ms

                    push_message(
                        "WORKER",
                        worker,
                        "worker",
                        f"Worker反馈 #{idx}",
                        content,
                        meta=meta or None,
                    )

            retry_history = item.get("retry_history", [])
            retry_count = len(retry_history) if isinstance(retry_history, list) else 0

            if isinstance(retry_history, list):
                for idx, retry_item in enumerate(retry_history, start=1):
                    content = (
                        self._stringify_compact(retry_item) or "执行失败，触发重试"
                    )

                    push_message(
                        "RETRY",
                        "System",
                        "system",
                        f"重试 #{idx}",
                        content,
                    )

            decision_raw = str(item.get("decision_raw", "")).strip()

            parsed_actions = (
                item.get("parsed_actions", [])
                if isinstance(item.get("parsed_actions"), list)
                else []
            )

            execution_logs = str(item.get("execution_logs", "")).strip()
            decision_lines: List[str] = []

            if decision_raw:
                decision_lines.append(f"决策输出：{decision_raw}")

            if parsed_actions:
                action_lines = [
                    self._stringify_compact(action) for action in parsed_actions[:5]
                ]

                decision_lines.append("解析动作：")

                for action_text in action_lines:
                    if action_text:
                        decision_lines.append(f"- {action_text}")

                if len(parsed_actions) > 5:
                    decision_lines.append(f"... 共 {len(parsed_actions)} 项")

            if execution_logs:
                log_preview = "\n".join(execution_logs.splitlines()[:3]).strip()

                if log_preview:
                    decision_lines.append(f"执行日志：{log_preview}")

            has_decision = len(decision_lines) > 0

            if has_decision:
                push_message(
                    "DECIDE",
                    "Controller",
                    "controller",
                    "决策执行",
                    "\n".join(decision_lines),
                    meta={"actions": len(parsed_actions)},
                )

            narrative_text = str(item.get("narrative_polished", "")).strip()

            if narrative_text:
                push_message(
                    "NARRATE",
                    "Narrator",
                    "narrator",
                    "叙事总结",
                    narrative_text,
                )

            for event_item in related_events:
                if not isinstance(event_item, dict):
                    continue

                event_name = str(event_item.get("event", "")).strip()

                if event_name not in {"session_warning", "session_error"}:
                    continue

                event_data = event_item.get("data", {})
                event_text = self._stringify_compact(event_data) or event_name
                event_ts = self._as_non_negative_int(event_item.get("ts_ms"), 0)

                push_message(
                    "SYSTEM",
                    "System",
                    "system",
                    event_name,
                    event_text,
                    ts_override=event_ts if event_ts > 0 else None,
                )

            if not messages:
                push_message(
                    "SYSTEM",
                    "System",
                    "system",
                    "无可视化数据",
                    "本回合暂无可结构化协作消息。",
                )

            status = "partial"

            if has_decision:
                status = "retry" if retry_count > 0 else "done"

            turns.append(
                {
                    "turn_uid": turn_uid,
                    "round_idx": round_idx,
                    "side": side,
                    "status": status,
                    "retry_count": retry_count,
                    "worker_count": len(worker_reports)
                    if isinstance(worker_reports, list)
                    else 0,
                    "message_count": len(messages),
                    "messages": messages,
                }
            )

        turns.sort(
            key=lambda row: (
                self._as_non_negative_int(row.get("round_idx"), 0),
                str(row.get("turn_uid", "")),
            )
        )

        return _to_json_safe(turns)

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
        """Resolve graph payload used by export endpoints.

        Args:
            session: Target debate session.
            round_idx: Optional round index for historical export.

        Returns:
            Graph payload dictionary containing nodes and edges.
        """
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

    @staticmethod
    def _as_non_negative_int(value: Any, fallback: int = 0) -> int:
        """Parse a non-negative integer with fallback on invalid input.

        Args:
            value: Candidate numeric value.
            fallback: Value returned when parse fails or number is negative.

        Returns:
            Parsed non-negative integer or `fallback`.
        """
        try:
            parsed = int(value)
            return parsed if parsed >= 0 else fallback

        except (TypeError, ValueError):
            return fallback

    def _ensure_frontend_snapshots_dir(self) -> Path:
        """Ensure the frontend snapshot directory exists on disk.

        Returns:
            Existing or newly created snapshot directory path.
        """
        self._frontend_snapshots_dir.mkdir(parents=True, exist_ok=True)
        return self._frontend_snapshots_dir

    def _frontend_snapshot_path(self, snapshot_id: str) -> Path:
        """Build a sanitized filesystem path for one snapshot ID.

        Args:
            snapshot_id: User- or system-provided snapshot identifier.

        Returns:
            Target snapshot JSON file path.
        """
        safe_id = "".join(
            ch for ch in str(snapshot_id).strip() if ch.isalnum() or ch in {"-", "_"}
        )

        if not safe_id:
            raise ValueError("Invalid frontend snapshot id")

        return self._ensure_frontend_snapshots_dir() / f"{safe_id}.json"

    def _extract_replay_metadata(self, bundle: Dict[str, Any]) -> Dict[str, int]:
        """Derive replay counts from bundle content and metadata fallback.

        Args:
            bundle: Replay bundle payload.

        Returns:
            Dictionary with event/artifact/snapshot counts.
        """
        metadata_raw = bundle.get("metadata", {})

        if not isinstance(metadata_raw, dict):
            metadata_raw = {}

        events = bundle.get("events", [])
        artifacts = bundle.get("turn_artifacts", [])
        snapshots = bundle.get("snapshots", [])

        event_count = (
            len(events)
            if isinstance(events, list)
            else self._as_non_negative_int(metadata_raw.get("event_count"), 0)
        )

        artifact_count = (
            len(artifacts)
            if isinstance(artifacts, list)
            else self._as_non_negative_int(metadata_raw.get("artifact_count"), 0)
        )

        snapshot_count = (
            len(snapshots)
            if isinstance(snapshots, list)
            else self._as_non_negative_int(metadata_raw.get("snapshot_count"), 0)
        )

        return {
            "event_count": event_count,
            "artifact_count": artifact_count,
            "snapshot_count": snapshot_count,
        }

    def _frontend_snapshot_item(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw snapshot record into API list-item shape.

        Args:
            record: Persisted frontend snapshot record.

        Returns:
            Normalized snapshot metadata for list/detail endpoints.
        """
        metadata = record.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        event_count = self._as_non_negative_int(metadata.get("event_count"), 0)
        artifact_count = self._as_non_negative_int(metadata.get("artifact_count"), 0)
        snapshot_count = self._as_non_negative_int(metadata.get("snapshot_count"), 0)

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

    def _write_frontend_snapshot_record(self, record: Dict[str, Any]) -> None:
        """Persist a frontend snapshot record as JSON.

        Args:
            record: Snapshot record payload to persist.
        """
        target = self._frontend_snapshot_path(str(record.get("snapshot_id", "")))

        with target.open("w", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False, indent=2)

    def _read_frontend_snapshot_record(self, snapshot_id: str) -> Dict[str, Any]:
        """Load and validate one frontend snapshot record from disk.

        Args:
            snapshot_id: Snapshot identifier.

        Returns:
            Parsed snapshot record payload.
        """
        target = self._frontend_snapshot_path(snapshot_id)

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
            raise ValueError(
                f"Frontend snapshot payload must be an object: {snapshot_id}"
            )

        if not str(payload.get("snapshot_id", "")).strip():
            payload["snapshot_id"] = str(snapshot_id).strip()

        return payload

    def _build_snapshot_payload(self, session: DebateSession) -> Dict[str, Any]:
        """Build the API-facing snapshot payload for a session.

        Args:
            session: Source session.

        Returns:
            JSON-safe snapshot payload with compatibility fields.
        """
        base = session.engine.get_serializable_snapshot()
        graph_stats = base.get("graph_stats", {})

        if not isinstance(graph_stats, dict):
            graph_stats = {}

        payload = {
            **base,
            "session_id": session.session_id,
            "status": session.status,
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

        return _to_json_safe(payload)

    def _frontend_snapshot_load_response(
        self,
        record: Dict[str, Any],
        session: DebateSession,
    ) -> Dict[str, Any]:
        """Build response payload returned after loading a frontend snapshot.

        Args:
            record: Snapshot record that was loaded.
            session: Session restored from that snapshot.

        Returns:
            Combined payload for frontend state and backend session state.
        """
        return {
            "snapshot": self._frontend_snapshot_item(record),
            "frontend_state": _to_json_safe(record.get("frontend_state", {})),
            "session": {
                "session_id": session.session_id,
                "status": session.status,
                "current_round": int(getattr(session.engine, "round_idx", 0)),
                "updated_at": session.updated_at,
            },
            "snapshot_payload": self._build_snapshot_payload(session),
        }

    def _normalize_import_bundle(
        self,
        bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize imported replay bundle into canonical structure.

        Args:
            bundle: Import payload from client.

        Returns:
            JSON-safe normalized replay bundle.
        """
        candidate: Any = bundle
        replay_bundle = bundle.get("replay_bundle")

        if isinstance(replay_bundle, dict):
            candidate = replay_bundle

        if not isinstance(candidate, dict):
            raise ValueError("Imported bundle must be an object")

        snapshots = candidate.get("snapshots")

        if not isinstance(snapshots, list) or len(snapshots) == 0:
            raise ValueError("Imported bundle missing non-empty snapshots")

        return _to_json_safe(candidate)

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
            "metadata": self._extract_replay_metadata(bundle),
        }

        self._write_frontend_snapshot_record(record)
        return self._frontend_snapshot_item(record)

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
        normalized_bundle = self._normalize_import_bundle(bundle)
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
            "metadata": self._extract_replay_metadata(normalized_bundle),
        }

        self._write_frontend_snapshot_record(record)
        return self._frontend_snapshot_item(record)

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
        limit_value = max(1, int(limit))
        offset_value = max(0, int(offset))
        snapshot_dir = self._ensure_frontend_snapshots_dir()
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

                items.append(self._frontend_snapshot_item(payload))

            except Exception:
                continue

        total = len(items)
        paged = items[offset_value : offset_value + limit_value]
        return {"items": paged, "total": total}

    def _normalize_restored_events(
        self,
        session_id: str,
        events: Any,
    ) -> List[Dict[str, Any]]:
        """Normalize replay events into internal event-envelope format.

        Args:
            session_id: Restored session identifier.
            events: Raw event list from replay bundle.

        Returns:
            Validated and normalized event envelopes.
        """
        if not isinstance(events, list):
            return []

        rows: List[Dict[str, Any]] = []

        for idx, item in enumerate(events):
            if not isinstance(item, dict):
                continue

            ts_ms = self._as_non_negative_int(
                item.get("ts_ms"), int(time.time() * 1000)
            )

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
                data = {"value": _to_json_safe(data)}

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
                    "data": _to_json_safe(data),
                }
            )

        return rows

    @staticmethod
    def _restore_round_index(bundle: Dict[str, Any], snapshots: Any) -> int:
        """Choose the best snapshot index to restore from replay data.

        Args:
            bundle: Replay bundle containing top-level snapshot metadata.
            snapshots: Snapshot list available for restore.

        Returns:
            Snapshot index to restore.
        """
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

    async def load_frontend_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Restore a persisted frontend snapshot into a live session.

        Args:
            snapshot_id: Snapshot identifier to restore.

        Returns:
            Frontend bootstrap payload containing restored session details.
        """
        record = self._read_frontend_snapshot_record(snapshot_id)
        now_ts = time.time()
        cache_entry = self._recent_frontend_snapshot_loads.get(snapshot_id, {})
        recent_session_id = str(cache_entry.get("session_id", "")).strip()
        recent_loaded_at = float(cache_entry.get("loaded_at", 0.0) or 0.0)

        if recent_session_id and (now_ts - recent_loaded_at) <= 8.0:
            recent_session = self._sessions.get(recent_session_id)

            if recent_session is not None:
                recent_session.updated_at = utc_now_iso()
                return self._frontend_snapshot_load_response(record, recent_session)

        replay_bundle = record.get("replay_bundle")

        if not isinstance(replay_bundle, dict):
            raise ValueError("Frontend snapshot has invalid replay bundle")

        normalized_bundle = self._normalize_import_bundle(replay_bundle)
        restored_session = await self.create_session(auto_setup=True)
        engine = restored_session.engine
        snapshots = normalized_bundle.get("snapshots", [])
        turn_artifacts = normalized_bundle.get("turn_artifacts", [])
        engine.round_snapshots = snapshots if isinstance(snapshots, list) else []

        engine.turn_artifacts = (
            _to_json_safe(turn_artifacts) if isinstance(turn_artifacts, list) else []
        )

        raw_snapshot = normalized_bundle.get("snapshot", {})

        if isinstance(raw_snapshot, dict):
            latest_turn_uid = str(raw_snapshot.get("latest_turn_uid", "")).strip()

            if latest_turn_uid:
                engine.latest_turn_uid = latest_turn_uid

        restore_round_idx = self._restore_round_index(
            normalized_bundle,
            engine.round_snapshots,
        )

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

        restored_events = self._normalize_restored_events(
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

        restored_session.status = self._derive_status(engine)
        restored_session.last_error = ""
        restored_session.updated_at = utc_now_iso()

        self._recent_frontend_snapshot_loads[snapshot_id] = {
            "session_id": restored_session.session_id,
            "loaded_at": time.time(),
        }

        return self._frontend_snapshot_load_response(record, restored_session)

    def _make_keyframe(
        self,
        session_id: str,
        event: str,
        reason: str,
        round_idx: Optional[int],
        turn_uid: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build one demo keyframe record.

        Args:
            session_id: Session identifier.
            event: Keyframe event type.
            reason: Deterministic reason label for this keyframe.
            round_idx: Round index associated with the event.
            turn_uid: Turn UID associated with the event.
            extra: Optional extra payload.

        Returns:
            Keyframe dictionary.
        """
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
        """Append one event envelope and fan it out to live subscribers.

        Args:
            session_id: Session identifier.
            event: Event type name.
            source: Event source namespace.
            data: Optional structured event payload.
        """
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
        """Derive API session status from engine runtime fields.

        Args:
            engine: Debate engine instance.

        Returns:
            Session status string consumed by API responses.
        """
        if getattr(engine, "is_finished", False):
            return "FINISHED"

        if getattr(engine, "is_ready_for_adjudication", False):
            return "READY_FOR_ADJUDICATION"

        if getattr(engine, "round_idx", 0) > 0:
            return "DEBATING"

        if getattr(engine, "graph", None) is not None:
            return "SETUP_DONE"

        return "CREATED"
