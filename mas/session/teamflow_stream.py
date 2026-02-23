"""Transform turn artifacts and events into TeamFlow timeline payloads."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from mas.common.serialization import as_non_negative_int


def stringify_compact(value: Any) -> str:
    """Render nested values into compact multiline text.

    Args:
        value: Arbitrary scalar/container value.

    Returns:
        Human-readable multiline text representation. Empty/`None` values become
        an empty string.
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
            item_text = stringify_compact(item)

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
            item_text = stringify_compact(item)

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


def build_teamflow_message(
    *,
    to_json_safe: Callable[[Any], Any],
    message_id: str,
    phase: str,
    actor: str,
    role: str,
    title: str,
    content: str,
    ts_ms: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create one normalized TeamFlow message payload.

    Args:
        to_json_safe: Serializer for optional `meta` payload.
        message_id: Unique message identifier within one turn.
        phase: Processing phase label.
        actor: Actor display name.
        role: Actor role key used by frontend.
        title: Message title.
        content: Message body text.
        ts_ms: Optional millisecond timestamp.
        meta: Optional structured metadata payload.

    Returns:
        One normalized TeamFlow message dict.
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
        payload["meta"] = to_json_safe(meta)

    return payload


def build_teamflow_stream(
    *,
    artifacts: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    limit: int,
    to_json_safe: Callable[[Any], Any],
) -> List[Dict[str, Any]]:
    """Transform turn artifacts and events into TeamFlow timeline rows.

    Args:
        artifacts: Turn artifact rows emitted by the engine.
        events: Session event rows.
        limit: Maximum number of turns to return.
        to_json_safe: Serializer for structured payload fields.

    Returns:
        Ordered TeamFlow turn rows for frontend timeline rendering.
    """
    safe_limit = max(1, int(limit))

    if not isinstance(artifacts, list) or len(artifacts) == 0:
        return []

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

        round_idx = as_non_negative_int(
            item.get("round_idx", item.get("round")),
            artifact_index,
        )

        side = str(item.get("side", "unknown")).strip() or "unknown"
        related_events = events_by_turn.get(turn_uid, [])
        artifact_ts = as_non_negative_int(item.get("ts_ms"), 0)
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
            nonlocal message_seq
            message_seq += 1

            payload = build_teamflow_message(
                to_json_safe=to_json_safe,
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
        assessment_text = stringify_compact(assessment)

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
                content = stringify_compact(row.get("instruction", instruction_item))

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
            instruction_text = stringify_compact(instructions)

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
                    str(report.get("worker", report.get("target", "Worker"))).strip()
                    or "Worker"
                )

                content = (
                    stringify_compact(report.get("content", report_item))
                    or "（无内容）"
                )

                meta: Dict[str, Any] = {}
                status = str(report.get("status", "")).strip()

                if status:
                    meta["status"] = status

                max_score = report.get("max_score")

                if isinstance(max_score, (int, float)):
                    meta["max_score"] = max_score

                duration_ms = as_non_negative_int(report.get("duration_ms"), -1)

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
                content = stringify_compact(retry_item) or "执行失败，触发重试"

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

        action_cache = (
            item.get("action_cache", [])
            if isinstance(item.get("action_cache"), list)
            else []
        )

        plan_attempts_used = as_non_negative_int(item.get("plan_attempts_used"), 0)
        push_attempts_used = as_non_negative_int(item.get("push_attempts_used"), 0)
        decision_lines: List[str] = []

        if decision_raw:
            decision_lines.append(f"决策输出：{decision_raw}")

        if parsed_actions:
            action_lines = [stringify_compact(action) for action in parsed_actions[:5]]
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

        if plan_attempts_used > 0 or push_attempts_used > 0:
            decision_lines.append(
                f"Plan/PUSH 统计：plan={plan_attempts_used}, push={push_attempts_used}"
            )

        if action_cache:
            latest_cache = action_cache[-1]
            latest_text = stringify_compact(latest_cache)

            if latest_text:
                decision_lines.append(f"最新动作 Cache：{latest_text}")

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
            event_text = stringify_compact(event_data) or event_name
            event_ts = as_non_negative_int(event_item.get("ts_ms"), 0)

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
            as_non_negative_int(row.get("round_idx"), 0),
            str(row.get("turn_uid", "")),
        )
    )

    return to_json_safe(turns)[-safe_limit:]
