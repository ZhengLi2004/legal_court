"""Serialization helpers for API responses."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from .session_manager import DebateSession


def _as_list(value: Any) -> List[Any]:
    """Return `value` when it is a list, otherwise return an empty list.

    Args:
        value: Candidate value from a weakly-typed payload.

    Returns:
        A safe list object for downstream iteration.
    """
    return value if isinstance(value, list) else []


def _safe_graph_from_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a normalized `graph_data` payload from a snapshot dict.

    Args:
        snapshot: Serialized snapshot payload returned by the engine.

    Returns:
        A dictionary containing `nodes` and `edges` lists.
    """
    graph_data = snapshot.get("graph_data")

    if isinstance(graph_data, dict):
        nodes = _as_list(graph_data.get("nodes"))
        edges = _as_list(graph_data.get("edges"))
        return {"nodes": nodes, "edges": edges}

    return {"nodes": [], "edges": []}


def _safe_focus_node_ids(snapshot: Dict[str, Any]) -> List[str]:
    """Read and sanitize `focus_node_ids` from a snapshot payload.

    Args:
        snapshot: Serialized snapshot payload returned by the engine.

    Returns:
        A list of non-empty node ID strings.
    """
    rows = snapshot.get("focus_node_ids")

    if not isinstance(rows, list):
        return []

    result: List[str] = []

    for item in rows:
        node_id = str(item).strip()

        if node_id:
            result.append(node_id)

    return result


def _edge_identifier(edge: Dict[str, Any], idx: int) -> str:
    """Build a stable edge identifier for diff computation.

    Args:
        edge: Edge dictionary from `graph_data["edges"]`.
        idx: Fallback positional index when edge ID is missing.

    Returns:
        A deterministic identifier string for the edge.
    """
    edge_id = edge.get("id")

    if isinstance(edge_id, str) and edge_id:
        return edge_id

    source = edge.get("source", "")
    target = edge.get("target", "")
    edge_type = edge.get("type", "RELATION")
    return f"{source}->{target}:{edge_type}:{idx}"


def snapshot_response(session: DebateSession) -> Dict[str, Any]:
    """Build a session snapshot response with normalized metrics.

    Args:
        session: In-memory session wrapper around a debate engine.

    Returns:
        A JSON-serializable payload for snapshot endpoints.
    """
    base = session.engine.get_serializable_snapshot()
    graph_stats = base.get("graph_stats", {})

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

    return payload


def graph_response(
    session: DebateSession, round_idx: Optional[int] = None
) -> Dict[str, Any]:
    """Build graph payload for the latest state or a historical round.

    Args:
        session: In-memory session wrapper around a debate engine.
        round_idx: Optional historical round index.

    Returns:
        A graph response payload with nodes, edges, and focus node IDs.
    """
    if round_idx is None:
        snap = snapshot_response(session)
        graph_data = _safe_graph_from_snapshot(snap)
        current_round = int(snap.get("current_round", 0))

        return {
            "session_id": session.session_id,
            "round_idx": current_round,
            "graph_data": graph_data,
            "focus_node_ids": _safe_focus_node_ids(snap),
        }

    snapshots = _as_list(getattr(session.engine, "round_snapshots", []))

    if 0 <= round_idx < len(snapshots):
        row = snapshots[round_idx]
        graph_data = _safe_graph_from_snapshot(row)

        return {
            "session_id": session.session_id,
            "round_idx": int(row.get("round_idx", round_idx)),
            "graph_data": graph_data,
            "focus_node_ids": _safe_focus_node_ids(row),
        }

    latest = snapshot_response(session)

    return {
        "session_id": session.session_id,
        "round_idx": int(latest.get("current_round", 0)),
        "graph_data": _safe_graph_from_snapshot(latest),
        "focus_node_ids": _safe_focus_node_ids(latest),
    }


def _enum_value(raw: Any) -> str:
    """Return enum `.value` text when present, otherwise string text."""
    if hasattr(raw, "value"):
        return str(getattr(raw, "value"))

    return str(raw or "")


def _serialize_shadow_graph(shadow_graph: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Serialize one shadow graph object to API `graph_data` shape."""
    graph = getattr(shadow_graph, "graph", None)

    if graph is None:
        return {"nodes": [], "edges": []}

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    try:
        graph_nodes = list(graph.nodes(data=True))

    except Exception:
        graph_nodes = []

    for idx, row in enumerate(graph_nodes):
        node_id = str(row[0]).strip() if len(row) > 0 else f"node-{idx}"
        data = row[1] if len(row) > 1 and isinstance(row[1], dict) else {}
        content = str(data.get("content", "") or "").strip()
        node_type = _enum_value(data.get("type", "UNKNOWN")) or "UNKNOWN"
        status = _enum_value(data.get("status", ""))
        agent_id = str(data.get("agent_id", "") or "").strip()
        metadata = data.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "label": content or node_id,
                "content": content,
                "status": status,
                "agent_id": agent_id,
                "metadata": metadata,
            }
        )

    try:
        graph_edges = list(graph.edges(data=True))

    except Exception:
        graph_edges = []

    for idx, row in enumerate(graph_edges):
        source = str(row[0]).strip() if len(row) > 0 else ""
        target = str(row[1]).strip() if len(row) > 1 else ""
        data = row[2] if len(row) > 2 and isinstance(row[2], dict) else {}
        edge_type = _enum_value(data.get("type", "RELATION")) or "RELATION"
        edge_id = str(data.get("id", "") or "").strip()

        if not source or not target:
            continue

        edges.append(
            {
                "id": edge_id or f"{source}->{target}:{edge_type}:{idx}",
                "source": source,
                "target": target,
                "type": edge_type,
            }
        )

    return {"nodes": nodes, "edges": edges}


def memory_case_graph_response(session: DebateSession, case_id: str) -> Dict[str, Any]:
    """Build one historical-case graph payload for memory-case inspection."""
    normalized_case_id = str(case_id or "").strip()

    if not normalized_case_id:
        raise KeyError("Case ID is required.")

    engine = session.engine
    current_case_id = str(getattr(engine, "current_case_id", "") or "").strip()

    if normalized_case_id == current_case_id:
        current_graph_data = _serialize_shadow_graph(getattr(engine, "graph", None))

        if current_graph_data["nodes"] or current_graph_data["edges"]:
            return {
                "session_id": session.session_id,
                "case_id": normalized_case_id,
                "case_summary": _normalize_case_summary(
                    getattr(engine, "raw_facts", "")
                ),
                "round_idx": int(getattr(engine, "round_idx", 0) or 0),
                "graph_data": current_graph_data,
                "focus_node_ids": [],
            }

    legal_sys = getattr(engine, "legal_sys", None)
    memory = getattr(legal_sys, "memory", None)
    fetch_messages = getattr(memory, "_fetch_messages_by_ids", None)

    if callable(fetch_messages):
        try:
            candidates = _as_list(fetch_messages([normalized_case_id]))

        except Exception:
            candidates = []

        for row in candidates:
            row_case_id = str(
                getattr(row, "case_id", None) or getattr(row, "id", None) or ""
            ).strip()

            if row_case_id != normalized_case_id:
                continue

            graph_data = _serialize_shadow_graph(getattr(row, "shadow_graph", None))

            case_summary = _normalize_case_summary(
                getattr(row, "case_context", None)
                or getattr(row, "context", None)
                or getattr(row, "summary", None)
                or ""
            )

            return {
                "session_id": session.session_id,
                "case_id": normalized_case_id,
                "case_summary": case_summary or "（无摘要）",
                "round_idx": 0,
                "graph_data": graph_data,
                "focus_node_ids": [],
            }

    raise KeyError(f"Case graph not found: {normalized_case_id}")


def graph_diff_response(
    session: DebateSession, from_round: int, to_round: int
) -> Dict[str, Any]:
    """Compute structural and status differences between two rounds.

    Args:
        session: In-memory session wrapper around a debate engine.
        from_round: Source round index.
        to_round: Target round index.

    Returns:
        A payload listing added, removed, and changed node/edge IDs.
    """
    from_graph = graph_response(session, from_round)["graph_data"]
    to_graph = graph_response(session, to_round)["graph_data"]
    from_nodes = _as_list(from_graph.get("nodes"))
    to_nodes = _as_list(to_graph.get("nodes"))
    from_node_ids = {str(node.get("id", "")) for node in from_nodes}
    to_node_ids = {str(node.get("id", "")) for node in to_nodes}

    from_edge_ids = {
        _edge_identifier(edge, idx)
        for idx, edge in enumerate(_as_list(from_graph.get("edges")))
    }

    to_edge_ids = {
        _edge_identifier(edge, idx)
        for idx, edge in enumerate(_as_list(to_graph.get("edges")))
    }

    status_changed_node_ids: List[str] = []

    from_node_by_id = {
        str(node.get("id", "")): node
        for node in from_nodes
        if isinstance(node, dict) and node.get("id") is not None
    }

    for node in to_nodes:
        if not isinstance(node, dict):
            continue

        node_id = str(node.get("id", ""))
        prev = from_node_by_id.get(node_id)

        if not isinstance(prev, dict):
            continue

        if str(prev.get("status", "")) != str(node.get("status", "")):
            status_changed_node_ids.append(node_id)

    added_nodes = sorted([item for item in to_node_ids if item not in from_node_ids])
    removed_nodes = sorted([item for item in from_node_ids if item not in to_node_ids])
    added_edges = sorted([item for item in to_edge_ids if item not in from_edge_ids])
    removed_edges = sorted([item for item in from_edge_ids if item not in to_edge_ids])
    status_changed_node_ids = sorted(list(set(status_changed_node_ids)))

    return {
        "session_id": session.session_id,
        "from_round": from_round,
        "to_round": to_round,
        "added_node_ids": added_nodes,
        "removed_node_ids": removed_nodes,
        "added_edge_ids": added_edges,
        "removed_edge_ids": removed_edges,
        "status_changed_node_ids": status_changed_node_ids,
        "changed_node_ids": sorted(
            list(set(added_nodes + removed_nodes + status_changed_node_ids))
        ),
        "changed_edge_ids": sorted(list(set(added_edges + removed_edges))),
    }


def _normalize_case_summary(text: Any, max_len: int = 120) -> str:
    """Convert raw case context into compact single-line summary text."""
    value = " ".join(str(text or "").split()).strip()

    if not value:
        return ""

    if len(value) <= max_len:
        return value

    return f"{value[: max_len - 3]}..."


def _extract_case_fields(row: Any) -> Dict[str, str]:
    """Extract `case_id` and best-effort summary text from row/object."""
    if isinstance(row, dict):
        case_id = str(row.get("case_id") or row.get("id") or "").strip()

        summary = _normalize_case_summary(
            row.get("case_context") or row.get("context") or row.get("summary") or ""
        )

        return {"case_id": case_id, "summary": summary}

    case_id = str(
        getattr(row, "case_id", None) or getattr(row, "id", None) or ""
    ).strip()

    summary = _normalize_case_summary(
        getattr(row, "case_context", None)
        or getattr(row, "context", None)
        or getattr(row, "summary", None)
        or ""
    )

    return {"case_id": case_id, "summary": summary}


def _iter_graph_nodes(graph: Any) -> List[Any]:
    """Return graph nodes for networkx-like objects and simple stubs."""
    if graph is None:
        return []

    node_view = getattr(graph, "nodes", None)

    if callable(node_view):
        try:
            return list(node_view())

        except Exception:
            return []

    if isinstance(node_view, list):
        return list(node_view)

    try:
        return list(node_view) if node_view is not None else []

    except Exception:
        return []


def _iter_graph_edges(graph: Any) -> List[Any]:
    """Return graph edges for networkx-like objects and simple stubs."""
    if graph is None:
        return []

    edge_view = getattr(graph, "edges", None)

    if callable(edge_view):
        try:
            return list(edge_view(data=True))

        except TypeError:
            try:
                return list(edge_view())

            except Exception:
                return []

        except Exception:
            return []

    if isinstance(edge_view, list):
        return list(edge_view)

    try:
        return list(edge_view) if edge_view is not None else []

    except Exception:
        return []


def _source_rank(source: str) -> int:
    """Return deterministic ordering rank for related-case source tags."""
    order = {
        "representative": 0,
        "insight_case": 1,
        "topology_neighbor": 2,
        "static_retrieved": 3,
        "dynamic_retrieved": 4,
    }

    return order.get(str(source), 99)


def _normalize_insight_text(text: Any) -> str:
    """Normalize one insight text row and drop placeholder-only values."""
    value = " ".join(str(text or "").strip().split())

    if not value:
        return ""

    value = re.sub(
        r"^\s*SIDE\s*[:：]\s*(PLAINTIFF|DEFENDANT|COMMON)\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"^(PLAINTIFF|DEFENDANT|COMMON)\s*[:：\-]\s*",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()

    for prefix in ("CONTENT:", "content:", "Insight:", "insight:", "策略：", "策略:"):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()

    token = value.upper()

    if token in {"PLAINTIFF", "DEFENDANT", "COMMON", "INSIGHT", "策略", "N/A"}:
        return ""

    if len(value) <= 3:
        return ""

    return value


def memory_response(session: DebateSession) -> Dict[str, Any]:
    """Build a compact memory panel payload for frontend rendering.

    Args:
        session: In-memory session wrapper around a debate engine.

    Returns:
        A summary of insights, historical cases, and task-layer topology.
    """
    engine = session.engine
    legal_sys = getattr(engine, "legal_sys", None)
    insight_summaries: List[str] = []
    insight_items: List[Dict[str, Any]] = []
    representative_case_ids: List[str] = []
    case_catalog: Dict[str, Dict[str, str]] = {}
    retrieved_static_case_ids: List[str] = []
    retrieved_dynamic_case_ids: List[str] = []
    recalled_case_ids: List[str] = []
    recalled_case_count = 0
    task_layer_graph: Dict[str, List[Dict[str, Any]]] = {"nodes": [], "edges": []}

    if legal_sys is not None:
        referenced_case_ids: Set[str] = set()
        current_case_id = str(getattr(engine, "current_case_id", "") or "").strip()
        current_case_summary = _normalize_case_summary(getattr(engine, "raw_facts", ""))

        if current_case_id:
            case_catalog[current_case_id] = {
                "summary": current_case_summary or "当前案件",
            }

        def append_unique(rows: List[str], seen: Set[str], case_id: str):
            """Append non-empty case IDs while preserving input order."""
            if not case_id or case_id in seen:
                return

            seen.add(case_id)
            rows.append(case_id)

        def upsert_case_summary(case_id: str, summary: str):
            """Insert or update one case summary in case catalog."""
            if not case_id:
                return

            compact = _normalize_case_summary(summary)

            if case_id not in case_catalog:
                case_catalog[case_id] = {"summary": compact or "（无摘要）"}
                return

            existing = str(case_catalog[case_id].get("summary", "")).strip()

            if (not existing or existing == "（无摘要）") and compact:
                case_catalog[case_id]["summary"] = compact

        insights_manager = getattr(legal_sys, "insights", None)
        raw_insights = getattr(insights_manager, "insights", [])

        for item in _as_list(raw_insights):
            content = getattr(item, "content", None)

            if isinstance(content, str):
                clean_content = _normalize_insight_text(content)

                if not clean_content:
                    continue

                insight_summaries.append(clean_content)
                side_value = getattr(item, "side", "COMMON")
                side_text = getattr(side_value, "value", side_value)
                raw_cases = _as_list(getattr(item, "cases", []))

                cases = sorted(
                    {
                        str(case_id).strip()
                        for case_id in raw_cases
                        if str(case_id).strip()
                    }
                )

                raw_representatives = _as_list(getattr(item, "representatives", []))

                representatives = sorted(
                    {
                        str(case_id).strip()
                        for case_id in raw_representatives
                        if str(case_id).strip()
                    }
                )

                if not representatives:
                    representatives = list(cases)

                representative_case_ids.extend(representatives)
                source_map: Dict[str, Set[str]] = {}

                for case_id in cases:
                    source_map.setdefault(case_id, set()).add("insight_case")
                    referenced_case_ids.add(case_id)

                for case_id in representatives:
                    source_map.setdefault(case_id, set()).add("representative")
                    referenced_case_ids.add(case_id)

                insight_items.append(
                    {
                        "content": clean_content,
                        "side": str(side_text or "COMMON"),
                        "cases": cases,
                        "representatives": representatives,
                        "case_count": len(cases),
                        "representative_count": len(representatives),
                        "linked_round": (
                            int(getattr(engine, "round_idx", 0) or 0)
                            if current_case_id and current_case_id in cases
                            else 0
                        ),
                        "_source_map": source_map,
                    }
                )

        static_seen: Set[str] = set()
        dynamic_seen: Set[str] = set()
        static_rows = _as_list(getattr(legal_sys, "_static_history_cases", []))
        dynamic_rows = _as_list(getattr(legal_sys, "_dynamic_law_cases", []))

        for row in static_rows:
            case_fields = _extract_case_fields(row)
            case_id = case_fields["case_id"]

            if not case_id:
                continue

            append_unique(retrieved_static_case_ids, static_seen, case_id)
            upsert_case_summary(case_id, case_fields["summary"])
            referenced_case_ids.add(case_id)

        for row in dynamic_rows:
            case_fields = _extract_case_fields(row)
            case_id = case_fields["case_id"]

            if not case_id:
                continue

            append_unique(retrieved_dynamic_case_ids, dynamic_seen, case_id)
            upsert_case_summary(case_id, case_fields["summary"])
            referenced_case_ids.add(case_id)

        active_case_ids = set(retrieved_static_case_ids + retrieved_dynamic_case_ids)

        if current_case_id:
            active_case_ids.add(current_case_id)

        relevant_insight_contents: Set[str] = set()

        get_relevant_insights = getattr(
            insights_manager, "get_relevant_insights_by_side", None
        )

        if callable(get_relevant_insights):
            try:
                query_context = current_case_summary or str(
                    getattr(engine, "raw_facts", "") or ""
                )

                top_k = max(3, min(len(insight_items), 6))
                p_rows, d_rows = get_relevant_insights(query_context, top_k=top_k)

                for row in _as_list(p_rows) + _as_list(d_rows):
                    normalized = _normalize_insight_text(row)

                    if normalized:
                        relevant_insight_contents.add(normalized)

            except Exception:
                relevant_insight_contents = set()

        filtered_insight_items: List[Dict[str, Any]] = []

        if insight_items and (active_case_ids or relevant_insight_contents):
            for item in insight_items:
                content = str(item.get("content", "")).strip()
                case_ids = set(_as_list(item.get("cases")))
                case_ids.update(_as_list(item.get("representatives")))

                is_relevant = bool(case_ids.intersection(active_case_ids)) or (
                    content in relevant_insight_contents
                )

                if is_relevant:
                    filtered_insight_items.append(item)

        if filtered_insight_items:
            insight_items = filtered_insight_items

        insight_summaries = [
            str(item.get("content", "")).strip()
            for item in insight_items
            if str(item.get("content", "")).strip()
        ]

        representative_case_ids = []

        for item in insight_items:
            representative_case_ids.extend(_as_list(item.get("representatives")))

        merged_representative_case_ids: List[str] = []
        representative_seen: Set[str] = set()

        for case_id in (
            retrieved_static_case_ids
            + retrieved_dynamic_case_ids
            + representative_case_ids
        ):
            normalized_id = str(case_id).strip()

            if not normalized_id or normalized_id in representative_seen:
                continue

            representative_seen.add(normalized_id)
            merged_representative_case_ids.append(normalized_id)

        representative_case_ids = merged_representative_case_ids
        recalled_case_ids = list(representative_case_ids)
        recalled_case_count = len(recalled_case_ids)
        memory = getattr(legal_sys, "memory", None)
        task_layer = getattr(memory, "task_layer", None)
        graph = getattr(task_layer, "graph", None)
        graph_nodes = _iter_graph_nodes(graph)
        graph_edges = _iter_graph_edges(graph)

        for node in graph_nodes:
            node_id = str(node).strip()

            if node_id:
                referenced_case_ids.add(node_id)

        static_set = set(retrieved_static_case_ids)
        dynamic_set = set(retrieved_dynamic_case_ids)

        for item in insight_items:
            source_map = item.get("_source_map", {})

            if not isinstance(source_map, dict):
                continue

            seed_ids = list(source_map.keys())

            for case_id in seed_ids:
                if not case_id:
                    continue

                if case_id in static_set:
                    source_map.setdefault(case_id, set()).add("static_retrieved")

                if case_id in dynamic_set:
                    source_map.setdefault(case_id, set()).add("dynamic_retrieved")

                referenced_case_ids.add(case_id)

                try:
                    has_case = bool(graph is not None and graph.has_node(case_id))

                except Exception:
                    has_case = False

                if not has_case:
                    continue

                try:
                    neighbors = list(graph.neighbors(case_id))

                except Exception:
                    neighbors = []

                for neighbor in neighbors:
                    neighbor_id = str(neighbor).strip()

                    if not neighbor_id:
                        continue

                    source_map.setdefault(neighbor_id, set()).add("topology_neighbor")
                    referenced_case_ids.add(neighbor_id)

        fetch_messages = getattr(memory, "_fetch_messages_by_ids", None)

        if callable(fetch_messages):
            missing_case_ids = [
                case_id
                for case_id in sorted(referenced_case_ids)
                if case_id
                and (
                    case_id not in case_catalog
                    or case_catalog[case_id].get("summary") == "（无摘要）"
                )
            ]

            try:
                fetched_rows = fetch_messages(missing_case_ids)

            except Exception:
                fetched_rows = []

            for row in _as_list(fetched_rows):
                case_fields = _extract_case_fields(row)
                upsert_case_summary(case_fields["case_id"], case_fields["summary"])

        for item in insight_items:
            source_map = item.get("_source_map", {})
            related_cases: List[Dict[str, Any]] = []

            if isinstance(source_map, dict):
                for case_id, sources in source_map.items():
                    if not case_id:
                        continue

                    source_rows = sorted(set(sources), key=_source_rank)
                    summary = case_catalog.get(case_id, {}).get("summary", "（无摘要）")

                    related_cases.append(
                        {
                            "case_id": case_id,
                            "summary": summary or "（无摘要）",
                            "sources": source_rows,
                        }
                    )

            related_cases.sort(
                key=lambda row: (
                    min(
                        [_source_rank(s) for s in _as_list(row.get("sources"))] or [99]
                    ),
                    str(row.get("case_id", "")),
                )
            )

            item["related_cases"] = related_cases
            item["related_case_count"] = len(related_cases)
            item.pop("_source_map", None)

        task_nodes: Dict[str, Dict[str, Any]] = {}
        task_edges: List[Dict[str, Any]] = []
        edge_seen: Set[str] = set()

        node_kind_priority = {
            "topology_case": 0,
            "static_retrieved": 1,
            "dynamic_retrieved": 2,
            "current": 3,
        }

        def upsert_task_node(case_id: str, kind: str):
            """Insert one task-layer node while preserving the strongest kind tag."""
            if not case_id:
                return

            summary = case_catalog.get(case_id, {}).get("summary", "（无摘要）")
            current = task_nodes.get(case_id)

            if current is None:
                task_nodes[case_id] = {"id": case_id, "label": summary, "kind": kind}
                return

            old_kind = str(current.get("kind", "topology_case"))

            if node_kind_priority.get(kind, -1) > node_kind_priority.get(old_kind, -1):
                current["kind"] = kind

            old_label = str(current.get("label", "")).strip()

            if (not old_label or old_label == "（无摘要）") and summary:
                current["label"] = summary

        def add_task_edge(source: str, target: str, edge_type: str):
            """Append one deduplicated task-layer edge."""
            if not source or not target or source == target:
                return

            pair_key = "||".join(sorted([source, target]))
            edge_key = f"{pair_key}:{edge_type}"

            if edge_key in edge_seen:
                return

            edge_seen.add(edge_key)
            edge_id = f"{source}->{target}:{edge_type}:{len(task_edges)}"

            task_edges.append(
                {
                    "id": edge_id,
                    "source": source,
                    "target": target,
                    "type": edge_type,
                }
            )

        for node in graph_nodes:
            node_id = str(node).strip()

            if node_id:
                upsert_task_node(node_id, "topology_case")

        for edge in graph_edges:
            source = str(edge[0]).strip() if len(edge) > 0 else ""
            target = str(edge[1]).strip() if len(edge) > 1 else ""
            edge_data = edge[2] if len(edge) > 2 and isinstance(edge[2], dict) else {}
            edge_type = str(edge_data.get("type", "reference")).strip() or "reference"
            add_task_edge(source, target, edge_type)

        for case_id in retrieved_static_case_ids:
            upsert_task_node(case_id, "static_retrieved")

        for case_id in retrieved_dynamic_case_ids:
            upsert_task_node(case_id, "dynamic_retrieved")

        if current_case_id:
            upsert_task_node(current_case_id, "current")

            for case_id in retrieved_static_case_ids:
                add_task_edge(current_case_id, case_id, "retrieved_static")

            for case_id in retrieved_dynamic_case_ids:
                add_task_edge(current_case_id, case_id, "retrieved_dynamic")

        task_layer_graph = {
            "nodes": list(task_nodes.values()),
            "edges": task_edges,
        }

    insight_items = sorted(
        insight_items,
        key=lambda item: (
            -int(item.get("case_count", 0)),
            str(item.get("content", "")),
        ),
    )

    return {
        "session_id": session.session_id,
        "insight_summaries": insight_summaries,
        "insight_items": insight_items,
        "representative_case_ids": representative_case_ids,
        "case_catalog": case_catalog,
        "recalled_case_ids": recalled_case_ids,
        "recalled_case_count": recalled_case_count,
        "task_layer_graph": task_layer_graph,
    }
