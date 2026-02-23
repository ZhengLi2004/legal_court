"""Memory and historical-case serializers."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from ..session_manager import DebateSession
from ._common import (
    _as_list,
    _extract_case_fields,
    _iter_graph_edges,
    _iter_graph_nodes,
    _normalize_case_summary,
    _normalize_insight_text,
    _serialize_shadow_graph,
    _source_rank,
)


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

        except (AttributeError, TypeError, ValueError, RuntimeError):
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


def memory_response(session: DebateSession) -> Dict[str, Any]:
    """Build a compact memory panel payload for frontend rendering."""
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

            except (AttributeError, TypeError, ValueError, RuntimeError):
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
                has_node = getattr(graph, "has_node", None)
                has_case = False

                if callable(has_node):
                    try:
                        has_case = bool(has_node(case_id))

                    except (TypeError, ValueError):
                        has_case = False

                if not has_case:
                    continue

                neighbors = []
                neighbors_fn = getattr(graph, "neighbors", None)

                if callable(neighbors_fn):
                    try:
                        neighbors = list(neighbors_fn(case_id))

                    except (KeyError, TypeError, ValueError):
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

            except (AttributeError, TypeError, ValueError, RuntimeError):
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
