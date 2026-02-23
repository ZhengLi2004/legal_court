"""Setup flow for `DebateEngine`."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from metagpt.logs import logger

from tools.fact_es_tool import FactEsTool
from tools.graph_tool import GraphTool
from tools.initializer import CaseInitializer
from tools.law_es_tool import LawEsTool
from tools.llm import GPTChat

from ..agents.narrator import GraphNarrator
from ..agents.team import DebateTeam
from ..core.graph import NodeType
from ..infrastructure import build_legal_system


async def run_engine_setup(
    engine: Any,
    *,
    case_data_path: Optional[str] = None,
    case_data: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> None:
    """Initialize the engine for a new case."""
    logger.info(">>> [Engine] Setting up...")
    engine._notify_state_change("setup_start", None)
    engine._is_running = True

    try:
        agent_llm = GPTChat(model_name=engine.cfg.llm.model_name)
        engine.narrator = GraphNarrator(llm=agent_llm)
        engine.legal_sys = build_legal_system(engine.cfg)

        engine.fact_es = FactEsTool(
            es_host=engine.cfg.es.host,
            embedding_func=engine.legal_sys.ef,
        )

        engine.law_es = LawEsTool(
            es_host=engine.cfg.es.host,
            embedding_func=engine.legal_sys.ef,
        )

        graph_tool = GraphTool(legal_system=engine.legal_sys, llm=agent_llm)
        resolved_case_data = case_data

        if resolved_case_data is None and case_data_path:
            with open(case_data_path, "r", encoding="utf-8") as file:
                resolved_case_data = json.loads(file.readline())

        if resolved_case_data is None:
            raise ValueError("Either case_data_path or case_data must be provided.")

        engine.raw_facts = resolved_case_data.get("fact_finding", "")
        engine.current_case_id = engine._resolve_case_id(resolved_case_data)
        engine.turn_artifacts = []
        engine.latest_turn_uid = ""
        cause = resolved_case_data.get("cause", ["未知案由"])

        if isinstance(cause, list):
            cause = cause[0] if cause else "未知案由"

        initializer = CaseInitializer(agent_llm)
        init_res = await initializer.initialize(engine.raw_facts, cause)

        engine.graph, (p_insights_list, d_insights_list) = engine.legal_sys.new_case(
            engine.raw_facts
        )

        p_insights_str = "\n".join([f"- {item}" for item in p_insights_list])
        d_insights_str = "\n".join([f"- {item}" for item in d_insights_list])
        engine.legal_sys.step_counter = 0
        engine.prev_stats["claim_nodes"] = engine._count_claim_nodes()
        engine.prev_stats["conflict_edges"] = 0
        graph_tool.set_current_graph(engine.graph)
        fact_count = 0
        fact_ids = []

        for fact_statement in init_res.fact_statements:
            node_id, is_new = engine.graph.add_node(
                content=fact_statement,
                node_type=NodeType.FACT,
                agent_id="System_Init",
                metadata={"is_objective_fact": True},
            )

            if is_new:
                fact_count += 1
                fact_ids.append(node_id)

        if fact_ids:
            engine.graph.touch_nodes(fact_ids, step_index=0)

        claim_ids = []

        for claim_statement in init_res.root_claim_actions:
            node_id, is_new = engine.graph.add_node(
                content=claim_statement,
                node_type=NodeType.CLAIM,
                agent_id="System_Init",
                metadata={"is_root_claim": True},
            )

            if is_new:
                claim_ids.append(node_id)

        if claim_ids:
            engine.graph.touch_nodes(claim_ids, step_index=0)

        engine.prev_stats["claim_nodes"] = engine._count_claim_nodes()
        engine.graph.refresh_context(current_step=0)

        engine.p_team = DebateTeam(
            "plaintiff",
            init_res.plaintiff_persona,
            graph_tool,
            engine.fact_es,
            engine.law_es,
            agent_llm,
            engine.legal_sys,
            insights=p_insights_str,
            verbose=verbose,
        )

        engine.p_team.on_state_change = engine._handle_team_state_change

        engine.d_team = DebateTeam(
            "defendant",
            init_res.defendant_persona,
            graph_tool,
            engine.fact_es,
            engine.law_es,
            agent_llm,
            engine.legal_sys,
            insights=d_insights_str,
            verbose=verbose,
        )

        engine.d_team.on_state_change = engine._handle_team_state_change
        logger.info(">>> [Engine] Setup complete.")

        init_narrative = (
            f"【系统初始化】\n"
            f"案件案由：{cause}\n"
            f"已注入 {fact_count} 个客观事实节点\n"
            f"已注入 {len(claim_ids)} 个根诉求节点\n"
            f"图谱状态：{engine.graph.graph.number_of_nodes()} 个节点，"
            f"{engine.graph.graph.number_of_edges()} 条边\n"
            f"系统已准备就绪，等待辩论开始。"
        )

        engine.transcript.append(init_narrative)

        engine.last_step_log = {
            "turn": "Setup",
            "action": "System Initialized",
            "details": f"{len(init_res.root_claim_actions)} initial claims/facts added.",
        }

        initial_snapshot = engine._create_snapshot(0, "Setup")
        engine.round_snapshots.append(initial_snapshot)

        engine._notify_state_change(
            "setup_complete",
            {
                "fact_count": fact_count,
                "claim_count": len(claim_ids),
                "node_count": engine.graph.graph.number_of_nodes(),
                "edge_count": engine.graph.graph.number_of_edges(),
            },
        )

    finally:
        engine._is_running = False
