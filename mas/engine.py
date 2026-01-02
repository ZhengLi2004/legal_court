import json
from enum import Enum
from typing import Any, Dict, List

from metagpt.logs import logger

from tools.fact_es_tool import FactEsTool
from tools.graph_tool import GraphTool
from tools.initializer import CaseInitializer
from tools.law_es_tool import LawEsTool

from .common import EdgeType, NodeStatus, NodeType, ShadowGraph
from .config import SystemConfig
from .legal_system import LegalSystem
from .llm import GPTChat
from .narrator import GraphNarrator
from .team import DebateTeam


class Turn(Enum):
    PLAINTIFF = "plaintiff"
    DEFENDANT = "defendant"


class DebateEngine:
    def __init__(self, config: SystemConfig, judge_config: Dict):
        self.cfg = config
        self.judge_config = judge_config
        self.legal_sys: LegalSystem = None
        self.p_team: DebateTeam = None
        self.d_team: DebateTeam = None
        self.graph: ShadowGraph = None
        self.raw_facts: str = ""
        self.current_turn: Turn = Turn.PLAINTIFF
        self.round_idx: int = 0
        self.max_rounds: int = 10
        self.is_finished: bool = False
        self.winner: str = "Unsettled"
        self.last_step_log: Dict[str, Any] = {}
        self.fact_es = None
        self.law_es = None
        self.judgment_document: str = ""
        self.root_claims_status: Dict[str, NodeStatus] = {}
        self.convergence_history: List[float] = []
        self.prev_stats: Dict[str, int] = {"claim_nodes": 0, "conflict_edges": 0}
        self.transcript: List[str] = []
        self.narrator: GraphNarrator = None

    def _count_claim_nodes(self) -> int:
        if not self.graph or not self.graph.graph:
            return 0

        count = 0

        for _, data in self.graph.graph.nodes(data=True):
            n_type = data.get("type")

            if n_type == NodeType.CLAIM or str(n_type) == "CLAIM":
                count += 1

        return count

    async def setup(
        self, case_data_path: str = None, case_data: Dict = None, verbose: bool = False
    ):
        logger.info(">>> [Engine] Setting up...")
        agent_llm = GPTChat(model_name=self.cfg.llm.model_name)
        self.narrator = GraphNarrator(llm=agent_llm)
        self.legal_sys = LegalSystem(config=self.cfg)

        self.fact_es = FactEsTool(
            es_host=self.cfg.es.host, embedding_func=self.legal_sys.ef
        )

        self.law_es = LawEsTool(
            es_host=self.cfg.es.host, embedding_func=self.legal_sys.ef
        )

        graph_tool = GraphTool(legal_system=self.legal_sys, llm=agent_llm)

        if case_data is None and case_data_path:
            with open(case_data_path, "r", encoding="utf-8") as f:
                case_data = json.loads(f.readline())

        if case_data is None:
            raise ValueError(
                "Either case_data_path or case_data must be provided to setup the engine."
            )

        self.raw_facts = case_data.get("fact_finding", "")
        cause = case_data.get("cause", ["未知案由"])[0]
        initializer = CaseInitializer(agent_llm)
        init_res = await initializer.initialize(self.raw_facts, cause)
        self.graph, insights = self.legal_sys.new_case(self.raw_facts)
        self.legal_sys.step_counter = 0
        self.prev_stats["claim_nodes"] = self._count_claim_nodes()
        self.prev_stats["conflict_edges"] = 0
        graph_tool.set_current_graph(self.graph)
        logger.info(">>> [System] Injecting immutable facts...")
        fact_count = 0
        fact_ids = []

        for fact_statement in init_res.fact_statements:
            node_id, is_new = self.graph.add_node(
                content=fact_statement,
                node_type=NodeType.FACT,
                agent_id="System_Init",
                metadata={"is_objective_fact": True},
            )

            if is_new:
                fact_count += 1
                fact_ids.append(node_id)

        if fact_ids:
            self.graph.touch_nodes(fact_ids, step_index=0)

        logger.info(f">>> [System] Injected {fact_count} fact nodes.")
        logger.info(">>> [System] Injecting root claims...")
        claim_ids = []

        for claim_statement in init_res.root_claim_actions:
            node_id, is_new = self.graph.add_node(
                content=claim_statement,
                node_type=NodeType.CLAIM,
                agent_id="System_Init",
                metadata={"is_root_claim": True},
            )

            if is_new:
                claim_ids.append(node_id)

        if claim_ids:
            self.graph.touch_nodes(claim_ids, step_index=0)

        logger.info(f">>> [System] Injected {len(claim_ids)} root claim nodes.")
        self.prev_stats["claim_nodes"] = self._count_claim_nodes()

        self.p_team = DebateTeam(
            "plaintiff",
            init_res.plaintiff_persona,
            graph_tool,
            self.fact_es,
            self.law_es,
            agent_llm,
            self.legal_sys,
            insights,
            verbose=verbose,
        )

        self.d_team = DebateTeam(
            "defendant",
            init_res.defendant_persona,
            graph_tool,
            self.fact_es,
            self.law_es,
            agent_llm,
            self.legal_sys,
            insights,
            verbose=verbose,
        )

        logger.info(">>> [Engine] Setup complete.")

        self.last_step_log = {
            "turn": "Setup",
            "action": "System Initialized",
            "details": f"{len(init_res.root_claim_actions)} initial claims/facts added.",
        }

    def _calculate_convergence(self) -> float:
        current_claim_nodes = self._count_claim_nodes()
        current_conflicts = 0

        for _, _, d in self.graph.graph.edges(data=True):
            e_type = d.get("type")

            if str(e_type) == "CONFLICT" or e_type == EdgeType.CONFLICT:
                current_conflicts += 1

        delta_v = max(0, current_claim_nodes - self.prev_stats["claim_nodes"])
        delta_e = max(0, current_conflicts - self.prev_stats["conflict_edges"])

        self.prev_stats = {
            "claim_nodes": current_claim_nodes,
            "conflict_edges": current_conflicts,
        }

        alpha = self.cfg.convergence.alpha
        delta_phi = (1 - alpha) * delta_v + alpha * delta_e
        return delta_phi

    async def open_resources(self):
        if self.fact_es:
            await self.fact_es.open()

        if self.law_es:
            await self.law_es.open()

    async def close_resources(self):
        if self.fact_es:
            await self.fact_es.close()
            logger.info("[Engine] FactEsTool connection closed.")

        if self.law_es:
            await self.law_es.close()
            logger.info("[Engine] LawEsTool connection closed.")

    async def step(self):
        if self.is_finished:
            return

        try:
            await self.open_resources()
            self.legal_sys.advance_step()
            is_plaintiff_turn = self.current_turn == Turn.PLAINTIFF
            turn_name_str = "plaintiff" if is_plaintiff_turn else "defendant"

            if is_plaintiff_turn:
                if self.round_idx == 0:
                    self.round_idx = 1

                logger.info(
                    f"\n>>> [Engine] Round {self.round_idx}, Plaintiff's Turn..."
                )

                team_to_run = self.p_team
                next_turn = Turn.DEFENDANT

            else:
                logger.info(
                    f"\n>>> [Engine] Round {self.round_idx}, Defendant's Turn..."
                )

                team_to_run = self.d_team
                next_turn = Turn.PLAINTIFF

            turn_result = await team_to_run.run_turn(self.graph)
            executed_actions = turn_result.get("actions", [])
            narrative_text = ""

            if executed_actions:
                logger.info(
                    f">>> [Narrator] Generating transcript for {len(executed_actions)} actions..."
                )

                narrative_text = await self.narrator.generate_narrative(
                    actions=executed_actions, graph=self.graph, turn=turn_name_str
                )

                if narrative_text:
                    self.transcript.append(narrative_text)
                    logger.info(f">>> [Transcript Updated]:\n{narrative_text}")

            else:
                logger.info(
                    ">>> [Narrator] No actions executed this turn. Skipping narrative."
                )

            delta_phi = self._calculate_convergence()
            self.convergence_history.append(delta_phi)
            window = self.cfg.convergence.window_size
            recent_history = self.convergence_history[-window:]
            sma = sum(recent_history) / len(recent_history) if recent_history else 0.0

            logger.info(
                f"[Convergence] Round {self.round_idx} | ΔΦ: {delta_phi:.4f} | SMA: {sma:.4f}"
            )

            self.last_step_log = {
                "turn": self.current_turn.value,
                "round": self.round_idx,
                "action": turn_result["summary"],
                "dialogue": turn_result["transcript"],
                "narrative": narrative_text,
                "convergence": {
                    "delta_phi": delta_phi,
                    "sma": sma,
                    "is_converged": False,
                    "gc_removed": 0,
                },
            }

            cond_max_rounds = self.round_idx >= self.max_rounds

            cond_converged = (
                self.round_idx >= self.cfg.convergence.min_rounds
                and sma < self.cfg.convergence.epsilon
            )

            should_adjudicate = cond_max_rounds or cond_converged

            if should_adjudicate:
                reason = (
                    "Max Rounds Reached" if cond_max_rounds else "Convergence Reached"
                )

                logger.info(f">>> [Engine] Adjudication triggered. Reason: {reason}")
                self.last_step_log["convergence"]["is_converged"] = True

                logger.info(
                    ">>> [Engine] Running Pre-Adjudication Garbage Collection..."
                )

                removed_count = self.graph.garbage_collect()

                if removed_count > 0:
                    logger.info(
                        f"✅ [GC] Cleaned {removed_count} isolated nodes before adjudication."
                    )

                    self.last_step_log["convergence"]["gc_removed"] = removed_count

                else:
                    logger.info("✅ [GC] Graph is clean. No nodes removed.")

                (
                    self.judgment_document,
                    self.root_claims_status,
                ) = await self.legal_sys.adjudicate(
                    self.raw_facts,
                    self.graph,
                    transcript=self.transcript,  # 传递笔录
                )

                logger.info(
                    f">>> [Engine] root_claims_status: {self.root_claims_status}"
                )

                self.last_step_log["adjudication_result"] = {
                    "document": self.judgment_document,
                    "claims_status": {
                        k: v.value for k, v in self.root_claims_status.items()
                    },
                }

                self.is_finished = True

            if not self.is_finished:
                self.current_turn = next_turn

                if not is_plaintiff_turn:
                    self.round_idx += 1

        finally:
            await self.close_resources()

    def get_snapshot(self) -> Dict[str, Any]:
        if not self.legal_sys:
            return {}

        serializable_claims_status = {
            k: v.value for k, v in self.root_claims_status.items()
        }

        p_mem = []

        if self.p_team and self.p_team.controller:
            p_mem = [m.model_dump() for m in self.p_team.controller.get_memories()]

        d_mem = []

        if self.d_team and self.d_team.controller:
            d_mem = [m.model_dump() for m in self.d_team.controller.get_memories()]

        return {
            "shadow_graph": self.graph,
            "insights_manager": self.legal_sys.insights,
            "task_layer": self.legal_sys.memory.task_layer,
            "last_log": self.last_step_log,
            "is_finished": self.is_finished,
            "winner": self.winner,
            "judgment_document": self.judgment_document,
            "root_claims_status": serializable_claims_status,
            "agent_memories": {"plaintiff": p_mem, "defendant": d_mem},
            "full_transcript": self.transcript,
        }
