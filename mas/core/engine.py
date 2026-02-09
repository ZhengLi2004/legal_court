"""The core orchestrator for the legal debate simulation.

This module defines the `DebateEngine`, the central class that manages the
entire lifecycle of a legal debate.
"""

import json
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from metagpt.logs import logger

from roles.controller import ControllerPipelineStep
from tools.fact_es_tool import FactEsTool
from tools.graph_tool import GraphTool
from tools.initializer import CaseInitializer
from tools.law_es_tool import LawEsTool
from tools.llm import GPTChat

from ..agents.narrator import GraphNarrator
from ..agents.team import DebateTeam
from ..config import SystemConfig
from .graph import EdgeType, NodeStatus, NodeType, ShadowGraph
from .system import LegalSystem


class Turn(Enum):
    """Enumeration to represent whose turn it is in the debate."""

    PLAINTIFF = "plaintiff"
    DEFENDANT = "defendant"


class DebateEngine:
    """Manages and executes the entire legal debate simulation.

    Attributes:
        cfg: The system configuration object.
        legal_sys: The `LegalSystem` instance.
        p_team: The `DebateTeam` for the plaintiff.
        d_team: The `DebateTeam` for the defendant.
        graph: The central `ShadowGraph` representing the debate state.
        current_turn: The `Turn` enum member indicating the active team.
        round_idx: The current round number of the debate.
        is_finished: A boolean flag indicating if the debate has concluded.
        transcript: A list of strings containing the narrated debate history.
        on_state_change: Optional callback for state changes.
    """

    def __init__(self, config: SystemConfig, judge_config: Dict):
        """Initialize the DebateEngine.

        Args:
            config: The main `SystemConfig` object.
            judge_config: A dictionary with configuration for the judge LLM.
        """
        self.cfg = config
        self.judge_config = judge_config
        self.legal_sys: Optional[LegalSystem] = None
        self.p_team: Optional[DebateTeam] = None
        self.d_team: Optional[DebateTeam] = None
        self.graph: Optional[ShadowGraph] = None
        self.raw_facts: str = ""
        self.current_turn: Turn = Turn.PLAINTIFF
        self.round_idx: int = 0
        self.max_rounds: int = 10
        self.is_finished: bool = False
        self.is_ready_for_adjudication: bool = False
        self.winner: str = "Unsettled"
        self.last_step_log: Dict[str, Any] = {}
        self.fact_es: Optional[FactEsTool] = None
        self.law_es: Optional[LawEsTool] = None
        self.judgment_document: str = ""
        self.root_claims_status: Dict[str, NodeStatus] = {}
        self.convergence_history: List[float] = []
        self.prev_stats: Dict[str, int] = {"claim_nodes": 0, "conflict_edges": 0}
        self.transcript: List[str] = []
        self.round_snapshots: List[Dict[str, Any]] = []
        self.narrator: Optional[GraphNarrator] = None
        self.baf_details: Dict[str, Any] = {}
        self.preferred_extension: Set[str] = set()
        self.on_state_change: Optional[Callable[[str, Any], None]] = None
        self._is_running: bool = False

    def set_state_callback(self, callback: Callable[[str, Any], None]):
        """Set a callback for state changes.

        Args:
            callback: A function that takes (event_type, data) as arguments.
        """
        self.on_state_change = callback

    def _notify_state_change(self, event: str, data: Any = None):
        """Notify listeners about state changes.

        Args:
            event: The event type.
            data: Additional data about the event.
        """
        if self.on_state_change:
            try:
                self.on_state_change(event, data)

            except Exception as e:
                logger.warning(f"State change callback error: {e}")

    def _count_claim_nodes(self) -> int:
        """Count the number of CLAIM nodes in the graph.

        Returns:
            The count of CLAIM nodes.
        """
        if not self.graph or not self.graph.graph:
            return 0

        count = 0

        for _, data in self.graph.graph.nodes(data=True):
            n_type = data.get("type")

            if n_type == NodeType.CLAIM or str(n_type) == "CLAIM":
                count += 1

        return count

    def _handle_team_state_change(self, side: str, event: str, data: dict):
        """Handle state change events from teams.

        Args:
            side: The team side ("plaintiff" or "defendant").
            event: The event type.
            data: Event data.
        """
        self._notify_state_change(f"team_{side}_{event}", data)

    async def setup(
        self, case_data_path: str = None, case_data: Dict = None, verbose: bool = False
    ):
        """Initialize the engine for a new case.

        Args:
            case_data_path: The file path to the case data (JSONL format).
            case_data: A dictionary containing the case data.
            verbose: Enable detailed transcript logging.

        Raises:
            ValueError: If neither case_data_path nor case_data is provided.
        """
        logger.info(">>> [Engine] Setting up...")
        self._notify_state_change("setup_start", None)
        self._is_running = True

        try:
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
                raise ValueError("Either case_data_path or case_data must be provided.")

            self.raw_facts = case_data.get("fact_finding", "")
            cause = case_data.get("cause", ["未知案由"])

            if isinstance(cause, list):
                cause = cause[0] if cause else "未知案由"

            initializer = CaseInitializer(agent_llm)
            init_res = await initializer.initialize(self.raw_facts, cause)

            self.graph, (p_insights_list, d_insights_list) = self.legal_sys.new_case(
                self.raw_facts
            )

            p_insights_str = "\n".join([f"- {s}" for s in p_insights_list])
            d_insights_str = "\n".join([f"- {s}" for s in d_insights_list])
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
            self.graph.refresh_context(current_step=0)

            logger.info(
                f">>> [System] Graph refreshed. "
                f"Nodes: {self.graph.graph.number_of_nodes()}, "
                f"Edges: {self.graph.graph.number_of_edges()}"
            )

            self.p_team = DebateTeam(
                "plaintiff",
                init_res.plaintiff_persona,
                graph_tool,
                self.fact_es,
                self.law_es,
                agent_llm,
                self.legal_sys,
                insights=p_insights_str,
                verbose=verbose,
            )

            self.p_team.on_state_change = self._handle_team_state_change

            self.d_team = DebateTeam(
                "defendant",
                init_res.defendant_persona,
                graph_tool,
                self.fact_es,
                self.law_es,
                agent_llm,
                self.legal_sys,
                insights=d_insights_str,
                verbose=verbose,
            )

            self.d_team.on_state_change = self._handle_team_state_change
            logger.info(">>> [Engine] Setup complete.")

            init_narrative = (
                f"【系统初始化】\n"
                f"案件案由：{cause}\n"
                f"已注入 {fact_count} 个客观事实节点\n"
                f"已注入 {len(claim_ids)} 个根诉求节点\n"
                f"图谱状态：{self.graph.graph.number_of_nodes()} 个节点，"
                f"{self.graph.graph.number_of_edges()} 条边\n"
                f"系统已准备就绪，等待辩论开始。"
            )

            self.transcript.append(init_narrative)
            logger.info(f">>> [Transcript Updated]:\n{init_narrative}")

            self.last_step_log = {
                "turn": "Setup",
                "action": "System Initialized",
                "details": f"{len(init_res.root_claim_actions)} initial claims/facts added.",
            }

            initial_snapshot = self._create_snapshot(0, "Setup")
            self.round_snapshots.append(initial_snapshot)

            logger.info(
                f">>> [Snapshot] Initial snapshot saved. "
                f"Total snapshots: {len(self.round_snapshots)}"
            )

            self._notify_state_change(
                "setup_complete",
                {
                    "fact_count": fact_count,
                    "claim_count": len(claim_ids),
                    "node_count": self.graph.graph.number_of_nodes(),
                    "edge_count": self.graph.graph.number_of_edges(),
                },
            )

        finally:
            self._is_running = False

    def _calculate_convergence(self) -> float:
        """Calculate the convergence score for the current turn.

        Returns:
            The convergence score (delta_phi) for the current turn.
        """
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

    def _create_snapshot(self, round_idx: int, turn: str) -> Dict[str, Any]:
        """Create a snapshot of the current debate state.

        Args:
            round_idx: The current round number.
            turn: The current turn name.

        Returns:
            A dictionary containing the snapshot data.
        """
        graph_data = {
            "nodes": [
                {"id": n, **{k: self._serialize_value(v) for k, v in d.items()}}
                for n, d in self.graph.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **{k: self._serialize_value(v) for k, v in d.items()},
                }
                for u, v, d in self.graph.graph.edges(data=True)
            ],
        }

        serializable_claims_status = {
            k: v.value if hasattr(v, "value") else str(v)
            for k, v in self.root_claims_status.items()
        }

        snapshot = {
            "round_idx": round_idx,
            "turn": turn,
            "timestamp": self.legal_sys.step_counter if self.legal_sys else 0,
            "graph_data": graph_data,
            "convergence": {
                "delta_phi": self.last_step_log.get("convergence", {}).get(
                    "delta_phi", 0.0
                ),
                "sma": self.last_step_log.get("convergence", {}).get("sma", 0.0),
                "history": list(self.convergence_history),
            },
            "transcript": list(self.transcript),
            "root_claims_status": serializable_claims_status,
            "stats": {
                "node_count": self.graph.graph.number_of_nodes(),
                "edge_count": self.graph.graph.number_of_edges(),
                "claim_nodes": self._count_claim_nodes(),
                "conflict_edges": sum(
                    1
                    for _, _, d in self.graph.graph.edges(data=True)
                    if str(d.get("type")) == "CONFLICT"
                ),
            },
            "action_summary": self.last_step_log.get("action", ""),
            "is_finished": self.is_finished,
        }

        return snapshot

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON compatibility.

        Args:
            value: The value to serialize.

        Returns:
            A JSON-serializable version of the value.
        """
        if hasattr(value, "value"):
            return value.value

        if hasattr(value, "name"):
            return value.name

        return value

    def restore_snapshot(self, round_idx: int) -> bool:
        """Restore the graph state from a snapshot.

        Args:
            round_idx: The index of the snapshot to restore.

        Returns:
            True if restoration was successful, False otherwise.
        """
        if round_idx < 0 or round_idx >= len(self.round_snapshots):
            return False

        snapshot = self.round_snapshots[round_idx]
        return True

    async def open_resources(self):
        """Open persistent connections, like to Elasticsearch."""
        if self.fact_es:
            await self.fact_es.open()

        if self.law_es:
            await self.law_es.open()

    async def close_resources(self):
        """Close any open persistent connections."""
        if self.fact_es:
            await self.fact_es.close()
            logger.info("[Engine] FactEsTool connection closed.")

        if self.law_es:
            await self.law_es.close()
            logger.info("[Engine] LawEsTool connection closed.")

    async def step(self):
        """Execute a single turn of the debate."""
        if self.is_finished:
            return

        if self._is_running:
            logger.warning("[Engine] Step already in progress, skipping.")
            return

        self._is_running = True

        try:
            await self.open_resources()
            self.legal_sys.advance_step()
            is_plaintiff_turn = self.current_turn == Turn.PLAINTIFF
            turn_name_str = "plaintiff" if is_plaintiff_turn else "defendant"

            self._notify_state_change(
                "turn_start",
                {
                    "turn": turn_name_str,
                    "round": self.round_idx,
                },
            )

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
                    f">>> [Narrator] Generating transcript for "
                    f"{len(executed_actions)} actions..."
                )

                narrative_text = await self.narrator.generate_narrative(
                    actions=executed_actions, graph=self.graph, turn=turn_name_str
                )

                if narrative_text:
                    self.transcript.append(narrative_text)
                    logger.info(f">>> [Transcript Updated]:\n{narrative_text}")

                    self._notify_state_change(
                        "transcript_update",
                        {
                            "text": narrative_text,
                            "turn": turn_name_str,
                        },
                    )

            else:
                logger.info(">>> [Narrator] No actions executed this turn.")

            delta_phi = self._calculate_convergence()
            self.convergence_history.append(delta_phi)
            window = self.cfg.convergence.window_size
            recent_history = self.convergence_history[-window:]
            sma = sum(recent_history) / len(recent_history) if recent_history else 0.0

            logger.info(
                f"[Convergence] Round {self.round_idx} | "
                f"ΔΦ: {delta_phi:.4f} | SMA: {sma:.4f}"
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

            self._notify_state_change(
                "turn_complete",
                {
                    "turn": turn_name_str,
                    "round": self.round_idx,
                    "delta_phi": delta_phi,
                    "sma": sma,
                },
            )

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

                logger.info(f">>> [Engine] Adjudication ready. Reason: {reason}")
                self.last_step_log["convergence"]["is_converged"] = True
                self.is_ready_for_adjudication = True
                self._notify_state_change("adjudication_ready", {"reason": reason})

            if not self.is_finished:
                self.current_turn = next_turn

                if not is_plaintiff_turn:
                    self.round_idx += 1

            if self.p_team and self.p_team.controller:
                self.p_team.controller.pipeline_step = ControllerPipelineStep.IDLE

            if self.d_team and self.d_team.controller:
                self.d_team.controller.pipeline_step = ControllerPipelineStep.IDLE

            turn_snapshot = self._create_snapshot(self.round_idx, turn_name_str)
            self.round_snapshots.append(turn_snapshot)

            logger.info(
                f">>> [Snapshot] Round {self.round_idx} ({turn_name_str}) saved. "
                f"Total: {len(self.round_snapshots)}"
            )

            self._notify_state_change(
                "snapshot_saved",
                {
                    "round": self.round_idx,
                    "turn": turn_name_str,
                    "total": len(self.round_snapshots),
                },
            )

        finally:
            self._is_running = False
            await self.close_resources()

    async def adjudicate(self):
        """Execute the adjudication process."""
        logger.info(">>> [Engine] Running Pre-Adjudication Garbage Collection...")
        removed_count = self.graph.garbage_collect()

        if removed_count > 0:
            logger.info(f"✅ [GC] Cleaned {removed_count} isolated nodes.")
            self.last_step_log["convergence"]["gc_removed"] = removed_count

        else:
            logger.info("✅ [GC] Graph is clean. No nodes removed.")

        self._notify_state_change("adjudication_start", None)

        (
            self.judgment_document,
            self.root_claims_status,
        ) = await self.legal_sys.adjudicate(
            self.raw_facts,
            self.graph,
            transcript=self.transcript,
        )

        logger.info(f">>> [Engine] root_claims_status: {self.root_claims_status}")

        self.last_step_log["adjudication_result"] = {
            "document": self.judgment_document,
            "claims_status": {
                k: v.value if hasattr(v, "value") else str(v)
                for k, v in self.root_claims_status.items()
            },
        }

        logger.info(">>> [Engine] Running BAF calculation...")

        from ..analysis.baf import BAFCalculator

        baf_calc = BAFCalculator(self.graph)
        preferred_extensions = baf_calc.find_preferred_extensions()

        llm_validated = {
            k for k, v in self.root_claims_status.items() if v == NodeStatus.VALIDATED
        }

        llm_defeated = {
            k for k, v in self.root_claims_status.items() if v == NodeStatus.DEFEATED
        }

        best_extension, matching_details = baf_calc.match_with_llm_judgment(
            preferred_extensions, llm_validated, llm_defeated
        )

        self.preferred_extension = best_extension

        self.baf_details = {
            "preferred_extensions_count": len(preferred_extensions),
            "chosen_extension_size": len(best_extension),
            "matching_score": matching_details.get("score", 0),
            "alignment_rate": matching_details.get("alignment_rate", 0.0),
            "matching_details": matching_details,
        }

        logger.info(
            f">>> [BAF] Found {len(preferred_extensions)} preferred extensions, "
            f"chose extension with {len(best_extension)} nodes, "
            f"alignment rate: {self.baf_details['alignment_rate']:.2%}"
        )

        self.is_finished = True

        self._notify_state_change(
            "adjudication_complete",
            {
                "judgment_length": len(self.judgment_document),
                "baf_details": self.baf_details,
            },
        )

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of the current state of the debate.

        Returns:
            A dictionary containing state information.
        """
        if not self.legal_sys:
            return {}

        serializable_claims_status = {
            k: v.value if hasattr(v, "value") else str(v)
            for k, v in self.root_claims_status.items()
        }

        p_mem = []

        if self.p_team and self.p_team.controller:
            try:
                p_mem = [m.model_dump() for m in self.p_team.controller.get_memories()]

            except Exception:
                pass

        d_mem = []

        if self.d_team and self.d_team.controller:
            try:
                d_mem = [m.model_dump() for m in self.d_team.controller.get_memories()]

            except Exception:
                pass

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
