"""The core orchestrator for the legal debate simulation.

This module defines the `DebateEngine`, the central class that manages the
entire lifecycle of a legal debate.
"""

import json
import time
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
        self.turn_artifacts: List[Dict[str, Any]] = []
        self.latest_turn_uid: str = ""
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
            self.turn_artifacts = []
            self.latest_turn_uid = ""
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
        graph_stats = self._build_graph_stats(graph_data)

        snapshot = {
            "round_idx": round_idx,
            "turn": turn,
            "timestamp": self.legal_sys.step_counter if self.legal_sys else 0,
            "ts_ms": int(time.time() * 1000),
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
                "node_count": graph_stats["node_count"],
                "edge_count": graph_stats["edge_count"],
                "claim_nodes": graph_stats["claim_nodes"],
                "conflict_edges": graph_stats["conflict_edges"],
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

    def _to_json_safe(self, value: Any) -> Any:
        """Recursively convert values into JSON-safe primitives."""
        if isinstance(value, dict):
            return {str(k): self._to_json_safe(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [self._to_json_safe(v) for v in value]

        if isinstance(value, set):
            return [self._to_json_safe(v) for v in sorted(value, key=str)]

        serialized = self._serialize_value(value)

        if isinstance(serialized, (str, int, float, bool)) or serialized is None:
            return serialized

        if isinstance(serialized, (dict, list, tuple, set)):
            return self._to_json_safe(serialized)

        return str(serialized)

    def _deserialize_node_type(self, value: Any) -> Any:
        """Best-effort convert a snapshot node type value back to NodeType."""
        if isinstance(value, NodeType):
            return value

        if isinstance(value, str):
            token = value.split(".")[-1].strip().upper()

            try:
                return NodeType[token]

            except Exception:
                try:
                    return NodeType(token)

                except Exception:
                    return value

        return value

    def _deserialize_edge_type(self, value: Any) -> Any:
        """Best-effort convert a snapshot edge type value back to EdgeType."""
        if isinstance(value, EdgeType):
            return value

        if isinstance(value, str):
            token = value.split(".")[-1].strip().upper()

            try:
                return EdgeType[token]

            except Exception:
                try:
                    return EdgeType(token)

                except Exception:
                    return value

        return value

    def _deserialize_node_status(self, value: Any) -> Any:
        """Best-effort convert a snapshot status value back to NodeStatus."""
        if isinstance(value, NodeStatus):
            return value

        if isinstance(value, str):
            token = value.split(".")[-1].strip().upper()

            try:
                return NodeStatus[token]

            except Exception:
                try:
                    return NodeStatus(token)

                except Exception:
                    return value

        return value

    def _count_conflict_edges(self) -> int:
        """Count CONFLICT edges in the current graph."""
        if not self.graph or not self.graph.graph:
            return 0

        count = 0

        for _, _, data in self.graph.graph.edges(data=True):
            edge_type = data.get("type")

            if edge_type == EdgeType.CONFLICT or str(edge_type) in {
                "CONFLICT",
                "EdgeType.CONFLICT",
            }:
                count += 1

        return count

    def _build_graph_data(self) -> Dict[str, Any]:
        """Build JSON-safe graph payload for API/transport usage."""
        if not self.graph:
            return {"nodes": [], "edges": []}

        return {
            "nodes": [
                {"id": nid, **{k: self._to_json_safe(v) for k, v in data.items()}}
                for nid, data in self.graph.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": src,
                    "target": dst,
                    **{k: self._to_json_safe(v) for k, v in data.items()},
                }
                for src, dst, data in self.graph.graph.edges(data=True)
            ],
        }

    def _count_support_edges_from_graph_data(self, graph_data: Dict[str, Any]) -> int:
        """Count SUPPORT edges from serialized graph data."""
        count = 0

        for edge in graph_data.get("edges", []):
            if str(edge.get("type")) in {"SUPPORT", "EdgeType.SUPPORT"}:
                count += 1

        return count

    def _build_graph_stats(
        self, graph_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """Build a unified graph stats payload for snapshots and APIs."""
        graph_payload = (
            graph_data if graph_data is not None else self._build_graph_data()
        )
        conflict_count = self._count_conflict_edges()
        support_count = self._count_support_edges_from_graph_data(graph_payload)

        return {
            "node_count": len(graph_payload.get("nodes", [])),
            "edge_count": len(graph_payload.get("edges", [])),
            "claim_nodes": self._count_claim_nodes(),
            "conflict_edges": conflict_count,
            "edge_conflict_count": conflict_count,
            "edge_attack_count": conflict_count,
            "edge_support_count": support_count,
        }

    def _collect_agent_memories(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect serializable controller memories from both teams."""
        p_mem: List[Dict[str, Any]] = []

        if self.p_team and self.p_team.controller:
            try:
                p_mem = [m.model_dump() for m in self.p_team.controller.get_memories()]

            except Exception:
                pass

        d_mem: List[Dict[str, Any]] = []

        if self.d_team and self.d_team.controller:
            try:
                d_mem = [m.model_dump() for m in self.d_team.controller.get_memories()]

            except Exception:
                pass

        return {"plaintiff": p_mem, "defendant": d_mem}

    def _build_turn_artifact(
        self, turn: str, turn_result: Dict[str, Any], narrative_text: str
    ) -> Dict[str, Any]:
        """Build a JSON-safe turn artifact record."""
        turn_uid = str(turn_result.get("turn_uid", "")).strip()

        if not turn_uid:
            turn_uid = f"turn_{self.round_idx}_{turn}_{int(time.time() * 1000)}"

        try:
            ts_ms = int(turn_result.get("ts_ms", int(time.time() * 1000)))

        except Exception:
            ts_ms = int(time.time() * 1000)

        artifact = {
            "turn_uid": turn_uid,
            "side": turn,
            "round_idx": self.round_idx,
            "controller_assessment": turn_result.get("controller_assessment", {}),
            "batch_instructions": turn_result.get("batch_instructions", []),
            "worker_reports": turn_result.get("worker_reports_raw", []),
            "decision_raw": turn_result.get("decision_raw", ""),
            "parsed_actions": turn_result.get("parsed_actions", []),
            "execution_logs": turn_result.get("execution_log", ""),
            "retry_history": turn_result.get("retry_history", []),
            "narrative_raw_sentences": turn_result.get("narrative_raw_sentences", []),
            "narrative_polished": narrative_text or "",
            "action_summary": turn_result.get("summary", ""),
            "ts_ms": ts_ms,
        }

        return self._to_json_safe(artifact)

    def get_turn_artifacts(
        self, turn_uid: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return stored turn artifacts, optionally filtered by turn UID."""
        rows = list(self.turn_artifacts)

        if turn_uid:
            rows = [item for item in rows if str(item.get("turn_uid", "")) == turn_uid]

        safe_limit = max(1, int(limit))
        return rows[-safe_limit:]

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
        graph_data = snapshot.get("graph_data", {})
        restored_graph = ShadowGraph()

        for node in graph_data.get("nodes", []):
            if not isinstance(node, dict):
                continue

            node_row = dict(node)
            node_id = node_row.pop("id", None)

            if not node_id:
                continue

            if "type" in node_row:
                node_row["type"] = self._deserialize_node_type(node_row["type"])

            if "status" in node_row:
                node_row["status"] = self._deserialize_node_status(node_row["status"])

            restored_graph.graph.add_node(node_id, **node_row)

        for edge in graph_data.get("edges", []):
            if not isinstance(edge, dict):
                continue

            edge_row = dict(edge)
            source = edge_row.pop("source", None)
            target = edge_row.pop("target", None)

            if not source or not target:
                continue

            if "type" in edge_row:
                edge_row["type"] = self._deserialize_edge_type(edge_row["type"])

            restored_graph.graph.add_edge(source, target, **edge_row)

        self.graph = restored_graph
        self.round_idx = int(snapshot.get("round_idx", round_idx))
        self.transcript = list(snapshot.get("transcript", []))
        self.is_finished = bool(snapshot.get("is_finished", False))
        self.is_ready_for_adjudication = False

        convergence = snapshot.get("convergence", {})
        history = convergence.get("history", [])

        if isinstance(history, list):
            self.convergence_history = list(history)

        else:
            self.convergence_history = []

        raw_claim_status = snapshot.get("root_claims_status", {})
        restored_claim_status: Dict[str, NodeStatus] = {}

        if isinstance(raw_claim_status, dict):
            for claim_id, status in raw_claim_status.items():
                restored_claim_status[claim_id] = self._deserialize_node_status(status)

        self.root_claims_status = restored_claim_status

        if not isinstance(self.last_step_log, dict):
            self.last_step_log = {}

        self.last_step_log["action"] = snapshot.get("action_summary", "")
        self.last_step_log["round"] = self.round_idx
        self.last_step_log["turn"] = snapshot.get("turn", "")

        if isinstance(convergence, dict):
            self.last_step_log["convergence"] = {
                "delta_phi": convergence.get("delta_phi", 0.0),
                "sma": convergence.get("sma", 0.0),
                "is_converged": self.last_step_log.get("convergence", {}).get(
                    "is_converged", False
                ),
                "gc_removed": self.last_step_log.get("convergence", {}).get(
                    "gc_removed", 0
                ),
            }

        turn_name = str(snapshot.get("turn", "")).lower()

        if turn_name == Turn.PLAINTIFF.value:
            self.current_turn = Turn.DEFENDANT

        elif turn_name == Turn.DEFENDANT.value:
            self.current_turn = Turn.PLAINTIFF

        else:
            self.current_turn = Turn.PLAINTIFF

        self.prev_stats = {
            "claim_nodes": self._count_claim_nodes(),
            "conflict_edges": self._count_conflict_edges(),
        }

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

            turn_artifact = self._build_turn_artifact(
                turn=turn_name_str,
                turn_result=turn_result,
                narrative_text=narrative_text,
            )

            self.turn_artifacts.append(turn_artifact)
            self.latest_turn_uid = str(turn_artifact.get("turn_uid", ""))

            self._notify_state_change(
                "turn_complete",
                {
                    "turn": turn_name_str,
                    "round": self.round_idx,
                    "delta_phi": delta_phi,
                    "sma": sma,
                    "turn_uid": self.latest_turn_uid,
                },
            )

            cond_converged = (
                self.round_idx >= self.cfg.convergence.min_rounds
                and sma < self.cfg.convergence.epsilon
            )

            should_adjudicate = cond_converged

            if should_adjudicate:
                reason = "Convergence Reached"
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
                    "turn_uid": self.latest_turn_uid,
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

        agent_memories = self._collect_agent_memories()

        return {
            "shadow_graph": self.graph,
            "insights_manager": self.legal_sys.insights,
            "task_layer": self.legal_sys.memory.task_layer,
            "last_log": self.last_step_log,
            "is_finished": self.is_finished,
            "winner": self.winner,
            "judgment_document": self.judgment_document,
            "root_claims_status": serializable_claims_status,
            "agent_memories": agent_memories,
            "full_transcript": self.transcript,
        }

    def get_serializable_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-safe state snapshot for API responses."""
        graph_data = self._build_graph_data()
        graph_stats = self._build_graph_stats(graph_data)
        transcript_rows = list(self.transcript)

        serializable_claims_status = {
            k: self._serialize_value(v) for k, v in self.root_claims_status.items()
        }

        state = {
            "current_round": self.round_idx,
            "current_turn": self.current_turn.value,
            "is_ready_for_adjudication": self.is_ready_for_adjudication,
            "is_finished": self.is_finished,
            "winner": self.winner,
            "judgment_document": self.judgment_document,
            "root_claims_status": serializable_claims_status,
            "baf_details": self.baf_details,
            "preferred_extension": list(self.preferred_extension),
            "last_log": self.last_step_log,
            "transcript": transcript_rows,
            "full_transcript": transcript_rows,
            "convergence_history": list(self.convergence_history),
            "latest_turn_uid": self.latest_turn_uid,
            "turn_artifact_count": len(self.turn_artifacts),
            "graph_data": graph_data,
            "graph_stats": {
                "node_count": graph_stats["node_count"],
                "edge_count": graph_stats["edge_count"],
                "claim_nodes": graph_stats["claim_nodes"],
                "edge_conflict_count": graph_stats["edge_conflict_count"],
                "edge_attack_count": graph_stats["edge_attack_count"],
                "edge_support_count": graph_stats["edge_support_count"],
            },
            "agent_memories": self._collect_agent_memories(),
        }

        return self._to_json_safe(state)
