"""The core orchestrator for the legal debate simulation.

This module defines the `DebateEngine`, the central class that manages the
entire lifecycle of a legal debate.
"""

import hashlib
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from metagpt.logs import logger

from mas.common.serialization import serialize_enum_like, to_json_safe

from ..config import SystemConfig
from .controller_pipeline import ControllerPipelineStep
from .engine_adjudication import run_engine_adjudication
from .engine_convergence import (
    build_engine_convergence_config,
    calculate_engine_convergence_delta,
)
from .engine_snapshot_codec import (
    build_engine_focus_node_ids,
    build_engine_graph_data,
    build_engine_graph_stats,
    build_engine_snapshot_state,
    build_engine_turn_artifact,
    collect_engine_agent_memories,
    count_engine_conflict_edges,
    count_support_edges_from_graph_data,
    create_engine_snapshot,
    get_engine_turn_artifacts,
    restore_engine_snapshot,
)
from .engine_turn_runner import run_engine_step
from .graph import EdgeType, NodeStatus, NodeType, ShadowGraph
from .system import LegalSystem


class Turn(Enum):
    """Enumeration to represent whose turn it is in the debate.

    Attributes:
        PLAINTIFF: Plaintiff turn.
        DEFENDANT: Defendant turn.
    """

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

    def __init__(self, config: SystemConfig):
        """Initialize the DebateEngine.

        Args:
            config: The main `SystemConfig` object.
        """
        self.cfg = config
        self.legal_sys: Optional[LegalSystem] = None
        self.p_team: Optional[Any] = None
        self.d_team: Optional[Any] = None
        self.graph: Optional[ShadowGraph] = None
        self.raw_facts: str = ""
        self.current_case_id: str = ""
        self.current_turn: Turn = Turn.PLAINTIFF
        self.round_idx: int = 0
        self.is_finished: bool = False
        self.is_ready_for_adjudication: bool = False
        self.winner: str = "Unsettled"
        self.last_step_log: Dict[str, Any] = {}
        self.fact_es: Optional[Any] = None
        self.law_es: Optional[Any] = None
        self.root_claims_status: Dict[str, NodeStatus] = {}
        self.convergence_history: List[float] = []
        self.prev_stats: Dict[str, int] = {"claim_nodes": 0, "conflict_edges": 0}
        self.transcript: List[str] = []
        self.round_snapshots: List[Dict[str, Any]] = []
        self.turn_artifacts: List[Dict[str, Any]] = []
        self.latest_turn_uid: str = ""
        self.narrator: Optional[Any] = None
        self.on_state_change: Optional[Callable[[str, Any], None]] = None
        self._is_running: bool = False

    def _resolve_case_id(self, case_data: Dict[str, Any]) -> str:
        """Resolve a stable case identifier from setup payload.

        Args:
            case_data: Raw case payload used for engine setup.

        Returns:
            Non-empty case identifier for post-adjudication learning.
        """
        candidate_keys = ("uid", "case_uid", "case_id", "id")

        for key in candidate_keys:
            candidate = str(case_data.get(key, "")).strip()

            if candidate:
                return candidate

        title = str(case_data.get("title", "")).strip()
        fact_finding = str(case_data.get("fact_finding", "")).strip()
        digest_src = f"{title}\n{fact_finding}".strip()

        if not digest_src:
            digest_src = str(time.time_ns())

        digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:12]
        return f"case_{digest}"

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

            except (TypeError, ValueError, RuntimeError) as e:
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
        from mas.application.engine_setup import run_engine_setup

        await run_engine_setup(
            self,
            case_data_path=case_data_path,
            case_data=case_data,
            verbose=verbose,
        )

    def _calculate_convergence(self) -> float:
        """Calculate the convergence score for the current turn.

        Returns:
            The convergence score (delta_phi) for the current turn.
        """
        return calculate_engine_convergence_delta(self)

    def _build_convergence_config_payload(self) -> Dict[str, Any]:
        """Return normalized convergence config values for snapshots/logs."""
        return build_engine_convergence_config(self)

    def _create_snapshot(self, round_idx: int, turn: str) -> Dict[str, Any]:
        """Create a snapshot of the current debate state.

        Args:
            round_idx: The current round number.
            turn: The current turn name.

        Returns:
            A dictionary containing the snapshot data.
        """
        return create_engine_snapshot(self, round_idx, turn)

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON compatibility.

        Args:
            value: The value to serialize.

        Returns:
            A JSON-serializable version of the value.
        """
        return serialize_enum_like(value)

    def _to_json_safe(self, value: Any) -> Any:
        """Recursively convert values into JSON-safe primitives."""
        return to_json_safe(value, scalar_serializer=self._serialize_value)

    @staticmethod
    def _deserialize_enum(value: Any, enum_cls: Any) -> Any:
        """Best-effort convert serialized enum text back to enum value."""
        if isinstance(value, enum_cls):
            return value

        if not isinstance(value, str):
            return value

        token = value.split(".")[-1].strip().upper()

        try:
            return enum_cls[token]

        except KeyError:
            pass

        try:
            return enum_cls(token)

        except ValueError:
            return value

    def _deserialize_node_type(self, value: Any) -> Any:
        """Best-effort convert a snapshot node type value back to NodeType."""
        return self._deserialize_enum(value, NodeType)

    def _deserialize_edge_type(self, value: Any) -> Any:
        """Best-effort convert a snapshot edge type value back to EdgeType."""
        return self._deserialize_enum(value, EdgeType)

    def _deserialize_node_status(self, value: Any) -> Any:
        """Best-effort convert a snapshot status value back to NodeStatus."""
        return self._deserialize_enum(value, NodeStatus)

    def _count_conflict_edges(self) -> int:
        """Count CONFLICT edges in the current graph."""
        return count_engine_conflict_edges(self)

    def _build_graph_data(self) -> Dict[str, Any]:
        """Build JSON-safe graph payload for API/transport usage."""
        return build_engine_graph_data(self)

    def _build_focus_node_ids(self) -> List[str]:
        """Return current focus-node ids used by tactical linearization."""
        return build_engine_focus_node_ids(self)

    def _count_support_edges_from_graph_data(self, graph_data: Dict[str, Any]) -> int:
        """Count SUPPORT edges from serialized graph data."""
        return count_support_edges_from_graph_data(graph_data)

    def _build_graph_stats(
        self, graph_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """Build a unified graph stats payload for snapshots and APIs."""
        return build_engine_graph_stats(self, graph_data)

    def _collect_agent_memories(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect serializable controller memories from both teams."""
        return collect_engine_agent_memories(self)

    def _build_turn_artifact(
        self, turn: str, turn_result: Dict[str, Any], narrative_text: str
    ) -> Dict[str, Any]:
        """Build a JSON-safe turn artifact record."""
        return build_engine_turn_artifact(self, turn, turn_result, narrative_text)

    def get_turn_artifacts(
        self, turn_uid: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return stored turn artifacts, optionally filtered by one turn UID.

        Args:
            turn_uid: Optional turn UID filter.
            limit: Maximum number of rows to return.

        Returns:
            Artifact rows in chronological order (tail slice).
        """
        return get_engine_turn_artifacts(self, turn_uid, limit)

    def restore_snapshot(self, round_idx: int) -> bool:
        """Restore the graph state from a snapshot.

        Args:
            round_idx: The index of the snapshot to restore.

        Returns:
            True if restoration was successful, False otherwise.
        """
        return restore_engine_snapshot(self, round_idx)

    async def open_resources(self):
        """Open persistent infrastructure resources.

        Side Effects:
            Opens `fact_es` and `law_es` connections when present.
        """
        if self.fact_es:
            await self.fact_es.open()

        if self.law_es:
            await self.law_es.open()

    async def close_resources(self):
        """Close persistent infrastructure resources.

        Side Effects:
            Closes `fact_es` and `law_es` connections when present.
        """
        if self.fact_es:
            await self.fact_es.close()

        if self.law_es:
            await self.law_es.close()

    async def step(self, persist_snapshot: bool = True):
        """Execute one debate turn.

        Args:
            persist_snapshot: Whether to append a round snapshot after execution.
        """
        await run_engine_step(
            self,
            turn_enum=Turn,
            controller_idle_step=ControllerPipelineStep.IDLE,
            persist_snapshot=persist_snapshot,
        )

    async def adjudicate(self):
        """Execute final adjudication workflow."""
        await run_engine_adjudication(self)

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
            "root_claims_status": serializable_claims_status,
            "agent_memories": agent_memories,
            "full_transcript": self.transcript,
        }

    def get_serializable_snapshot(self) -> Dict[str, Any]:
        """Return JSON-safe state snapshot for API responses.

        Returns:
            Snapshot dict generated by engine snapshot codec helpers.
        """
        return build_engine_snapshot_state(self)
