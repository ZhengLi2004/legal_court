"""Defines common data structures and enumerations for the legal MAS.

This module contains the core data structures used to represent the state of a
legal debate, including enumerations for node and edge types (`NodeType`,
`EdgeType`), node statuses (`NodeStatus`), and the primary graph data structure,
`ShadowGraph`. It also defines `LegalMessage`, the standard format for
exchanging and storing case information within the system.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional, TypedDict, Union

import networkx as nx
from networkx.readwrite import json_graph


class BaseMetadata(TypedDict, total=False):
    """Base metadata for graph nodes.

    Attributes:
        created_at: Creation timestamp.
        source_doc_id: Source document identifier.
        last_modified_step: Debate step index of the most recent modification.
    """

    created_at: float
    source_doc_id: str
    last_modified_step: int


class ClaimMetadata(BaseMetadata, total=False):
    """Metadata specific to CLAIM nodes.

    Attributes:
        is_root_claim: Whether the claim is a root debate claim.
        claim_index: Claim order in the originating case.
        verdict_status: Final verdict status attached during adjudication.
    """

    is_root_claim: bool
    claim_index: int
    verdict_status: str


NodeMetadata = Union[BaseMetadata, ClaimMetadata]


class NodeType(str, Enum):
    """Enumeration for the types of nodes in the debate graph.

    Attributes:
        FACT: Evidence or factual proposition node.
        LAW: Legal rule or statute node.
        CLAIM: Debate claim or argument node.
    """

    FACT = "FACT"
    LAW = "LAW"
    CLAIM = "CLAIM"


class NodeStatus(str, Enum):
    """Enumeration for the status of a node, updated during adjudication.

    Attributes:
        HYPOTHETICAL: Node has not yet been accepted or defeated.
        VALIDATED: Node is accepted by the adjudication stage.
        DEFEATED: Node is rejected by the adjudication stage.
    """

    HYPOTHETICAL = "HYPOTHETICAL"
    VALIDATED = "VALIDATED"
    DEFEATED = "DEFEATED"


class EdgeType(str, Enum):
    """Enumeration for the types of edges (relationships) in the debate graph.

    Attributes:
        SUPPORT: Source supports the target node.
        CONFLICT: Source rebuts or conflicts with the target node.
    """

    SUPPORT = "SUPPORT"
    CONFLICT = "CONFLICT"


class EdgeAddResult(Enum):
    """Enumeration for the result of an attempt to add an edge to the graph.

    Attributes:
        CREATED: Edge was inserted successfully.
        DUPLICATE: Same edge already existed.
        TYPE_CLASH: Opposite edge type already existed.
        SELF_LOOP: Attempted self-loop was rejected.
    """

    CREATED = auto()
    DUPLICATE = auto()
    TYPE_CLASH = auto()
    SELF_LOOP = auto()


@dataclass
class ShadowNode:
    """Represents a single node within the ShadowGraph.

    Attributes:
        id: Stable node identifier.
        content: Node content text.
        type: Node type.
        agent_id: Agent that created the node.
        status: Current adjudication status.
        metadata: Additional graph metadata for the node.
    """

    id: str
    content: str
    type: NodeType
    agent_id: str
    status: NodeStatus = NodeStatus.HYPOTHETICAL
    metadata: NodeMetadata = field(default_factory=dict)


def ensure_node_type(val: Any) -> NodeType:
    """Convert a value to `NodeType`.

    Args:
        val: Candidate value, expected as `NodeType` or matching string key.

    Returns:
        Parsed `NodeType` enum value.

    Raises:
        ValueError: If the value cannot be resolved to any `NodeType`.
    """
    if isinstance(val, NodeType):
        return val

    if isinstance(val, str):
        return NodeType[val.upper()]

    raise ValueError(f"Invalid NodeType: {val}")


def ensure_edge_type(val: Any) -> EdgeType:
    """Convert a value to `EdgeType`.

    Args:
        val: Candidate value, expected as `EdgeType` or matching string key.

    Returns:
        Parsed `EdgeType` enum value.

    Raises:
        ValueError: If the value cannot be resolved to any `EdgeType`.
    """
    if isinstance(val, EdgeType):
        return val

    if isinstance(val, str):
        return EdgeType[val.upper()]

    raise ValueError(f"Invalid EdgeType: {val}")


@dataclass
class ShadowGraph:
    """A wrapper around a networkx.DiGraph to represent the debate state.

    This class provides the central data structure for the debate simulation.
    It contains the directed graph of arguments, facts, and laws, and provides
    methods to manipulate the graph, serialize it to text, and manage its context.

    Attributes:
        graph: The underlying `networkx.DiGraph` object.
        latest_context: A string containing a textual representation of the most
            recent or relevant parts of the graph, used as context for the LLM agents.
    """

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    latest_context: str = field(default="")

    def __post_init__(self):
        """Ensure graph-level attribute storage exists."""
        if not hasattr(self.graph, "graph"):
            self.graph.graph = {}

    def _calculate_focus_nodes(self, current_step: int) -> List[str]:
        """Determine which nodes are currently most relevant for context.

        The focus set includes all root claims plus any nodes that have been
        modified in the last few turns.

        Args:
            current_step: The current step index of the debate engine.

        Returns:
            A list of node IDs that are considered in focus.
        """
        focus_nodes = set(
            [
                nid
                for nid, data in self.graph.nodes(data=True)
                if data.get("metadata", {}).get("is_root_claim", False)
            ]
        )

        lookback_window = [current_step - i for i in range(4)]

        for step in lookback_window:
            if step >= 0:
                step_nodes = self.get_nodes_by_step(step)
                focus_nodes.update(step_nodes)

        return list(focus_nodes)

    def refresh_context(self, current_step: int):
        """Update the `latest_context` attribute with the current tactical view.

        It generates a textual representation of the subgraph surrounding the
        current focus nodes. If no nodes are in focus, it falls back to a
        recursive text representation of the entire graph.

        Args:
            current_step: The current step index of the debate engine.
        """
        focus_nodes = self._calculate_focus_nodes(current_step)

        if focus_nodes:
            self.latest_context = self.to_tactical_text(focus_nodes)

        else:
            self.latest_context = self.to_recursive_text()

        if not self.latest_context:
            self.latest_context = "（当前辩论图谱为空）"

    def touch_nodes(self, node_ids: List[str], step_index: int):
        """Update the 'last_modified_step' metadata for a list of nodes.

        Args:
            node_ids: A list of node IDs to update.
            step_index: The current step index to set as the last modified time.
        """
        for nid in node_ids:
            if self.graph.has_node(nid):
                current_step = (
                    self.graph.nodes[nid]
                    .get("metadata", {})
                    .get("last_modified_step", -1)
                )

                if step_index > current_step:
                    if "metadata" not in self.graph.nodes[nid]:
                        self.graph.nodes[nid]["metadata"] = {}

                    self.graph.nodes[nid]["metadata"]["last_modified_step"] = step_index

    def _get_node_type(self, node_id: str) -> NodeType:
        """Retrieve the NodeType of a node by its ID."""
        t = self.graph.nodes[node_id]["type"]

        if isinstance(t, str):
            return NodeType(t)

        return t

    def add_node(
        self,
        content: str,
        node_type: NodeType,
        agent_id: str,
        matcher: Any = None,
        metadata: dict = None,
    ) -> str:
        """Add a node to the graph, handling deduplication.

        Before adding a new node, it checks for existing nodes with identical
        or semantically similar content to avoid redundancy.

        Args:
            content: The text content of the node.
            node_type: The type of the node (FACT, LAW, or CLAIM).
            agent_id: The ID of the agent creating the node.
            matcher: An optional semantic matcher for deduplication.
            metadata: Optional dictionary of metadata for the node.

        Returns:
            A tuple of `(node_id, created)` where `created` indicates whether a
            new node was inserted (`True`) or an existing node was reused
            (`False`).
        """
        node_type = ensure_node_type(node_type)
        existing_id = None

        if node_type == NodeType.LAW:
            existing_id = self._find_exact_match_node(content, node_type)

        else:
            if matcher:
                candidates = [
                    (n, d["content"])
                    for n, d in self.graph.nodes(data=True)
                    if d["type"] == node_type
                ]

                existing_id = matcher.find_match(content, candidates)

            else:
                existing_id = self._find_exact_match_node(content, node_type)

        if existing_id:
            return existing_id, False

        node_id = self._generate_id(node_type)

        node = ShadowNode(
            id=node_id,
            content=content,
            type=node_type,
            agent_id=agent_id,
            metadata=metadata or {},
        )

        self.graph.add_node(node_id, **asdict(node))
        return node_id, True

    def _find_exact_match_node(
        self, content: str, node_type: NodeType
    ) -> Optional[str]:
        """Find a node with exactly matching content and type."""
        norm_content = content.strip().lower()

        for nid, data in self.graph.nodes(data=True):
            current_type = data.get("type")

            if hasattr(current_type, "value"):
                current_type = current_type.value

            target_type = node_type.value if hasattr(node_type, "value") else node_type

            if current_type == target_type:
                if data.get("content", "").strip().lower() == norm_content:
                    return nid

        return None

    def garbage_collect(self) -> int:
        """Remove isolated nodes that are not connected to any root claims.

        This is useful for cleaning up the graph before final adjudication,
        removing any speculative or irrelevant arguments that were not
        integrated into the main debate structure.

        Returns:
            The number of nodes that were removed.
        """
        all_roots = [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("metadata", {}).get("is_root_claim")
        ]

        active_roots = []

        for r in all_roots:
            if self.graph.degree(r) > 0:
                active_roots.append(r)

        valid_nodes = set(active_roots)

        for r in active_roots:
            valid_nodes.update(nx.ancestors(self.graph, r))

        all_nodes = set(self.graph.nodes())
        garbage = all_nodes - valid_nodes

        if garbage:
            self.graph.remove_nodes_from(garbage)
            self.refresh_context(current_step=9999)

        return len(garbage)

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType) -> None:
        """Add a directed edge between two nodes.

        Performs checks for existence of nodes, self-loops, and logical validity
        before adding the edge.

        Args:
            source_id: The ID of the source node.
            target_id: The ID of the target node.
            edge_type: The type of the edge (SUPPORT or CONFLICT).

        Returns:
            `EdgeAddResult` indicating `CREATED`, `DUPLICATE`, `TYPE_CLASH`,
            or `SELF_LOOP`.

        Raises:
            ValueError: If the source or target node does not exist, or if the
                connection is logically invalid.
        """
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            raise ValueError(
                f"Source {source_id} or Target {target_id} does not exist."
            )

        if source_id == target_id:
            return EdgeAddResult.SELF_LOOP

        edge_type = ensure_edge_type(edge_type)

        if self.graph.has_edge(source_id, target_id):
            existing_data = self.graph.get_edge_data(source_id, target_id)
            existing_type = existing_data.get("type")

            if hasattr(existing_type, "value"):
                existing_type = existing_type.value

            target_type_val = edge_type.value

            if existing_type == target_type_val:
                return EdgeAddResult.DUPLICATE

            else:
                return EdgeAddResult.TYPE_CLASH

        self.graph.add_edge(source_id, target_id, type=edge_type)
        return EdgeAddResult.CREATED

    def get_subgraph(self, node_ids: List[str]) -> "ShadowGraph":
        """Create a new `ShadowGraph` containing only the specified nodes.

        Args:
            node_ids: Node ids retained in the subgraph.

        Returns:
            A copied `ShadowGraph` subgraph over `node_ids`.
        """
        sub_nx = self.graph.subgraph(node_ids).copy()
        new_sg = ShadowGraph()
        new_sg.graph = sub_nx
        return new_sg

    def _find_semantically_identical_node(
        self, content: str, node_type: NodeType
    ) -> Optional[str]:
        """Find a node with semantically identical content."""
        for nid, data in self.graph.nodes(data=True):
            if (
                data.get("type") == node_type
                and data.get("content", "").strip() == content.strip()
            ):
                return nid

        return None

    def _generate_id(self, node_type: NodeType) -> str:
        """Generate a unique ID for a new node."""
        uid = uuid.uuid4().hex[:8]
        return f"{node_type.value}_{uid}"

    @staticmethod
    def to_dict(sg: "ShadowGraph") -> dict:
        """Serialize a shadow graph to a dictionary payload.

        Args:
            sg: Shadow graph instance.

        Returns:
            Dict containing node-link graph data under `graph_data`.
        """
        graph_data = json_graph.node_link_data(sg.graph)
        return {"graph_data": graph_data}

    @staticmethod
    def from_dict(data: dict) -> "ShadowGraph":
        """Deserialize dictionary payload into `ShadowGraph`.

        Args:
            data: Dict produced by `ShadowGraph.to_dict`.

        Returns:
            Reconstructed `ShadowGraph` instance.
        """
        sg = ShadowGraph()
        sg.graph = json_graph.node_link_graph(data["graph_data"])
        return sg

    def to_json(self) -> str:
        """Serialize this graph to a JSON string.

        Returns:
            UTF-8 JSON text generated from `ShadowGraph.to_dict`.
        """
        return json.dumps(self.to_dict(self), ensure_ascii=False)

    def to_recursive_text(self) -> str:
        """Generate a structured, recursive text representation of the graph.

        This method is ideal for providing a comprehensive view of the entire
        debate structure to an LLM. It clusters related arguments together
        and uses indentation to show logical relationships.

        Returns:
            A formatted string representing the entire graph.
        """
        if self.graph.number_of_nodes() == 0:
            return "无辩论记录。"

        root_claims = [
            nid
            for nid, data in self.graph.nodes(data=True)
            if data.get("metadata", {}).get("is_root_claim")
        ]

        try:
            centrality = nx.pagerank(self.graph)

        except (nx.NetworkXException, ZeroDivisionError):
            centrality = nx.degree_centrality(self.graph)

        clusters = []
        covered_nodes = set()
        root_claims.sort(key=lambda n: -centrality.get(n, 0))

        for root_id in root_claims:
            if root_id in covered_nodes:
                continue

            support_cone = set(nx.ancestors(self.graph, root_id))
            support_cone.add(root_id)
            attackers = set()

            for node_in_cone in support_cone:
                for pred in self.graph.predecessors(node_in_cone):
                    if (
                        self.graph.get_edge_data(pred, node_in_cone)["type"]
                        == EdgeType.CONFLICT
                    ):
                        attacker_cone = set(nx.ancestors(self.graph, pred))
                        attacker_cone.add(pred)
                        attackers.update(attacker_cone)

            cluster_nodes = support_cone.union(attackers)
            clusters.append(list(cluster_nodes))
            covered_nodes.update(cluster_nodes)

        orphan_nodes = list(set(self.graph.nodes()) - covered_nodes)

        if orphan_nodes:
            clusters.append(orphan_nodes)

        final_text_blocks = []
        visited = set()

        for i, cluster in enumerate(clusters):
            cluster_roots_in_this_cluster = [n for n in cluster if n in root_claims]

            if cluster_roots_in_this_cluster:
                cluster_roots_in_this_cluster.sort(key=lambda n: -centrality.get(n, 0))

                title_node_content = self.graph.nodes[cluster_roots_in_this_cluster[0]][
                    "content"
                ]

                cluster_title = f"议题 {i + 1}: 关于 “{title_node_content[:20]}...”"

            else:
                cluster_title = f"议题 {i + 1}: 其他相关论点"

            final_text_blocks.append(f"\n--- {cluster_title} ---\n")

            local_roots = [
                n
                for n in cluster
                if not any(
                    successor in cluster for successor in self.graph.successors(n)
                )
            ]

            if not local_roots:
                local_roots = cluster

            local_roots.sort(key=lambda n: -centrality.get(n, 0))

            for local_root_id in local_roots:
                if local_root_id not in visited:
                    block = self._serialize_node(local_root_id, visited)

                    if block:
                        final_text_blocks.append(block)

        return "\n".join(final_text_blocks).strip()

    def get_tactical_subgraph(self, focus_nodes: List[str]) -> "ShadowGraph":
        """Extract tactical subgraph around focus nodes and root claims.

        Args:
            focus_nodes: Candidate focus node ids.

        Returns:
            Subgraph containing root claims and neighborhood of valid focus nodes.
        """
        nodes_to_keep = set()

        roots = [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("metadata", {}).get("is_root_claim")
        ]

        nodes_to_keep.update(roots)

        if not focus_nodes:
            if not nodes_to_keep:
                return self.get_subgraph(list(self.graph.nodes()))

            return self.get_subgraph(list(nodes_to_keep))

        valid_focus = [n for n in focus_nodes if self.graph.has_node(n)]
        nodes_to_keep.update(valid_focus)

        for node in valid_focus:
            ancestors = nx.ancestors(self.graph, node)
            nodes_to_keep.update(ancestors)
            successors = self.graph.successors(node)
            nodes_to_keep.update(successors)

        return self.get_subgraph(list(nodes_to_keep))

    def to_tactical_text(self, focus_nodes: List[str]) -> str:
        """Render tactical subgraph as recursive text.

        Args:
            focus_nodes: Candidate focus node ids.

        Returns:
            Recursive textual representation of tactical subgraph.
        """
        subgraph = self.get_tactical_subgraph(focus_nodes)
        return subgraph.to_recursive_text()

    def _serialize_node(self, node_id: str, visited: set) -> str:
        """Recursively serializes a node and its predecessors into text."""
        if node_id in visited:
            return ""

        visited.add(node_id)
        data = self.graph.nodes[node_id]

        status_map = {
            NodeStatus.HYPOTHETICAL: "",
            NodeStatus.VALIDATED: "【已采信】",
            NodeStatus.DEFEATED: "【已驳回】",
        }

        current_status_str = status_map.get(data.get("status"), "")
        node_type = data["type"]

        if hasattr(node_type, "value"):
            node_type = node_type.value

        node_type = NodeType(node_type)
        agent_id = data.get("agent_id", "").lower()

        if node_type == NodeType.FACT:
            type_cn = "事实"

        elif node_type == NodeType.LAW:
            type_cn = "法条"

        elif node_type == NodeType.CLAIM:
            if "plaintiff" in agent_id:
                type_cn = "原告观点"

            elif "defendant" in agent_id:
                type_cn = "被告观点"

            else:
                type_cn = "核心诉求"

        else:
            type_cn = "节点"

        content = f"[{node_id}] [{type_cn}] {data['content']}{current_status_str}"
        preds = sorted(list(self.graph.predecessors(node_id)))
        supporting_texts = []
        conflicting_texts = []

        for pred_id in preds:
            edge_data = self.graph.get_edge_data(pred_id, node_id)
            edge_type = edge_data.get("type")
            sub_text = self._serialize_node(pred_id, visited)

            if not sub_text:
                continue

            if edge_type == EdgeType.SUPPORT:
                supporting_texts.append(sub_text)

            elif edge_type == EdgeType.CONFLICT:
                conflicting_texts.append(sub_text)

        result = content

        if supporting_texts:
            indented = [s.replace("\n", "\n  ") for s in supporting_texts]
            result += "\n  支持依据:\n    - " + "\n    - ".join(indented)

        if conflicting_texts:
            indented = [s.replace("\n", "\n  ") for s in conflicting_texts]
            result += "\n  受到反驳:\n    - " + "\n    - ".join(indented)

        return result

    def get_simple_id_list(self) -> str:
        """Generate a simple list of node ids and node types.

        Returns:
            Multiline text where each line is `- <id> (<type>)`.
        """
        ids = []

        for n, d in self.graph.nodes(data=True):
            n_type = d.get("type", "UNKNOWN")

            if hasattr(n_type, "value"):
                n_type = n_type.value

            ids.append(f"- {n} ({n_type})")

        if not ids:
            return "（当前图谱为空）"

        return "\n".join(ids)

    def get_nodes_by_step(self, step_index: int) -> List[str]:
        """Get node ids last modified at one specific step.

        Args:
            step_index: Target step index.

        Returns:
            Node id list whose metadata `last_modified_step` equals `step_index`.
        """
        target_nodes = []

        for nid, data in self.graph.nodes(data=True):
            meta = data.get("metadata", {})
            node_step = meta.get("last_modified_step")

            if node_step is not None and node_step == step_index:
                target_nodes.append(nid)

        return target_nodes


@dataclass
class LegalMessage:
    """A standardized message for storing and retrieving case data from memory.

    This dataclass encapsulates all necessary information about a completed case
    for the purpose of long-term memory and learning.

    Attributes:
        case_id: The unique identifier for the case.
        case_context: A concise, natural language summary of the case facts.
        shadow_graph: The final state of the `ShadowGraph` after adjudication.
    """

    case_id: str
    case_context: str
    shadow_graph: ShadowGraph = field(default_factory=ShadowGraph)

    @staticmethod
    def to_dict(msg: "LegalMessage") -> dict:
        """Serialize a `LegalMessage` object to dict.

        Args:
            msg: Message object to serialize.

        Returns:
            Dict with `case_id`, `case_context`, and serialized graph JSON.
        """
        return {
            "case_id": msg.case_id,
            "case_context": msg.case_context,
            "graph_json": msg.shadow_graph.to_json(),
        }

    @staticmethod
    def from_dict(data: dict) -> "LegalMessage":
        """Deserialize dict payload into `LegalMessage`.

        Args:
            data: Dict with `case_id`, `case_context`, and `graph_json`.

        Returns:
            Reconstructed `LegalMessage` instance.
        """
        sg = ShadowGraph.from_dict(json.loads(data["graph_json"]))

        return LegalMessage(
            case_id=data["case_id"], case_context=data["case_context"], shadow_graph=sg
        )
