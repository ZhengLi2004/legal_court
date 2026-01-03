import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional, TypedDict, Union

import networkx as nx
from networkx.readwrite import json_graph


class BaseMetadata(TypedDict, total=False):
    created_at: float
    source_doc_id: str
    last_modified_step: int


class ClaimMetadata(BaseMetadata, total=False):
    is_root_claim: bool
    claim_index: int
    verdict_status: str


NodeMetadata = Union[BaseMetadata, ClaimMetadata]


class NodeType(str, Enum):
    FACT = "FACT"
    LAW = "LAW"
    CLAIM = "CLAIM"


class NodeStatus(str, Enum):
    HYPOTHETICAL = "HYPOTHETICAL"
    VALIDATED = "VALIDATED"
    DEFEATED = "DEFEATED"


class EdgeType(str, Enum):
    SUPPORT = "SUPPORT"
    CONFLICT = "CONFLICT"


class EdgeAddResult(Enum):
    CREATED = auto()
    DUPLICATE = auto()
    TYPE_CLASH = auto()
    SELF_LOOP = auto()


@dataclass
class ShadowNode:
    id: str
    content: str
    type: NodeType
    agent_id: str
    status: NodeStatus = NodeStatus.HYPOTHETICAL
    metadata: NodeMetadata = field(default_factory=dict)


def ensure_node_type(val: Any) -> NodeType:
    if isinstance(val, NodeType):
        return val

    if isinstance(val, str):
        return NodeType[val.upper()]

    raise ValueError(f"Invalid NodeType: {val}")


def ensure_edge_type(val: Any) -> EdgeType:
    if isinstance(val, EdgeType):
        return val

    if isinstance(val, str):
        return EdgeType[val.upper()]

    raise ValueError(f"Invalid EdgeType: {val}")


@dataclass
class ShadowGraph:
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    latest_context: str = field(default="")

    def __post_init__(self):
        if not hasattr(self.graph, "graph"):
            self.graph.graph = {}

    def get_id_inventory(self) -> str:
        lines = []

        for n, d in self.graph.nodes(data=True):
            n_type = d.get("type", "UNKNOWN")

            if hasattr(n_type, "value"):
                n_type = n_type.value

            content = d.get("content", "")[:30].replace("\n", " ")
            lines.append(f"- [{n}] {n_type}: {content}...")

        return "\n".join(lines)

    def _calculate_focus_nodes(self, current_step: int) -> List[str]:
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
        focus_nodes = self._calculate_focus_nodes(current_step)

        if focus_nodes:
            self.latest_context = self.to_tactical_text(focus_nodes)

        else:
            self.latest_context = self.to_recursive_text()

        if not self.latest_context:
            self.latest_context = "（当前辩论图谱为空）"

    def touch_nodes(self, node_ids: List[str], step_index: int):
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

    def is_valid_connection(
        self, src_id: str, tgt_id: str, edge_type: EdgeType
    ) -> bool:
        if not self.graph.has_node(src_id) or not self.graph.has_node(tgt_id):
            return False

        src_type = self._get_node_type(src_id)
        tgt_type = self._get_node_type(tgt_id)
        s, t = src_type.value, tgt_type.value
        F, _, C = NodeType.FACT.value, NodeType.LAW.value, NodeType.CLAIM.value

        if tgt_type == NodeType.LAW:
            return False

        if edge_type == EdgeType.SUPPORT and s == C and t == F:
            return False

        return True

    def is_valid_evidence(self, node_id: str) -> bool:
        if not self.graph.has_node(node_id):
            return False

        t = self._get_node_type(node_id)
        return t.value in [NodeType.FACT.value, NodeType.LAW.value]

    def _get_node_type(self, node_id: str) -> NodeType:
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

        if not self.is_valid_connection(source_id, target_id, edge_type):
            raise ValueError(
                f"Invalid connection: {edge_type} from {self._get_node_type(source_id)} to {self._get_node_type(target_id)}"
            )

        self.graph.add_edge(source_id, target_id, type=edge_type)
        return EdgeAddResult.CREATED

    def get_subgraph(self, node_ids: List[str]) -> "ShadowGraph":
        sub_nx = self.graph.subgraph(node_ids).copy()
        new_sg = ShadowGraph()
        new_sg.graph = sub_nx
        return new_sg

    def _find_semantically_identical_node(
        self, content: str, node_type: NodeType
    ) -> Optional[str]:
        for nid, data in self.graph.nodes(data=True):
            if (
                data.get("type") == node_type
                and data.get("content", "").strip() == content.strip()
            ):
                return nid

        return None

    def _generate_id(self, node_type: NodeType) -> str:
        uid = uuid.uuid4().hex[:8]
        return f"{node_type.value}_{uid}"

    @staticmethod
    def to_dict(sg: "ShadowGraph") -> dict:
        graph_data = json_graph.node_link_data(sg.graph)
        return {"graph_data": graph_data}

    @staticmethod
    def from_dict(data: dict) -> "ShadowGraph":
        sg = ShadowGraph()
        sg.graph = json_graph.node_link_graph(data["graph_data"])
        return sg

    def to_json(self) -> str:
        return json.dumps(self.to_dict(self), ensure_ascii=False)

    def to_recursive_text(self) -> str:
        if self.graph.number_of_nodes() == 0:
            return "无辩论记录。"

        root_claims = [
            nid
            for nid, data in self.graph.nodes(data=True)
            if data.get("metadata", {}).get("is_root_claim")
        ]

        try:
            centrality = nx.pagerank(self.graph)

        except Exception:
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

    def get_tactical_subgraph(
        self, focus_nodes: List[str], history_window: int = 1
    ) -> "ShadowGraph":
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
            pass

        return self.get_subgraph(list(nodes_to_keep))

    def to_tactical_text(self, focus_nodes: List[str]) -> str:
        subgraph = self.get_tactical_subgraph(focus_nodes)
        return subgraph.to_recursive_text()

    def _serialize_node(self, node_id: str, visited: set) -> str:
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
        type_map = {NodeType.FACT: "事实", NodeType.LAW: "法条", NodeType.CLAIM: "观点"}
        type_cn = type_map.get(data["type"], data["type"].value)
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

    def get_nodes_by_step(self, step_index: int) -> List[str]:
        target_nodes = []

        for nid, data in self.graph.nodes(data=True):
            meta = data.get("metadata", {})
            node_step = meta.get("last_modified_step")

            if node_step is not None and node_step == step_index:
                target_nodes.append(nid)

        return target_nodes


@dataclass
class LegalMessage:
    case_id: str
    case_context: str
    shadow_graph: ShadowGraph = field(default_factory=ShadowGraph)
    task_main: str = field(init=False)

    def __post_init__(self):
        self.task_main = self.case_context

    @staticmethod
    def to_dict(msg: "LegalMessage") -> dict:
        return {
            "case_id": msg.case_id,
            "case_context": msg.case_context,
            "graph_json": msg.shadow_graph.to_json(),
        }

    @staticmethod
    def from_dict(data: dict) -> "LegalMessage":
        sg = ShadowGraph.from_dict(json.loads(data["graph_json"]))

        return LegalMessage(
            case_id=data["case_id"], case_context=data["case_context"], shadow_graph=sg
        )
