import networkx as nx
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Iterator, Dict, List
from enum import Enum
from networkx.readwrite import json_graph
# 枚举定义
class NodeType(str, Enum):
    FACT = "FACT"
    LAW = "LAW"
    CLAIM = "CLAIM"

class NodeStatus(str, Enum):
    HYPOTHETICAL = "HYPOTHETICAL"  # 初始状态
    VALIDATED = "VALIDATED"        # 被 Verdict 确认为真
    DEFEATED = "DEFEATED"          # 被反驳成功

class EdgeType(str, Enum):
    SUPPORT = "SUPPORT"    # 支持关系
    CONFLICT = "CONFLICT"  # 冲突/反驳关系
# 核心数据结构
@dataclass
class ShadowNode:
    id: str
    content: str
    type: NodeType
    agent_id: str   # 提出节点的 Agent
    status: NodeStatus = NodeStatus.HYPOTHETICAL
    metadata: Dict[str, Any] = field(default_factory=dict)

def ensure_node_type(val: Any) -> NodeType:
    if isinstance(val, NodeType): return val
    if isinstance(val, str): return NodeType[val.upper()]
    raise ValueError(f"Invalid NodeType: {val}")

def ensure_edge_type(val: Any) -> EdgeType:
    if isinstance(val, EdgeType): return val
    if isinstance(val, str): return EdgeType[val.upper()]
    raise ValueError(f"Invalid EdgeType: {val}")
# 法律论辩过程中的语义图结构（G_t）
@dataclass
class ShadowGraph:
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    def __post_init__(self):
        if not hasattr(self.graph, "graph"): self.graph.graph = {}
        if "id_counter" not in self.graph.graph: self.graph.graph["id_counter"] = 0
    
    def add_node(self, content: str, node_type: NodeType, agent_id: str, 
                matcher: Any = None,
                metadata: dict = None) -> str:
        node_type = ensure_node_type(node_type)

        if matcher:
            candidates = [
                (n, d['content']) 
                for n, d in self.graph.nodes(data=True) 
                if d['type'] == node_type
            ]

            existing_id = matcher.find_match(content, candidates)
            if existing_id: return existing_id
        
        else:
            existing_id = self._find_semantically_identical_node(content, node_type)
            if existing_id: return existing_id
        
        node_id = self._generate_id(node_type)
        
        node = ShadowNode(
            id=node_id,
            content=content,
            type=node_type,
            agent_id=agent_id,
            metadata=metadata or {}
        )
        
        self.graph.add_node(node_id, **asdict(node))
        return node_id

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType) -> None:
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id): raise ValueError(f"Source {source_id} or Target {target_id} does not exist.")
        edge_type = ensure_edge_type(edge_type)
        self.graph.add_edge(source_id, target_id, type=edge_type)

    def get_subgraph(self, node_ids: List[str]) -> "ShadowGraph":
        sub_nx = self.graph.subgraph(node_ids).copy()
        new_sg = ShadowGraph()
        new_sg.graph = sub_nx
        new_sg.graph.graph["id_counter"] = self.graph.graph.get("id_counter", 0)
        return new_sg

    def _find_semantically_identical_node(self, content: str, node_type: NodeType) -> Optional[str]:
        for nid, data in self.graph.nodes(data=True):
            if data.get('type') == node_type and data.get('content', '').strip() == content.strip(): return nid
        
        return None

    def _generate_id(self, node_type: NodeType) -> str:
        if "id_counter" not in self.graph.graph: self.graph.graph["id_counter"] = self.graph.number_of_nodes()
        self.graph.graph["id_counter"] += 1
        return f"{node_type.value}_{self.graph.graph['id_counter']}"
    # 序列化方法
    @staticmethod
    def to_dict(sg: "ShadowGraph") -> dict: return json_graph.node_link_data(sg.graph)

    @staticmethod
    def from_dict(data: dict) -> "ShadowGraph":
        sg = ShadowGraph()
        loaded_graph = json_graph.node_link_graph(data)
        sg.graph = loaded_graph
        if "id_counter" not in sg.graph.graph: sg.graph.graph["id_counter"] = loaded_graph.number_of_nodes()
        return sg

    def to_json(self) -> str: return json.dumps(self.to_dict(self), ensure_ascii=False)
# 传输法律案件上下文
@dataclass
class LegalMessage:
    case_id: str
    case_context: str
    shadow_graph: ShadowGraph = field(default_factory=ShadowGraph)
    task_main: str = field(init=False) 
    def __post_init__(self): self.task_main = self.case_context

    @staticmethod
    def to_dict(msg: "LegalMessage") -> dict:
        return {
            "case_id": msg.case_id,
            "case_context": msg.case_context,
            "graph_json": msg.shadow_graph.to_json()
        }
    
    @staticmethod
    def from_dict(data: dict) -> "LegalMessage":
        sg = ShadowGraph.from_dict(json.loads(data["graph_json"]))
        return LegalMessage(
            case_id=data["case_id"],
            case_context=data["case_context"],
            shadow_graph=sg
        )