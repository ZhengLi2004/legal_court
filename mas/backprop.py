from typing import Set
from .common import ShadowGraph, NodeType, NodeStatus, EdgeType
# 根据判决结果标记属性
class BackPropagator:
    def propagate(self, graph: ShadowGraph, winner_role: str) -> ShadowGraph:
        for nid, data in graph.graph.nodes(data=True):
            owner = data.get('agent_id')
            if owner == winner_role: data['status'] = NodeStatus.VALIDATED
            elif owner == "projection": data['status'] = NodeStatus.VALIDATED
            else: data['status'] = NodeStatus.DEFEATED
        
        return graph