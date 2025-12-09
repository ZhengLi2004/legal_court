from typing import List, Set
from .common import ShadowGraph, NodeStatus, EdgeType
# 根据判决结果标记属性
class BackPropagator:
    def propagate(self, graph: ShadowGraph, explicit_validated_ids: List[str]) -> ShadowGraph:
        nx_graph = graph.graph
        for nid in nx_graph.nodes(): nx_graph.nodes[nid]['status'] = NodeStatus.HYPOTHETICAL
        queue = []
        validated_set = set()

        for nid in explicit_validated_ids:
            if nx_graph.has_node(nid):
                self._mark_validated(nx_graph, nid)
                queue.append(nid)
                validated_set.add(nid)

        while queue:
            curr_id = queue.pop()

            for pred_id in nx_graph.predecessors(curr_id):
                edge_data = nx_graph.get_edge_data(pred_id, curr_id)
                edge_type = edge_data.get('type')
                # 只有 SUPPORT 边传递真值
                if edge_type == EdgeType.SUPPORT:
                    if pred_id not in validated_set:
                        self._mark_validated(nx_graph, pred_id)
                        validated_set.add(pred_id)
                        queue.append(pred_id)

        for val_id in validated_set:
            for pred_id in nx_graph.predecessors(val_id):
                edge_data = nx_graph.get_edge_data(pred_id, val_id)
                edge_type = edge_data.get('type')

                if edge_type == EdgeType.CONFLICT:
                    attacker_status = nx_graph.nodes[pred_id]['status']
                    if attacker_status != NodeStatus.VALIDATED: self._mark_defeated(nx_graph, pred_id)

            for succ_id in nx_graph.successors(val_id):
                edge_data = nx_graph.get_edge_data(val_id, succ_id)
                edge_type = edge_data.get('type')

                if edge_type == EdgeType.CONFLICT:
                    attacked_status = nx_graph.nodes[succ_id]['status']
                    if attacked_status != NodeStatus.VALIDATED: self._mark_defeated(nx_graph, succ_id)

        return graph
    
    def _mark_validated(self, nx_graph, node_id): nx_graph.nodes[node_id]['status'] = NodeStatus.VALIDATED
    def _mark_defeated(self, nx_graph, node_id): nx_graph.nodes[node_id]['status'] = NodeStatus.DEFEATED