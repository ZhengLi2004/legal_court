from typing import List, Dict, Set
from .common import ShadowGraph, NodeType
from .semantic_matcher import SemanticMatcher

class GraphProjector:
    def __init__(self, matcher: SemanticMatcher): self.matcher = matcher

    def project(self, current_graph: ShadowGraph, history_graphs: List[ShadowGraph]) -> ShadowGraph:
        anchors = [
            (nid, data['content'])
            for nid, data in current_graph.graph.nodes(data=True)
        ]

        if not anchors: return current_graph
        for h_graph in history_graphs: self._project_single_graph(current_graph, h_graph, anchors)
        return current_graph

    def _project_single_graph(self, target_graph: ShadowGraph, source_graph: ShadowGraph, anchors: List[tuple]):
        source_candidates = [
            (nid, data['content'])
            for nid, data in source_graph.graph.nodes(data=True)
        ]

        matched_pairs = []

        for tgt_id, tgt_content in anchors:
            src_id = self.matcher.find_match(tgt_content, source_candidates)
            if src_id: matched_pairs.append((src_id, tgt_id))

        if not matched_pairs: return
        id_map: Dict[str, str] = {}
        for src_id, tgt_id in matched_pairs: id_map[src_id] = tgt_id
        nodes_to_copy: Dict[str, Dict] = {}

        for src_id, _ in matched_pairs:
            neighbors_to_process = set(source_graph.graph.successors(src_id)) | \
                                   set(source_graph.graph.predecessors(src_id))

            for neighbor_id in neighbors_to_process:
                if neighbor_id in id_map: continue  # 锚点本身，不复制
                neighbor_data = source_graph.graph.nodes[neighbor_id]
                node_type_val = neighbor_data.get('type')
                if hasattr(node_type_val, 'value'): node_type_val = node_type_val.value
                if str(node_type_val) == NodeType.FACT.value: continue
                nodes_to_copy[neighbor_id] = neighbor_data

        for nid, data in nodes_to_copy.items():
            new_id = target_graph.add_node(
                content=data['content'],
                node_type=data['type'],
                agent_id="projection",
                matcher=self.matcher
            )

            id_map[nid] = new_id

        for src_id, _ in matched_pairs:
            for successor_id in source_graph.graph.successors(src_id):
                if src_id in id_map and successor_id in id_map:
                    new_u, new_v = id_map[src_id], id_map[successor_id]
                    
                    if not target_graph.graph.has_edge(new_u, new_v):
                        edge_data = source_graph.graph.get_edge_data(src_id, successor_id)
                        edge_type_val = edge_data.get('type')
                        target_graph.add_edge(new_u, new_v, edge_type=edge_type_val)

            for predecessor_id in source_graph.graph.predecessors(src_id):
                if predecessor_id in id_map and src_id in id_map:
                    new_u, new_v = id_map[predecessor_id], id_map[src_id]
                    
                    if not target_graph.graph.has_edge(new_u, new_v):
                        edge_data = source_graph.graph.get_edge_data(predecessor_id, src_id)
                        edge_type_val = edge_data.get('type')
                        target_graph.add_edge(new_u, new_v, edge_type=edge_type_val)