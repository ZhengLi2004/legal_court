from typing import List
from .common import ShadowGraph, NodeType, EdgeType
from .semantic_matcher import SemanticMatcher
# 实现 Associative Projection
class GraphProjector:
    def __init__(self, matcher: SemanticMatcher): self.matcher = matcher

    def project(self, current_graph: ShadowGraph, history_graphs: List[ShadowGraph]) -> ShadowGraph:
        current_facts = [
            (nid, data['content'])
            for nid, data in current_graph.graph.nodes(data=True)
            if data['type'] == NodeType.FACT
        ]

        if not current_facts: return current_graph
        for h_graph in history_graphs: self._project_single_graph(current_graph, h_graph, current_facts)
        return current_graph
    
    def _project_single_graph(self, target_graph: ShadowGraph, source_graph: ShadowGraph, anchors: List[tuple]):
        source_candidates = [
            (nid, data['content'])
            for nid, data in source_graph.graph.nodes(data=True)
            if data['type'] == NodeType.FACT
        ]

        matched_pairs = []

        for tgt_id, tgt_content in anchors:
            src_id = self.matcher.find_match(tgt_content, source_candidates)
            if src_id: matched_pairs.append((src_id, tgt_id))

        for src_id, tgt_id in matched_pairs:
            for neighbor_id in source_graph.graph.successors(src_id):
                neighbor_data = source_graph.graph.nodes[neighbor_id]
                edge_data = source_graph.graph.get_edge_data(src_id, neighbor_id)
                
                if neighbor_data['type'] not in [NodeType.LAW, NodeType.CLAIM]: continue
                if edge_data['type'] != EdgeType.SUPPORT: continue

                new_node_id = target_graph.add_node(
                    content=neighbor_data['content'],
                    node_type=neighbor_data['type'],
                    agent_id="projection", # 标记来源
                    matcher=self.matcher   # 确保语义去重
                )

                if not target_graph.graph.has_edge(tgt_id, new_node_id): target_graph.add_edge(tgt_id, new_node_id, edge_type=edge_data['type'])