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
        id_map: Dict[str, str] = {src_id: tgt_id for src_id, tgt_id in matched_pairs}
        nodes_to_project_ids: Set[str] = set(id_map.keys())

        for src_id, _ in matched_pairs:
            neighbors_to_process = set(source_graph.graph.successors(src_id)) | \
                                   set(source_graph.graph.predecessors(src_id))

            for neighbor_id in neighbors_to_process:
                neighbor_data = source_graph.graph.nodes[neighbor_id]
                node_type_val = neighbor_data.get('type')
                if hasattr(node_type_val, 'value'): node_type_val = node_type_val.value
                if str(node_type_val) == NodeType.FACT.value: continue
                nodes_to_project_ids.add(neighbor_id)

        for nid in nodes_to_project_ids:
            if nid in id_map: continue
            data = source_graph.graph.nodes[nid]

            new_id = target_graph.add_node(
                content=data['content'],
                node_type=data['type'],
                agent_id="projection",
                matcher=self.matcher
            )

            id_map[nid] = new_id

        for u, v, data in source_graph.graph.edges(data=True):
            if u in nodes_to_project_ids and v in nodes_to_project_ids:
                new_u = id_map.get(u)
                new_v = id_map.get(v)
                if new_u == new_v: continue

                if new_u and new_v and not target_graph.graph.has_edge(new_u, new_v):
                    edge_type_val = data.get('type')
                    target_graph.add_edge(new_u, new_v, edge_type=edge_type_val)