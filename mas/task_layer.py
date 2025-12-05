import os
import pickle
import networkx as nx
from dataclasses import dataclass
from typing import List, Set, Tuple

@dataclass
class TaskLayer:
    working_dir: str
    similarity_threshold: float = 0.75

    def __post_init__(self):
        self._graph_path = os.path.join(self.working_dir, "case_graph.pkl")
        self._load_graph()
    # 图的存取
    def _load_graph(self):
        if os.path.exists(self._graph_path):
            with open(self._graph_path, 'rb') as f: self.graph = pickle.load(f)
        
        else: self.graph = nx.Graph()

    def _save_graph(self):
        with open(self._graph_path, 'wb') as f: pickle.dump(self.graph, f)
    # 根据外部信息更新图拓扑
    def update_topology(self, new_case_id: str, neighbors: List[Tuple[str, float]]) -> None:
        if new_case_id not in self.graph: self.graph.add_node(new_case_id)
        
        for neighbor_id, score in neighbors:
            if neighbor_id == new_case_id: continue
            
            if score >= self.similarity_threshold:
                if neighbor_id not in self.graph: self.graph.add_node(neighbor_id)
                self.graph.add_edge(new_case_id, neighbor_id, weight=score)

        self._save_graph()
    # 添加案件节点
    def add_case_node(self, case_context: str, case_id: str) -> None:
        if case_id in self.graph: return
        # TODO: 实际生产中要配合 ChromaDB 查询，避免存两份。这里复刻逻辑，先存内存
        embedding = self.embedding_func.embed_query(case_context)
        self.graph.add_node(case_id, content=case_context, embedding=embedding)
        # TODO: 先做全量对比，后续优化
        for existing_id, data in self.graph.nodes(data=True):
            if existing_id == case_id: continue
            sim = cosine_similarity(embedding, data['embedding'])
            if sim >= self.similarity_threshold: self.graph.add_edge(case_id, existing_id, weight=sim)

        self._save_graph()
    # K-Hop 检索逻辑
    def retrieve_related_cases(self, query_context: str, top_k: int = 3, hop: int = 1) -> List[str]:
        query_emb = self.embedding_func.embed_query(query_context)
        scores = []
        
        for nid, data in self.graph.nodes(data=True):
            sim = cosine_similarity(query_emb, data['embedding'])
            scores.append((nid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        anchors = [nid for nid, _ in scores[:top_k]]
        result_set: Set[str] = set(anchors)

        for anchor in anchors:
            neighbors = nx.single_source_shortest_path_length(self.graph, anchor, cutoff=hop).keys()
            result_set.update(neighbors)

        return list(result_set)