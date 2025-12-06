import os
import pickle
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple
from .utils import simple_file_lock

@dataclass
class TaskLayer:
    working_dir: str
    similarity_threshold: float = 0.6

    def __post_init__(self):
        self._graph_path = os.path.join(self.working_dir, "case_graph.pkl")
        self._load_graph()
    # 图的存取
    def _load_graph(self):
        if os.path.exists(self._graph_path):
            with open(self._graph_path, 'rb') as f: self.graph = pickle.load(f)
        
        else: self.graph = nx.Graph()

    def _save_graph(self):
        lock_file = self._graph_path + ".lock"
        
        with simple_file_lock(lock_file):
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

    def get_k_hop_neighbors(self, anchor_ids: List[str], hop: int = 1) -> List[str]:
        result_set = set(anchor_ids)
        
        for anchor in anchor_ids:
            if anchor in self.graph:
                neighbors = nx.single_source_shortest_path_length(self.graph, anchor, cutoff=hop).keys()
                result_set.update(neighbors)
        
        return list(result_set)