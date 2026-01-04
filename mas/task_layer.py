import os
import pickle
from dataclasses import dataclass
from typing import List

import networkx as nx

from .utils import file_lock


@dataclass
class TaskLayer:
    working_dir: str
    filename: str = "case_reference_graph.pkl"

    def __post_init__(self):
        self._graph_path = os.path.join(self.working_dir, self.filename)
        self._load_graph()

    def _load_graph(self):
        if os.path.exists(self._graph_path):
            try:
                with open(self._graph_path, "rb") as f:
                    self.graph = pickle.load(f)

            except Exception:
                self.graph = nx.Graph()

        else:
            self.graph = nx.Graph()

    def _save_graph(self):
        lock_file = self._graph_path + ".lock"

        with file_lock(lock_file):
            with open(self._graph_path, "wb") as f:
                pickle.dump(self.graph, f)

    def add_node(self, case_id: str):
        if case_id not in self.graph:
            self.graph.add_node(case_id)
            self._save_graph()

    def add_link(self, case_a: str, case_b: str):
        if case_a == case_b:
            return

        if not self.graph.has_node(case_a):
            self.graph.add_node(case_a)

        if not self.graph.has_node(case_b):
            self.graph.add_node(case_b)

        if not self.graph.has_edge(case_a, case_b):
            self.graph.add_edge(case_a, case_b)
            self._save_graph()

    def get_subgraph_components(self, case_ids: List[str]) -> List[List[str]]:
        if not case_ids:
            return []

        valid_nodes = [n for n in case_ids if self.graph.has_node(n)]

        if not valid_nodes:
            return [[n] for n in case_ids]

        subgraph = self.graph.subgraph(valid_nodes)
        components = list(nx.connected_components(subgraph))
        return [list(comp) for comp in components]

    def get_central_node(self, component_nodes: List[str]) -> str:
        if not component_nodes:
            return None

        if len(component_nodes) == 1:
            return component_nodes[0]

        subgraph = self.graph.subgraph(component_nodes)
        degrees = dict(subgraph.degree())

        central_node = max(
            degrees.items(),
            key=lambda x: (x[1], x[0] * -1 if isinstance(x[0], int) else x[0]),
        )[0]

        central_node = max(degrees.items(), key=lambda x: (x[1], x[0]))[0]

        return central_node
