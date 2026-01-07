"""Manages a graph representing the relationships between legal cases.

This module provides the `TaskLayer` class, which maintains a persistent,
undirected graph where nodes are cases and edges represent a relationship
(e.g., they share the same winning strategy). This "topology" allows the
system to reason about clusters of related cases.
"""

import os
import pickle
from dataclasses import dataclass
from typing import List

import networkx as nx

from .utils import file_lock


@dataclass
class TaskLayer:
    """Manages the case topology graph.

    This class is responsible for loading, saving, and manipulating a graph of
    case relationships. It uses a pickled `networkx.Graph` object for
    persistence. The graph helps in retrieving a broader context of related
    cases when a single historical case is found to be relevant.

    Attributes:
        working_dir: The directory where the graph file is stored.
        filename: The name of the pickle file for the graph.
        graph: The `networkx.Graph` instance.
    """

    working_dir: str
    filename: str = "case_reference_graph.pkl"

    def __post_init__(self):
        """Initialize the TaskLayer by loading the graph from the file."""
        self._graph_path = os.path.join(self.working_dir, self.filename)
        self._load_graph()

    def _load_graph(self):
        """Load the graph from a pickle file, creating an empty one if not found."""
        if os.path.exists(self._graph_path):
            try:
                with open(self._graph_path, "rb") as f:
                    self.graph = pickle.load(f)

            except Exception:
                self.graph = nx.Graph()

        else:
            self.graph = nx.Graph()

    def _save_graph(self):
        """Save the current graph to a pickle file using a file lock."""
        lock_file = self._graph_path + ".lock"

        with file_lock(lock_file):
            with open(self._graph_path, "wb") as f:
                pickle.dump(self.graph, f)

    def add_node(self, case_id: str):
        """Add a new case node to the graph.

        Args:
            case_id: The unique identifier for the case.
        """
        if case_id not in self.graph:
            self.graph.add_node(case_id)
            self._save_graph()

    def add_link(self, case_a: str, case_b: str):
        """Add an edge between two case nodes.

        Args:
            case_a: The ID of the first case.
            case_b: The ID of the second case.
        """
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
        """Find all connected components within a subgraph of specified cases.

        Given a list of case IDs, this method finds which of them are connected
        to each other in the larger topology graph and returns them as a list
        of components (each component being a list of case IDs).

        Args:
            case_ids: A list of case IDs to analyze.

        Returns:
            A list of lists, where each inner list is a connected component of cases.
        """
        if not case_ids:
            return []

        valid_nodes = [n for n in case_ids if self.graph.has_node(n)]

        if not valid_nodes:
            return [[n] for n in case_ids]  # Return as singletons if not in graph

        subgraph = self.graph.subgraph(valid_nodes)
        components = list(nx.connected_components(subgraph))
        return [list(comp) for comp in components]

    def get_central_node(self, component_nodes: List[str]) -> str:
        """Find the most central node within a connected component.

        Centrality is determined primarily by the node's degree (number of
        connections) within the component.

        Args:
            component_nodes: A list of case IDs forming a single connected component.

        Returns:
            The ID of the most central case in the component.
        """
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
