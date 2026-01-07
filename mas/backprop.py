"""Implements the backpropagation algorithm for updating node statuses in the debate graph.

This module provides the `BackPropagator` class, which is responsible for
propagating the final verdict (e.g., which root claims were accepted)
back through the argument graph. This process updates the status of all nodes
to `VALIDATED`, `DEFEATED`, or `HYPOTHETICAL` based on their relationship
to the final accepted claims.
"""

from typing import List

from .common import EdgeType, NodeStatus, ShadowGraph


class BackPropagator:
    """Propagates final claim statuses through the argument graph.

    This class implements a graph traversal algorithm to determine the final
    status of every node based on an initial set of validated nodes (typically
    the root claims accepted by the judge).

    The propagation logic is as follows:
    1.  All nodes are initially reset to `HYPOTHETICAL`.
    2.  An initial set of nodes (e.g., judge-accepted root claims) are marked
        as `VALIDATED`.
    3.  Validation status propagates backward through `SUPPORT` edges. Any node
        that supports a `VALIDATED` node also becomes `VALIDATED`.
    4.  Once all validation propagation is complete, defeat status is
        propagated through `CONFLICT` edges. A node is marked as `DEFEATED` if
        it is attacked by a `VALIDATED` node, or if it attacks a `VALIDATED` node.
    """

    def propagate(
        self, graph: ShadowGraph, explicit_validated_ids: List[str]
    ) -> ShadowGraph:
        """Perform backpropagation on the graph to update all node statuses.

        Args:
            graph: The `ShadowGraph` instance representing the final state of
                the debate.
            explicit_validated_ids: A list of node IDs that are explicitly
                marked as validated, typically the root claims accepted by the judge.

        Returns:
            The same `ShadowGraph` instance with all node statuses updated.
        """
        nx_graph = graph.graph

        for nid in nx_graph.nodes():
            nx_graph.nodes[nid]["status"] = NodeStatus.HYPOTHETICAL

        queue = []
        validated_set = set()

        for nid in explicit_validated_ids:
            if nx_graph.has_node(nid):
                self._mark_validated(nx_graph, nid)
                queue.append(nid)
                validated_set.add(nid)

        initial_anchors = set(explicit_validated_ids)

        while queue:
            curr_id = queue.pop()

            for pred_id in nx_graph.predecessors(curr_id):
                edge_data = nx_graph.get_edge_data(pred_id, curr_id)
                edge_type = edge_data.get("type")

                if edge_type == EdgeType.SUPPORT:
                    if pred_id not in validated_set:
                        self._mark_validated(nx_graph, pred_id)
                        validated_set.add(pred_id)
                        queue.append(pred_id)

        for val_id in validated_set:
            for pred_id in nx_graph.predecessors(val_id):
                edge_data = nx_graph.get_edge_data(pred_id, val_id)
                edge_type = edge_data.get("type")

                if edge_type == EdgeType.CONFLICT:
                    attacker_status = nx_graph.nodes[pred_id]["status"]

                    if attacker_status != NodeStatus.VALIDATED:
                        self._mark_defeated(nx_graph, pred_id)

            for succ_id in nx_graph.successors(val_id):
                edge_data = nx_graph.get_edge_data(val_id, succ_id)
                edge_type = edge_data.get("type")

                if edge_type == EdgeType.CONFLICT:
                    if succ_id not in initial_anchors:
                        self._mark_defeated(nx_graph, succ_id)

        return graph

    def _mark_validated(self, nx_graph, node_id):
        """Set a node's status to VALIDATED."""
        nx_graph.nodes[node_id]["status"] = NodeStatus.VALIDATED

    def _mark_defeated(self, nx_graph, node_id):
        """Set a node's status to DEFEATED."""
        nx_graph.nodes[node_id]["status"] = NodeStatus.DEFEATED
