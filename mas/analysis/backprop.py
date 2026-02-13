"""Implements BAF-guided propagation for updating node statuses.

This module provides the `BackPropagator` class, which updates graph node
statuses according to a chosen BAF preferred extension.
"""

from typing import Optional, Set

from metagpt.logs import logger

from ..core.graph import EdgeType, NodeStatus, ShadowGraph


class BackPropagator:
    """Propagates final claim statuses through the argument graph via BAF."""

    def propagate_with_baf(
        self,
        graph: ShadowGraph,
        baf_extension: Set[str],
        root_claims_status: Optional[dict] = None,
    ) -> ShadowGraph:
        """Perform BAF-guided backpropagation on the graph.

        This method uses the BAF preferred extension as the guide for propagation,
        providing logically consistent status assignments based on formal BAF semantics.

        Propagation strategy:
        1. All nodes in BAF extension are marked VALIDATED
        2. VALIDATED status propagates backward through SUPPORT edges
        3. DEFEATED status is applied based on collective attack relationships:
           - If a node attacks a VALIDATED node → DEFEATED
           - If a node is attacked by a VALIDATED node → DEFEATED

        Args:
            graph: The ShadowGraph instance representing the final state
            baf_extension: Set of node IDs in the preferred extension
            root_claims_status: Optional dict of explicit root claim status for verification

        Returns:
            The same ShadowGraph instance with all node statuses updated
        """
        logger.info("[BackPropagator] Starting BAF-guided propagation...")
        nx_graph = graph.graph

        for nid in nx_graph.nodes():
            nx_graph.nodes[nid]["status"] = NodeStatus.HYPOTHETICAL

        validated_set = set()
        queue = []

        for nid in baf_extension:
            if nx_graph.has_node(nid):
                self._mark_validated(nx_graph, nid)
                validated_set.add(nid)
                queue.append(nid)

        logger.info(
            f"[BackPropagator] Marked {len(validated_set)} nodes as VALIDATED from BAF extension"
        )

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

        logger.info(
            f"[BackPropagator] After support propagation: {len(validated_set)} VALIDATED nodes"
        )
        defeated_count = 0

        for val_id in validated_set:
            for pred_id in nx_graph.predecessors(val_id):
                edge_data = nx_graph.get_edge_data(pred_id, val_id)
                edge_type = edge_data.get("type")

                if edge_type == EdgeType.CONFLICT:
                    if pred_id not in validated_set:
                        self._mark_defeated(nx_graph, pred_id)
                        defeated_count += 1

        for val_id in validated_set:
            for succ_id in nx_graph.successors(val_id):
                edge_data = nx_graph.get_edge_data(val_id, succ_id)
                edge_type = edge_data.get("type")

                if edge_type == EdgeType.CONFLICT:
                    if succ_id not in validated_set:
                        self._mark_defeated(nx_graph, succ_id)
                        defeated_count += 1

        logger.info(f"[BackPropagator] Marked {defeated_count} nodes as DEFEATED")

        if root_claims_status:
            self._verify_root_claim_alignment(graph, root_claims_status, validated_set)

        return graph

    def _verify_root_claim_alignment(
        self, graph: ShadowGraph, root_claims_status: dict, validated_set: Set[str]
    ):
        """Verify that root claim statuses align with propagation results.

        This is a consistency check to ensure the BAF-guided propagation
        produces results compatible with the expected root claim statuses.

        Args:
            graph: The ShadowGraph instance
            root_claims_status: Expected root claim status mapping
            validated_set: Set of nodes marked as VALIDATED
        """
        mismatches = []

        for root_id, expected_status in root_claims_status.items():
            if not graph.graph.has_node(root_id):
                continue

            actual_status = graph.graph.nodes[root_id]["status"]

            if expected_status != actual_status:
                mismatches.append(
                    {
                        "root_id": root_id,
                        "expected": expected_status,
                        "actual": actual_status,
                    }
                )

        if mismatches:
            logger.warning(
                f"[BackPropagator] Found {len(mismatches)} root claim status mismatches"
            )

            for mismatch in mismatches:
                logger.warning(
                    f"[BackPropagator] - {mismatch['root_id']}: "
                    f"expected {mismatch['expected']}, got {mismatch['actual']}"
                )

    def _mark_validated(self, nx_graph, node_id):
        """Set a node's status to VALIDATED."""
        nx_graph.nodes[node_id]["status"] = NodeStatus.VALIDATED

    def _mark_defeated(self, nx_graph, node_id):
        """Set a node's status to DEFEATED."""
        nx_graph.nodes[node_id]["status"] = NodeStatus.DEFEATED
