"""Implements propagation strategies for updating node statuses.

This module provides the `BackPropagator` class, which updates graph node
statuses according to a chosen BAF preferred extension.
"""

from typing import Dict, Optional, Set

from metagpt.logs import logger

from ..core.graph import EdgeType, NodeStatus, NodeType, ShadowGraph


class BackPropagator:
    """Propagates final claim statuses through the argument graph via BAF."""

    def _node_type_name(self, nx_graph, node_id: str) -> str:
        raw_type = nx_graph.nodes[node_id].get("type")

        if isinstance(raw_type, NodeType):
            return raw_type.value

        text = str(raw_type).strip().upper()

        if text.startswith("NODETYPE."):
            text = text.split(".", 1)[1]

        return text

    def _is_fact_or_law(self, nx_graph, node_id: str) -> bool:
        return self._node_type_name(nx_graph, node_id) in {
            NodeType.FACT.value,
            NodeType.LAW.value,
        }

    def _coerce_status(self, value: object) -> NodeStatus:
        if isinstance(value, NodeStatus):
            return value

        text = str(value).strip().upper()

        if text.startswith("NODESTATUS."):
            text = text.split(".", 1)[1]

        try:
            return NodeStatus(text)

        except ValueError:
            return NodeStatus.HYPOTHETICAL

    def _propagate_defeated_from_validated(
        self,
        nx_graph,
        validated_set: Set[str],
        skip_defeat_for_fact_law: bool = True,
    ) -> int:
        defeated_nodes: Set[str] = set()

        for val_id in validated_set:
            for succ_id in nx_graph.successors(val_id):
                edge_data = nx_graph.get_edge_data(val_id, succ_id)

                if edge_data.get("type") != EdgeType.CONFLICT:
                    continue

                if succ_id in validated_set:
                    continue

                if skip_defeat_for_fact_law and self._is_fact_or_law(nx_graph, succ_id):
                    continue

                self._mark_defeated(nx_graph, succ_id)
                defeated_nodes.add(str(succ_id))

        return len(defeated_nodes)

    def propagate_from_root_status(
        self,
        graph: ShadowGraph,
        root_claims_status: Dict[str, object],
        reset_first: bool = True,
        skip_defeat_for_fact_law: bool = True,
    ) -> ShadowGraph:
        """Propagate statuses from root-claim verdicts before preferred search."""
        logger.info("[BackPropagator] Starting root-status pre-propagation...")
        nx_graph = graph.graph

        if reset_first:
            for nid in nx_graph.nodes():
                nx_graph.nodes[nid]["status"] = NodeStatus.HYPOTHETICAL

        validated_set: Set[str] = set()
        queue: list[str] = []

        for root_id, raw_status in (root_claims_status or {}).items():
            if not nx_graph.has_node(root_id):
                continue

            status = self._coerce_status(raw_status)

            if status == NodeStatus.VALIDATED:
                self._mark_validated(nx_graph, root_id)
                validated_set.add(str(root_id))
                queue.append(str(root_id))

            elif status == NodeStatus.DEFEATED:
                self._mark_defeated(nx_graph, root_id)

        while queue:
            curr_id = queue.pop()

            for pred_id in nx_graph.predecessors(curr_id):
                edge_data = nx_graph.get_edge_data(pred_id, curr_id)

                if edge_data.get("type") != EdgeType.SUPPORT:
                    continue

                if pred_id in validated_set:
                    continue

                current_status = self._coerce_status(
                    nx_graph.nodes[pred_id].get("status")
                )

                if current_status == NodeStatus.DEFEATED:
                    continue

                self._mark_validated(nx_graph, pred_id)
                validated_set.add(str(pred_id))
                queue.append(str(pred_id))

        defeated_count = self._propagate_defeated_from_validated(
            nx_graph=nx_graph,
            validated_set=validated_set,
            skip_defeat_for_fact_law=skip_defeat_for_fact_law,
        )

        logger.info(
            f"[BackPropagator] Pre-propagation done: "
            f"validated={len(validated_set)} defeated={defeated_count}"
        )

        return graph

    def propagate_with_baf(
        self,
        graph: ShadowGraph,
        baf_extension: Set[str],
        root_claims_status: Optional[dict] = None,
        reset_first: bool = True,
        skip_defeat_for_fact_law: bool = True,
    ) -> ShadowGraph:
        """Perform BAF-guided backpropagation on the graph.

        This method uses the BAF preferred extension as the guide for propagation,
        providing logically consistent status assignments based on formal BAF semantics.

        Propagation strategy:
        1. All nodes in BAF extension are marked VALIDATED
        2. VALIDATED status propagates backward through SUPPORT edges
        3. DEFEATED status is applied to direct CONFLICT targets of VALIDATED nodes

        Args:
            graph: The ShadowGraph instance representing the final state
            baf_extension: Set of node IDs in the preferred extension
            root_claims_status: Optional dict of explicit root claim status for verification

        Returns:
            The same ShadowGraph instance with all node statuses updated
        """
        logger.info("[BackPropagator] Starting BAF-guided propagation...")
        nx_graph = graph.graph

        if reset_first:
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

        defeated_count = self._propagate_defeated_from_validated(
            nx_graph=nx_graph,
            validated_set=validated_set,
            skip_defeat_for_fact_law=skip_defeat_for_fact_law,
        )

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
