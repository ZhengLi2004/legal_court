"""Implements graph-status propagation after root-claim adjudication."""

from typing import Dict, Set

from ..core.graph import EdgeType, NodeStatus, NodeType, ShadowGraph


class BackPropagator:
    """Propagates final claim statuses through the argument graph."""

    def _edge_type_name(self, edge_type: object) -> str:
        """Normalize edge type payload to uppercase enum name text.

        Args:
            edge_type: Raw edge type value stored in graph edge metadata.

        Returns:
            Canonical edge type name such as `SUPPORT` or `CONFLICT`.
        """
        if isinstance(edge_type, EdgeType):
            return edge_type.value

        text = str(edge_type).strip().upper()

        if text.startswith("EDGETYPE."):
            text = text.split(".", 1)[1]

        return text

    def _node_type_name(self, nx_graph, node_id: str) -> str:
        """Normalize a node's stored type value to an uppercase enum name.

        Args:
            nx_graph: Underlying NetworkX graph instance.
            node_id: Node ID whose type should be read.

        Returns:
            Canonical node-type text such as `CLAIM`, `FACT`, or `LAW`.
        """
        raw_type = nx_graph.nodes[node_id].get("type")

        if isinstance(raw_type, NodeType):
            return raw_type.value

        text = str(raw_type).strip().upper()

        if text.startswith("NODETYPE."):
            text = text.split(".", 1)[1]

        return text

    def _is_fact_or_law(self, nx_graph, node_id: str) -> bool:
        """Check whether a node is a FACT or LAW node.

        Args:
            nx_graph: Underlying NetworkX graph instance.
            node_id: Node ID to classify.

        Returns:
            `True` when the node type is FACT or LAW.
        """
        return self._node_type_name(nx_graph, node_id) in {
            NodeType.FACT.value,
            NodeType.LAW.value,
        }

    def _coerce_status(self, value: object) -> NodeStatus:
        """Convert a raw status payload into a `NodeStatus` enum value.

        Args:
            value: Raw status object from external input or graph metadata.

        Returns:
            A valid `NodeStatus`, defaulting to `HYPOTHETICAL` on parse failure.
        """
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
        """Mark direct conflict targets of validated claims as defeated.

        Args:
            nx_graph: Underlying NetworkX graph instance.
            validated_set: IDs currently marked as validated.
            skip_defeat_for_fact_law: Whether FACT/LAW nodes are exempted from
                automatic defeat.

        Returns:
            Number of nodes newly marked as defeated.
        """
        defeated_nodes: Set[str] = set()

        for val_id in validated_set:
            for succ_id in nx_graph.successors(val_id):
                edge_data = nx_graph.get_edge_data(val_id, succ_id)

                if (
                    self._edge_type_name(edge_data.get("type"))
                    != EdgeType.CONFLICT.value
                ):
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
        """Propagate statuses from adjudicated root claims across the graph.

        Args:
            graph: Debate graph to update in place.
            root_claims_status: Mapping from root-claim IDs to expected statuses.
            reset_first: Whether to reset all nodes to `HYPOTHETICAL` first.
            skip_defeat_for_fact_law: Whether FACT/LAW nodes are exempt from
                automatic conflict defeat.

        Returns:
            The same graph instance with updated node statuses.
        """
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

                if (
                    self._edge_type_name(edge_data.get("type"))
                    != EdgeType.SUPPORT.value
                ):
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

        self._propagate_defeated_from_validated(
            nx_graph=nx_graph,
            validated_set=validated_set,
            skip_defeat_for_fact_law=skip_defeat_for_fact_law,
        )

        return graph

    def _mark_validated(self, nx_graph, node_id):
        """Set a node's status to VALIDATED."""
        nx_graph.nodes[node_id]["status"] = NodeStatus.VALIDATED

    def _mark_defeated(self, nx_graph, node_id):
        """Set a node's status to DEFEATED."""
        nx_graph.nodes[node_id]["status"] = NodeStatus.DEFEATED
