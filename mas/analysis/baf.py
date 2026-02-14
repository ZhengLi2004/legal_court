"""Implements Bipolar Argumentation Framework (BAF) semantics for legal debate.

This module provides `BAFCalculator` that:
- builds collective attacks (direct, support-based, indirect),
- computes preferred extensions (maximal admissible sets),
- exposes context-selection utilities shared by judge prompting.
"""

from __future__ import annotations

import time
from collections import deque
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Set, Tuple

from metagpt.logs import logger

from ..core.graph import EdgeType, NodeType, ShadowGraph


class BAFComputationError(RuntimeError):
    """Raised when BAF computation encounters a hard failure."""

    def __init__(
        self,
        code: str,
        message: str,
        stats: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a BAF computation error with structured metadata.

        Args:
            code: Stable error code used by API and logging layers.
            message: Human-readable failure message.
            stats: Optional search or graph statistics captured at failure time.
        """
        super().__init__(message)
        self.code = str(code).strip() or "BAF_ERROR"
        self.stats = stats or {}

    def __str__(self) -> str:
        """Render the error as `<code>: <message>` when possible.

        Returns:
            A formatted error string suitable for logs and responses.
        """
        base = super().__str__()
        return f"{self.code}: {base}" if base else self.code


class CollectiveAttackType:
    """Types of collective attacks in BAF."""

    DIRECT = "direct"
    SUPPORT_BASED = "support_based"
    INDIRECT = "indirect"


class BAFCalculator:
    """Calculates BAF semantics for the debate graph.

    Preferred-extension search is exact.
    """

    def __init__(self, graph: ShadowGraph):
        """Initialize indexes and attack closures for a debate graph.

        Args:
            graph: Debate graph on which BAF semantics will be computed.
        """
        self.graph = graph
        self.algorithm_version = "baf_exact_v2"
        self.collective_attacks: Dict[str, Dict[str, List[str]]] = {}
        self.attack_matrix: Dict[str, Set[str]] = {}
        self.attacks_from: Dict[str, Set[str]] = {}
        self.support_successors: Dict[str, Set[str]] = {}
        self.support_predecessors: Dict[str, Set[str]] = {}
        self.direct_conflicts: Dict[str, Set[str]] = {}
        self._support_reachability: Dict[str, Set[str]] = {}
        self._all_nodes_sorted: List[str] = []
        self._undirected_neighbors: Dict[str, Set[str]] = {}
        self._search_started_ms = 0.0
        self._searched_states = 0
        self._pruned_states = 0
        self._last_search_stats: Dict[str, Any] = {}
        self._last_context_selection: Dict[str, Any] = {}
        self._build_edge_indexes()
        self._compute_all_attacks()

    def _edge_type_name(self, value: Any) -> str:
        """Normalize a raw edge type value to canonical uppercase text.

        Args:
            value: Raw edge type value from graph metadata.

        Returns:
            Canonical edge type name, such as `SUPPORT` or `CONFLICT`.
        """
        if isinstance(value, EdgeType):
            return value.value

        text = str(value).strip().upper()

        if text.startswith("EDGETYPE."):
            text = text.split(".", 1)[1]

        return text

    def _node_type_name(self, node_id: str) -> str:
        """Normalize a node's raw type field to canonical uppercase text.

        Args:
            node_id: Node ID whose type should be resolved.

        Returns:
            Canonical node type name, such as `CLAIM`, `FACT`, or `LAW`.
        """
        data = self.graph.graph.nodes.get(node_id, {})
        raw_type = data.get("type")

        if isinstance(raw_type, NodeType):
            return raw_type.value

        text = str(raw_type).strip().upper()

        if text.startswith("NODETYPE."):
            text = text.split(".", 1)[1]

        return text

    def _is_fact_or_law_node(self, node_id: str) -> bool:
        """Check whether a node is a FACT or LAW node.

        Args:
            node_id: Node ID to classify.

        Returns:
            `True` when the node type is FACT or LAW.
        """
        return self._node_type_name(node_id) in {
            NodeType.FACT.value,
            NodeType.LAW.value,
        }

    def _is_claim_node(self, node_id: str) -> bool:
        """Check whether a node is a CLAIM node.

        Args:
            node_id: Node ID to classify.

        Returns:
            `True` when the node type is CLAIM.
        """
        return self._node_type_name(node_id) == NodeType.CLAIM.value

    def _build_edge_indexes(self) -> None:
        """Precompute adjacency indexes used by BAF attack derivations."""
        graph = self.graph.graph
        nodes = [str(n) for n in graph.nodes()]
        self._all_nodes_sorted = sorted(nodes)
        self.support_successors = {node: set() for node in self._all_nodes_sorted}
        self.support_predecessors = {node: set() for node in self._all_nodes_sorted}
        self.direct_conflicts = {node: set() for node in self._all_nodes_sorted}
        self._undirected_neighbors = {node: set() for node in self._all_nodes_sorted}

        for src_raw, tgt_raw, data in graph.edges(data=True):
            src = str(src_raw)
            tgt = str(tgt_raw)
            edge_name = self._edge_type_name(data.get("type"))
            self._undirected_neighbors.setdefault(src, set()).add(tgt)
            self._undirected_neighbors.setdefault(tgt, set()).add(src)

            if edge_name == EdgeType.SUPPORT.value:
                self.support_successors.setdefault(src, set()).add(tgt)
                self.support_predecessors.setdefault(tgt, set()).add(src)

            elif edge_name == EdgeType.CONFLICT.value:
                self.direct_conflicts.setdefault(src, set()).add(tgt)

        self._support_reachability = {
            node: self._compute_support_reachability(node)
            for node in self._all_nodes_sorted
        }

    def _compute_support_reachability(self, source: str) -> Set[str]:
        """Return SUPPORT* reachability set including source itself."""
        visited = {source}
        queue = deque([source])

        while queue:
            current = queue.popleft()

            for nxt in self.support_successors.get(current, set()):
                if nxt in visited:
                    continue

                visited.add(nxt)
                queue.append(nxt)

        return visited

    def _register_attack(
        self,
        source: str,
        target: str,
        attack_type: str,
        type_buckets: Dict[str, Dict[str, Set[str]]],
    ) -> None:
        """Register one computed attack relation in all internal indexes.

        Args:
            source: Attacker node ID.
            target: Target node ID.
            attack_type: Attack category name.
            type_buckets: Mutable map used to aggregate typed attacker sets.
        """
        if source == target:
            return

        self.attacks_from.setdefault(source, set()).add(target)
        self.attack_matrix.setdefault(target, set()).add(source)
        type_buckets.setdefault(target, {}).setdefault(attack_type, set()).add(source)

    def _compute_all_attacks(self) -> None:
        """Compute collective attack closures."""
        logger.info("[BAF] Computing collective attacks...")

        self.collective_attacks = {
            node: {"direct": [], "support_based": [], "indirect": []}
            for node in self._all_nodes_sorted
        }

        self.attack_matrix = {node: set() for node in self._all_nodes_sorted}
        self.attacks_from = {node: set() for node in self._all_nodes_sorted}

        typed_attackers: Dict[str, Dict[str, Set[str]]] = {
            node: {
                CollectiveAttackType.DIRECT: set(),
                CollectiveAttackType.SUPPORT_BASED: set(),
                CollectiveAttackType.INDIRECT: set(),
            }
            for node in self._all_nodes_sorted
        }

        for src in self._all_nodes_sorted:
            direct_targets = self.direct_conflicts.get(src, set())

            for tgt in direct_targets:
                self._register_attack(
                    source=src,
                    target=tgt,
                    attack_type=CollectiveAttackType.DIRECT,
                    type_buckets=typed_attackers,
                )

            support_reachable = self._support_reachability.get(src, {src})

            for mid in support_reachable:
                for tgt in self.direct_conflicts.get(mid, set()):
                    self._register_attack(
                        source=src,
                        target=tgt,
                        attack_type=CollectiveAttackType.SUPPORT_BASED,
                        type_buckets=typed_attackers,
                    )

            for mid in direct_targets:
                downstream = self._support_reachability.get(mid, {mid})

                for tgt in downstream:
                    self._register_attack(
                        source=src,
                        target=tgt,
                        attack_type=CollectiveAttackType.INDIRECT,
                        type_buckets=typed_attackers,
                    )

        total_direct = 0
        total_support = 0
        total_indirect = 0

        for target in self._all_nodes_sorted:
            direct = sorted(typed_attackers[target][CollectiveAttackType.DIRECT])

            support_based = sorted(
                typed_attackers[target][CollectiveAttackType.SUPPORT_BASED]
                - set(direct)
            )

            indirect = sorted(
                typed_attackers[target][CollectiveAttackType.INDIRECT]
                - set(direct)
                - set(support_based)
            )

            self.collective_attacks[target] = {
                "direct": direct,
                "support_based": support_based,
                "indirect": indirect,
            }

            total_direct += len(direct)
            total_support += len(support_based)
            total_indirect += len(indirect)

        logger.info(
            f"[BAF] Found {total_direct} direct, {total_support} support-based, "
            f"{total_indirect} indirect attacks"
        )

    def _has_internal_conflict(self, node: str, node_set: Set[str]) -> bool:
        """Check whether `node` attacks or is attacked by nodes in `node_set`.

        Args:
            node: Candidate node to test.
            node_set: Nodes currently included in a candidate extension.

        Returns:
            `True` if adding or keeping `node` violates conflict-freeness.
        """
        if not node_set:
            return False

        return bool(
            self.attack_matrix.get(node, set()) & node_set
            or self.attacks_from.get(node, set()) & node_set
        )

    def is_conflict_free(self, node_set: Set[str]) -> bool:
        """Check if a set of nodes is conflict-free."""
        for node in node_set:
            if self._has_internal_conflict(node, node_set - {node}):
                return False

        return True

    def defends(self, defender_set: Set[str], node: str) -> bool:
        """Check whether `defender_set` defends `node`."""
        if not self._is_claim_node(node):
            return True

        attackers = self.attack_matrix.get(node, set())

        for attacker in attackers:
            has_defender = False

            for defender in defender_set:
                if attacker in self.attacks_from.get(defender, set()):
                    has_defender = True
                    break

            if not has_defender:
                return False

        return True

    def is_admissible(self, node_set: Set[str]) -> bool:
        """Check if a set is admissible."""
        if not self.is_conflict_free(node_set):
            return False

        for node in node_set:
            if not self.defends(node_set, node):
                return False

        return True

    def _start_search(self, stage: str) -> None:
        """Reset search timers and counters for a new solving stage.

        Args:
            stage: Logical stage name used in telemetry output.
        """
        self._search_started_ms = time.perf_counter() * 1000.0
        self._searched_states = 0
        self._pruned_states = 0

        self._last_search_stats = {
            "stage": stage,
            "algorithm_version": self.algorithm_version,
        }

    def _elapsed_ms(self) -> int:
        """Return elapsed milliseconds since the current search started.

        Returns:
            Non-negative elapsed time in milliseconds.
        """
        if self._search_started_ms <= 0:
            return 0

        return max(0, int(time.perf_counter() * 1000.0 - self._search_started_ms))

    def _search_stats(self, termination_reason: str) -> Dict[str, Any]:
        """Build the common telemetry payload for search completion.

        Args:
            termination_reason: Reason string describing why search ended.

        Returns:
            Dictionary with counters, timing, and termination metadata.
        """
        return {
            **self._last_search_stats,
            "searched_states": int(self._searched_states),
            "pruned_states": int(self._pruned_states),
            "search_time_ms": self._elapsed_ms(),
            "termination_reason": termination_reason,
        }

    def _can_still_defend(self, chosen: Set[str], undecided: Set[str]) -> bool:
        """Prune when a chosen node can no longer be defended by any future choice."""
        if not chosen:
            return True

        potential_defenders = chosen | undecided

        for node in chosen:
            for attacker in self.attack_matrix.get(node, set()):
                defendable = False

                for candidate in potential_defenders:
                    if attacker in self.attacks_from.get(candidate, set()):
                        defendable = True
                        break

                if not defendable:
                    return False

        return True

    def _contentious_nodes(self) -> List[str]:
        """List nodes that participate in at least one attack relation.

        Returns:
            Sorted node IDs that either attack or are attacked.
        """
        return sorted(
            [
                node
                for node in self._all_nodes_sorted
                if self.attack_matrix.get(node) or self.attacks_from.get(node)
            ]
        )

    def find_all_admissible_sets(self) -> List[Set[str]]:
        """Find admissible sets (mainly for debugging/tests)."""
        logger.info("[BAF] Finding admissible sets...")
        self._start_search(stage="all_admissible_sets")
        contentious = self._contentious_nodes()
        neutral = set(self._all_nodes_sorted) - set(contentious)
        admissible_sets: List[Set[str]] = []

        for mask in range(1 << len(contentious)):
            self._searched_states += 1
            subset = set(neutral)

            for idx, node in enumerate(contentious):
                if (mask >> idx) & 1:
                    subset.add(node)

            if self.is_admissible(subset):
                admissible_sets.append(subset)

        admissible_sets.sort(key=lambda row: (-len(row), tuple(sorted(row))))
        self._last_search_stats = self._search_stats(termination_reason="completed")
        logger.info(f"[BAF] Found {len(admissible_sets)} admissible sets")
        return admissible_sets

    def _is_maximal_with_respect_to(
        self,
        chosen: Set[str],
        excluded: Set[str],
        neutral: Set[str],
    ) -> bool:
        """Check maximality of a candidate extension under one DFS branch.

        Args:
            chosen: Currently selected contentious nodes.
            excluded: Contention nodes explicitly excluded in this branch.
            neutral: Always-included neutral nodes.

        Returns:
            `True` if no excluded node can be added while remaining admissible.
        """
        base = set(chosen) | neutral

        for node in excluded:
            trial = set(base)
            trial.add(node)

            if self.is_admissible(trial):
                return False

        return True

    @staticmethod
    def _extension_sort_key(extension: Set[str]) -> Tuple[int, Tuple[str, ...]]:
        """Build deterministic ordering key for preferred extensions.

        Args:
            extension: Candidate extension.

        Returns:
            Tuple that sorts by size descending and lexicographic node order.
        """
        return (-len(extension), tuple(sorted(extension)))

    def _score_extension(
        self,
        extension: Set[str],
        llm_validated: Set[str],
        llm_defeated: Set[str],
    ) -> int:
        """Score an extension against LLM-validated and defeated sets.

        Args:
            extension: Candidate preferred extension.
            llm_validated: Nodes judged as validated by LLM extraction.
            llm_defeated: Nodes judged as defeated by LLM extraction.

        Returns:
            Integer agreement score; higher values indicate better alignment.
        """
        validated_in_ext = len(extension & llm_validated)
        validated_out_ext = len(llm_validated - extension)
        defeated_in_ext = len(extension & llm_defeated)
        defeated_out_ext = len(llm_defeated - extension)
        return validated_in_ext - validated_out_ext - defeated_in_ext + defeated_out_ext

    def _build_match_details(
        self,
        extension: Set[str],
        llm_validated: Set[str],
        llm_defeated: Set[str],
        extension_index: int = -1,
    ) -> Dict[str, Any]:
        """Build detailed alignment statistics for one extension candidate.

        Args:
            extension: Candidate preferred extension.
            llm_validated: Nodes judged as validated by LLM extraction.
            llm_defeated: Nodes judged as defeated by LLM extraction.
            extension_index: Original index in the candidate list.

        Returns:
            Dictionary containing score components and alignment metrics.
        """
        validated_in_ext = extension & llm_validated
        validated_out_ext = llm_validated - extension
        defeated_in_ext = extension & llm_defeated
        defeated_out_ext = llm_defeated - extension

        details = {
            "extension_index": int(extension_index),
            "score": int(self._score_extension(extension, llm_validated, llm_defeated)),
            "size": len(extension),
            "validated_in_ext": len(validated_in_ext),
            "validated_out_ext": len(validated_out_ext),
            "defeated_in_ext": len(defeated_in_ext),
            "defeated_out_ext": len(defeated_out_ext),
            "hypothetical_in_ext": len(extension - llm_validated - llm_defeated),
            "chosen_extension": sorted(extension),
            "alignment_rate": self._calculate_alignment_rate(
                extension, llm_validated, llm_defeated
            ),
        }

        return details

    def _maybe_log_search_progress(self, stage: str) -> None:
        """Log coarse-grained DFS progress at fixed search intervals.

        Args:
            stage: Stage label included in log output.
        """
        if self._searched_states <= 0:
            return

        if (self._searched_states % 250000) != 0:
            return

        logger.info(
            f"[BAF] {stage} progress: "
            f"searched_states={self._searched_states} "
            f"pruned_states={self._pruned_states} "
            f"elapsed_ms={self._elapsed_ms()}"
        )

    def _contentious_components(self, contentious: List[str]) -> List[Set[str]]:
        """Split contentious nodes into undirected connected components.

        Args:
            contentious: Sorted contentious node IDs.

        Returns:
            Components sorted by size descending and lexical tie-breaker.
        """
        if not contentious:
            return []

        scope = set(contentious)
        neighbors: Dict[str, Set[str]] = {node: set() for node in scope}

        for src in contentious:
            for tgt in self.attacks_from.get(src, set()):
                if tgt not in scope:
                    continue

                neighbors[src].add(tgt)
                neighbors[tgt].add(src)

        components: List[Set[str]] = []
        visited: Set[str] = set()

        for start in contentious:
            if start in visited:
                continue

            queue = deque([start])
            visited.add(start)
            component: Set[str] = set()

            while queue:
                node = queue.popleft()
                component.add(node)

                for nxt in neighbors.get(node, set()):
                    if nxt in visited:
                        continue

                    visited.add(nxt)
                    queue.append(nxt)

            components.append(component)

        components.sort(key=lambda item: (-len(item), tuple(sorted(item))))
        return components

    def _topological_order_if_dag(
        self,
        component_nodes: Set[str],
    ) -> Optional[List[str]]:
        """Try building a topological order for a component attack subgraph.

        Args:
            component_nodes: Node IDs in one contentious component.

        Returns:
            Topological order when the component is a DAG, otherwise `None`.
        """
        if not component_nodes:
            return []

        indegree: Dict[str, int] = {
            node: len(self.attack_matrix.get(node, set()) & component_nodes)
            for node in component_nodes
        }

        heap: List[str] = [node for node, degree in indegree.items() if degree == 0]
        heap.sort()
        topo_order: List[str] = []

        while heap:
            node = heappop(heap)
            topo_order.append(node)

            for target in self.attacks_from.get(node, set()):
                if target not in component_nodes:
                    continue

                indegree[target] = max(0, indegree[target] - 1)

                if indegree[target] == 0:
                    heappush(heap, target)

        if len(topo_order) != len(component_nodes):
            return None

        return topo_order

    def _solve_preferred_for_attack_dag(
        self,
        component_nodes: Set[str],
        topo_order: List[str],
    ) -> Set[str]:
        """Compute one preferred extension for an acyclic attack component.

        Args:
            component_nodes: Node IDs in one contentious component.
            topo_order: Topological ordering of that component.

        Returns:
            Accepted node IDs selected by linear DAG pass.
        """
        accepted: Set[str] = set()

        for node in topo_order:
            attackers = self.attack_matrix.get(node, set()) & component_nodes

            if any(attacker in accepted for attacker in attackers):
                continue

            accepted.add(node)

        return accepted

    def _enumerate_preferred_for_scope(
        self,
        contentious: List[str],
        neutral: Set[str],
        stage_label: str,
    ) -> List[Set[str]]:
        """Enumerate all preferred extensions for a scoped node subset.

        Args:
            contentious: Nodes that require include/exclude search.
            neutral: Nodes that are always included.
            stage_label: Stage label used for progress logging.

        Returns:
            Sorted list of maximal admissible sets for the scope.
        """
        preferred_sets: List[Set[str]] = []
        seen: Set[Tuple[str, ...]] = set()

        if not contentious:
            return [set(neutral)]

        def dfs(
            idx: int,
            chosen: Set[str],
            undecided: Set[str],
            excluded: Set[str],
        ) -> None:
            """Depth-first branch search over include/exclude decisions.

            Args:
                idx: Current index within `contentious`.
                chosen: Nodes currently selected in this branch.
                undecided: Nodes not yet decided in this branch.
                excluded: Nodes rejected in this branch.
            """
            self._searched_states += 1
            self._maybe_log_search_progress(stage=stage_label)

            if not self._can_still_defend(chosen, undecided):
                self._pruned_states += 1
                return

            if idx >= len(contentious):
                full_set = set(chosen) | neutral

                if not self.is_admissible(full_set):
                    self._pruned_states += 1
                    return

                if not self._is_maximal_with_respect_to(chosen, excluded, neutral):
                    self._pruned_states += 1
                    return

                key = tuple(sorted(full_set))

                if key in seen:
                    return

                seen.add(key)
                preferred_sets.append(set(full_set))
                return

            node = contentious[idx]
            undecided.remove(node)

            if not self._has_internal_conflict(node, chosen):
                chosen.add(node)
                dfs(idx + 1, chosen, undecided, excluded)
                chosen.remove(node)

            excluded.add(node)
            dfs(idx + 1, chosen, undecided, excluded)
            excluded.remove(node)
            undecided.add(node)

        dfs(
            idx=0,
            chosen=set(),
            undecided=set(contentious),
            excluded=set(),
        )

        preferred_sets.sort(key=self._extension_sort_key)
        return preferred_sets

    def _solve_component_preferred_extensions(
        self,
        component_nodes: Set[str],
        component_index: int,
    ) -> Tuple[List[Set[str]], Dict[str, Any]]:
        """Solve preferred extensions for one contentious component.

        Args:
            component_nodes: Node IDs that belong to one component.
            component_index: Stable index used in telemetry payloads.

        Returns:
            Tuple of `(preferred_extensions, component_detail_dict)`.
        """
        component_sorted = sorted(component_nodes)
        before_states = self._searched_states
        before_pruned = self._pruned_states
        solver_path = "component_dfs"
        topological_order = self._topological_order_if_dag(component_nodes)

        if topological_order is not None:
            solver_path = "dag_linear"
            self._searched_states += len(component_sorted)

            candidate = self._solve_preferred_for_attack_dag(
                component_nodes, topological_order
            )

            if self.is_admissible(candidate) and self._is_maximal_with_respect_to(
                chosen=set(candidate),
                excluded=set(component_nodes) - set(candidate),
                neutral=set(),
            ):
                preferred = [set(candidate)]

                return preferred, {
                    "component_index": int(component_index),
                    "component_size": len(component_sorted),
                    "solver_path": solver_path,
                    "preferred_count": 1,
                    "searched_states": int(self._searched_states - before_states),
                    "pruned_states": int(self._pruned_states - before_pruned),
                    "nodes": component_sorted,
                }

            solver_path = "dag_fallback_to_dfs"

        preferred = self._enumerate_preferred_for_scope(
            contentious=component_sorted,
            neutral=set(),
            stage_label=f"component_{component_index}_preferred_search",
        )

        return preferred, {
            "component_index": int(component_index),
            "component_size": len(component_sorted),
            "solver_path": solver_path,
            "preferred_count": len(preferred),
            "searched_states": int(self._searched_states - before_states),
            "pruned_states": int(self._pruned_states - before_pruned),
            "nodes": component_sorted,
        }

    def find_preferred_extensions(self) -> List[Set[str]]:
        """Find preferred extensions (maximal admissible sets)."""
        logger.info("[BAF] Finding preferred extensions...")
        self._start_search(stage="preferred_extensions")
        contentious = self._contentious_nodes()
        neutral = set(self._all_nodes_sorted) - set(contentious)

        if not contentious:
            single = set(neutral)

            self._last_search_stats = {
                **self._search_stats(termination_reason="completed"),
                "preferred_extensions_count": 1,
                "preferred_extensions_count_estimated": 1,
                "component_count": 0,
                "component_sizes": [],
                "component_solver_paths": [],
                "solver_path": "trivial_no_contention",
            }

            logger.info("[BAF] No contentious nodes; single preferred extension.")
            return [single]

        components = self._contentious_components(contentious)
        component_extensions: List[List[Set[str]]] = []
        component_details: List[Dict[str, Any]] = []
        preferred_count_estimated = 1

        for idx, component in enumerate(components):
            local_extensions, detail = self._solve_component_preferred_extensions(
                component_nodes=component,
                component_index=idx,
            )

            component_extensions.append(local_extensions)
            component_details.append(detail)
            preferred_count_estimated *= max(1, len(local_extensions))

        preferred_sets: List[Set[str]] = [set(neutral)]

        for local_extensions in component_extensions:
            combined: List[Set[str]] = []

            for base in preferred_sets:
                for ext in local_extensions:
                    combined.append(set(base) | set(ext))

            preferred_sets = combined

        preferred_sets.sort(key=self._extension_sort_key)

        self._last_search_stats = {
            **self._search_stats(termination_reason="completed"),
            "preferred_extensions_count": len(preferred_sets),
            "preferred_extensions_count_estimated": int(preferred_count_estimated),
            "component_count": len(components),
            "component_sizes": [len(component) for component in components],
            "component_solver_paths": [
                str(item.get("solver_path", "unknown")) for item in component_details
            ],
            "component_details": component_details,
            "solver_path": "componentized_exact",
            "contentious_count": len(contentious),
        }

        logger.info(
            f"[BAF] Found {len(preferred_sets)} preferred extensions in "
            f"{self._last_search_stats.get('search_time_ms', 0)}ms "
            f"(components={len(components)})"
        )

        return preferred_sets

    def find_best_preferred_extension(
        self,
        llm_validated: Set[str],
        llm_defeated: Set[str],
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """Find one score-optimal preferred extension without global enumeration."""
        logger.info("[BAF] Finding best preferred extension...")
        self._start_search(stage="best_preferred_extension")
        contentious = self._contentious_nodes()
        neutral = set(self._all_nodes_sorted) - set(contentious)
        llm_decided = set(llm_validated) | set(llm_defeated)

        if not contentious:
            chosen = set(neutral)

            details = self._build_match_details(
                extension=chosen,
                llm_validated=llm_validated,
                llm_defeated=llm_defeated,
                extension_index=0,
            )

            details["selection_strategy"] = "trivial_no_contention"
            details["component_selection"] = []
            details["preferred_extensions_count"] = 1

            self._last_search_stats = {
                **self._search_stats(termination_reason="completed"),
                "preferred_extensions_count": 1,
                "preferred_extensions_count_estimated": 1,
                "component_count": 0,
                "component_sizes": [],
                "component_solver_paths": [],
                "relevant_component_count": 0,
                "solver_path": "trivial_no_contention",
                "contentious_count": 0,
            }

            return chosen, details

        components = self._contentious_components(contentious)
        chosen_extension = set(neutral)
        preferred_count_estimated = 1
        relevant_component_count = 0
        component_details: List[Dict[str, Any]] = []

        for idx, component in enumerate(components):
            local_extensions, detail = self._solve_component_preferred_extensions(
                component_nodes=component,
                component_index=idx,
            )

            preferred_count_estimated *= max(1, len(local_extensions))

            if not local_extensions:
                error_details = {
                    "error": f"No local preferred extension for component {idx}",
                    "component_index": idx,
                }

                self._last_search_stats = {
                    **self._search_stats(termination_reason="failed"),
                    "preferred_extensions_count": 0,
                    "preferred_extensions_count_estimated": 0,
                    "component_count": len(components),
                    "component_sizes": [len(item) for item in components],
                    "component_solver_paths": [
                        str(item.get("solver_path", "unknown"))
                        for item in component_details
                    ],
                    "relevant_component_count": relevant_component_count,
                    "solver_path": "componentized_exact",
                    "contentious_count": len(contentious),
                }

                return set(), error_details

            local_validated = llm_validated & component
            local_defeated = llm_defeated & component
            is_relevant = bool(component & llm_decided)

            if is_relevant:
                relevant_component_count += 1

            local_best = set(local_extensions[0])

            local_best_score = self._score_extension(
                local_best, local_validated, local_defeated
            )

            local_best_key = tuple(sorted(local_best))

            for candidate in local_extensions[1:]:
                candidate_score = self._score_extension(
                    candidate, local_validated, local_defeated
                )

                candidate_key = tuple(sorted(candidate))
                should_replace = candidate_score > local_best_score

                if not should_replace and candidate_score == local_best_score:
                    if len(candidate) > len(local_best):
                        should_replace = True

                    elif len(candidate) == len(local_best):
                        should_replace = candidate_key < local_best_key

                if should_replace:
                    local_best = set(candidate)
                    local_best_score = candidate_score
                    local_best_key = candidate_key

            chosen_extension.update(local_best)
            detail["is_relevant"] = is_relevant
            detail["local_score"] = int(local_best_score)
            detail["selected_extension"] = sorted(local_best)
            detail["selected_extension_size"] = len(local_best)
            component_details.append(detail)

        chosen_extension = set(chosen_extension)

        match_details = self._build_match_details(
            extension=chosen_extension,
            llm_validated=llm_validated,
            llm_defeated=llm_defeated,
            extension_index=0,
        )

        match_details["selection_strategy"] = "componentized_exact"
        match_details["component_selection"] = component_details
        match_details["component_count"] = len(components)
        match_details["component_sizes"] = [len(item) for item in components]
        match_details["relevant_component_count"] = relevant_component_count
        match_details["preferred_extensions_count"] = int(preferred_count_estimated)

        self._last_search_stats = {
            **self._search_stats(termination_reason="completed"),
            "preferred_extensions_count": int(preferred_count_estimated),
            "preferred_extensions_count_estimated": int(preferred_count_estimated),
            "component_count": len(components),
            "component_sizes": [len(item) for item in components],
            "component_solver_paths": [
                str(item.get("solver_path", "unknown")) for item in component_details
            ],
            "component_details": component_details,
            "relevant_component_count": relevant_component_count,
            "solver_path": "componentized_exact",
            "contentious_count": len(contentious),
        }

        logger.info(
            "[BAF] Best preferred extension selected in "
            f"{self._last_search_stats.get('search_time_ms', 0)}ms "
            f"(components={len(components)}, "
            f"estimated_preferred={preferred_count_estimated})"
        )

        return chosen_extension, match_details

    def get_search_stats(self) -> Dict[str, Any]:
        """Return stats from the latest search call."""
        return dict(self._last_search_stats)

    def _shortest_distance_from_roots(self, roots: Set[str]) -> Dict[str, int]:
        """Compute undirected BFS distance from root nodes to reachable nodes.

        Args:
            roots: Root node IDs used as BFS sources.

        Returns:
            Mapping from node ID to minimum hop distance from any root.
        """
        dist: Dict[str, int] = {}
        queue = deque()

        for root in roots:
            if root not in self._undirected_neighbors:
                continue

            dist[root] = 0
            queue.append(root)

        while queue:
            current = queue.popleft()
            current_dist = dist[current]

            for nxt in self._undirected_neighbors.get(current, set()):
                if nxt in dist:
                    continue

                dist[nxt] = current_dist + 1
                queue.append(nxt)

        return dist

    def _expand_support_predecessors(self, seeds: Set[str], max_hop: int) -> Set[str]:
        """Expand upstream SUPPORT predecessors up to a hop limit.

        Args:
            seeds: Seed node IDs from which expansion starts.
            max_hop: Maximum predecessor hop distance.

        Returns:
            Predecessor node IDs discovered within `max_hop`.
        """
        max_depth = max(0, int(max_hop))

        if max_depth <= 0 or not seeds:
            return set()

        result: Set[str] = set()
        queue = deque([(seed, 0) for seed in seeds])
        visited = set(seeds)

        while queue:
            node, depth = queue.popleft()

            if depth >= max_depth:
                continue

            for pred in self.support_predecessors.get(node, set()):
                if pred in visited:
                    continue

                visited.add(pred)
                result.add(pred)
                queue.append((pred, depth + 1))

        return result

    def build_root_anchored_context(
        self,
        root_ids: Set[str],
        k_hop: int = 3,
        max_nodes: Optional[int] = None,
    ) -> Set[str]:
        """Build context nodes around root claims using BAF-consistent relations."""
        roots = {str(node) for node in root_ids if str(node) in self._all_nodes_sorted}
        k_value = max(1, int(k_hop))
        max_keep = None if max_nodes is None else max(1, int(max_nodes))

        if not roots:
            fallback = set(self._all_nodes_sorted)

            self._last_context_selection = {
                "mode": "fallback",
                "max_nodes": max_keep,
                "selected_count": len(fallback),
                "selected_nodes": sorted(fallback),
            }

            return fallback

        support_cone = self._expand_support_predecessors(roots, k_value)
        selected = set(roots) | set(support_cone)
        attackers: Set[str] = set()

        for node in selected:
            attackers.update(self.attack_matrix.get(node, set()))

        defenders: Set[str] = set()

        for attacker in attackers:
            defenders.update(self.attack_matrix.get(attacker, set()))

        selected.update(attackers)
        selected.update(defenders)

        selected.update(
            self._expand_support_predecessors(
                attackers | defenders, max(1, k_value - 1)
            )
        )

        if max_keep is not None and len(selected) > max_keep:
            dist = self._shortest_distance_from_roots(roots)
            ranked: List[Tuple[float, str]] = []
            graph = self.graph.graph

            for node in selected:
                score = 0.0

                if node in roots:
                    score += 1000.0

                if node in support_cone:
                    score += 120.0

                if node in attackers:
                    score += 80.0

                if node in defenders:
                    score += 60.0

                node_dist = dist.get(node, 100)
                score += max(0.0, 40.0 - float(node_dist) * 6.0)
                data = graph.nodes.get(node, {})
                metadata = data.get("metadata", {})

                try:
                    last_modified = int(metadata.get("last_modified_step", 0) or 0)

                except (TypeError, ValueError):
                    last_modified = 0

                score += min(20.0, max(0.0, float(last_modified) * 0.5))
                ranked.append((score, node))

            ranked.sort(key=lambda row: (-row[0], row[1]))
            keep = set(sorted(roots))

            for _, node in ranked:
                if len(keep) >= max_keep:
                    break

                keep.add(node)

            selected = keep

        self._last_context_selection = {
            "mode": "root_evidence_cone",
            "k_hop": k_value,
            "max_nodes": max_keep,
            "root_count": len(roots),
            "support_cone_count": len(support_cone),
            "attacker_count": len(attackers),
            "defender_count": len(defenders),
            "selected_count": len(selected),
            "selected_nodes": sorted(selected),
        }

        return selected

    def build_root_k_hop_context(
        self,
        root_ids: Set[str],
        k_hop: int = 3,
    ) -> Set[str]:
        """Build an undirected k-hop neighborhood around root claims."""
        roots = {str(node) for node in root_ids if str(node) in self._all_nodes_sorted}
        k_value = max(0, int(k_hop))

        if not roots:
            fallback = set(self._all_nodes_sorted)

            self._last_context_selection = {
                "mode": "fallback",
                "k_hop": k_value,
                "selected_count": len(fallback),
                "selected_nodes": sorted(fallback),
            }

            return fallback

        selected: Set[str] = set(roots)
        queue = deque((root, 0) for root in roots)
        visited = set(roots)

        while queue:
            node, depth = queue.popleft()

            if depth >= k_value:
                continue

            for nxt in self._undirected_neighbors.get(node, set()):
                if nxt in visited:
                    continue

                visited.add(nxt)
                selected.add(nxt)
                queue.append((nxt, depth + 1))

        self._last_context_selection = {
            "mode": "root_k_hop",
            "k_hop": k_value,
            "root_count": len(roots),
            "selected_count": len(selected),
            "selected_nodes": sorted(selected),
        }

        return selected

    def explain_context_selection(self) -> Dict[str, Any]:
        """Return metadata for the latest context-selection call."""
        return dict(self._last_context_selection)

    def match_with_llm_judgment(
        self,
        preferred_extensions: List[Set[str]],
        llm_validated: Set[str],
        llm_defeated: Set[str],
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """Select the preferred extension that best aligns with LLM judgment."""
        logger.info("[BAF] Matching preferred extensions with LLM judgment...")

        if not preferred_extensions:
            logger.warning("[BAF] No preferred extensions to match!")
            return set(), {"error": "No preferred extensions"}

        best_extension: Set[str] = set()
        best_score = float("-inf")
        best_details: Dict[str, Any] = {}

        for i, ext in enumerate(preferred_extensions):
            details = self._build_match_details(
                extension=ext,
                llm_validated=llm_validated,
                llm_defeated=llm_defeated,
                extension_index=i,
            )

            score = int(details.get("score", 0))

            if score > best_score:
                best_score = score
                best_extension = set(ext)
                best_details = details

        return best_extension, best_details

    def _calculate_alignment_rate(
        self,
        extension: Set[str],
        llm_validated: Set[str],
        llm_defeated: Set[str],
    ) -> float:
        """Calculate ratio of LLM decisions that match one extension.

        Args:
            extension: Candidate preferred extension.
            llm_validated: Nodes judged as validated by LLM extraction.
            llm_defeated: Nodes judged as defeated by LLM extraction.

        Returns:
            Agreement ratio in `[0.0, 1.0]`.
        """
        total_decided = len(llm_validated) + len(llm_defeated)

        if total_decided == 0:
            return 1.0

        validated_in_ext = len(extension & llm_validated)
        defeated_out_ext = len(llm_defeated - extension)
        agreements = validated_in_ext + defeated_out_ext
        return agreements / total_decided

    def get_attack_report(self) -> Dict[str, Dict[str, List[str]]]:
        """Get a detailed report of all attacks in the graph."""
        return {
            node: {
                "direct": list(values.get("direct", [])),
                "support_based": list(values.get("support_based", [])),
                "indirect": list(values.get("indirect", [])),
            }
            for node, values in self.collective_attacks.items()
        }

    def validate_consistency(
        self,
        llm_validated: Set[str],
        llm_defeated: Set[str],
    ) -> Dict[str, Any]:
        """Validate consistency of LLM judgment with BAF semantics."""
        issues: List[Dict[str, Any]] = []
        overlap = llm_validated & llm_defeated

        if overlap:
            issues.append(
                {
                    "type": "overlap",
                    "message": (
                        "Nodes marked as both VALIDATED and DEFEATED: "
                        f"{sorted(overlap)}"
                    ),
                }
            )

        if not self.is_conflict_free(llm_validated):
            issues.append(
                {
                    "type": "internal_conflict",
                    "message": "VALIDATED nodes attack each other",
                }
            )

        for node in sorted(llm_validated):
            if not self.defends(llm_validated, node):
                issues.append(
                    {
                        "type": "undefended",
                        "message": f"VALIDATED node {node} is not defended",
                    }
                )

        return {
            "is_consistent": len(issues) == 0,
            "issues": issues,
            "validated_count": len(llm_validated),
            "defeated_count": len(llm_defeated),
        }
