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

from ..core.graph import EdgeType, ShadowGraph


class BAFComputationError(RuntimeError):
    """Raised when BAF computation encounters a hard failure."""

    def __init__(
        self,
        code: str,
        message: str,
        stats: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = str(code).strip() or "BAF_ERROR"
        self.stats = stats or {}

    def __str__(self) -> str:
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
        if isinstance(value, EdgeType):
            return value.value

        text = str(value).strip().upper()

        if text.startswith("EDGETYPE."):
            text = text.split(".", 1)[1]

        return text

    def _build_edge_indexes(self) -> None:
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
        self._search_started_ms = time.perf_counter() * 1000.0
        self._searched_states = 0
        self._pruned_states = 0

        self._last_search_stats = {
            "stage": stage,
            "algorithm_version": self.algorithm_version,
        }

    def _elapsed_ms(self) -> int:
        if self._search_started_ms <= 0:
            return 0

        return max(0, int(time.perf_counter() * 1000.0 - self._search_started_ms))

    def _search_stats(self, termination_reason: str) -> Dict[str, Any]:
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
        base = set(chosen) | neutral

        for node in excluded:
            trial = set(base)
            trial.add(node)

            if self.is_admissible(trial):
                return False

        return True

    @staticmethod
    def _extension_sort_key(extension: Set[str]) -> Tuple[int, Tuple[str, ...]]:
        return (-len(extension), tuple(sorted(extension)))

    def _score_extension(
        self,
        extension: Set[str],
        llm_validated: Set[str],
        llm_defeated: Set[str],
    ) -> int:
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
