"""Implements Bipolar Argumentation Framework (BAF) semantics for legal debate.

This module provides the `BAFCalculator` class, which implements formal BAF
semantics including collective attack detection, admissible set calculation,
and preferred extension finding. This adds a logical verification layer to the
LLM-based judgment process.

Key Concepts:
- Collective Attacks: Direct, support-based, and indirect attacks
- Conflict-Free Set: No mutual attacks within the set
- Defense: Every external attacker is defended by the set
- Admissible Set: Conflict-free + defensive
- Preferred Extension: Maximal admissible set
- Matching Function: Selects extension best aligned with LLM judgment
"""

import itertools
from typing import Dict, List, Set, Tuple

from metagpt.logs import logger

from .common import EdgeType, NodeStatus, ShadowGraph


class CollectiveAttackType:
    """Types of collective attacks in BAF."""

    DIRECT = "direct"
    SUPPORT_BASED = "support_based"
    INDIRECT = "indirect"


class BAFCalculator:
    """Calculates BAF semantics for the debate graph.

    This class implements the formal BAF semantics as defined in the methodology.
    It can:
    1. Detect all collective attacks in the graph
    2. Calculate conflict-free sets
    3. Verify defense properties
    4. Find all admissible sets
    5. Find preferred extensions (maximal admissible sets)
    6. Match preferred extensions with LLM judgments

    Attributes:
        graph: The ShadowGraph to analyze
        collective_attacks: Cache of all collective attacks
        attack_matrix: Matrix representation of attacks for efficient computation
    """

    def __init__(self, graph: ShadowGraph):
        """Initialize the BAF calculator with a graph.

        Args:
            graph: The ShadowGraph to analyze
        """
        self.graph = graph
        self.collective_attacks: Dict[str, Dict[str, List[str]]] = {}
        self.attack_matrix: Dict[str, Set[str]] = {}
        self._compute_all_attacks()

    def _compute_all_attacks(self):
        """Compute all types of collective attacks in the graph.

        This includes:
        - Direct attacks (CONFLICT edges)
        - Support-based attacks (A supports B, B attacks C → A attacks C)
        - Indirect attacks (A attacks B, B attacks C → A attacks C)
        """
        logger.info("[BAF] Computing collective attacks...")
        all_nodes = set(self.graph.graph.nodes())

        self.collective_attacks = {
            node_id: {
                "direct": [],
                "support_based": [],
                "indirect": []
            }
            for node_id in all_nodes
        }

        for src, tgt, data in self.graph.graph.edges(data=True):
            if data.get("type") == EdgeType.CONFLICT:
                self.collective_attacks[tgt]["direct"].append(src)

        for src, tgt, data in self.graph.graph.edges(data=True):
            if data.get("type") == EdgeType.SUPPORT:
                for tgt_attacks in self.collective_attacks[tgt]["direct"]:
                    if src not in self.collective_attacks[tgt_attacks]["support_based"]:
                        self.collective_attacks[tgt_attacks]["support_based"].append(src)

        for node_id in all_nodes:
            direct_attackers = self.collective_attacks[node_id]["direct"]

            for attacker in direct_attackers:
                for attacker_target in self.collective_attacks[attacker]["direct"]:
                    if attacker_target != node_id:  # Avoid self-attack
                        if node_id not in self.collective_attacks[attacker_target]["indirect"]:
                            self.collective_attacks[attacker_target]["indirect"].append(node_id)

        for node_id in all_nodes:
            all_attackers = (
                self.collective_attacks[node_id]["direct"] +
                self.collective_attacks[node_id]["support_based"] +
                self.collective_attacks[node_id]["indirect"]
            )

            self.attack_matrix[node_id] = set(all_attackers)

        total_direct = sum(len(v["direct"]) for v in self.collective_attacks.values())
        total_support = sum(len(v["support_based"]) for v in self.collective_attacks.values())
        total_indirect = sum(len(v["indirect"]) for v in self.collective_attacks.values())

        logger.info(
            f"[BAF] Found {total_direct} direct, {total_support} support-based, "
            f"{total_indirect} indirect attacks"
        )

    def is_conflict_free(self, node_set: Set[str]) -> bool:
        """Check if a set of nodes is conflict-free.

        A set is conflict-free if no node in the set attacks another node
        in the set (considering all types of collective attacks).

        Args:
            node_set: Set of node IDs to check

        Returns:
            True if the set is conflict-free, False otherwise
        """
        for node in node_set:
            attackers = self.attack_matrix.get(node, set())

            if attackers & node_set:
                return False

        return True

    def defends(self, defender_set: Set[str], node: str) -> bool:
        """Check if a set defends a node.

        A set S defends node X if for every attacker Y of X, there exists
        some Z in S such that Z attacks Y.

        Args:
            defender_set: Set of potential defender nodes
            node: The node being defended

        Returns:
            True if defender_set defends node, False otherwise
        """
        attackers = self.attack_matrix.get(node, set())

        for attacker in attackers:
            has_defender = False

            for defender in defender_set:
                if attacker in self.attack_matrix.get(defender, set()):
                    has_defender = True
                    break

            if not has_defender:
                return False

        return True

    def is_admissible(self, node_set: Set[str]) -> bool:
        """Check if a set is admissible.

        A set is admissible if it is conflict-free and every node in the set
        is defended by the set.

        Args:
            node_set: Set of node IDs to check

        Returns:
            True if the set is admissible, False otherwise
        """
        if not self.is_conflict_free(node_set):
            return False

        for node in node_set:
            if not self.defends(node_set, node):
                return False

        return True

    def find_all_admissible_sets(self) -> List[Set[str]]:
        """Find all admissible sets in the graph.

        This uses a brute-force approach, which is acceptable for typical
        legal debate graphs (usually < 50 nodes). For larger graphs, consider
        using more efficient algorithms.

        Returns:
            List of all admissible sets, sorted by size (descending)
        """
        logger.info("[BAF] Finding all admissible sets...")
        all_nodes = list(self.graph.graph.nodes())
        admissible_sets = []

        for size in range(len(all_nodes), 0, -1):
            for subset in itertools.combinations(all_nodes, size):
                subset_set = set(subset)

                if self.is_admissible(subset_set):
                    admissible_sets.append(subset_set)

        logger.info(f"[BAF] Found {len(admissible_sets)} admissible sets")
        admissible_sets.sort(key=len, reverse=True)
        return admissible_sets

    def find_preferred_extensions(self) -> List[Set[str]]:
        """Find all preferred extensions.

        Preferred extensions are the maximal admissible sets - they cannot be
        extended by adding any node without losing admissibility.

        Returns:
            List of preferred extensions (maximal admissible sets)
        """
        logger.info("[BAF] Finding preferred extensions...")
        admissible_sets = self.find_all_admissible_sets()

        if not admissible_sets:
            logger.warning("[BAF] No admissible sets found!")
            return []

        preferred = []
        max_size = max(len(s) for s in admissible_sets)

        for adm_set in admissible_sets:
            if len(adm_set) == max_size:
                preferred.append(adm_set)

        logger.info(f"[BAF] Found {len(preferred)} preferred extensions (size={max_size})")
        return preferred

    def match_with_llm_judgment(
        self,
        preferred_extensions: List[Set[str]],
        llm_validated: Set[str],
        llm_defeated: Set[str]
    ) -> Tuple[Set[str], Dict[str, str]]:
        """Match preferred extensions with LLM judgment.

        Uses a scoring function to find the preferred extension that best
        aligns with the LLM judgment.

        Scoring:
        - +1 for each node that is VALIDATED in LLM and included in extension
        - -1 for each node that is VALIDATED in LLM but excluded from extension
        - -1 for each node that is DEFEATED in LLM but included in extension
        - +1 for each node that is DEFEATED in LLM and excluded from extension

        Args:
            preferred_extensions: List of preferred extensions
            llm_validated: Nodes marked as VALIDATED by LLM
            llm_defeated: Nodes marked as DEFEATED by LLM

        Returns:
            Tuple of (best_extension, matching_details)
            - best_extension: The extension with highest alignment score
            - matching_details: Dict with score and alignment info
        """
        logger.info("[BAF] Matching preferred extensions with LLM judgment...")

        if not preferred_extensions:
            logger.warning("[BAF] No preferred extensions to match!")
            return set(), {"error": "No preferred extensions"}

        best_extension = None
        best_score = float('-inf')
        best_details = {}

        for i, ext in enumerate(preferred_extensions):
            validated_in_ext = ext & llm_validated
            validated_out_ext = llm_validated - ext
            defeated_in_ext = ext & llm_defeated
            defeated_out_ext = llm_defeated - ext

            score = (
                len(validated_in_ext) * 1 +
                len(validated_out_ext) * -1 +
                len(defeated_in_ext) * -1 +
                len(defeated_out_ext) * 1
            )

            details = {
                "extension_index": i,
                "score": score,
                "size": len(ext),
                "validated_in_ext": len(validated_in_ext),
                "validated_out_ext": len(validated_out_ext),
                "defeated_in_ext": len(defeated_in_ext),
                "defeated_out_ext": len(defeated_out_ext),
                "hypothetical_in_ext": len(ext - llm_validated - llm_defeated)
            }

            logger.debug(f"[BAF] Extension {i}: score={score}, {details}")

            if score > best_score:
                best_score = score
                best_extension = ext
                best_details = details

        logger.info(
            f"[BAF] Best match: extension {best_details.get('extension_index')} "
            f"with score {best_score}"
        )

        best_details["chosen_extension"] = list(best_extension)

        best_details["alignment_rate"] = self._calculate_alignment_rate(
            best_extension, llm_validated, llm_defeated
        )

        return best_extension, best_details

    def _calculate_alignment_rate(
        self,
        extension: Set[str],
        llm_validated: Set[str],
        llm_defeated: Set[str]
    ) -> float:
        """Calculate alignment rate between extension and LLM judgment.

        Args:
            extension: The chosen extension
            llm_validated: Nodes marked as VALIDATED by LLM
            llm_defeated: Nodes marked as DEFEATED by LLM

        Returns:
            Alignment rate between 0 and 1
        """
        total_decided = len(llm_validated) + len(llm_defeated)

        if total_decided == 0:
            return 1.0

        validated_in_ext = len(extension & llm_validated)
        defeated_out_ext = len(llm_defeated - extension)
        agreements = validated_in_ext + defeated_out_ext
        return agreements / total_decided

    def get_attack_report(self) -> Dict[str, Dict]:
        """Get a detailed report of all attacks in the graph.

        Returns:
            Dictionary mapping node IDs to their attack information
        """
        return self.collective_attacks

    def validate_consistency(self, llm_validated: Set[str], llm_defeated: Set[str]) -> Dict:
        """Validate consistency of LLM judgment with BAF semantics.

        Checks if the LLM judgment is logically consistent according to BAF:
        1. No node is both VALIDATED and DEFEATED
        2. VALIDATED nodes do not attack each other
        3. All VALIDATED nodes are defended

        Args:
            llm_validated: Nodes marked as VALIDATED by LLM
            llm_defeated: Nodes marked as DEFEATED by LLM

        Returns:
            Dictionary with validation results and issues found
        """
        issues = []
        overlap = llm_validated & llm_defeated

        if overlap:
            issues.append({
                "type": "overlap",
                "message": f"Nodes marked as both VALIDATED and DEFEATED: {overlap}"
            })

        if not self.is_conflict_free(llm_validated):
            issues.append({
                "type": "internal_conflict",
                "message": "VALIDATED nodes attack each other"
            })

        for node in llm_validated:
            if not self.defends(llm_validated, node):
                issues.append({
                    "type": "undefended",
                    "message": f"VALIDATED node {node} is not defended"
                })

        return {
            "is_consistent": len(issues) == 0,
            "issues": issues,
            "validated_count": len(llm_validated),
            "defeated_count": len(llm_defeated)
        }
