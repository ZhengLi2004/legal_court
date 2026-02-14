"""Defines the Judge agent responsible for adjudicating the debate.

This module provides the `LLMJudge` class, which acts as an impartial
adjudicator. Its primary roles are to generate a final judgment document based
on the debate transcript and to extract a structured verdict on the root
claims from that document.

The judge now supports BAF (Bipolar Argumentation Framework) semantics for
logical verification of judgments.
"""

import asyncio
from abc import ABC
from typing import Any, Dict, Optional, Set, Tuple

from metagpt.logs import logger

from prompts.common_prompts import JUDGE_EVALUATE_PROMPT, JUDGE_EXTRACT_VERDICT_PROMPT
from tools.llm import GPTChat, Message

from ..analysis.baf import BAFCalculator
from ..core.graph import NodeStatus, ShadowGraph

_VERDICT_STATUS_SCHEMA = {
    "name": "verdict_status",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["ACCEPTED", "REJECTED", "UNMENTIONED"],
            }
        },
        "required": ["status"],
        "additionalProperties": False,
    },
}


class BaseJudge(ABC):
    """Abstract base class for a Judge."""

    def evaluate(
        self,
        context: str,
        graph: ShadowGraph,
        baf_calculator: Optional[BAFCalculator] = None,
        baf_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a final judgment document.

        Args:
            context: The initial facts of the case.
            graph: The final state of the debate graph.
            baf_calculator: Optional BAF calculator shared with adjudication.
            baf_config: Optional BAF config dict for context-budget controls.

        Returns:
            A string containing the full text of the judgment document.
        """
        pass

    async def extract_verdict(
        self, judgment_document: str, graph: ShadowGraph
    ) -> Dict[str, NodeStatus]:
        """Extract the status of each root claim from the judgment document.

        Args:
            judgment_document: The full text of the judgment.
            graph: The debate graph, used to identify the root claims.

        Returns:
            A dictionary mapping each root claim ID to its final `NodeStatus`
            (VALIDATED, DEFEATED, or HYPOTHETICAL).
        """
        pass


class LLMJudge(BaseJudge):
    """An implementation of the Judge using Large Language Models.

    This class uses one LLM (the `judge_llm`) to write the main legal
    analysis and another (`extraction_llm`) to perform the structured task
    of extracting claim statuses. This separation of concerns can improve
    reliability.

    Attributes:
        judge_llm: The `GPTChat` instance for generating the judgment document.
        extraction_llm: The `GPTChat` instance for extracting verdicts.
    """

    def __init__(self, judge_llm: GPTChat, extraction_llm: GPTChat):
        """Initialize the LLMJudge.

        Args:
            judge_llm: An LLM client configured for the main judgment task.
            extraction_llm: An LLM client configured for the extraction task.
        """
        self.judge_llm = judge_llm
        self.extraction_llm = extraction_llm

    def evaluate(
        self,
        context: str,
        graph: ShadowGraph,
        baf_calculator: Optional[BAFCalculator] = None,
        baf_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a judgment document by prompting an LLM."""
        del context

        payload = self._build_adjudication_input(
            graph=graph,
            baf_calculator=baf_calculator,
            baf_config=baf_config,
        )

        prompt = JUDGE_EVALUATE_PROMPT.format(
            issue_list=payload["issue_list"],
            graph_context=payload["graph_context"],
        )

        logger.info(
            "[Judge] Prompt assembled. graph_chars={} roots={}",
            payload["graph_chars"],
            payload["root_count"],
        )

        logger.info(">>> [Judge] Generating Judgment Document...")
        response = self.judge_llm([Message(role="user", content=prompt)])
        return response

    def _cfg_value(
        self,
        baf_config: Optional[Dict[str, Any]],
        key: str,
        default: Any,
    ) -> Any:
        if isinstance(baf_config, dict) and key in baf_config:
            return baf_config[key]

        return default

    def _build_adjudication_input(
        self,
        graph: ShadowGraph,
        baf_calculator: Optional[BAFCalculator] = None,
        baf_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        root_claims: Dict[str, str] = {}

        for node_id, data in graph.graph.nodes(data=True):
            metadata = data.get("metadata", {})

            if metadata.get("is_root_claim"):
                root_claims[str(node_id)] = str(data.get("content", "未知诉求"))

        if not root_claims:
            issue_list_str = "（未检测到明确的根诉求，请根据证据上下文归纳争议焦点）"
            logger.warning("[Judge] No root claims found in graph metadata.")

        else:
            issue_list_str = "\n".join(
                [
                    f"{idx + 1}. [{claim_id}] {content}"
                    for idx, (claim_id, content) in enumerate(
                        sorted(root_claims.items(), key=lambda row: row[0])
                    )
                ]
            )

        k_hop = max(
            1,
            int(self._cfg_value(baf_config, "judge_context_k_hop", 3)),
        )

        selected_nodes: Optional[Set[str]] = None

        if baf_calculator is not None and root_claims:
            selected_nodes = baf_calculator.build_root_k_hop_context(
                root_ids=set(root_claims.keys()),
                k_hop=k_hop,
            )

        if selected_nodes:
            graph_context = graph.to_recursive_text_for_nodes(selected_nodes)

        else:
            graph_context = graph.to_recursive_text()

        return {
            "issue_list": issue_list_str,
            "graph_context": graph_context,
            "graph_chars": len(graph_context),
            "root_count": len(root_claims),
        }

    async def extract_verdict(
        self, judgment_document: str, graph: ShadowGraph
    ) -> Dict[str, NodeStatus]:
        """Extract verdicts for all root claims in parallel.

        For each root claim identified in the graph, it creates a separate
        asynchronous task to query the `extraction_llm`. This task asks the LLM
        to read the judgment document and determine if that specific claim was
        'ACCEPTED', 'REJECTED', or 'UNMENTIONED'.

        Args:
            judgment_document: The full text of the judgment.
            graph: The debate graph, used to identify the root claims.

        Returns:
            A dictionary mapping each root claim ID to its final `NodeStatus`.
        """
        root_claims_status: Dict[str, NodeStatus] = {}

        all_root_claims = {
            nid: data.get("content")
            for nid, data in graph.graph.nodes(data=True)
            if data.get("metadata", {}).get("is_root_claim", False)
        }

        if not all_root_claims:
            return {}

        extraction_tasks = []

        for claim_id, claim_content in all_root_claims.items():
            extraction_prompt = JUDGE_EXTRACT_VERDICT_PROMPT.format(
                judgment_document=judgment_document,
                claim_id=claim_id,
                claim_content=claim_content,
            )

            extraction_tasks.append(
                self.extraction_llm.aask_json_schema(
                    extraction_prompt,
                    schema=_VERDICT_STATUS_SCHEMA,
                    temperature=0.0,
                )
            )

        extraction_responses = await asyncio.gather(
            *extraction_tasks, return_exceptions=True
        )

        for i, (claim_id, claim_content) in enumerate(all_root_claims.items()):
            response = extraction_responses[i]

            if isinstance(response, Exception):
                root_claims_status[claim_id] = NodeStatus.HYPOTHETICAL
                logger.warning(f"[Judge-Extract] Extraction failed for {claim_id}: {response}")
                continue

            status = ""

            if isinstance(response, dict):
                status = str(response.get("status", "")).upper().strip()

            if status == "ACCEPTED":
                root_claims_status[claim_id] = NodeStatus.VALIDATED

            elif status == "REJECTED":
                root_claims_status[claim_id] = NodeStatus.DEFEATED

            else:
                root_claims_status[claim_id] = NodeStatus.HYPOTHETICAL

                logger.debug(
                    f"[Judge-Extract] Unclear status for {claim_id}: {response}"
                )

        return root_claims_status

    async def extract_verdict_with_baf(
        self,
        judgment_document: str,
        graph: ShadowGraph,
        use_baf: bool = True,
        baf_config: Optional[Dict] = None,
        baf_calculator: Optional[BAFCalculator] = None,
    ) -> Tuple[Dict[str, NodeStatus], Optional[Dict]]:
        """Extract verdicts with BAF semantic verification.

        This implements a three-phase judgment process:
        1. LLM Phase: Extract initial verdicts from judgment document
        2. BAF Phase: Calculate preferred extensions using formal BAF semantics
        3. Fusion Phase: Match preferred extension with LLM judgment and apply corrections

        Args:
            judgment_document: The full text of the judgment
            graph: The debate graph
            use_baf: Whether to use BAF semantics (default: True)
            baf_config: Optional configuration for BAF parameters

        Returns:
            Tuple of (root_claims_status, baf_details)
            - root_claims_status: Dictionary mapping root claim IDs to NodeStatus
            - baf_details: Optional dictionary with BAF calculation details
        """
        if not use_baf:
            llm_verdict = await self.extract_verdict(judgment_document, graph)
            return llm_verdict, None

        logger.info("[Judge] Starting BAF-enhanced verdict extraction...")
        llm_verdict = await self.extract_verdict(judgment_document, graph)

        llm_validated = {
            nid for nid, status in llm_verdict.items() if status == NodeStatus.VALIDATED
        }

        llm_defeated = {
            nid for nid, status in llm_verdict.items() if status == NodeStatus.DEFEATED
        }

        logger.info(
            f"[Judge] LLM extracted: {len(llm_validated)} VALIDATED, "
            f"{len(llm_defeated)} DEFEATED"
        )

        calculator = baf_calculator or BAFCalculator(graph=graph)

        consistency_report = calculator.validate_consistency(
            llm_validated, llm_defeated
        )

        if not consistency_report["is_consistent"]:
            logger.warning(
                f"[Judge] LLM judgment inconsistencies detected: "
                f"{len(consistency_report['issues'])} issues"
            )

            for issue in consistency_report["issues"]:
                logger.warning(f"[Judge] - {issue['type']}: {issue['message']}")

        best_extension, match_details = calculator.find_best_preferred_extension(
            llm_validated=llm_validated,
            llm_defeated=llm_defeated,
        )

        search_stats = calculator.get_search_stats()
        context_selection = calculator.explain_context_selection()

        preferred_extensions_count = int(
            search_stats.get("preferred_extensions_count", 0) or 0
        )

        if match_details.get("error"):
            logger.warning("[Judge] No preferred extensions found, using LLM verdict")
            empty_stats = calculator.get_search_stats()

            return llm_verdict, {
                "baf_used": True,
                "error": "No preferred extensions",
                "llm_root_claims_status": {
                    claim_id: (
                        status.value if hasattr(status, "value") else str(status)
                    )
                    for claim_id, status in llm_verdict.items()
                },
                "fused_root_claims_status": {
                    claim_id: (
                        status.value if hasattr(status, "value") else str(status)
                    )
                    for claim_id, status in llm_verdict.items()
                },
                "consistency_report": consistency_report,
                "context_selection": context_selection,
                "search_stats": empty_stats,
                "match_details": match_details,
                "chosen_extension": [],
                "chosen_extension_size": 0,
                "algorithm_version": empty_stats.get("algorithm_version", "unknown"),
                "search_time_ms": int(empty_stats.get("search_time_ms", 0) or 0),
                "searched_states": int(empty_stats.get("searched_states", 0) or 0),
                "pruned_states": int(empty_stats.get("pruned_states", 0) or 0),
                "preferred_extensions_count": int(
                    empty_stats.get("preferred_extensions_count", 0) or 0
                ),
                "preferred_extensions_count_estimated": int(
                    empty_stats.get("preferred_extensions_count_estimated", 0) or 0
                ),
                "termination_reason": str(
                    empty_stats.get("termination_reason", "completed")
                ),
            }

        logger.info(
            f"[Judge] BAF selected extension with alignment rate: "
            f"{match_details.get('alignment_rate', 0):.2%}"
        )

        fusion_verdict = self._fuse_llm_and_baf(
            llm_verdict, best_extension, graph, match_details
        )

        baf_details = {
            "baf_used": True,
            "llm_validated": list(llm_validated),
            "llm_defeated": list(llm_defeated),
            "llm_root_claims_status": {
                claim_id: (status.value if hasattr(status, "value") else str(status))
                for claim_id, status in llm_verdict.items()
            },
            "fused_root_claims_status": {
                claim_id: (status.value if hasattr(status, "value") else str(status))
                for claim_id, status in fusion_verdict.items()
            },
            "consistency_report": consistency_report,
            "preferred_extensions_count": preferred_extensions_count,
            "preferred_extensions_count_estimated": int(
                search_stats.get("preferred_extensions_count_estimated", 0) or 0
            ),
            "chosen_extension": match_details.get("chosen_extension", []),
            "chosen_extension_size": len(match_details.get("chosen_extension", [])),
            "match_score": match_details.get("score", 0),
            "match_details": match_details,
            "alignment_rate": match_details.get("alignment_rate", 0),
            "fusion_corrections": self._count_corrections(llm_verdict, fusion_verdict),
            "context_selection": context_selection,
            "search_stats": search_stats,
            "algorithm_version": search_stats.get("algorithm_version", "unknown"),
            "search_time_ms": int(search_stats.get("search_time_ms", 0) or 0),
            "searched_states": int(search_stats.get("searched_states", 0) or 0),
            "pruned_states": int(search_stats.get("pruned_states", 0) or 0),
            "termination_reason": str(
                search_stats.get("termination_reason", "completed")
            ),
        }

        return fusion_verdict, baf_details

    def _fuse_llm_and_baf(
        self,
        llm_verdict: Dict[str, NodeStatus],
        baf_extension: Set[str],
        graph: ShadowGraph,
        match_details: Dict,
    ) -> Dict[str, NodeStatus]:
        """Fuse LLM judgment with BAF preferred extension.

        Fusion strategy:
        - Root claims in both LLM VALIDATED and BAF extension → VALIDATED
        - Root claims in LLM DEFEATED but in BAF extension → VALIDATED (BAF correction)
        - Root claims in LLM VALIDATED but not in BAF extension → DEFEATED (BAF correction)
        - Root claims not decided by LLM but in BAF extension → VALIDATED (BAF inference)
        - Root claims not decided by LLM and not in BAF extension → HYPOTHETICAL

        Args:
            llm_verdict: Initial verdict from LLM
            baf_extension: Best-matching BAF preferred extension
            graph: The debate graph
            match_details: Matching details from BAF calculator

        Returns:
            Fused verdict dictionary
        """
        fused_verdict = {}

        root_claims = {
            nid: data.get("content")
            for nid, data in graph.graph.nodes(data=True)
            if data.get("metadata", {}).get("is_root_claim", False)
        }

        for claim_id in root_claims:
            llm_status = llm_verdict.get(claim_id, NodeStatus.HYPOTHETICAL)
            in_baf_extension = claim_id in baf_extension

            if llm_status == NodeStatus.VALIDATED and in_baf_extension:
                fused_verdict[claim_id] = NodeStatus.VALIDATED

            elif llm_status == NodeStatus.DEFEATED and in_baf_extension:
                logger.info(
                    f"[Judge] BAF correction: {claim_id} was DEFEATED, now VALIDATED"
                )

                fused_verdict[claim_id] = NodeStatus.VALIDATED

            elif llm_status == NodeStatus.VALIDATED and not in_baf_extension:
                logger.info(
                    f"[Judge] BAF correction: {claim_id} was VALIDATED, now DEFEATED"
                )

                fused_verdict[claim_id] = NodeStatus.DEFEATED

            elif llm_status == NodeStatus.HYPOTHETICAL and in_baf_extension:
                logger.info(
                    f"[Judge] BAF inference: {claim_id} was UNMENTIONED, now VALIDATED"
                )

                fused_verdict[claim_id] = NodeStatus.VALIDATED

            else:
                fused_verdict[claim_id] = NodeStatus.HYPOTHETICAL

        return fused_verdict

    def _count_corrections(
        self, llm_verdict: Dict[str, NodeStatus], fused_verdict: Dict[str, NodeStatus]
    ) -> Dict[str, int]:
        """Count corrections made by BAF.

        Args:
            llm_verdict: Original LLM verdict
            fused_verdict: Fused verdict after BAF

        Returns:
            Dictionary with correction counts
        """
        validated_to_defeated = 0
        defeated_to_validated = 0
        hypothetical_to_validated = 0

        for claim_id, llm_status in llm_verdict.items():
            fused_status = fused_verdict.get(claim_id)

            if (
                llm_status == NodeStatus.VALIDATED
                and fused_status == NodeStatus.DEFEATED
            ):
                validated_to_defeated += 1

            elif (
                llm_status == NodeStatus.DEFEATED
                and fused_status == NodeStatus.VALIDATED
            ):
                defeated_to_validated += 1

            elif (
                llm_status == NodeStatus.HYPOTHETICAL
                and fused_status == NodeStatus.VALIDATED
            ):
                hypothetical_to_validated += 1

        return {
            "validated_to_defeated": validated_to_defeated,
            "defeated_to_validated": defeated_to_validated,
            "hypothetical_to_validated": hypothetical_to_validated,
            "total_corrections": (
                validated_to_defeated
                + defeated_to_validated
                + hypothetical_to_validated
            ),
        }

    async def extract_verdict_wrapper(
        self,
        judgment_document: str,
        graph: ShadowGraph,
        use_baf_semantics: bool = False,
    ) -> Tuple[Dict[str, NodeStatus], Optional[Dict]]:
        """Wrapper method for verdict extraction with optional BAF semantics.

        This method is kept as a compatibility entry point for callers that
        route adjudication through a single wrapper.
        It routes to either the standard LLM-only extraction or the BAF-enhanced
        extraction based on the use_baf_semantics parameter.

        Args:
            judgment_document: The full text of the judgment
            graph: The debate graph
            use_baf_semantics: Whether to use BAF semantics (default: False)

        Returns:
            Tuple of (root_claims_status, baf_details)
            - root_claims_status: Dictionary mapping root claim IDs to NodeStatus
            - baf_details: Optional dictionary with BAF calculation details
        """
        if use_baf_semantics:
            return await self.extract_verdict_with_baf(
                judgment_document, graph, use_baf=True
            )

        else:
            verdict = await self.extract_verdict(judgment_document, graph)
            return verdict, None
