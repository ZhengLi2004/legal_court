"""Defines judge abstractions used for final adjudication."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict

from metagpt.logs import logger

from prompts.common_prompts import JUDGE_EVALUATE_PROMPT, JUDGE_EXTRACT_VERDICT_PROMPT
from tools.llm import GPTChat, Message

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
    """Abstract interface for judgment generation and verdict extraction."""

    @abstractmethod
    def evaluate(self, context: str, graph: ShadowGraph) -> str:
        """Generate a final judgment document.

        Args:
            context: Input case facts.
            graph: Final debate graph before adjudication.

        Returns:
            Generated judgment document.
        """

    @abstractmethod
    async def extract_verdict(
        self, judgment_document: str, graph: ShadowGraph
    ) -> Dict[str, NodeStatus]:
        """Extract root-claim statuses from the generated judgment.

        Args:
            judgment_document: Final judgment text.
            graph: Debate graph used to locate root claims.

        Returns:
            Mapping from root-claim IDs to extracted statuses.
        """


class LLMJudge(BaseJudge):
    """Judge implementation backed by LLM calls."""

    def __init__(self, judge_llm: GPTChat, extraction_llm: GPTChat):
        """Initialize the judge with generation and extraction models.

        Args:
            judge_llm: LLM used to write the final judgment.
            extraction_llm: LLM used to classify each root claim.
        """
        self.judge_llm = judge_llm
        self.extraction_llm = extraction_llm

    def evaluate(self, context: str, graph: ShadowGraph) -> str:
        """Generate a judgment document from graph context.

        Args:
            context: Input case facts.
            graph: Final debate graph before adjudication.

        Returns:
            Generated judgment document.
        """
        del context
        payload = self._build_adjudication_input(graph=graph)

        prompt = JUDGE_EVALUATE_PROMPT.format(
            issue_list=payload["issue_list"],
            graph_context=payload["graph_context"],
        )

        return self.judge_llm([Message(role="user", content=prompt)])

    def _build_adjudication_input(self, graph: ShadowGraph) -> Dict[str, Any]:
        """Assemble issue list and graph context for the judgment prompt.

        Args:
            graph: Debate graph to summarize for adjudication.

        Returns:
            Prompt payload containing issue list, graph context, and metrics.
        """
        root_claims: Dict[str, str] = {}

        for node_id, data in graph.graph.nodes(data=True):
            metadata = data.get("metadata", {})

            if metadata.get("is_root_claim", False):
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

        Args:
            judgment_document: Full text of the generated judgment.
            graph: Debate graph containing root-claim metadata.

        Returns:
            Mapping from root-claim ID to `NodeStatus`.
        """
        root_claims_status: Dict[str, NodeStatus] = {}

        all_root_claims = {
            str(node_id): data.get("content")
            for node_id, data in graph.graph.nodes(data=True)
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

        for idx, claim_id in enumerate(all_root_claims):
            response = extraction_responses[idx]

            if isinstance(response, Exception):
                root_claims_status[claim_id] = NodeStatus.HYPOTHETICAL

                logger.warning(
                    f"[Judge-Extract] Extraction failed for {claim_id}: {response}"
                )

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

        return root_claims_status
