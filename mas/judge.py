import asyncio
from abc import ABC
from typing import Dict, List

from metagpt.logs import logger

from prompts.common_prompts import JUDGE_EVALUATE_PROMPT, JUDGE_EXTRACT_VERDICT_PROMPT

from .common import NodeStatus, ShadowGraph
from .llm import GPTChat, Message


class BaseJudge(ABC):
    def evaluate(self, context: str, graph: ShadowGraph, transcript: List[str]) -> str:
        pass

    async def extract_verdict(
        self, judgment_document: str, graph: ShadowGraph
    ) -> Dict[str, NodeStatus]:
        pass


class LLMJudge(BaseJudge):
    def __init__(self, judge_llm: GPTChat, extraction_llm: GPTChat):
        self.judge_llm = judge_llm
        self.extraction_llm = extraction_llm

    def evaluate(self, context: str, graph: ShadowGraph, transcript: List[str]) -> str:
        root_claims = []

        for _, data in graph.graph.nodes(data=True):
            if data.get("metadata", {}).get("is_root_claim"):
                root_claims.append(data.get("content", "未知诉求"))

        if not root_claims:
            issue_list_str = "（未检测到明确的根诉求，请根据庭审笔录自行归纳争议焦点）"
            logger.warning("[Judge] No root claims found in graph metadata.")

        else:
            issue_list_str = "\n".join(
                [f"{i + 1}. {content}" for i, content in enumerate(root_claims)]
            )

        if not transcript:
            transcript_str = "（本案无庭审辩论记录）"

            logger.warning(
                "[Judge] Transcript is empty! Judgment might be hallucinated."
            )

        else:
            transcript_str = "\n\n".join(transcript)
            logger.info(f"[Judge] compiled transcript with {len(transcript)} segments.")

        prompt = JUDGE_EVALUATE_PROMPT.format(
            issue_list=issue_list_str, transcript=transcript_str
        )

        logger.info(">>> [Judge] Generating Judgment Document...")

        response = self.judge_llm(
            [Message(role="user", content=prompt)], max_tokens=4096
        )

        return response

    async def extract_verdict(
        self, judgment_document: str, graph: ShadowGraph
    ) -> Dict[str, NodeStatus]:
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
                self.extraction_llm.aask(extraction_prompt, max_tokens=1024)
            )

        extraction_responses = await asyncio.gather(*extraction_tasks)

        for i, (claim_id, claim_content) in enumerate(all_root_claims.items()):
            response = extraction_responses[i]

            if "STATUS: ACCEPTED" in response:
                root_claims_status[claim_id] = NodeStatus.VALIDATED

            elif "STATUS: REJECTED" in response:
                root_claims_status[claim_id] = NodeStatus.DEFEATED

            elif "STATUS: UNMENTIONED" in response:
                root_claims_status[claim_id] = NodeStatus.HYPOTHETICAL

            else:
                root_claims_status[claim_id] = NodeStatus.HYPOTHETICAL

                logger.debug(
                    f"[Judge-Extract] Unclear status for {claim_id}: {response}"
                )

        return root_claims_status
