from abc import ABC
from typing import Tuple, Dict, List
import re
import asyncio
from .common import ShadowGraph, NodeType, NodeStatus
from .llm import GPTChat, Message
from metagpt.logs import logger
from prompts.common_prompts import JUDGE_EVALUATE_PROMPT, JUDGE_EXTRACT_VERDICT_PROMPT

class BaseJudge(ABC):
    def evaluate(self, context: str, graph: ShadowGraph) -> str: pass
    async def extract_verdict(self, judgment_document: str, graph: ShadowGraph) -> Dict[str, NodeStatus]: pass
# LLM 模拟法官进行判决
class LLMJudge(BaseJudge):
    def __init__(self, judge_llm: GPTChat, extraction_llm: GPTChat):
        self.judge_llm = judge_llm
        self.extraction_llm = extraction_llm

    def evaluate(self, context: str, graph: ShadowGraph) -> str:
        prompt = JUDGE_EVALUATE_PROMPT.format(graph_text=graph.to_recursive_text()).strip()
        response = self.judge_llm([Message(role="user", content=prompt)], max_tokens=4096)
        return response

    async def extract_verdict(self, judgment_document: str, graph: ShadowGraph) -> Dict[str, NodeStatus]:
        root_claims_status: Dict[str, NodeStatus] = {}
        

        all_root_claims = {
            nid: data.get('content') 
            for nid, data in graph.graph.nodes(data=True) 
            if data.get('metadata', {}).get('is_root_claim', False)
        }

        if not all_root_claims:
            logger.info("No root claims found in the graph for extraction.")
            return {}

        extraction_tasks = []
        
        for claim_id, claim_content in all_root_claims.items():
            extraction_prompt = JUDGE_EXTRACT_VERDICT_PROMPT.format(
                judgment_document=judgment_document,
                claim_id=claim_id,
                claim_content=claim_content
            )
            
            extraction_tasks.append(self.extraction_llm.aask(extraction_prompt, max_tokens=1024))

        extraction_responses = await asyncio.gather(*extraction_tasks)

        for i, (claim_id, claim_content) in enumerate(all_root_claims.items()):
            response = extraction_responses[i]
            if "STATUS: ACCEPTED" in response: root_claims_status[claim_id] = NodeStatus.VALIDATED
            elif "STATUS: REJECTED" in response: root_claims_status[claim_id] = NodeStatus.DEFEATED
            
            elif "STATUS: UNMENTIONED" in response:
                root_claims_status[claim_id] = NodeStatus.HYPOTHETICAL # Treat as not yet settled
                logger.info(f"Extraction LLM: Claim {claim_id} explicitly marked as UNMENTIONED.")
            
            else:
                root_claims_status[claim_id] = NodeStatus.HYPOTHETICAL
                logger.warning(f"Extraction LLM returned unclear/unexpected status for claim {claim_id}: '{response}'. Marking as HYPOTHETICAL.")
        
        return root_claims_status