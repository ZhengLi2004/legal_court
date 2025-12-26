from abc import ABC
from typing import Tuple, Dict, List
import re
import asyncio
from .common import ShadowGraph, NodeType, NodeStatus
from .llm import GPTChat, Message
from metagpt.logs import logger

class BaseJudge(ABC):
    def evaluate(self, context: str, graph: ShadowGraph) -> str: pass
    async def extract_verdict(self, judgment_document: str, graph: ShadowGraph) -> Dict[str, NodeStatus]: pass
# LLM 模拟法官进行判决
class LLMJudge(BaseJudge):
    def __init__(self, judge_llm: GPTChat, extraction_llm: GPTChat):
        self.judge_llm = judge_llm
        self.extraction_llm = extraction_llm

    def evaluate(self, context: str, graph: ShadowGraph) -> str:
        prompt = f"""
        你是一名公正的法官。请根据以下案件事实，进行法律分析并给出判决结果。

        【辩论记录】
        {graph.to_recursive_text()}
        """.strip()

        response = self.judge_llm([Message(role="user", content=prompt)], max_tokens=4096)
        return response

    async def extract_verdict(self, judgment_document: str, graph: ShadowGraph) -> Dict[str, NodeStatus]:
        root_claims_status: Dict[str, NodeStatus] = {}
        
        plaintiff_root_claims = {
            nid: data.get('content') 
            for nid, data in graph.graph.nodes(data=True) 
            if data.get('metadata', {}).get('is_root_claim', False) and data.get('agent_id', '').startswith('plaintiff')
        }

        if not plaintiff_root_claims:
            logger.info("No plaintiff root claims found in the graph for extraction.")
            return {}

        extraction_tasks = []
        
        for claim_id, claim_content in plaintiff_root_claims.items():
            extraction_prompt = f"""
            以下是一份法官撰写的法律分析报告：
            ---
            {judgment_document}
            ---
            请根据这份报告的原文内容，判断以下原告诉求是否被法官明确地采纳（ACCEPTED）或驳回（REJECTED）。
            如果报告中明确提到了该诉求并对其做出了“采纳”或“驳回”的判断，请给出相应的状态。
            如果报告中没有明确提及或判断，请输出“UNMENTIONED”。
            
            原告诉求 ID: {claim_id}
            原告诉求内容: {claim_content}
            
            请严格按照以下格式输出你的判断：
            STATUS: [ACCEPTED|REJECTED|UNMENTIONED]
            """
            extraction_tasks.append(self.extraction_llm.aask(extraction_prompt, max_tokens=1024))

        extraction_responses = await asyncio.gather(*extraction_tasks)

        for i, (claim_id, claim_content) in enumerate(plaintiff_root_claims.items()):
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