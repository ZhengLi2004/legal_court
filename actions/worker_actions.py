from typing import List

from metagpt.actions import Action
from metagpt.logs import logger

from mas.common import ShadowGraph
from mas.legal_system import LegalSystem
from mas.utils import cosine_similarity
from prompts.common_prompts import (
    ANALYZE_RECALL_PROMPT,
    DECOMPOSE_SEARCH_INTENT_PROMPT,
)
from tools.json_utils import extract_json_from_text


class AnalyzeSearchResults(Action):
    name: str = "AnalyzeSearchResults"

    async def run(
        self, user_query: str, search_result: str, prompt_template: str
    ) -> str:
        prompt = prompt_template.format(
            user_query=user_query,
            search_result=search_result,
        )

        return await self.llm.aask(prompt)


class FormulateSearchQueries(Action):
    name: str = "FormulateSearchQueries"

    async def run(
        self, intent: str, prompt_template: str = DECOMPOSE_SEARCH_INTENT_PROMPT
    ) -> List[str]:
        prompt = prompt_template.format(intent=intent)

        try:
            response = await self.llm.aask(prompt, temperature=0.5)
            queries = extract_json_from_text(response)

            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries

            logger.warning(
                "[FormulateSearchQueries] Output format invalid. Fallback to intent."
            )

            return [intent]

        except Exception as e:
            logger.error(f"[FormulateSearchQueries] Failed: {e}")
            return [intent]


class ProjectAndAnalyze(Action):
    name: str = "ProjectAndAnalyze"

    async def run(
        self,
        query: str,
        legal_system: LegalSystem,
        current_graph: ShadowGraph,
        top_k: int = 3,
    ) -> str:
        logger.info(
            f"Running historical projection retrieval (cached) for intent: {query[:50]}..."
        )

        if current_graph.graph.number_of_nodes() == 0:
            return "无法执行历史映射：当前图谱为空，没有可用的锚点。"

        query_emb = legal_system.ef.embed_query(query)
        candidates = []

        for nid, data in current_graph.graph.nodes(data=True):
            content = data.get("content", "")

            if not content:
                continue

            node_emb = legal_system.ef.embed_query(content)
            sim = cosine_similarity(query_emb, node_emb)
            candidates.append((sim, nid))

        candidates.sort(key=lambda x: x[0], reverse=True)
        focus_node_ids = [nid for _, nid in candidates[:top_k]]

        if not focus_node_ids:
            return (
                f"未能根据意图 '{query}' 在当前图谱中找到足够相关的节点作为映射锚点。"
            )

        history_messages = legal_system.active_history_cases

        if not history_messages:
            return "系统初始化时未检索到相关历史案例，暂无经验可供借鉴。"

        historical_context_text = legal_system.projector.retrieve_historical_context(
            current_graph=current_graph,
            focus_node_ids=focus_node_ids,
            history_messages=history_messages,
        )

        if "未找到" in historical_context_text or not historical_context_text.strip():
            return "在预加载的历史案例中，未找到与当前锚点结构相似的论点路径。"

        logger.info("Successfully retrieved historical context text from cache.")

        prompt = ANALYZE_RECALL_PROMPT.format(
            user_query=query, projection_context=historical_context_text
        )

        return await self.llm.aask(prompt)
