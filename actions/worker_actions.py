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
        logger.info(f"Running memory projection for query: {query[:50]}...")

        if current_graph.graph.number_of_nodes() == 0:
            return "无法执行投影：当前图谱为空，没有可用的锚点。"

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
        anchor_ids = [nid for _, nid in candidates[:top_k]]

        if not anchor_ids:
            return f"未能根据查询 '{query}' 在当前图谱中找到足够相关的锚点以进行历史案例投影。"

        history_messages, _ = legal_system.memory.retrieve_memory(query, top_k=3)

        if not history_messages:
            return "根据你的查询，在记忆库中没有找到相关的历史案例可供借鉴。"

        nodes_before = set(current_graph.graph.nodes())
        legal_system.projector.project(current_graph, history_messages)
        nodes_after = set(current_graph.graph.nodes())
        new_node_ids = list(nodes_after - nodes_before)

        if not new_node_ids:
            return "检索到了历史案例，但其中没有与当前战局可关联的新论点被成功投影。"

        logger.info(
            f"Successfully projected {len(new_node_ids)} new nodes into the graph."
        )

        projection_context = current_graph.to_tactical_text(new_node_ids)

        prompt = ANALYZE_RECALL_PROMPT.format(
            user_query=query, projection_context=projection_context
        )

        return await self.llm.aask(prompt)
