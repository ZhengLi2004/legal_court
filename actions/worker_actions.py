from metagpt.actions import Action
from metagpt.logs import logger

from mas.common import ShadowGraph
from mas.legal_system import LegalSystem
from mas.utils import cosine_similarity
from prompts.common_prompts import ANALYZE_RECALL_PROMPT


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


class InjectLawsToGraph(Action):
    name: str = "InjectLawsToGraph"

    async def run(
        self, query: str, es_tool, graph_tool, threshold: float = 0.6, top_k: int = 3
    ) -> str:
        try:
            hits = await es_tool.search_laws_raw(query, top_k=top_k)

        except Exception as e:
            return f"检索失败: {str(e)}"

        if not hits:
            return "未检索到相关法条。"

        law_contents = []
        injected_details = []

        for hit in hits:
            score = hit["_score"] - 1.0

            if score < threshold:
                continue

            source = hit["_source"]
            content = f"《{source.get('law_name')}》{source.get('article_id')}: {source.get('content')}"
            law_contents.append(content)

            injected_details.append(
                f"•《{source.get('law_name')}》{source.get('article_id')}: {source.get('content')}"
            )

        if not law_contents:
            return f"检索到法条但相似度均低于阈值 ({threshold})。"

        graph_tool.inject_law_nodes(law_contents)
        return "\n".join(injected_details)


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
