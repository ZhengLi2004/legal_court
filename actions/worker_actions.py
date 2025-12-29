from metagpt.actions import Action

from prompts.common_prompts import ANALYZE_SEARCH_RESULTS_PROMPT, SUGGEST_PIVOT_PROMPT


class AnalyzeSearchResults(Action):
    name: str = "AnalyzeSearchResults"

    async def run(
        self, role_type: str, graph_context: str, user_query: str, search_result: str
    ):
        prompt = ANALYZE_SEARCH_RESULTS_PROMPT.format(
            role_type=role_type,
            graph_context=graph_context,
            user_query=user_query,
            search_result=search_result,
        )

        return await self.llm.aask(prompt)


class SuggestPivot(Action):
    name: str = "SuggestPivot"

    async def run(self, graph_context: str, user_query: str):
        prompt = SUGGEST_PIVOT_PROMPT.format(
            graph_context=graph_context, user_query=user_query
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

        for hit in hits:
            score = hit["_score"] - 1.0

            if score < threshold:
                continue

            source = hit["_source"]
            content = f"《{source.get('law_name')}》{source.get('article_id')}: {source.get('content')}"
            law_contents.append(content)

        if not law_contents:
            return f"检索到法条但相似度均低于阈值 ({threshold})。"

        log = graph_tool.inject_law_nodes(law_contents)
        return f"✅ 检索完成。{log}\n内容示例: {law_contents[0][:20]}..."
