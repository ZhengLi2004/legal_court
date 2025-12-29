from metagpt.actions import Action

from prompts.common_prompts import ANALYZE_SEARCH_RESULTS_PROMPT


class AnalyzeSearchResults(Action):
    name: str = "AnalyzeSearchResults"

    async def run(self, role_type: str, user_query: str, search_result: str):
        prompt = ANALYZE_SEARCH_RESULTS_PROMPT.format(
            role_type=role_type,
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
                f"•《{source.get('law_name')}》{source.get('article_id')}: {source.get('content')[:50]}..."
            )

        if not law_contents:
            return f"检索到法条但相似度均低于阈值 ({threshold})。"

        graph_tool.inject_law_nodes(law_contents)

        report = (
            f"✅ 已成功将 {len(law_contents)} 条相关法条注入图谱。\n"
            f"注入内容摘要：\n" + "\n".join(injected_details) + "\n"
            f"这些法条为解决'{query}'提供了直接的法律依据。"
        )

        return report
