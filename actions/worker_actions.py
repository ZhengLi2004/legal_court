from metagpt.actions import Action

class AnalyzeSearchResults(Action):
    name: str = "AnalyzeSearchResults"

    PROMPT_TEMPLATE: str = """
    你是一个法律辩论团队的【{role_type}参谋】。
    
    【当前战局 (Graph Context)】:
    {graph_context}
    
    【指挥官的查询意图】:
    "{user_query}"
    
    【检索到的情报】:
    {search_result}
    
    你的任务：结合情报，为指挥官提供具体的图谱操作建议。
    你的建议应该清晰地指出要“添加”什么（事实/主张/法条），或者要“支持”/“反驳”哪个已有节点。
    如果涉及关系，请明确指出源节点和目标节点（使用其ID）。

    【建议示例】:
    - 基于检索结果，添加事实：“王某于2023年3月15日向李某借款人民币10万元。”
    - 基于检索结果，引用法条：“中华人民共和国合同法 第二百零六条”，并用其支持主张 CLAIM_abcde123。
    - 基于检索结果，添加主张：“被告无证据证明其已履行还款义务”，并用其反驳 CLAIM_fghij456。

    请生成建议:
    """

    async def run(self, role_type: str, graph_context: str, user_query: str, search_result: str):
        prompt = self.PROMPT_TEMPLATE.format(
            role_type=role_type,
            graph_context=graph_context,
            user_query=user_query,
            search_result=search_result,
        )

        return await self.llm.aask(prompt)
    
class SuggestPivot(Action):
    name: str = "SuggestPivot"

    PROMPT_TEMPLATE: str = """
    你是一个法律参谋。指挥官想查询 "{user_query}"，但在数据库中**未找到**强相关的内容。
    
    【当前战局】:
    {graph_context}
    
    请分析可能的失败原因，并提供一个**新的查询方向**或**更有效的关键词**。
    
    【输出示例】:
    "关于'口头变更'的直接先例较少。建议调整方向，尝试搜索 '合同实际履行' 或 '默认行为' 相关的判例。"
    
    请生成转型建议:
    """

    async def run(self, graph_context: str, user_query: str):
        prompt = self.PROMPT_TEMPLATE.format(
            graph_context=graph_context, 
            user_query=user_query
        )

        return await self.llm.aask(prompt)
    
class InjectLawsToGraph(Action):
    name: str = "InjectLawsToGraph"

    async def run(self, query: str, es_tool, graph_tool, threshold: float = 0.6, top_k: int = 3) -> str:
        try:
            query_vector = es_tool.embedding_func.embed_query(query)

            hits = await es_tool._search(
                index_name=es_tool.INDEX_NAME,
                query_vector=query_vector,
                vector_field=es_tool.VECTOR_FIELD,
                source_fields=es_tool.SOURCE_FIELDS,
                top_k=top_k
            )
        
        except Exception as e: return f"检索失败: {str(e)}"
        if not hits: return "未检索到相关法条。"
        law_contents = []

        for hit in hits:
            score = hit['_score'] - 1.0
            if score < threshold: continue
            source = hit['_source']
            content = f"《{source.get('law_name')}》{source.get('article_id')}: {source.get('content')}"
            law_contents.append(content)

        if not law_contents: return f"检索到法条但相似度均低于阈值 ({threshold})。"
        log = graph_tool.inject_law_nodes(law_contents)
        return f"✅ 检索完成。{log}\n内容示例: {law_contents[0][:20]}..."