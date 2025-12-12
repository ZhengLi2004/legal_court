from metagpt.actions import Action
from prompts.shared import get_shared_prompt, OutputMode

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
    
    {graph_rules_section}
    
    请生成建议:
    """

    async def run(self, role_type: str, graph_context: str, user_query: str, search_result: str):
        rules = get_shared_prompt(mode=OutputMode.LOGIC_ONLY)

        prompt = self.PROMPT_TEMPLATE.format(
            role_type=role_type,
            graph_context=graph_context,
            user_query=user_query,
            search_result=search_result,
            graph_rules_section=rules
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