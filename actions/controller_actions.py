from metagpt.actions import Action

class PlanTactics(Action):
    name: str = "PlanTactics"
    
    PROMPT_TEMPLATE: str = """
    你是【{role_name}】的辩护律师。
    
    【你的核心人设 (BDI)】:
    - 风格: {style}
    - 信念: {belief}
    - 战略重心: {strategic_focus}
    
    【历史策略锦囊】:
    {insights}
    
    【当前战局】:
    {graph_context}
    
    你的任务：为了推进我方战略，你需要指示【事实参谋(FactWorker)】或【法律参谋(LawWorker)】去寻找什么情报？
    
    请输出一个简练的查询指令。
    示例: "指示 FactWorker 查询关于'借条备注'的先例"
    """

    async def run(self, role_name: str, persona: object, insights: str, graph_context: str):
        prompt = self.PROMPT_TEMPLATE.format(
            role_name=role_name,
            style=persona.intention,
            belief=persona.belief,
            strategic_focus=persona.initial_strategy,
            insights=insights,
            graph_context=graph_context
        )
        
        return await self.llm.aask(prompt)
    
class VerifyAndDecide(Action):
    name: str = "VerifyAndDecide"

    PROMPT_TEMPLATE: str = """
    你是【{role_name}】。你的参谋提供了一条建议。
    
    【参谋建议】:
    "{worker_advice}"
    
    【当前战局】:
    {graph_context}
    
    请判断：这条建议是否有利于你的【战略重心】({focus})？
    
    - 如果有利且逻辑通顺 -> 输出 "ADOPT: <基于建议生成最终的自然语言操作意图>"
      示例: "ADOPT: 采纳建议，添加观点'被告违约'，并用[FACT_1]支持它。"
      
    - 如果无关或有害 -> 输出 "REJECT: <理由>"
    - 如果需要进一步挖掘 -> 输出 "RETRY: <新方向>"
    """

    async def run(self, role_name: str, worker_advice: str, graph_context: str, focus: str):
        prompt = self.PROMPT_TEMPLATE.format(
            role_name=role_name,
            worker_advice=worker_advice,
            graph_context=graph_context,
            focus=focus
        )
        
        return await self.llm.aask(prompt)