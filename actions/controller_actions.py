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

    {feedback_section}
    
    你的任务：为了推进我方战略，你需要指示哪位参谋去寻找什么情报？
    
    【可用参谋】:
    - **事实参谋 (FactWorker)**: 负责检索相似的历史判例，提供战术参考。
    - **法律参谋 (LawWorker)**: 负责检索具体的法律条款。
    
    请输出一个简练的查询指令，并明确指出由谁执行。
    
    【输出格式】:
    指示 [FactWorker/LawWorker] 查询 [具体内容]
    
    【示例】:
    指示 FactWorker 查询关于'借条备注'的先例
    指示 LawWorker 查询《民法典》中关于'债务加入'的规定
    """

    async def run(self, role_name: str, persona: object, insights: str, graph_context: str, feedback: str = ""):
        feedback_text = ""
        if feedback: feedback_text = f"【⚠️ 上次尝试失败反馈】:\n{feedback}\n请分析失败原因，并重新规划。如果是指令错误，请修正格式；如果是逻辑错误，请调整策略。"
        
        prompt = self.PROMPT_TEMPLATE.format(
            role_name=role_name,
            style=persona.intention,
            belief=persona.belief,
            strategic_focus=persona.initial_strategy,
            insights=insights,
            graph_context=graph_context,
            feedback_section=feedback_text
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
    
    【输出规范】:
    1. 使用自然语言描述逻辑。
    2. 对于**新添加**的内容，请使用 `<new_claim>内容</new_claim>` 或 `<new_law>内容</new_law>` 标记。
    3. 对于**已有**的节点，使用 `[ID]` 引用。
    
    示例: 
    "ADOPT: 采纳建议。添加观点 <new_claim>被告违约</new_claim>，并使用 [FACT_u8a9] 支持它。"
    """

    async def run(self, role_name: str, worker_advice: str, graph_context: str, focus: str):
        prompt = self.PROMPT_TEMPLATE.format(
            role_name=role_name,
            worker_advice=worker_advice,
            graph_context=graph_context,
            focus=focus
        )
        
        return await self.llm.aask(prompt)