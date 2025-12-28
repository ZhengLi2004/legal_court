from metagpt.actions import Action
from mas.schema import AGENT_ACTION_SCHEMA_DESC

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
    你是【{role_name}】。你的参谋已完成任务。

    【参谋建议】:
    "{worker_advice}"

    【当前战局】:
    {graph_context}

    请决策：基于战局和报告，生成图谱操作指令以推进【战略重心】({focus})？
    
    {action_schema_desc}

    如果认为不应采纳建议，请直接输出 "REJECT:" 加上拒绝理由。
    """

    async def run(self, role_name: str, worker_advice: str, graph_context: str, focus: str):
        prompt = self.PROMPT_TEMPLATE.format(
            role_name=role_name,
            worker_advice=worker_advice,
            graph_context=graph_context,
            focus=focus,
            action_schema_desc=AGENT_ACTION_SCHEMA_DESC
        )
        
        return await self.llm.aask(prompt, max_tokens=8192)