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
    
    你的任务：判断当前是否需要补充外部情报（法条或事实），还是可以直接进行论证。
    
    【决策选项】:
    1. **LawWorker**: 图谱中缺少支撑你观点的【法条节点】。填写法条检索需求。
    2. **FactWorker**: 图谱中缺少【历史判例】或需要确认某些事实细节。填写事实检索需求。
    3. **Self**: 图谱中已有足够的法条和事实，或者是时候发起攻击/总结了。简述你的论证思路。
    
    请严格按照以下 JSON 格式输出决策：
    ```json
    {{
        "target": "LawWorker" | "FactWorker" | "Self",
        "content": "..."
    }}
    ```
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
    你是【{role_name}】。你的参谋已完成任务并提交了报告。

    【参谋报告/行动结果】:
    "{worker_advice}"

    【当前战局】:
    {graph_context}

    任务：请结合【参谋报告】中提到的成果和【当前战局】中的已有节点，生成具体的图谱操作 JSON 数组，以推进【战略重心】({focus})。

    {action_schema_desc}

    请严格按照 JSON 格式输出决策：
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