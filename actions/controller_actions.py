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

    【近期团队对话】:
    {recent_history}

    【当前战局】:
    {graph_context}

    {feedback_section}
    
    【决策分析】
    1. **审视目标**：为了推进你的主张，下一步最合乎逻辑的论点是什么？
    2. **检查可用证据**：仔细查看上方的图谱总结。你是否拥有一个*已存在的、具体的、可引用的* `FACT_...` 或 `LAW_...` 节点ID来直接支持你的论点？
    3. **评估风险**：在图谱中没有直接证据支持的情况下贸然行动是高风险的，很可能会被系统拒绝。如果你不是100%确定，寻求外部信息（调用Worker）是更明智的选择。
    
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

    async def run(self, role_name: str, persona: object, insights: str, graph_context: str, feedback: str = "", history: str = ""):
        feedback_text = ""
        if feedback: feedback_text = f"【⚠️ 上次尝试失败反馈】:\n{feedback}\n请分析失败原因，并重新规划。如果是指令错误，请修正格式；如果是逻辑错误，请调整策略。"
        history_text = history if history else "（暂无近期对话）"
        
        prompt = self.PROMPT_TEMPLATE.format(
            role_name=role_name,
            style=persona.intention,
            belief=persona.belief,
            strategic_focus=persona.initial_strategy,
            insights=insights,
            recent_history=history_text,
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

    {feedback_section}

    任务：请结合【参谋报告】中提到的成果和【当前战局】中的已有节点，生成具体的图谱操作 JSON 数组，以推进【战略重心】({focus})。

    {action_schema_desc}

    请严格按照 JSON 格式输出决策：
    """

    async def run(self, role_name: str, worker_advice: str, graph_context: str, focus: str, feedback: str = ""):
        feedback_text = ""
        if feedback: feedback_text = f"【⚠️ 警告：之前的尝试被拒绝】\n错误原因: {feedback}\n请务必避免犯同样的错误（例如：不要建立自环，不要重复添加已存在的边）。"

        prompt = self.PROMPT_TEMPLATE.format(
            role_name=role_name,
            worker_advice=worker_advice,
            graph_context=graph_context,
            focus=focus,
            action_schema_desc=AGENT_ACTION_SCHEMA_DESC,
            feedback_section=feedback_text
        )
        
        return await self.llm.aask(prompt, max_tokens=8192)