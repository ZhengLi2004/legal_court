from metagpt.actions import Action
from mas.schema import AGENT_ACTION_SCHEMA_DESC
from prompts.common_prompts import PLAN_TACTICS_PROMPT, VERIFY_AND_DECIDE_PROMPT

class PlanTactics(Action):
    name: str = "PlanTactics"
    
    async def run(self, role_name: str, persona: object, insights: str, graph_context: str, feedback: str = "", history: str = ""):
        feedback_text = ""
        if feedback: feedback_text = f"【⚠️ 上次尝试失败反馈】:\n{feedback}\n请分析失败原因，并重新规划。如果是指令错误，请修正格式；如果是逻辑错误，请调整策略。"
        history_text = history if history else "（暂无近期对话）"
        
        prompt = PLAN_TACTICS_PROMPT.format(
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

    async def run(self, role_name: str, worker_advice: str, graph_context: str, focus: str, feedback: str = ""):
        feedback_text = ""
        if feedback: feedback_text = f"【⚠️ 警告：之前的尝试被拒绝】\n错误原因: {feedback}\n请务必避免犯同样的错误（例如：不要建立自环，不要重复添加已存在的边）。"

        prompt = VERIFY_AND_DECIDE_PROMPT.format(
            role_name=role_name,
            worker_advice=worker_advice,
            graph_context=graph_context,
            focus=focus,
            action_schema_desc=AGENT_ACTION_SCHEMA_DESC,
            feedback_section=feedback_text
        )
        
        return await self.llm.aask(prompt, max_tokens=8192)