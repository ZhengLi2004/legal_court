from metagpt.actions import Action

from mas.schema import AGENT_ACTION_SCHEMA_DESC
from prompts.common_prompts import (
    ASSESS_FACT_NEEDS_PROMPT,
    ASSESS_LAW_NEEDS_PROMPT,
    VERIFY_AND_DECIDE_PROMPT,
)


class AssessFactNeeds(Action):
    name: str = "AssessFactNeeds"

    async def run(self, role_name: str, persona: object, graph_context: str) -> str:
        prompt = ASSESS_FACT_NEEDS_PROMPT.format(
            role_name=role_name,
            belief=persona.belief,
            intention=persona.intention,
            strategy=persona.initial_strategy,
            graph_context=graph_context,
        )

        return await self.llm.aask(prompt, temperature=0.1)


class AssessLawNeeds(Action):
    name: str = "AssessLawNeeds"

    async def run(self, role_name: str, persona: object, graph_context: str) -> str:
        prompt = ASSESS_LAW_NEEDS_PROMPT.format(
            role_name=role_name,
            belief=persona.belief,
            intention=persona.intention,
            strategy=persona.initial_strategy,
            graph_context=graph_context,
        )

        return await self.llm.aask(prompt, temperature=0.1)


class VerifyAndDecide(Action):
    name: str = "VerifyAndDecide"

    async def run(
        self,
        role_name: str,
        worker_advice: str,
        graph_context: str,
        focus: str,
        feedback: str = "",
    ):
        feedback_text = ""

        if feedback:
            feedback_text = f"【⚠️ 警告：之前的尝试被拒绝】\n错误原因: {feedback}\n请务必避免犯同样的错误（例如：不要建立自环，不要重复添加已存在的边）。"

        prompt = VERIFY_AND_DECIDE_PROMPT.format(
            role_name=role_name,
            worker_advice=worker_advice,
            graph_context=graph_context,
            focus=focus,
            action_schema_desc=AGENT_ACTION_SCHEMA_DESC,
            feedback_section=feedback_text,
        )

        return await self.llm.aask(prompt, max_tokens=8192)
