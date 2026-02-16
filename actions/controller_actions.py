"""Define high-level strategic actions for the argument controller.

The actions in this module use strict OpenAI function-calling contracts for
routing. They do not rely on JSON text extraction and do not fall back to
legacy JSON-only prompting.
"""

from typing import Any, Dict, List

from metagpt.actions import Action

from mas.core.schemas import AGENT_ACTION_SCHEMA_DESC
from prompts.common_prompts import (
    ASSESS_FACT_NEEDS_PROMPT,
    ASSESS_LAW_NEEDS_PROMPT,
    ASSESS_RECALL_NEEDS_PROMPT,
    VERIFY_AND_DECIDE_PROMPT,
)

_RESOURCE_REQUIREMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "need": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "intent": {"type": ["string", "null"]},
    },
    "required": ["need", "reasoning", "intent"],
    "additionalProperties": False,
}

_AGENT_ACTION_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": [
                            "cite_fact",
                            "cite_law",
                            "support_claim",
                            "rebut_claim",
                        ],
                    },
                    "content": {"type": "string"},
                    "target_id": {"type": ["string", "null"]},
                    "source_id": {"type": ["string", "null"]},
                    "metadata": {
                        "type": "object",
                        "properties": {"reason_brief": {"type": "string"}},
                        "required": ["reason_brief"],
                        "additionalProperties": True,
                    },
                },
                "required": [
                    "action_type",
                    "content",
                    "target_id",
                    "source_id",
                    "metadata",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["actions"],
    "additionalProperties": False,
}

_ASSESS_FACT_NEEDS_TOOL = {
    "type": "function",
    "function": {
        "name": "assess_fact_needs",
        "description": "Assess whether factual retrieval is needed this turn.",
        "parameters": _RESOURCE_REQUIREMENT_SCHEMA,
    },
}

_ASSESS_LAW_NEEDS_TOOL = {
    "type": "function",
    "function": {
        "name": "assess_law_needs",
        "description": "Assess whether legal-article retrieval is needed this turn.",
        "parameters": _RESOURCE_REQUIREMENT_SCHEMA,
    },
}

_ASSESS_RECALL_NEEDS_TOOL = {
    "type": "function",
    "function": {
        "name": "assess_recall_needs",
        "description": "Assess whether historical-case recall is needed this turn.",
        "parameters": _RESOURCE_REQUIREMENT_SCHEMA,
    },
}

_VERIFY_AND_DECIDE_TOOL = {
    "type": "function",
    "function": {
        "name": "verify_and_decide",
        "description": "Return finalized graph-operation actions for this turn.",
        "parameters": _AGENT_ACTION_LIST_SCHEMA,
    },
}


class AssessFactNeeds(Action):
    """An action to assess the need for additional factual evidence.

    This action uses the agent's persona and the current graph context to
    determine if searching for more facts is strategically necessary.
    """

    name: str = "AssessFactNeeds"

    async def run(
        self, role_name: str, persona: object, graph_context: str
    ) -> Dict[str, Any]:
        """Assess whether fact retrieval is needed.

        Args:
            role_name: The name of the role executing the action (e.g., "plaintiff").
            persona: The BDI (Belief, Desire, Intention) persona object of the agent.
            graph_context: A textual representation of the current debate graph.

        Returns:
            Parsed tool-call arguments with `need`, `reasoning`, and `intent`.

        Raises:
            ToolCallContractError: If the model violates the tool-call contract.
        """
        prompt = ASSESS_FACT_NEEDS_PROMPT.format(
            role_name=role_name,
            belief=persona.belief,
            intention=persona.intention,
            strategy=persona.initial_strategy,
            graph_context=graph_context,
        )

        result = await self.llm.aask_tool_call(
            prompt=prompt,
            tools=[_ASSESS_FACT_NEEDS_TOOL],
            tool_choice="assess_fact_needs",
            temperature=0.5,
        )

        return result.arguments


class AssessLawNeeds(Action):
    """An action to assess the need for legal statute retrieval.

    This action uses the agent's persona and the current graph context to
    determine if searching for legal articles is strategically necessary.
    """

    name: str = "AssessLawNeeds"

    async def run(
        self, role_name: str, persona: object, graph_context: str
    ) -> Dict[str, Any]:
        """Assess whether law retrieval is needed.

        Args:
            role_name: The name of the role executing the action (e.g., "defendant").
            persona: The BDI (Belief, Desire, Intention) persona object of the agent.
            graph_context: A textual representation of the current debate graph.

        Returns:
            Parsed tool-call arguments with `need`, `reasoning`, and `intent`.

        Raises:
            ToolCallContractError: If the model violates the tool-call contract.
        """
        prompt = ASSESS_LAW_NEEDS_PROMPT.format(
            role_name=role_name,
            belief=persona.belief,
            intention=persona.intention,
            strategy=persona.initial_strategy,
            graph_context=graph_context,
        )

        result = await self.llm.aask_tool_call(
            prompt=prompt,
            tools=[_ASSESS_LAW_NEEDS_TOOL],
            tool_choice="assess_law_needs",
            temperature=0.5,
        )

        return result.arguments


class AssessRecallNeeds(Action):
    """An action to assess the need to recall strategies from historical cases.

    This action uses the agent's persona and the current graph context to
    determine if consulting similar past cases for strategic insights is necessary.
    """

    name: str = "AssessRecallNeeds"

    async def run(
        self, role_name: str, persona: object, graph_context: str
    ) -> Dict[str, Any]:
        """Assess whether historical-case recall is needed.

        Args:
            role_name: The name of the role executing the action.
            persona: The BDI (Belief, Desire, Intention) persona object of the agent.
            graph_context: A textual representation of the current debate graph.

        Returns:
            Parsed tool-call arguments with `need`, `reasoning`, and `intent`.

        Raises:
            ToolCallContractError: If the model violates the tool-call contract.
        """
        prompt = ASSESS_RECALL_NEEDS_PROMPT.format(
            role_name=role_name,
            belief=persona.belief,
            intention=persona.intention,
            strategy=persona.initial_strategy,
            graph_context=graph_context,
        )

        result = await self.llm.aask_tool_call(
            prompt=prompt,
            tools=[_ASSESS_RECALL_NEEDS_TOOL],
            tool_choice="assess_recall_needs",
            temperature=0.5,
        )

        return result.arguments


class VerifyAndDecide(Action):
    """An action to verify worker advice and decide on final graph operations.

    This action synthesizes the information gathered by worker agents with the
    current debate state to generate a sequence of concrete `AgentAction`s to
    modify the debate graph. It also incorporates feedback from previous failed
    attempts to avoid repeating mistakes.
    """

    name: str = "VerifyAndDecide"

    async def run(
        self,
        role_name: str,
        worker_advice: str,
        graph_context: str,
        focus: str,
        id_inventory: str,
        feedback: str = "",
    ) -> List[Dict[str, Any]]:
        """Generate final graph actions for the current turn.

        Args:
            role_name: The name of the role executing the action.
            worker_advice: The consolidated summary or advice from worker agents.
            graph_context: A textual representation of the current debate graph.
            focus: The agent's high-level strategic focus for the current turn.
            id_inventory: A string listing all valid node IDs in the graph to
                prevent hallucination of non-existent nodes.
            feedback: Optional feedback from a previously failed execution attempt.

        Returns:
            A list of action dictionaries parsed from function arguments.
            Each action payload includes `metadata.reason_brief`.

        Raises:
            ToolCallContractError: If the model violates the tool-call contract.
            ValueError: If `actions` field is missing or has an invalid type.
        """
        feedback_text = ""

        if feedback:
            feedback_text = (
                "【上一轮结果反馈】\n"
                f"错误原因: {feedback}\n"
                "请修正动作字段，并确保每个动作的 metadata.reason_brief 与动作内容一致。"
            )

        prompt = VERIFY_AND_DECIDE_PROMPT.format(
            role_name=role_name,
            worker_advice=worker_advice,
            graph_context=graph_context,
            focus=focus,
            action_schema_desc=AGENT_ACTION_SCHEMA_DESC,
            feedback_section=feedback_text,
            id_inventory=id_inventory,
        )

        result = await self.llm.aask_tool_call(
            prompt=prompt,
            tools=[_VERIFY_AND_DECIDE_TOOL],
            tool_choice="verify_and_decide",
            temperature=0.5,
        )

        payload = result.arguments
        actions = payload.get("actions", [])

        if not isinstance(actions, list):
            raise ValueError("`verify_and_decide` tool arguments must include list `actions`.")

        return actions
