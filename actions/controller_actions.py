"""Defines high-level strategic actions for the ArgumentController role.

This module contains Action classes derived from metagpt.actions.Action. These
actions represent the decision-making and strategic planning layer of a
controller agent in the legal debate system. They are responsible for
assessing the state of the debate, determining resource needs (facts, laws,
historical cases), and generating the final graph operations for a turn.
"""

import json

from metagpt.actions import Action

from mas.core.schemas import AGENT_ACTION_SCHEMA_DESC
from prompts.common_prompts import (
    ASSESS_FACT_NEEDS_PROMPT,
    ASSESS_LAW_NEEDS_PROMPT,
    ASSESS_RECALL_NEEDS_PROMPT,
    VERIFY_AND_DECIDE_PROMPT,
)

_RESOURCE_REQUIREMENT_SCHEMA = {
    "name": "resource_requirement",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "need": {"type": "boolean"},
            "reasoning": {"type": "string"},
            "intent": {"type": ["string", "null"]},
        },
        "required": ["need", "reasoning", "intent"],
        "additionalProperties": False,
    },
}

_AGENT_ACTION_LIST_SCHEMA = {
    "name": "agent_action_list",
    "strict": True,
    "schema": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["cite_fact", "cite_law", "support_claim", "rebut_claim"],
                },
                "content": {"type": "string"},
                "target_id": {"type": ["string", "null"]},
                "source_id": {"type": ["string", "null"]},
                "metadata": {"type": "object", "additionalProperties": True},
            },
            "required": ["action_type", "content", "target_id", "source_id"],
            "additionalProperties": False,
        },
    },
}


class AssessFactNeeds(Action):
    """An action to assess the need for additional factual evidence.

    This action uses the agent's persona and the current graph context to
    determine if searching for more facts is strategically necessary.
    """

    name: str = "AssessFactNeeds"

    async def run(self, role_name: str, persona: object, graph_context: str) -> str:
        """Format a prompt and queries the LLM to assess fact-finding needs.

        Args:
            role_name: The name of the role executing the action (e.g., "plaintiff").
            persona: The BDI (Belief, Desire, Intention) persona object of the agent.
            graph_context: A textual representation of the current debate graph.

        Returns:
            A raw string response from the language model, expected to be a JSON
            object detailing whether a fact search is needed, the reasoning,
            and the search intent.
        """
        prompt = ASSESS_FACT_NEEDS_PROMPT.format(
            role_name=role_name,
            belief=persona.belief,
            intention=persona.intention,
            strategy=persona.initial_strategy,
            graph_context=graph_context,
        )

        result = await self.llm.aask_json_schema(
            prompt,
            schema=_RESOURCE_REQUIREMENT_SCHEMA,
            temperature=0.5,
        )

        return json.dumps(result, ensure_ascii=False)


class AssessLawNeeds(Action):
    """An action to assess the need for legal statute retrieval.

    This action uses the agent's persona and the current graph context to
    determine if searching for legal articles is strategically necessary.
    """

    name: str = "AssessLawNeeds"

    async def run(self, role_name: str, persona: object, graph_context: str) -> str:
        """Format a prompt and queries the LLM to assess legal research needs.

        Args:
            role_name: The name of the role executing the action (e.g., "defendant").
            persona: The BDI (Belief, Desire, Intention) persona object of the agent.
            graph_context: A textual representation of the current debate graph.

        Returns:
            A raw string response from the language model, expected to be a JSON
            object detailing whether a law search is needed, the reasoning,
            and the search intent.
        """
        prompt = ASSESS_LAW_NEEDS_PROMPT.format(
            role_name=role_name,
            belief=persona.belief,
            intention=persona.intention,
            strategy=persona.initial_strategy,
            graph_context=graph_context,
        )

        result = await self.llm.aask_json_schema(
            prompt,
            schema=_RESOURCE_REQUIREMENT_SCHEMA,
            temperature=0.5,
        )

        return json.dumps(result, ensure_ascii=False)


class AssessRecallNeeds(Action):
    """An action to assess the need to recall strategies from historical cases.

    This action uses the agent's persona and the current graph context to
    determine if consulting similar past cases for strategic insights is necessary.
    """

    name: str = "AssessRecallNeeds"

    async def run(self, role_name: str, persona: object, graph_context: str) -> str:
        """Format a prompt and queries the LLM to assess historical recall needs.

        Args:
            role_name: The name of the role executing the action.
            persona: The BDI (Belief, Desire, Intention) persona object of the agent.
            graph_context: A textual representation of the current debate graph.

        Returns:
            A raw string response from the language model, expected to be a JSON
            object detailing whether recalling past cases is needed, the reasoning,
            and the strategic intent.
        """
        prompt = ASSESS_RECALL_NEEDS_PROMPT.format(
            role_name=role_name,
            belief=persona.belief,
            intention=persona.intention,
            strategy=persona.initial_strategy,
            graph_context=graph_context,
        )

        result = await self.llm.aask_json_schema(
            prompt,
            schema=_RESOURCE_REQUIREMENT_SCHEMA,
            temperature=0.5,
        )

        return json.dumps(result, ensure_ascii=False)


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
    ):
        """Format a prompt and queries the LLM to generate final agent actions.

        Args:
            role_name: The name of the role executing the action.
            worker_advice: The consolidated summary or advice from worker agents.
            graph_context: A textual representation of the current debate graph.
            focus: The agent's high-level strategic focus for the current turn.
            id_inventory: A string listing all valid node IDs in the graph to
                prevent hallucination of non-existent nodes.
            feedback: Optional feedback from a previously failed execution attempt.

        Returns:
            A raw string response from the language model, expected to be a JSON
            array of `AgentAction` objects to be executed on the debate graph.
        """
        feedback_text = ""

        if feedback:
            feedback_text = f"【⚠️ 警告：之前的尝试被拒绝】\n错误原因: {feedback}\n请务必避免犯同样的错误。"

        prompt = VERIFY_AND_DECIDE_PROMPT.format(
            role_name=role_name,
            worker_advice=worker_advice,
            graph_context=graph_context,
            focus=focus,
            action_schema_desc=AGENT_ACTION_SCHEMA_DESC,
            feedback_section=feedback_text,
            id_inventory=id_inventory,
        )

        result = await self.llm.aask_json_schema(
            prompt,
            schema=_AGENT_ACTION_LIST_SCHEMA,
            temperature=0.5,
        )

        return json.dumps(result, ensure_ascii=False)
