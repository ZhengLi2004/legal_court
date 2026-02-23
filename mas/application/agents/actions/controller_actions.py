"""Define high-level strategic actions for the argument controller.

The actions in this module use strict OpenAI function-calling contracts for
routing. They do not rely on JSON text extraction and do not fall back to
legacy JSON-only prompting.
"""

import asyncio
from typing import Any, Dict, List, Tuple

from metagpt.actions import Action

from mas.core.schemas import AGENT_ACTION_SCHEMA_DESC, AgentAction
from prompts.common_prompts import (
    ASSESS_FACT_NEEDS_PROMPT,
    ASSESS_LAW_NEEDS_PROMPT,
    ASSESS_RECALL_NEEDS_PROMPT,
    CHOOSE_PLAN_OR_PUSH_PROMPT,
    SYSTEM_PROMPT_CONTROLLER_ROUTING,
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

_PLAN_OR_PUSH_SCHEMA = {
    "type": "object",
    "properties": {
        "next_step": {"type": "string", "enum": ["plan", "push"]},
        "reason": {"type": "string"},
    },
    "required": ["next_step", "reason"],
    "additionalProperties": False,
}

_CHOOSE_PLAN_OR_PUSH_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_plan_or_push",
        "description": "Choose whether to continue planning or push validated actions.",
        "parameters": _PLAN_OR_PUSH_SCHEMA,
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
            desire=persona.desire,
            intention=persona.intention,
            graph_context=graph_context,
        )

        result = await self.llm.aask_tool_call(
            prompt=prompt,
            tools=[_ASSESS_FACT_NEEDS_TOOL],
            tool_choice="assess_fact_needs",
            system_msgs=[SYSTEM_PROMPT_CONTROLLER_ROUTING],
            temperature=0.4,
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
            desire=persona.desire,
            intention=persona.intention,
            graph_context=graph_context,
        )

        result = await self.llm.aask_tool_call(
            prompt=prompt,
            tools=[_ASSESS_LAW_NEEDS_TOOL],
            tool_choice="assess_law_needs",
            system_msgs=[SYSTEM_PROMPT_CONTROLLER_ROUTING],
            temperature=0.4,
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
            desire=persona.desire,
            intention=persona.intention,
            graph_context=graph_context,
        )

        result = await self.llm.aask_tool_call(
            prompt=prompt,
            tools=[_ASSESS_RECALL_NEEDS_TOOL],
            tool_choice="assess_recall_needs",
            system_msgs=[SYSTEM_PROMPT_CONTROLLER_ROUTING],
            temperature=0.4,
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
            system_msgs=[SYSTEM_PROMPT_CONTROLLER_ROUTING],
            temperature=0.4,
        )

        payload = result.arguments
        actions = payload.get("actions", [])

        if not isinstance(actions, list):
            raise ValueError(
                "`verify_and_decide` tool arguments must include list `actions`."
            )

        return actions


class ChoosePlanOrPush(Action):
    """Route controller next step between plan and push."""

    name: str = "ChoosePlanOrPush"

    async def run(
        self,
        role_name: str,
        worker_advice: str,
        graph_context: str,
        action_cache_context: str,
        has_validated_plan: bool,
        plan_attempt: int,
        max_plan_attempts: int,
        recent_errors: str = "",
    ) -> Dict[str, Any]:
        """Choose whether controller should continue planning or push actions.

        Args:
            role_name: Current side role name.
            worker_advice: Consolidated worker analysis text.
            graph_context: Current graph context text.
            action_cache_context: Cached candidate actions context.
            has_validated_plan: Whether a valid action plan already exists.
            plan_attempt: Current planning-attempt index.
            max_plan_attempts: Maximum allowed planning attempts.
            recent_errors: Optional recent validation/runtime errors.

        Returns:
            Dict containing `next_step` (`plan` or `push`) and decision reason.

        Raises:
            ValueError: If model returns unsupported `next_step`.
        """
        prompt = CHOOSE_PLAN_OR_PUSH_PROMPT.format(
            role_name=role_name,
            worker_advice=worker_advice,
            graph_context=graph_context,
            action_cache_context=action_cache_context,
            has_validated_plan=has_validated_plan,
            plan_attempt=plan_attempt,
            max_plan_attempts=max_plan_attempts,
            recent_errors=recent_errors or "（无）",
        )

        result = await self.llm.aask_tool_call(
            prompt=prompt,
            tools=[_CHOOSE_PLAN_OR_PUSH_TOOL],
            tool_choice="choose_plan_or_push",
            system_msgs=[SYSTEM_PROMPT_CONTROLLER_ROUTING],
        )

        payload = result.arguments
        next_step = str(payload.get("next_step", "plan")).strip().lower()
        reason = str(payload.get("reason", "")).strip()

        if next_step not in {"plan", "push"}:
            raise ValueError(
                "`choose_plan_or_push` must return next_step in {plan,push}."
            )

        return {"next_step": next_step, "reason": reason}


class PlanTool(Action):
    """Plan tool: assess needs, synthesize worker advice, and validate JSON actions."""

    name: str = "PlanTool"

    async def assess_needs(
        self, role_name: str, persona: object, graph_context: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Run fact/law/recall requirement assessments concurrently.

        Args:
            role_name: Current side role name.
            persona: Persona object containing belief/desire/intention fields.
            graph_context: Current graph context text.

        Returns:
            Tuple of `(fact_need, law_need, recall_need)` assessment payloads.
        """
        fact_task = AssessFactNeeds(llm=self.llm).run(role_name, persona, graph_context)
        law_task = AssessLawNeeds(llm=self.llm).run(role_name, persona, graph_context)

        recall_task = AssessRecallNeeds(llm=self.llm).run(
            role_name, persona, graph_context
        )

        return await asyncio.gather(fact_task, law_task, recall_task)

    async def build_actions(
        self,
        role_name: str,
        worker_advice: str,
        graph_context: str,
        focus: str,
        id_inventory: str,
        feedback: str = "",
    ) -> List[Dict[str, Any]]:
        """Generate candidate graph actions via strict tool-calling.

        Args:
            role_name: Current side role name.
            worker_advice: Consolidated worker analysis text.
            graph_context: Current graph context text.
            focus: Current strategic focus text.
            id_inventory: Valid node id inventory text.
            feedback: Optional feedback from previous failed attempt.

        Returns:
            Candidate action payloads.
        """
        return await VerifyAndDecide(llm=self.llm).run(
            role_name=role_name,
            worker_advice=worker_advice,
            graph_context=graph_context,
            focus=focus,
            id_inventory=id_inventory,
            feedback=feedback,
        )

    def validate_json_actions(
        self, graph_tool: Any, raw_actions: List[Dict[str, Any]]
    ) -> Tuple[bool, List[AgentAction], List[str]]:
        """Validate raw action payload with schema and graph constraints.

        Args:
            graph_tool: Graph tool exposing `validate_actions`.
            raw_actions: Raw action payload rows from model output.

        Returns:
            Tuple of `(is_valid, parsed_actions, errors)`.
        """
        try:
            parsed_actions = [AgentAction.model_validate(item) for item in raw_actions]

        except (TypeError, ValueError) as exc:
            return False, [], [f"Action Validation Failed: {exc}"]

        is_valid, errors = graph_tool.validate_actions(parsed_actions)

        if not is_valid:
            return False, parsed_actions, list(errors)

        return True, parsed_actions, []

    async def run(
        self,
        role_name: str,
        graph_tool: Any,
        worker_advice: str,
        graph_context: str,
        focus: str,
        id_inventory: str,
        feedback: str = "",
    ) -> Dict[str, Any]:
        """Build and validate candidate actions for controller push stage.

        Args:
            role_name: Current side role name.
            graph_tool: Graph tool used for static validation.
            worker_advice: Consolidated worker analysis text.
            graph_context: Current graph context text.
            focus: Current strategic focus text.
            id_inventory: Valid node id inventory text.
            feedback: Optional feedback from previous failed attempt.

        Returns:
            Validation payload containing raw actions, parsed actions, and errors.
        """
        raw_actions = await self.build_actions(
            role_name=role_name,
            worker_advice=worker_advice,
            graph_context=graph_context,
            focus=focus,
            id_inventory=id_inventory,
            feedback=feedback,
        )

        validated, parsed_actions, errors = self.validate_json_actions(
            graph_tool=graph_tool,
            raw_actions=raw_actions,
        )

        return {
            "validated": validated,
            "raw_actions": raw_actions,
            "parsed_actions": [
                item.model_dump(exclude_none=True) for item in parsed_actions
            ],
            "validation_errors": errors,
            "sandbox_actions": [
                item.model_dump(exclude_none=True) for item in parsed_actions
            ],
        }


class PushTool(Action):
    """Push tool: apply validated actions to the debate graph."""

    name: str = "PushTool"

    async def run(
        self, role_name: str, graph_tool: Any, actions: List[AgentAction]
    ) -> str:
        """Execute validated actions and return graph execution logs.

        Args:
            role_name: Current side role name.
            graph_tool: Graph tool exposing `apply_actions`.
            actions: Validated action objects.

        Returns:
            Execution log text returned by graph tool.
        """
        return await graph_tool.apply_actions(agent_id=role_name, actions=actions)
