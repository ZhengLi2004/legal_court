"""Provides a tool for initializing a legal case before the debate begins.

This module defines the `CaseInitializer` class, which takes raw case data
(facts and cause of action) and uses an LLM to preprocess it into the structured
formats needed to start the debate simulation. This includes decomposing facts,
generating root claims, and creating BDI personas for the agents.
"""

import json
from dataclasses import dataclass
from typing import List

from metagpt.logs import logger

from mas.infrastructure.llm import GPTChat
from prompts.common_prompts import (
    DECOMPOSE_FACTS_PROMPT,
    GENERATE_PERSONA_PROMPT,
    GENERATE_ROOT_CLAIM_PROMPT,
    SYSTEM_PROMPT_CASE_INITIALIZER,
)

_FACT_STATEMENTS_SCHEMA = {
    "name": "fact_statements",
    "strict": True,
    "schema": {
        "type": "array",
        "items": {"type": "string"},
    },
}

_ROOT_CLAIMS_SCHEMA = {
    "name": "root_claim_actions",
    "strict": True,
    "schema": {
        "type": "array",
        "items": {"type": "string"},
    },
}

_PERSONA_SCHEMA = {
    "name": "agent_persona",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "belief": {"type": "string"},
            "desire": {"type": "string"},
            "intention": {"type": "string"},
        },
        "required": ["belief", "desire", "intention"],
        "additionalProperties": False,
    },
}


@dataclass
class AgentPersona:
    """A structured representation of an agent's BDI (Belief, Desire, Intention) model.

    Attributes:
        role_name: The role of the agent (e.g., "plaintiff").
        belief: A description of the agent's core beliefs about the case.
        desire: What the agent wants to achieve.
        intention: The agent's executable approach for current and subsequent turns.
    """

    role_name: str
    belief: str
    desire: str
    intention: str


@dataclass
class InitializationResult:
    """A container for all the structured data produced by the CaseInitializer.

    Attributes:
        plaintiff_persona: The generated persona for the plaintiff agent.
        defendant_persona: The generated persona for the defendant agent.
        fact_statements: A list of decomposed, atomic fact statements.
        root_claim_actions: A list of the plaintiff's core legal claims.
    """

    plaintiff_persona: AgentPersona
    defendant_persona: AgentPersona
    fact_statements: List[str]
    root_claim_actions: List[str]


class CaseInitializer:
    """A tool to preprocess raw case data into a debate-ready format.

    This class orchestrates several LLM calls to perform the key setup tasks
    required before the `DebateEngine` can start a simulation.
    """

    def __init__(self, llm: GPTChat):
        """Initialize the CaseInitializer.

        Args:
            llm: The language model client to use for processing.
        """
        self.llm = llm

    async def initialize(self, fact_finding: str, cause: str) -> InitializationResult:
        """Run the full case initialization pipeline.

        This method asynchronously calls the helper methods to decompose facts,
        generate claims, and create personas for both sides, then bundles the
        results into a single `InitializationResult` object.

        Args:
            fact_finding: The raw text of the "facts found by the court" section.
            cause: The cause of action for the case (e.g., "contract dispute").

        Returns:
            An `InitializationResult` object containing all the processed data.
        """
        fact_statements = await self._decompose_facts(fact_finding)
        root_claim_texts = await self._generate_root_claim(fact_finding, cause)

        p_persona = await self._generate_persona(
            fact_statements=fact_statements,
            root_claim_actions=root_claim_texts,
            cause=cause,
            role="plaintiff",
        )

        d_persona = await self._generate_persona(
            fact_statements=fact_statements,
            root_claim_actions=root_claim_texts,
            cause=cause,
            role="defendant",
        )

        return InitializationResult(
            plaintiff_persona=p_persona,
            defendant_persona=d_persona,
            fact_statements=fact_statements,
            root_claim_actions=root_claim_texts,
        )

    async def _decompose_facts(self, text: str) -> List[str]:
        """Use strict JSON-schema output to derive atomic fact statements."""
        prompt = DECOMPOSE_FACTS_PROMPT.format(text=text)

        try:
            facts = await self.llm.aask_json_schema(
                prompt,
                schema=_FACT_STATEMENTS_SCHEMA,
                system_msgs=[SYSTEM_PROMPT_CASE_INITIALIZER],
            )

            if not isinstance(facts, list) or not all(
                isinstance(item, str) for item in facts
            ):
                raise ValueError("LLM did not return a JSON array of strings.")

            normalized = [item.strip() for item in facts if item and item.strip()]

            if not normalized:
                raise ValueError("LLM returned empty fact list.")

            return normalized

        except Exception as e:
            logger.error(f"Fact JSON-schema decomposition failed: {e}")
            raise ValueError(f"Fact decomposition failed under strict JSON: {e}") from e

    async def _generate_root_claim(self, facts: str, cause: str) -> List[str]:
        """Use strict JSON-schema output to generate plaintiff root claims."""
        prompt = GENERATE_ROOT_CLAIM_PROMPT.format(cause=cause, facts=facts)

        try:
            claims_list = await self.llm.aask_json_schema(
                prompt,
                schema=_ROOT_CLAIMS_SCHEMA,
                system_msgs=[SYSTEM_PROMPT_CASE_INITIALIZER],
            )

            if not isinstance(claims_list, list) or not all(
                isinstance(item, str) for item in claims_list
            ):
                raise ValueError("LLM did not return a JSON array of strings.")

            normalized = [item.strip() for item in claims_list if item and item.strip()]

            if not normalized:
                raise ValueError("LLM returned empty root claim list.")

            return normalized

        except Exception as e:
            logger.error(f"Error parsing root claim texts: {e}")

            raise ValueError(
                f"Root claim generation failed under strict JSON: {e}"
            ) from e

    async def _generate_persona(
        self,
        fact_statements: List[str],
        root_claim_actions: List[str],
        cause: str,
        role: str,
    ) -> AgentPersona:
        """Use strict JSON-schema output to generate one BDI persona."""
        role_cn = "原告" if role == "plaintiff" else "被告"
        fact_text = json.dumps(fact_statements or [], ensure_ascii=False, indent=2)
        claim_text = json.dumps(root_claim_actions or [], ensure_ascii=False, indent=2)

        prompt = GENERATE_PERSONA_PROMPT.format(
            cause=cause,
            role_cn=role_cn,
            fact_statements=fact_text,
            root_claim_actions=claim_text,
        )

        try:
            data = await self.llm.aask_json_schema(
                prompt,
                schema=_PERSONA_SCHEMA,
                system_msgs=[SYSTEM_PROMPT_CASE_INITIALIZER],
                temperature=0.7,
            )

            if not isinstance(data, dict):
                raise ValueError("LLM did not return a JSON object for persona.")

            belief = str(data.get("belief", "")).strip()
            desire = str(data.get("desire", "")).strip()
            intention = str(data.get("intention", "")).strip()

            if not (belief and desire and intention):
                raise ValueError("Persona JSON fields cannot be empty.")

            return AgentPersona(
                role_name=role,
                belief=belief,
                desire=desire,
                intention=intention,
            )

        except Exception as e:
            logger.error(f"Error parsing persona for {role}: {e}")
            raise ValueError(f"Persona generation failed under strict JSON: {e}") from e
