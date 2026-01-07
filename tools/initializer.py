"""Provides a tool for initializing a legal case before the debate begins.

This module defines the `CaseInitializer` class, which takes raw case data
(facts and cause of action) and uses an LLM to preprocess it into the structured
formats needed to start the debate simulation. This includes decomposing facts,
generating root claims, and creating BDI personas for the agents.
"""

import re
from dataclasses import dataclass
from typing import List

from metagpt.logs import logger

from mas.llm import GPTChat, Message
from prompts.common_prompts import (
    DECOMPOSE_FACTS_PROMPT,
    GENERATE_PERSONA_PROMPT,
    GENERATE_ROOT_CLAIM_PROMPT,
)
from tools.json_utils import extract_json_from_text


@dataclass
class AgentPersona:
    """A structured representation of an agent's BDI (Belief, Desire, Intention) model.

    Attributes:
        role_name: The role of the agent (e.g., "plaintiff").
        belief: A description of the agent's core beliefs about the case.
        desire: What the agent wants to achieve.
        intention: The agent's general approach or style.
        initial_strategy: A high-level starting strategy.
    """

    role_name: str
    belief: str
    desire: str
    intention: str
    initial_strategy: str


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
        p_persona = await self._generate_persona(fact_finding, cause, "plaintiff")
        d_persona = await self._generate_persona(fact_finding, cause, "defendant")

        return InitializationResult(
            plaintiff_persona=p_persona,
            defendant_persona=d_persona,
            fact_statements=fact_statements,
            root_claim_actions=root_claim_texts,
        )

    async def _parse_numbered_list_to_agent_actions(self, text: str) -> List[str]:
        """Parse a numbered list string into a list of strings."""
        facts = []
        lines = text.strip().split("\n")
        current_fact_content = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            match = re.match(r"^\d+\.\s*(.*)", line)

            if match:
                if current_fact_content:
                    facts.append(" ".join(current_fact_content).strip())

                current_fact_content = [match.group(1)]

            elif current_fact_content:
                current_fact_content.append(line)

        if current_fact_content:
            facts.append(" ".join(current_fact_content).strip())

        return facts

    async def _decompose_facts(self, text: str) -> List[str]:
        """Use an LLM to break down a block of text into atomic fact statements."""
        prompt = DECOMPOSE_FACTS_PROMPT.format(text=text)
        response = self.llm([Message(role="user", content=prompt)])

        try:
            return await self._parse_numbered_list_to_agent_actions(response)

        except Exception as e:
            logger.error(
                f"Error parsing decomposed facts from numbered list: {e}\nResponse: {response}"
            )

            return []

    async def _generate_root_claim(self, facts: str, cause: str) -> List[str]:
        """Use an LLM to generate the plaintiff's primary legal claims."""
        prompt = GENERATE_ROOT_CLAIM_PROMPT.format(cause=cause, facts=facts)
        response = self.llm([Message(role="user", content=prompt)])

        try:
            claims_list = extract_json_from_text(response)

            if not isinstance(claims_list, list) or not all(
                isinstance(item, str) for item in claims_list
            ):
                raise ValueError("LLM did not return a JSON array of strings.")

            return claims_list

        except Exception as e:
            logger.error(f"Error parsing root claim texts: {e}\nResponse: {response}")
            return []

    async def _generate_persona(
        self, facts: str, cause: str, role: str
    ) -> AgentPersona:
        """Use an LLM to generate a BDI persona for a given role."""
        role_cn = "原告" if role == "plaintiff" else "被告"

        prompt = GENERATE_PERSONA_PROMPT.format(
            cause=cause, role_cn=role_cn, facts=facts
        )

        response = self.llm([Message(role="user", content=prompt)], temperature=0.7)

        try:
            data = extract_json_from_text(response)

            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                data = data[0]

            return AgentPersona(
                role_name=role,
                belief=data.get("belief", "N/A"),
                desire=data.get("desire", "N/A"),
                intention=data.get("intention", "N/A"),
                initial_strategy=data.get("strategy", "N/A"),
            )

        except Exception as e:
            logger.error(f"Error parsing persona for {role}: {e}\nResponse: {response}")

            return AgentPersona(
                role,
                "Default Belief",
                "Default Desire",
                "Default Intention",
                "Default Strategy",
            )
