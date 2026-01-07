"""Provides a component for generating human-readable debate transcripts.

This module defines the `GraphNarrator` class, which is responsible for
translating the structured, logical `AgentAction` objects into fluid,
natural language prose. This creates a more understandable and engaging
record of the debate.
"""

from typing import List

from metagpt.logs import logger

from mas.common import ShadowGraph
from mas.llm import GPTChat
from mas.schema import AgentAction, AgentActionType
from prompts.common_prompts import NARRATOR_POLISH_PROMPT


class GraphNarrator:
    """Translates agent actions into a polished, human-readable narrative.

    This class takes a list of `AgentAction` objects from a single turn,
    converts each action into a somewhat mechanical sentence describing the
    logical operation, and then uses an LLM to "polish" these sentences into a
    coherent, well-written paragraph representing that turn's argument.
    """

    def __init__(self, llm: GPTChat):
        """Initialize the GraphNarrator.

        Args:
            llm: The `GPTChat` instance used for polishing the text.
        """
        self.llm = llm

    def _get_node_text(self, graph: ShadowGraph, node_id: str) -> str:
        """Safely retrieves the content of a node from the graph by its ID."""
        if not node_id:
            return "（未知节点）"

        if not graph.graph.has_node(node_id):
            return f"（缺失节点: {node_id}）"

        data = graph.graph.nodes[node_id]
        content = data.get("content", "")
        return content

    def _action_to_sentence(
        self, action: AgentAction, graph: ShadowGraph, turn: str
    ) -> str:
        """Convert a single AgentAction into a descriptive sentence.

        This is a rule-based conversion that creates a structured but potentially
        stilted sentence describing the action. For example, "Plaintiff cited
        fact [FACT_123] to support claim [CLAIM_456]."

        Args:
            action: The `AgentAction` to convert.
            graph: The current `ShadowGraph` to look up node content.
            turn: The current turn ("plaintiff" or "defendant").

        Returns:
            A string sentence describing the action.
        """
        role = "原告" if turn == "plaintiff" else "被告"
        sentence = ""
        reasoning = action.content.strip() if action.content else ""

        if action.action_type in [AgentActionType.CITE_FACT, AgentActionType.CITE_LAW]:
            source_text = self._get_node_text(graph, action.source_id)

            type_cn = (
                "事实" if action.action_type == AgentActionType.CITE_FACT else "法条"
            )

            if action.target_id:
                target_text = self._get_node_text(graph, action.target_id)
                base = f"{role}引用了{type_cn}【{source_text}】，旨在支持观点：【{target_text}】"

                if reasoning:
                    sentence = f"{base}，理由是：{reasoning}。"

                else:
                    sentence = f"{base}。"

            else:
                sentence = f"{role}引用了{type_cn}【{source_text}】，据此提出观点：【{action.content}】。"

        elif action.action_type in [
            AgentActionType.SUPPORT_CLAIM,
            AgentActionType.REBUT_CLAIM,
        ]:
            relation = (
                "支持"
                if action.action_type == AgentActionType.SUPPORT_CLAIM
                else "反驳"
            )

            target_text = self._get_node_text(graph, action.target_id)

            if action.source_id:
                source_text = self._get_node_text(graph, action.source_id)
                base = f"{role}重申观点【{source_text}】，用于{relation}观点：【{target_text}】"

                if reasoning:
                    sentence = f"{base}，理由是：{reasoning}。"

                else:
                    sentence = f"{base}。"

            else:
                sentence = f"{role}提出新观点：【{action.content}】，用于{relation}观点：【{target_text}】。"

        return sentence

    async def generate_narrative(
        self, actions: List[AgentAction], graph: ShadowGraph, turn: str
    ) -> str:
        """Generate a polished, narrative paragraph for a turn's actions.

        It first converts all actions to raw sentences, then feeds the collection
        of raw sentences to an LLM with a prompt asking it to rewrite them into
        a smooth and persuasive legal statement.

        Args:
            actions: The list of `AgentAction`s performed in the turn.
            graph: The current `ShadowGraph`.
            turn: The current turn ("plaintiff" or "defendant").

        Returns:
            A string containing the polished narrative for the turn.
        """
        if not actions:
            return ""

        raw_sentences = []

        for act in actions:
            try:
                sent = self._action_to_sentence(act, graph, turn)

                if sent:
                    raw_sentences.append(f"- {sent}")

            except Exception as e:
                logger.warning(f"Narrator failed to parse action: {e}")

        if not raw_sentences:
            return f"（{turn} 方本轮未产生有效逻辑陈述）"

        raw_text = "\n".join(raw_sentences)
        logger.debug(f"[Narrator] Raw Logic:\n{raw_text}")

        prompt = NARRATOR_POLISH_PROMPT.format(
            turn="原告" if turn == "plaintiff" else "被告", raw_sentences=raw_text
        )

        narrative = await self.llm.aask(prompt)
        return narrative
