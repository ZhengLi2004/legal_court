"""Defines tactical actions for the Worker roles (Fact, Law, Recall).

This module contains Action classes derived from metagpt.actions.Action. These
actions are designed to be executed by worker agents and are focused on
information retrieval, processing, and analysis tasks. They form the building
blocks for the research and analysis phase of a debate turn, such as formulating
search queries, analyzing search results, and projecting the current debate
state onto historical cases.
"""

from typing import List, Optional

from metagpt.actions import Action

from mas.core.graph import ShadowGraph
from mas.core.system import LegalSystem
from mas.infrastructure.embedding import cosine_similarity
from prompts.common_prompts import (
    ANALYZE_RECALL_PROMPT,
    DECOMPOSE_SEARCH_INTENT_PROMPT,
    SYSTEM_PROMPT_QUERY_DECOMPOSER,
    SYSTEM_PROMPT_WORKER_ANALYST,
)

_SEARCH_QUERY_ARGS_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 3,
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}

_FORMULATE_SEARCH_QUERIES_TOOL = {
    "type": "function",
    "function": {
        "name": "formulate_search_queries",
        "description": "Decompose one legal intent into 2-3 natural-language queries.",
        "parameters": _SEARCH_QUERY_ARGS_SCHEMA,
    },
}


class AnalyzeSearchResults(Action):
    """An action to analyze raw search results and provide a summary.

    This action takes a user's query and the raw text from a search tool,
    then uses a language model with a specific prompt template to generate a
    concise, actionable analysis.
    """

    name: str = "AnalyzeSearchResults"

    async def run(
        self, user_query: str, search_result: str, prompt_template: str
    ) -> str:
        """Format a prompt and queries the LLM to analyze search results.

        Args:
            user_query: The original query or intent from the user.
            search_result: The raw string of search results to be analyzed.
            prompt_template: The string template for the prompt, which should
                contain placeholders for `user_query` and `search_result`.

        Returns:
            A string containing the language model's analysis of the search results.
        """
        prompt = prompt_template.format(
            user_query=user_query,
            search_result=search_result,
        )

        return await self.llm.aask(
            prompt,
            system_msgs=[SYSTEM_PROMPT_WORKER_ANALYST],
            temperature=0.4,
        )


class FormulateSearchQueries(Action):
    """An action to decompose a high-level intent into specific search queries.

    This action takes a natural language statement of intent and uses a
    language model to break it down into multiple, concrete query strings
    suitable for a search engine.
    """

    name: str = "FormulateSearchQueries"

    async def run(
        self, intent: str, prompt_template: str = DECOMPOSE_SEARCH_INTENT_PROMPT
    ) -> List[str]:
        """Generate search queries from high-level legal intent.

        The method enforces strict function-calling output via
        `formulate_search_queries`.

        Args:
            intent: The high-level search intent string.
            prompt_template: The string template for the prompt, which should
                contain a placeholder for `intent`.

        Returns:
            A list of formulated search query strings.

        Raises:
            ValueError: If tool-call payload is missing or invalid.
        """
        prompt = prompt_template.format(intent=intent)

        result = await self.llm.aask_tool_call(
            prompt=prompt,
            tools=[_FORMULATE_SEARCH_QUERIES_TOOL],
            tool_choice="formulate_search_queries",
            system_msgs=[SYSTEM_PROMPT_QUERY_DECOMPOSER],
            temperature=0.4,
        )

        payload = result.arguments
        queries = payload.get("queries", [])

        if not (isinstance(queries, list) and all(isinstance(q, str) for q in queries)):
            raise ValueError(
                "FormulateSearchQueries expected payload.queries as list[str]."
            )

        normalized = [q.strip() for q in queries if q and q.strip()]

        if not normalized:
            raise ValueError("FormulateSearchQueries received empty queries list.")

        return normalized


class ProjectAndAnalyze(Action):
    """An action to find and analyze analogous arguments from historical cases.

    This action implements a "projection" mechanism. It first identifies key
    nodes in the current debate graph that are semantically similar to the
    given query. These nodes act as anchors. It then searches through a cached
    set of historical cases to find similar argument structures connected to
    those anchors. Finally, it uses an LLM to analyze the retrieved historical
    context and provide strategic advice.
    """

    @staticmethod
    def _coerce_positive_int(value: object) -> Optional[int]:
        """Parse a positive integer from a raw value."""
        try:
            parsed = int(value)

        except (TypeError, ValueError):
            return None

        return parsed if parsed > 0 else None

    @classmethod
    def _resolve_projection_top_k(
        cls,
        legal_system: LegalSystem,
        primary_key: str,
        default: int,
    ) -> int:
        """Resolve projection top-k from config.

        The paper describes projection over a bounded set of anchors and
        candidate history cases for engineering feasibility. This helper ensures
        those bounds are actually enforced from runtime config.
        """
        retrieval_cfg = getattr(getattr(legal_system, "cfg", None), "retrieval", None)

        if retrieval_cfg is not None:
            parsed = cls._coerce_positive_int(getattr(retrieval_cfg, primary_key, None))

            if parsed is not None:
                return parsed

        return max(1, int(default))

    name: str = "ProjectAndAnalyze"

    async def run(
        self,
        query: str,
        legal_system: LegalSystem,
        current_graph: ShadowGraph,
        my_role: str = "Unknown",
        top_k: Optional[int] = None,
    ) -> str:
        """Execute the projection and analysis workflow.

        Args:
            query: The strategic intent or question to investigate.
            legal_system: The main LegalSystem object providing access to
                resources like embedding functions and historical cases.
            current_graph: The current state of the debate as a ShadowGraph.
            my_role: The role of the agent executing the action (e.g., "plaintiff"),
                used to tailor the analysis.
            top_k: Optional explicit override for anchor top-k. When omitted,
                `cfg.retrieval.projection_anchor_top_k` is used.

        Returns:
            A string containing strategic advice derived from analyzing
            analogous historical arguments, or an explanatory message if
            no relevant history is found.
        """
        current_step = legal_system.step_counter
        focus_node_ids = current_graph._calculate_focus_nodes(current_step)
        tactical_subgraph = current_graph.get_subgraph(focus_node_ids)

        if tactical_subgraph.graph.number_of_nodes() == 0:
            return "无法执行历史映射：当前战术视图为空，没有可用的锚点。"

        query_emb = legal_system.ef.embed_query(query)

        configured_anchor_top_k = self._resolve_projection_top_k(
            legal_system=legal_system,
            primary_key="projection_anchor_top_k",
            default=3,
        )

        runtime_anchor_top_k = self._coerce_positive_int(top_k)
        anchor_top_k = runtime_anchor_top_k or configured_anchor_top_k
        candidates = []

        for nid, data in tactical_subgraph.graph.nodes(data=True):
            content = data.get("content", "")

            if not content:
                continue

            node_emb = legal_system.ef.embed_query(content)
            sim = cosine_similarity(query_emb, node_emb)
            candidates.append((sim, nid))

        candidates.sort(key=lambda x: x[0], reverse=True)
        anchor_node_ids = [nid for _, nid in candidates[:anchor_top_k]]

        if not anchor_node_ids:
            return f"未能根据意图 '{query}' 在当前战术视图中找到足够相关的节点作为映射锚点。"

        history_messages = legal_system.active_history_cases

        if not history_messages:
            return "系统初始化时未检索到相关历史案例，暂无经验可供借鉴。"

        case_top_k = self._resolve_projection_top_k(
            legal_system=legal_system,
            primary_key="projection_case_top_k",
            default=3,
        )

        scored_history = []

        for msg in history_messages:
            case_context = str(getattr(msg, "case_context", "") or "")

            if not case_context:
                continue

            case_emb = legal_system.ef.embed_query(case_context)
            sim = cosine_similarity(query_emb, case_emb)
            scored_history.append((sim, msg))

        scored_history.sort(key=lambda x: x[0], reverse=True)
        selected_history_messages = [msg for _, msg in scored_history[:case_top_k]]

        if not selected_history_messages:
            return "候选历史案例为空，无法执行投影映射。"

        historical_context_text = legal_system.projector.retrieve_historical_context(
            current_graph=current_graph,
            focus_node_ids=anchor_node_ids,
            history_messages=selected_history_messages,
        )

        if "未找到" in historical_context_text or not historical_context_text.strip():
            return "在预加载的历史案例中，未找到与当前锚点结构相似的论点路径。"

        prompt = ANALYZE_RECALL_PROMPT.format(
            user_query=query,
            projection_context=historical_context_text,
            my_role=my_role,
        )

        return await self.llm.aask(
            prompt,
            system_msgs=[SYSTEM_PROMPT_WORKER_ANALYST],
            temperature=0.4,
        )
