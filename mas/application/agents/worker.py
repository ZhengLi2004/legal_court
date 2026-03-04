"""Defines the specialized Worker agents for tactical information processing.

This module provides the `BaseWorker` class and its concrete implementations:
- `FactWorker`: Searches for relevant historical cases.
- `LawWorker`: Searches for legal statutes and injects them into the debate graph.
- `RecallWorker`: Uses historical cases to provide strategic advice.

These workers are directed by an `ArgumentController` and are responsible for
executing specific, well-defined tasks.
"""

import asyncio
from typing import Any, Dict, List, Union

from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message

from mas.application.agents.actions.worker_actions import (
    AnalyzeSearchResults,
    FormulateSearchQueries,
    ProjectAndAnalyze,
)
from mas.common.token_utils import truncate_by_tokens
from mas.core.schemas import WorkerInstruction, WorkerReport, WorkerReportStatus
from mas.core.system import LegalSystem
from mas.infrastructure.embedding import deduplicate_and_rerank
from mas.infrastructure.fact_es_tool import FactEsTool
from mas.infrastructure.law_es_tool import LawEsTool
from mas.infrastructure.llm import GPTChat
from prompts.common_prompts import (
    ANALYZE_FACT_PROMPT,
    ANALYZE_LAW_PROMPT,
    DECOMPOSE_LAW_INTENT_PROMPT,
)


def truncate_text(text: str, max_len: int = 768) -> str:
    """Truncate text by token budget with ellipsis suffix.

    Args:
        text: Raw text to truncate.
        max_len: Maximum token count (proxy tokenizer).

    Returns:
        Trimmed string. Empty input returns an empty string.
    """
    return truncate_by_tokens(text, max_tokens=max_len)


class BaseWorker(Role):
    """An abstract base class for all worker roles.

    This class provides common functionality for workers, most importantly the
    `_parse_instruction` method, which safely extracts and validates the
    instruction received from the controller.
    """

    name: str = "Worker"
    profile: str = "Legal Researcher"

    def __init__(
        self,
        name: str,
        profile: str,
        llm: GPTChat = None,
    ):
        """Initialize the BaseWorker.

        Args:
            name: The specific name of the worker role.
            profile: A short description of the worker's function.
            llm: The language model client for the agent.
        """
        super().__init__(name=name, profile=profile)
        self.llm = llm

    def _parse_instruction(self) -> Union[WorkerInstruction, Message]:
        """Parse the latest message in memory as a WorkerInstruction.

        It retrieves the last message, assumes it's a JSON string conforming
        to the `WorkerInstruction` schema, and parses it.

        Returns:
            A `WorkerInstruction` object on success, or an error `Message`
            object on failure.
        """
        try:
            memories = self.get_memories(k=1)

            if not memories:
                return Message(
                    content="ERROR: No instruction received", role=self.profile
                )

            instruction_json = memories[0].content

            if hasattr(instruction_json, "content"):
                instruction_json = instruction_json.content

            return WorkerInstruction.model_validate_json(instruction_json)

        except Exception as e:
            logger.error(f"Failed to parse instruction: {e}")

            report = WorkerReport(
                status=WorkerReportStatus.ERROR,
                content=f"Instruction parse error: {str(e)}",
            )

            return Message(content=report.to_json(), role=self.profile)

    async def _act(self) -> Message:
        """Abstract method for the worker's main action logic."""
        raise NotImplementedError


class FactWorker(BaseWorker):
    """A worker specialized in searching for relevant factual evidence from past cases.

    This worker takes a high-level intent, breaks it into multiple search
    queries, executes them in parallel against an Elasticsearch case database,
    deduplicates and reranks the results, and finally analyzes the top results
    to provide a concise summary.
    """

    def __init__(
        self,
        name: str = "FactWorker",
        es_tool: FactEsTool = None,
        llm: GPTChat = None,
        threshold: float = 0.6,
    ):
        """Initialize the FactWorker.

        Args:
            name: The name of the worker.
            es_tool: The `FactEsTool` for searching the case database.
            llm: The language model client.
            threshold: The minimum similarity score for a result to be considered relevant.
        """
        super().__init__(name, "Fact Researcher", llm)
        self.es_tool = es_tool
        self.threshold = threshold
        self.set_actions([FormulateSearchQueries, AnalyzeSearchResults])

    def _format_hits(self, hits: List[Dict[str, Any]]) -> str:
        """Format a list of Elasticsearch hits into a readable string for an LLM."""
        if not hits:
            return "未找到相关案例。"

        lines = []

        for i, hit in enumerate(hits):
            source = hit.get("_source", {})
            score = hit.get("_score", 1.0) - 1.0  # Adjust cosine score
            title = source.get("case_title", "无标题")
            analysis = source.get("analysis", "").strip().replace("\n", " ")[:150]

            lines.append(
                f"案例 {i + 1} [相似度: {score:.4f}] 《{title}》\n摘要: {analysis}..."
            )

        return "\n\n".join(lines)

    async def _act(self) -> Message:
        """Execute the full workflow for fact-finding.

        The workflow consists of:
        1. Decomposing the intent into search queries.
        2. Searching the database in parallel.
        3. Reranking and filtering results based on the threshold.
        4. Analyzing the best results to generate a report.

        Returns:
            A `Message` containing a `WorkerReport` in JSON format.
        """
        result = self._parse_instruction()

        if isinstance(result, Message):
            return result

        instruction = result

        try:
            formulate_action = FormulateSearchQueries(llm=self.llm)
            queries = await formulate_action.run(instruction.intent)
            search_tasks = [self.es_tool.search_cases_raw(q, top_k=3) for q in queries]
            search_results_list = await asyncio.gather(*search_tasks)

            final_top_hits = deduplicate_and_rerank(
                search_results_list,
                key_extractor=lambda hit, src: src.get("case_title", hit.get("_id")),
                top_k=3,
            )

            max_score = 0.0

            if final_top_hits:
                max_score = final_top_hits[0].get("_score", 1.0) - 1.0

            if not final_top_hits or max_score < self.threshold:
                return Message(
                    content=WorkerReport(
                        status=WorkerReportStatus.NOT_FOUND,
                        content=f"经过多角度检索与重排序，仍未找到符合阈值的历史案例 (Max Score: {max_score:.2f} < {self.threshold})。\n尝试的查询：{queries}",
                        max_score=max_score,
                    ).to_json(),
                    role=self.profile,
                )

            formatted_context = self._format_hits(final_top_hits)
            analyze_action = AnalyzeSearchResults(llm=self.llm)

            advice = await analyze_action.run(
                user_query=instruction.intent,
                search_result=formatted_context,
                prompt_template=ANALYZE_FACT_PROMPT,
            )

            advice = truncate_text(advice)

            report = WorkerReport(
                status=WorkerReportStatus.FOUND, content=advice, max_score=max_score
            )

            return Message(content=report.to_json(), role=self.profile)

        except Exception as e:
            logger.error(f"FactWorker failed: {e}")

            return Message(
                content=WorkerReport(
                    status=WorkerReportStatus.ERROR, content=str(e)
                ).to_json(),
                role=self.profile,
            )


class LawWorker(BaseWorker):
    """A worker specialized in searching for and injecting legal statutes.

    This worker's key distinction is that after finding relevant laws in the
    Elasticsearch database, it uses the `GraphTool` to directly inject these
    laws as `LAW` nodes into the main debate graph for all agents to see.
    """

    def __init__(
        self,
        name: str = "LawWorker",
        es_tool: LawEsTool = None,
        llm: GPTChat = None,
        legal_system: LegalSystem = None,
        threshold: float = 0.6,
    ):
        """Initialize the LawWorker.

        Args:
            name: The name of the worker.
            es_tool: The `LawEsTool` for searching the statute database.
            llm: The language model client.
            legal_system: The `LegalSystem` object.
            threshold: The minimum similarity score for a law to be considered relevant.
        """
        super().__init__(name, "Law Researcher", llm)
        self.es_tool = es_tool
        self.threshold = threshold
        self.graph_tool = None
        self.legal_system = legal_system
        self.set_actions([FormulateSearchQueries, AnalyzeSearchResults])

    async def _act(self) -> Message:
        """Execute the full workflow for legal research and injection.

        The workflow is:
        1. Formulate law-specific search queries.
        2. Search the database.
        3. Filter results by threshold.
        4. **Inject** the found laws as nodes into the debate graph.
        5. Analyze the injected laws to provide a summary report.

        Returns:
            A `Message` containing a `WorkerReport` in JSON format.
        """
        result = self._parse_instruction()

        if isinstance(result, Message):
            return result

        instruction = result

        if not self.graph_tool:
            return Message(
                content=WorkerReport(
                    status=WorkerReportStatus.ERROR,
                    content="System Error: GraphTool missing",
                ).to_json(),
                role=self.profile,
            )

        try:
            formulate_action = FormulateSearchQueries(llm=self.llm)

            queries = await formulate_action.run(
                instruction.intent, prompt_template=DECOMPOSE_LAW_INTENT_PROMPT
            )

            search_tasks = [self.es_tool.search_laws_raw(q, top_k=3) for q in queries]
            search_results_list = await asyncio.gather(*search_tasks)

            final_top_hits = deduplicate_and_rerank(
                search_results_list,
                key_extractor=lambda hit, src: (
                    f"{src.get('law_name')}_{src.get('article_id')}"
                ),
                top_k=3,
            )

            max_score = 0.0

            if final_top_hits:
                max_score = final_top_hits[0].get("_score", 1.0) - 1.0

            if not final_top_hits or max_score < self.threshold:
                return Message(
                    content=WorkerReport(
                        status=WorkerReportStatus.NOT_FOUND,
                        content=f"未检索到符合阈值的法条 (Max Score: {max_score:.2f})",
                        max_score=max_score,
                    ).to_json(),
                    role=self.profile,
                )

            law_contents_for_injection = []
            formatted_context_for_analysis = ""

            for i, hit in enumerate(final_top_hits):
                source = hit.get("_source", {})
                content = f"《{source.get('law_name')}》{source.get('article_id')}: {source.get('content')}"
                law_contents_for_injection.append(content)
                score = hit.get("_score", 1.0) - 1.0

                formatted_context_for_analysis += (
                    f"{i + 1}. [Sim: {score:.4f}] {content[:100]}...\n"
                )

            self.graph_tool.inject_law_nodes(law_contents_for_injection)

            if self.legal_system and self.graph_tool.current_graph:
                self.legal_system.inject_jurisprudential_context(
                    self.graph_tool.current_graph
                )

            analyze_action = AnalyzeSearchResults(llm=self.llm)

            analysis = await analyze_action.run(
                user_query=instruction.intent,
                search_result=formatted_context_for_analysis,
                prompt_template=ANALYZE_LAW_PROMPT,
            )

            final_content = (
                f"已注入 {len(law_contents_for_injection)} 条相关法条。\n{analysis}"
            )

            final_content = truncate_text(final_content)

            report = WorkerReport(
                status=WorkerReportStatus.FOUND,
                content=final_content,
                max_score=max_score,
            )

            return Message(content=report.to_json(), role=self.profile)

        except Exception as e:
            logger.error(f"LawWorker failed: {e}")

            return Message(
                content=WorkerReport(
                    status=WorkerReportStatus.ERROR, content=str(e)
                ).to_json(),
                role=self.profile,
            )


class RecallWorker(BaseWorker):
    """A worker specialized in recalling and analyzing strategies from past cases.

    This worker uses the `ProjectAndAnalyze` action to find analogous argument
    structures in historical cases and generates strategic advice based on them.
    """

    def __init__(
        self,
        name: str = "RecallWorker",
        legal_system: LegalSystem = None,
        llm: GPTChat = None,
        role_name: str = "Unknown",
    ):
        """Initialize the RecallWorker.

        Args:
            name: The name of the worker.
            legal_system: The `LegalSystem` object, which provides access to
                the projector and historical cases.
            llm: The language model client.
            role_name: The side the worker is on ("plaintiff" or "defendant").
        """
        super().__init__(name, "Strategy Researcher", llm)
        self.legal_system = legal_system
        self.graph_tool = None
        self.role_name = role_name
        self.set_actions([ProjectAndAnalyze])

    async def _act(self) -> Message:
        """Execute the projection and analysis workflow.

        It directly calls the `ProjectAndAnalyze` action, which encapsulates
        the complex logic of finding anchors, retrieving historical context,
        and generating strategic advice.

        Returns:
            A `Message` containing a `WorkerReport` in JSON format.
        """
        result = self._parse_instruction()

        if isinstance(result, Message):
            return result

        instruction = result

        if not self.legal_system or not self.graph_tool:
            return Message(
                content=WorkerReport(
                    status=WorkerReportStatus.ERROR,
                    content="System Error: Dependencies missing",
                ).to_json(),
                role=self.profile,
            )

        try:
            action = ProjectAndAnalyze(llm=self.llm)

            advice = await action.run(
                query=instruction.intent,
                legal_system=self.legal_system,
                current_graph=self.graph_tool.current_graph,
                my_role=self.role_name,
            )

            final_content = truncate_text(advice)

            report = WorkerReport(
                status=WorkerReportStatus.FOUND, content=final_content, max_score=1.0
            )

            return Message(content=report.to_json(), role=self.profile)

        except Exception as e:
            logger.error(f"RecallWorker failed: {e}")

            return Message(
                content=WorkerReport(
                    status=WorkerReportStatus.ERROR, content=str(e)
                ).to_json(),
                role=self.profile,
            )
