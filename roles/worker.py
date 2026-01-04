import asyncio
from typing import Any, Dict, List, Union

from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message

from actions.worker_actions import (
    AnalyzeSearchResults,
    FormulateSearchQueries,
    ProjectAndAnalyze,
)
from mas.legal_system import LegalSystem
from mas.llm import GPTChat
from mas.schema import WorkerInstruction, WorkerReport, WorkerReportStatus
from mas.utils import deduplicate_and_rerank
from prompts.common_prompts import (
    ANALYZE_FACT_PROMPT,
    ANALYZE_LAW_PROMPT,
    DECOMPOSE_LAW_INTENT_PROMPT,
)
from tools.fact_es_tool import FactEsTool
from tools.law_es_tool import LawEsTool


def truncate_text(text: str, max_len: int = 500) -> str:
    if not text:
        return ""

    text = text.strip()

    if len(text) > max_len:
        return text[: max_len - 3] + "..."

    return text


class BaseWorker(Role):
    name: str = "Worker"
    profile: str = "Legal Researcher"

    def __init__(
        self,
        name: str,
        profile: str,
        llm: GPTChat = None,
    ):
        super().__init__(name=name, profile=profile)
        self.llm = llm

    def _parse_instruction(self) -> Union[WorkerInstruction, Message]:
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
        raise NotImplementedError


class FactWorker(BaseWorker):
    def __init__(
        self,
        name: str = "FactWorker",
        es_tool: FactEsTool = None,
        llm: GPTChat = None,
        threshold: float = 0.6,
    ):
        super().__init__(name, "Fact Researcher", llm)
        self.es_tool = es_tool
        self.threshold = threshold
        self.set_actions([FormulateSearchQueries, AnalyzeSearchResults])

    def _format_hits(self, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return "未找到相关案例。"

        lines = []

        for i, hit in enumerate(hits):
            source = hit.get("_source", {})
            score = hit.get("_score", 1.0) - 1.0
            title = source.get("case_title", "无标题")
            analysis = source.get("analysis", "").strip().replace("\n", " ")[:150]

            lines.append(
                f"案例 {i + 1} [相似度: {score:.4f}] 《{title}》\n摘要: {analysis}..."
            )

        return "\n\n".join(lines)

    async def _act(self) -> Message:
        logger.info(
            f"{self.name} is acting (Plan -> Parallel Search -> Rerank -> Analyze)..."
        )

        result = self._parse_instruction()

        if isinstance(result, Message):
            return result

        instruction = result

        try:
            formulate_action = FormulateSearchQueries(llm=self.llm)
            queries = await formulate_action.run(instruction.intent)
            logger.info(f"[{self.name}] Formulated {len(queries)} queries: {queries}")
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
            logger.info(f"[{self.name}] Reranked Top-3 Results Selected.")
            analyze_action = AnalyzeSearchResults(llm=self.llm)

            advice = await analyze_action.run(
                user_query=instruction.intent,
                search_result=formatted_context,
                prompt_template=ANALYZE_FACT_PROMPT,
            )

            advice = truncate_text(advice, 600)

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
    def __init__(
        self,
        name: str = "LawWorker",
        es_tool: LawEsTool = None,
        llm: GPTChat = None,
        legal_system: LegalSystem = None,
        threshold: float = 0.6,
    ):
        super().__init__(name, "Law Researcher", llm)
        self.es_tool = es_tool
        self.threshold = threshold
        self.graph_tool = None
        self.legal_system = legal_system
        self.set_actions([FormulateSearchQueries, AnalyzeSearchResults])

    async def _act(self) -> Message:
        logger.info(
            f"{self.name} is acting (Plan -> Parallel Search -> Rerank -> Inject -> Analyze)..."
        )

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

            logger.info(f"[{self.name}] Formulated {len(queries)} queries: {queries}")
            search_tasks = [self.es_tool.search_laws_raw(q, top_k=3) for q in queries]
            search_results_list = await asyncio.gather(*search_tasks)

            final_top_hits = deduplicate_and_rerank(
                search_results_list,
                key_extractor=lambda hit,
                src: f"{src.get('law_name')}_{src.get('article_id')}",
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

            injection_log = self.graph_tool.inject_law_nodes(law_contents_for_injection)

            if self.legal_system and law_contents_for_injection:
                self.legal_system.inject_jurisprudential_context(
                    law_contents_for_injection
                )

            logger.info(f"[{self.name}] Injection Log: {injection_log}")
            analyze_action = AnalyzeSearchResults(llm=self.llm)

            analysis = await analyze_action.run(
                user_query=instruction.intent,
                search_result=formatted_context_for_analysis,
                prompt_template=ANALYZE_LAW_PROMPT,
            )

            final_content = (
                f"已注入 {len(law_contents_for_injection)} 条相关法条。\n{analysis}"
            )

            final_content = truncate_text(final_content, 500)

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
    def __init__(
        self,
        name: str = "RecallWorker",
        legal_system: LegalSystem = None,
        llm: GPTChat = None,
    ):
        super().__init__(name, "Strategy Researcher", llm)
        self.legal_system = legal_system
        self.graph_tool = None
        self.set_actions([ProjectAndAnalyze])

    async def _act(self) -> Message:
        logger.info(f"{self.name} is acting (Project -> Strategize)...")
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
            )

            final_content = truncate_text(advice, 500)

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
