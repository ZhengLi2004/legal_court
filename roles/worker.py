import re
from typing import Union

from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message

from actions.worker_actions import (
    AnalyzeSearchResults,
    InjectLawsToGraph,
    ProjectAndAnalyze,
)
from mas.legal_system import LegalSystem
from mas.llm import GPTChat
from mas.schema import WorkerInstruction, WorkerReport, WorkerReportStatus
from prompts.common_prompts import (
    ANALYZE_FACT_PROMPT,
    ANALYZE_LAW_PROMPT,
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
        self.set_actions([AnalyzeSearchResults])

    async def _act(self) -> Message:
        logger.info(f"{self.name} is acting (Evidence Analysis)...")
        result = self._parse_instruction()

        if isinstance(result, Message):
            return result

        instruction = result

        try:
            search_text = await self.es_tool.search_cases(instruction.query)
            max_score = 0.0
            scores = re.findall(r"\[相似度: (0\.\d+)\]", search_text)

            if scores:
                max_score = max([float(s) for s in scores])

            if max_score < self.threshold:
                return Message(
                    content=WorkerReport(
                        status=WorkerReportStatus.NOT_FOUND,
                        content=f"未检索到相关历史案例 (Score: {max_score:.2f} < {self.threshold})",
                        max_score=max_score,
                    ).to_json(),
                    role=self.profile,
                )

            action = AnalyzeSearchResults(llm=self.llm)

            advice = await action.run(
                user_query=instruction.query,
                search_result=search_text,
                prompt_template=ANALYZE_FACT_PROMPT,
            )

            advice = truncate_text(advice, 500)

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
        threshold: float = 0.6,
    ):
        super().__init__(name, "Law Researcher", llm)
        self.es_tool = es_tool
        self.threshold = threshold
        self.graph_tool = None
        self.set_actions([InjectLawsToGraph, AnalyzeSearchResults])

    async def _act(self) -> Message:
        logger.info(f"{self.name} is acting (Inject -> Analyze)...")
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
            inject_action = InjectLawsToGraph()

            injected_details = await inject_action.run(
                query=instruction.query,
                es_tool=self.es_tool,
                graph_tool=self.graph_tool,
                threshold=self.threshold,
            )

            if (
                "未检索到" in injected_details
                or "低于阈值" in injected_details
                or "检索失败" in injected_details
            ):
                return Message(
                    content=WorkerReport(
                        status=WorkerReportStatus.NOT_FOUND, content=injected_details
                    ).to_json(),
                    role=self.profile,
                )

            analyze_action = AnalyzeSearchResults(llm=self.llm)

            analysis = await analyze_action.run(
                user_query=instruction.query,
                search_result=injected_details,
                prompt_template=ANALYZE_LAW_PROMPT,
            )

            final_content = f"已注入相关法条。\n{analysis}"
            final_content = truncate_text(final_content, 500)

            report = WorkerReport(
                status=WorkerReportStatus.FOUND,
                content=final_content,
                max_score=1.0,
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
                query=instruction.query,
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
