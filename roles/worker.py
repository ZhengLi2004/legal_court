import re
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.logs import logger
from actions.worker_actions import AnalyzeSearchResults, SuggestPivot, InjectLawsToGraph
from mas.schema import WorkerInstruction, WorkerReport, WorkerReportStatus
from tools.fact_es_tool import FactEsTool
from mas.llm import GPTChat

class BaseWorker(Role):
    name: str = "Worker"
    profile: str = "Legal Researcher"

    def __init__(self, name: str, profile: str, es_tool: object, llm: GPTChat, threshold: float = 0.6):
        super().__init__(name=name, profile=profile)
        self.es_tool = es_tool
        self.threshold = threshold
        self.llm = llm
        self.set_actions([AnalyzeSearchResults, SuggestPivot])

    async def _perform_search(self, query: str) -> str: raise NotImplementedError
    # 抽取最高相似度
    async def _extract_max_score(self, search_text: str) -> float:
        try:
            scores = re.findall(r"\[相似度: (0\.\d+)\]", search_text)
            if scores: return max([float(s) for s in scores])
            return 0.0
        
        except Exception as e:
            logger.warning(f"Error extracting score: {e}")
            return 0.0

    async def _act(self) -> Message:
        logger.info(f"{self.name} is acting...")

        try:
            memories = self.get_memories(k=1)
            if not memories: return Message(content="ERROR: No instruction received", role=self.profile)
            instruction_json = memories[0].content
            if hasattr(instruction_json, 'content'):  instruction_json = instruction_json.content
            instruction = WorkerInstruction.model_validate_json(instruction_json)

        except Exception as e:
            logger.error(f"Failed to parse instruction: {e}")
            
            report = WorkerReport(
                status=WorkerReportStatus.ERROR, 
                content=f"Instruction parse error: {str(e)}"
            )
            
            return Message(content=report.to_json(), role=self.profile)
        
        try:
            search_text = await self._perform_search(instruction.query)
            max_score = await self._extract_max_score(search_text)
            logger.info(f"Search completed. Max score: {max_score} (Threshold: {self.threshold})")

        except Exception as e:
            logger.error(f"Search failed: {e}")

            report = WorkerReport(
                status=WorkerReportStatus.ERROR, 
                content=f"ES Search failed: {str(e)}"
            )

            return Message(content=report.to_json(), role=self.profile)
        
        try:
            if max_score >= self.threshold:
                action = AnalyzeSearchResults()
                action.llm = self.llm

                advice = await action.run(
                    role_type=self.profile,
                    graph_context=instruction.graph_context,
                    user_query=instruction.query,
                    search_result=search_text
                )

                report = WorkerReport(
                    status=WorkerReportStatus.FOUND,
                    content=advice,
                    max_score=max_score
                )
            
            else:
                action = SuggestPivot()
                action.llm = self.llm

                pivot_advice = await action.run(
                    graph_context=instruction.graph_context,
                    user_query=instruction.query
                )

                report = WorkerReport(
                    status=WorkerReportStatus.NOT_FOUND,
                    content=pivot_advice,
                    max_score=max_score
                )

            return Message(content=report.to_json(), role=self.profile)
        
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            
            report = WorkerReport(
                status=WorkerReportStatus.ERROR, 
                content=f"Reasoning failed: {str(e)}"
            )
            
            return Message(content=report.to_json(), role=self.profile)

class FactWorker(BaseWorker):
    def __init__(self, name: str = "FactWorker", es_tool: FactEsTool = None, llm: GPTChat = None, threshold: float = 0.6): super().__init__(name, "Fact Researcher", es_tool, llm, threshold)
    async def _perform_search(self, query: str) -> str: return await self.es_tool.search_cases(query)

class LawWorker(BaseWorker):
    def __init__(self, name: str = "LawWorker", es_tool=None, llm=None, threshold: float = 0.6):
        super().__init__(name, "Law Researcher", es_tool, llm, threshold)
        self.set_actions([InjectLawsToGraph])
        self.graph_tool = None

    async def _act(self) -> Message:
        logger.info(f"{self.name} is acting (Injection Mode)...")

        try:
            memories = self.get_memories(k=1)
            if not memories: return Message(content="ERROR: No instruction", role=self.profile)
            instruction_json = memories[0].content
            if hasattr(instruction_json, 'content'): instruction_json = instruction_json.content
            instruction = WorkerInstruction.model_validate_json(instruction_json)
        
        except Exception as e: return Message(content=WorkerReport(status=WorkerReportStatus.ERROR, content=str(e)).to_json(), role=self.profile)
        if not self.graph_tool: return Message(content=WorkerReport(status=WorkerReportStatus.ERROR, content="System Error: GraphTool not bound to LawWorker").to_json(), role=self.profile)

        try:
            action = InjectLawsToGraph()

            report_content = await action.run(
                query=instruction.query,
                es_tool=self.es_tool,
                graph_tool=self.graph_tool,
                threshold=self.threshold
            )

            status = WorkerReportStatus.FOUND if "已底层注入" in report_content else WorkerReportStatus.NOT_FOUND

            report = WorkerReport(
                status=status,
                content=report_content,
                max_score=1.0
            )

        except Exception as e:
            logger.error(f"LawWorker injection failed: {e}")
            report = WorkerReport(status=WorkerReportStatus.ERROR, content=f"Injection failed: {e}")

        return Message(content=report.to_json(), role=self.profile)