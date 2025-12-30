from typing import Set, Union

from metagpt.logs import logger
from metagpt.schema import Message

from roles.controller import ArgumentController
from roles.worker import FactWorker, LawWorker, RecallWorker
from tools.fact_es_tool import FactEsTool
from tools.graph_tool import GraphTool
from tools.initializer import AgentPersona
from tools.law_es_tool import LawEsTool

from .common import ShadowGraph
from .legal_system import LegalSystem
from .llm import GPTChat


class DebateTeam:
    def __init__(
        self,
        side: str,
        persona: AgentPersona,
        graph_tool: GraphTool,
        fact_es: FactEsTool,
        law_es: LawEsTool,
        llm: GPTChat,
        legal_system: LegalSystem,
        insights: str = "",
        verbose: bool = False,
    ):
        self.side = side
        self.persona = persona
        self.graph_tool = graph_tool
        self.verbose = verbose

        self.controller = ArgumentController(
            name=f"{side}_Controller",
            persona=persona,
            graph_tool=graph_tool,
            insights=insights,
        )

        self.controller.llm = llm

        for action in self.controller.actions:
            action.llm = llm

        self.fact_worker = FactWorker(
            name=f"{side}_FactWorker", es_tool=fact_es, llm=llm
        )

        self.law_worker = LawWorker(name=f"{side}_LawWorker", es_tool=law_es, llm=llm)
        self.law_worker.graph_tool = graph_tool

        self.recall_worker = RecallWorker(
            name=f"{side}_RecallWorker", legal_system=legal_system, llm=llm
        )

        self.recall_worker.graph_tool = graph_tool
        self.max_internal_steps = 15

    def _get_target_worker(
        self, send_to: Union[Set, str]
    ) -> Union[FactWorker, LawWorker, RecallWorker, None]:
        if not send_to:
            return None

        targets = send_to if isinstance(send_to, set) else {send_to}

        for t in targets:
            if "FactWorker" in t:
                return self.fact_worker

            if "LawWorker" in t:
                return self.law_worker

            if "RecallWorker" in t:
                return self.recall_worker

        return None

    async def run_turn(self, graph: ShadowGraph) -> dict:
        logger.info(f"--- Team {self.side} Turn Start ---")
        self.graph_tool.set_current_graph(graph)
        start_msg = Message(content="SYSTEM_START", role="System")
        self.controller.rc.memory.add(start_msg)
        transcript = []
        final_result = None
        step_count = 0

        while step_count < self.max_internal_steps:
            step_count += 1
            logger.debug(f"[{self.side}] Internal Step {step_count}")
            ctrl_msg = await self.controller._act()
            content = str(ctrl_msg.content)

            if self.verbose:
                transcript.append(
                    {
                        "from": self.controller.name,
                        "to": str(ctrl_msg.send_to) if ctrl_msg.send_to else "System",
                        "content": content,
                    }
                )

            if "EXECUTION_FAILURE" in content or "ERROR" in content:
                logger.warning(
                    f"[{self.side}] Controller Action Failed. Providing Feedback..."
                )

                feedback_msg = Message(
                    content=f"SYSTEM_FEEDBACK: 上一步操作失败。错误详情: {content}。请根据图谱现状和错误提示，修正你的动作。",
                    role="System",
                )

                self.controller.rc.memory.add(feedback_msg)
                continue

            if "Action Completed" in content:
                final_result = content
                logger.info(f"[{self.side}] Turn Successfully Completed.")
                break

            target_worker = self._get_target_worker(ctrl_msg.send_to)

            if target_worker:
                logger.info(f"[{self.side}] Routing to {target_worker.name}...")
                target_worker.rc.memory.add(ctrl_msg)
                worker_msg = await target_worker._act()

                if self.verbose:
                    transcript.append(
                        {
                            "from": target_worker.name,
                            "to": self.controller.name,
                            "content": worker_msg.content,
                        }
                    )

                self.controller.rc.memory.add(worker_msg)

            else:
                logger.warning(f"[{self.side}] Unknown Controller Output: {content}")

                feedback_msg = Message(
                    content="SYSTEM_FEEDBACK: 你的输出无法被识别。请检查你是否处于正确的状态。",
                    role="System",
                )

                self.controller.rc.memory.add(feedback_msg)

        if final_result is None:
            final_result = (
                f"Timeout: Internal steps exhausted ({self.max_internal_steps})."
            )

        return {"summary": final_result, "transcript": transcript}
