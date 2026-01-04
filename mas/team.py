import asyncio
from typing import Union

from metagpt.logs import logger
from metagpt.schema import Message

from roles.controller import ArgumentController, ControllerPipelineStep
from roles.worker import FactWorker, LawWorker, RecallWorker
from tools.fact_es_tool import FactEsTool
from tools.graph_tool import GraphTool
from tools.initializer import AgentPersona
from tools.json_utils import extract_json_from_text
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

        self.law_worker = LawWorker(
            name=f"{side}_LawWorker", es_tool=law_es, llm=llm, legal_system=legal_system
        )

        self.law_worker.graph_tool = graph_tool

        self.recall_worker = RecallWorker(
            name=f"{side}_RecallWorker", legal_system=legal_system, llm=llm
        )

        self.recall_worker.graph_tool = graph_tool
        self.max_internal_steps = 15

    def _get_worker_by_name(
        self, target_name: str
    ) -> Union[FactWorker, LawWorker, RecallWorker, None]:
        if "FactWorker" in target_name:
            return self.fact_worker

        if "LawWorker" in target_name:
            return self.law_worker

        if "RecallWorker" in target_name:
            return self.recall_worker

        return None

    async def run_turn(self, graph: ShadowGraph) -> dict:
        logger.info(f"--- Team {self.side} Turn Start ---")
        self.graph_tool.set_current_graph(graph)
        self.controller.reset_turn_state()
        self.controller.pipeline_step = ControllerPipelineStep.ASSESS_NEEDS
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
                        "to": "Team/Workers",
                        "content": content,
                    }
                )

            if "Action Completed" in content:
                final_result = content
                logger.info(f"[{self.side}] Turn Successfully Completed.")
                break

            if "EXECUTION_FAILURE_RETRY" in content:
                logger.warning(
                    f"[{self.side}] Action failed. Controller will retry with internal error history."
                )

                continue

            if "batch_instructions" in content:
                try:
                    data = extract_json_from_text(content)
                    instructions_list = data.get("batch_instructions", [])

                    if not instructions_list:
                        logger.info(
                            f"[{self.side}] No workers needed. Triggering ingest with empty list."
                        )

                        self.controller.ingest_results([])
                        continue

                    tasks = []
                    worker_names = []

                    logger.info(
                        f"[{self.side}] Dispatching {len(instructions_list)} parallel tasks..."
                    )

                    for item in instructions_list:
                        target_name = item.get("target")
                        inst_json = item.get("instruction")
                        worker = self._get_worker_by_name(target_name)

                        if worker:
                            worker.rc.memory.add(
                                Message(content=inst_json, role=self.controller.profile)
                            )

                            tasks.append(worker._act())
                            worker_names.append(worker.name)

                        else:
                            logger.warning(
                                f"[{self.side}] Unknown target worker: {target_name}"
                            )

                    if tasks:
                        results = await asyncio.gather(*tasks)
                        results_payload = []

                        for i, res_msg in enumerate(results):
                            w_name = worker_names[i]

                            if self.verbose:
                                transcript.append(
                                    {
                                        "from": w_name,
                                        "to": self.controller.name,
                                        "content": res_msg.content,
                                    }
                                )

                            results_payload.append(
                                {"worker": w_name, "content": str(res_msg.content)}
                            )

                        logger.info(
                            f"[{self.side}] Workers finished. Calling ingest_results()."
                        )

                        self.controller.ingest_results(results_payload)

                    continue

                except Exception as e:
                    logger.error(
                        f"[{self.side}] Failed to process batch instructions: {e}"
                    )

                    feedback_msg = Message(
                        content=f"SYSTEM_FEEDBACK: 指令分发系统发生严重错误: {str(e)}",
                        role="System",
                    )

                    self.controller.rc.memory.add(feedback_msg)
                    continue

            feedback_msg = Message(
                content="SYSTEM_FEEDBACK: 你的输出无法被识别。请确保输出 batch_instructions JSON。",
                role="System",
            )

            self.controller.rc.memory.add(feedback_msg)

        if final_result is None:
            final_result = (
                f"Timeout: Internal steps exhausted ({self.max_internal_steps})."
            )

        return {
            "summary": final_result,
            "transcript": transcript,
            "actions": self.controller.last_executed_actions,
        }
