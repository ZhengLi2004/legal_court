"""Defines the structure and orchestration of an agent team.

This module provides the `DebateTeam` class, which encapsulates one side of the
legal debate (either plaintiff or defendant). A team consists of a
`ArgumentController` agent and several specialized `Worker` agents.
"""

import asyncio
from typing import Callable, Optional, Union

from metagpt.logs import logger
from metagpt.schema import Message

from roles.controller import ArgumentController, ControllerPipelineStep
from roles.worker import FactWorker, LawWorker, RecallWorker
from tools.fact_es_tool import FactEsTool
from tools.graph_tool import GraphTool
from tools.initializer import AgentPersona
from tools.json_utils import extract_json_from_text
from tools.law_es_tool import LawEsTool
from tools.llm import GPTChat

from ..core.graph import ShadowGraph
from ..core.system import LegalSystem


class DebateTeam:
    """Orchestrates the agents of one side (plaintiff or defendant) for a debate turn.

    Attributes:
        side: The side the team represents, e.g., "plaintiff".
        controller: The `ArgumentController` agent for the team.
        fact_worker: The `FactWorker` agent.
        law_worker: The `LawWorker` agent.
        recall_worker: The `RecallWorker` agent.
        on_state_change: Optional callback for state changes.
    """

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
        """Initialize the DebateTeam and its constituent agents.

        Args:
            side: The side of the debate ("plaintiff" or "defendant").
            persona: The `AgentPersona` defining the team's goals and strategy.
            graph_tool: The tool for interacting with the debate graph.
            fact_es: The tool for searching fact databases.
            law_es: The tool for searching law databases.
            llm: The language model client for the agents.
            legal_system: The main `LegalSystem` object.
            insights: A string of initial strategic insights for the controller.
            verbose: A flag to enable detailed transcript logging.
        """
        self.side = side
        self.persona = persona
        self.graph_tool = graph_tool
        self.verbose = verbose
        self.on_state_change: Optional[Callable[[str, str, dict], None]] = None

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
            name=f"{side}_RecallWorker",
            legal_system=legal_system,
            llm=llm,
            role_name=side,
        )

        self.recall_worker.graph_tool = graph_tool
        self.max_internal_steps = 15

    @property
    def pipeline_step(self) -> ControllerPipelineStep:
        """Get the current pipeline step from the controller."""
        if self.controller and hasattr(self.controller, "pipeline_step"):
            return self.controller.pipeline_step

        return ControllerPipelineStep.IDLE

    def _notify_state_change(self, event: str, data: dict = None):
        """Notify listeners about state changes.

        Args:
            event: The event type (e.g., "step_start", "worker_dispatch").
            data: Additional data about the event.
        """
        if self.on_state_change:
            try:
                self.on_state_change(self.side, event, data or {})

            except Exception as e:
                logger.warning(f"State change callback failed: {e}")

    def _get_worker_by_name(
        self, target_name: str
    ) -> Union[FactWorker, LawWorker, RecallWorker, None]:
        """Retrieve a worker instance by its name string.

        Args:
            target_name: The name of the target worker.

        Returns:
            The worker instance or None if not found.
        """
        if "FactWorker" in target_name:
            return self.fact_worker

        if "LawWorker" in target_name:
            return self.law_worker

        if "RecallWorker" in target_name:
            return self.recall_worker

        return None

    async def run_turn(self, graph: ShadowGraph) -> dict:
        """Execute the workflow for a single debate turn.

        Args:
            graph: The current `ShadowGraph` of the debate.

        Returns:
            A dictionary containing a summary of the turn's outcome, a detailed
            transcript of internal messages (if verbose), and the list of
            executed `AgentAction` objects.
        """
        logger.info(f"--- Team {self.side} Turn Start ---")
        self.graph_tool.set_current_graph(graph)
        self.controller.reset_turn_state()
        self.controller.pipeline_step = ControllerPipelineStep.ASSESS_NEEDS
        self._notify_state_change("turn_start", {"step": "ASSESS_NEEDS"})
        transcript = []
        final_result = None
        step_count = 0

        while step_count < self.max_internal_steps:
            step_count += 1
            logger.debug(f"[{self.side}] Internal Step {step_count}")

            self._notify_state_change(
                "internal_step",
                {
                    "step_count": step_count,
                    "pipeline_step": self.controller.pipeline_step.name,
                },
            )

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
                self.controller.pipeline_step = ControllerPipelineStep.DONE
                self._notify_state_change("turn_complete", {"result": "success"})
                logger.info(f"[{self.side}] Turn Successfully Completed.")
                break

            if "EXECUTION_FAILURE_RETRY" in content:
                logger.warning(f"[{self.side}] Action failed. Controller will retry.")
                self._notify_state_change("retry", {"reason": "execution_failure"})
                continue

            if "batch_instructions" in content:
                try:
                    data = extract_json_from_text(content)
                    instructions_list = data.get("batch_instructions", [])

                    if not instructions_list:
                        logger.info(
                            f"[{self.side}] No workers needed. Triggering ingest."
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

            self.controller.pipeline_step = ControllerPipelineStep.DONE
            self._notify_state_change("turn_complete", {"result": "timeout"})

        return {
            "summary": final_result,
            "transcript": transcript,
            "actions": self.controller.last_executed_actions,
        }
