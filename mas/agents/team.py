"""Defines the structure and orchestration of an agent team.

This module provides the `DebateTeam` class, which encapsulates one side of the
legal debate (either plaintiff or defendant). A team consists of a
`ArgumentController` agent and several specialized `Worker` agents.
"""

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional, Union

from metagpt.logs import logger
from metagpt.schema import Message

from mas.application.agents.controller import ArgumentController
from mas.application.agents.worker import FactWorker, LawWorker, RecallWorker
from mas.core.controller_pipeline import ControllerPipelineStep
from mas.infrastructure.fact_es_tool import FactEsTool
from mas.infrastructure.graph_tool import GraphTool
from mas.infrastructure.initializer import AgentPersona
from mas.infrastructure.law_es_tool import LawEsTool
from mas.infrastructure.llm import GPTChat

from ..core.graph import ShadowGraph
from ..core.schemas import WorkerReport
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

        self.disable_recall_worker = bool(
            getattr(legal_system.cfg.experiment, "disable_recall_worker", False)
        )

        self.controller = ArgumentController(
            name=f"{side}_Controller",
            persona=persona,
            graph_tool=graph_tool,
            insights=insights,
        )

        self.controller.llm = llm

        for action in self.controller.actions:
            action.llm = llm

        worker_cfg = legal_system.cfg.worker_threshold
        fact_threshold = float(worker_cfg.fact_worker_threshold)
        law_threshold = float(worker_cfg.law_worker_threshold)

        self.fact_worker = FactWorker(
            name=f"{side}_FactWorker",
            es_tool=fact_es,
            llm=llm,
            threshold=fact_threshold,
        )

        self.law_worker = LawWorker(
            name=f"{side}_LawWorker",
            es_tool=law_es,
            llm=llm,
            legal_system=legal_system,
            threshold=law_threshold,
        )

        self.law_worker.graph_tool = graph_tool
        self.recall_worker: RecallWorker | None = None

        if not self.disable_recall_worker:
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
        """Return current controller pipeline step.

        Returns:
            Current controller pipeline step, or `IDLE` when unavailable.
        """
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

    def _parse_worker_report(
        self, worker_name: str, raw_content: str
    ) -> Dict[str, Any]:
        """Extract normalized worker report fields from raw response text."""
        content = raw_content
        status = "UNKNOWN"
        max_score = 0.0
        report: WorkerReport | None = None

        try:
            report = WorkerReport.model_validate_json(raw_content)

        except Exception:
            report = None

        if report is not None:
            if isinstance(report.content, str) and report.content.strip():
                content = report.content

            status = report.status.value

            try:
                max_score = float(report.max_score)

            except Exception:
                max_score = 0.0

        return {
            "worker": worker_name,
            "status": status,
            "content": content,
            "max_score": max_score,
            "raw": raw_content,
        }

    async def run_turn(self, graph: ShadowGraph) -> dict:
        """Execute the workflow for a single debate turn.

        Args:
            graph: The current `ShadowGraph` of the debate.

        Returns:
            A dictionary containing a summary of the turn's outcome, a detailed
            transcript of internal messages (if verbose), and the list of
            executed `AgentAction` objects.

        Raises:
            Assumption/Unverified: `ValueError` may be raised during internal
                batch payload validation but is caught and converted into system
                feedback; this coroutine normally returns a result dict.
        """
        logger.info(f"--- Team {self.side} Turn Start ---")
        self.graph_tool.set_current_graph(graph)
        self.controller.reset_turn_state()
        self.controller.pipeline_step = ControllerPipelineStep.ASSESS_NEEDS
        self._notify_state_change("turn_start", {"step": "ASSESS_NEEDS"})
        transcript = []
        final_result = None
        step_count = 0
        turn_uid = f"turn_{self.side}_{int(time.time() * 1000)}"
        worker_reports_raw: List[Dict[str, Any]] = []
        error_history_start = len(getattr(self.controller, "error_history", []))

        while step_count < self.max_internal_steps:
            step_count += 1

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
                if any(
                    marker in content
                    for marker in [
                        "PLAN_READY_FOR_PUSH",
                        "PLAN_VALIDATED",
                        "ROUTED_TO_PUSH",
                    ]
                ):
                    self._notify_state_change("plan_progress", {"content": content})

                else:
                    self._notify_state_change("retry", {"reason": "execution_failure"})

                continue

            parsed_payload = None

            if content:
                try:
                    parsed_payload = json.loads(content)

                except Exception:
                    parsed_payload = None

            if (
                isinstance(parsed_payload, dict)
                and "batch_instructions" in parsed_payload
            ):
                try:
                    data = parsed_payload

                    if not isinstance(data, dict):
                        raise ValueError("batch payload must be a JSON object")

                    instructions_list = data.get("batch_instructions", [])

                    if not isinstance(instructions_list, list):
                        raise ValueError("batch_instructions must be a list")

                    if not instructions_list:
                        self.controller.ingest_results([])
                        continue

                    tasks = []
                    worker_names = []
                    dispatch_times: List[float] = []

                    for item in instructions_list:
                        if not isinstance(item, dict):
                            raise ValueError("instruction item must be a JSON object")

                        target_name = item.get("target")
                        inst_json = item.get("instruction")
                        worker = self._get_worker_by_name(target_name)

                        if worker:
                            inst_payload = (
                                json.dumps(inst_json, ensure_ascii=False)
                                if isinstance(inst_json, dict)
                                else str(inst_json)
                            )

                            worker.rc.memory.add(
                                Message(
                                    content=inst_payload, role=self.controller.profile
                                )
                            )

                            tasks.append(worker._act())
                            worker_names.append(worker.name)
                            dispatch_times.append(time.perf_counter())

                        else:
                            logger.warning(
                                f"[{self.side}] Unknown target worker: {target_name}"
                            )

                    if tasks:
                        results = await asyncio.gather(*tasks)
                        results_payload = []
                        batch_durations_ms: List[int] = []

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

                            report_row = self._parse_worker_report(
                                w_name, str(res_msg.content)
                            )

                            report_row["duration_ms"] = int(
                                (time.perf_counter() - dispatch_times[i]) * 1000
                            )

                            batch_durations_ms.append(int(report_row["duration_ms"]))
                            worker_reports_raw.append(report_row)

                        avg_ms = (
                            int(sum(batch_durations_ms) / len(batch_durations_ms))
                            if batch_durations_ms
                            else 0
                        )

                        max_ms = max(batch_durations_ms) if batch_durations_ms else 0

                        logger.info(
                            f"[{self.side}] Worker batch completed: workers={len(results_payload)}, avg_ms={avg_ms}, max_ms={max_ms}"
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
                content=(
                    "SYSTEM_FEEDBACK: 你的输出无法被识别。"
                    "请确保输出合法的 batch_instructions 协议消息。"
                ),
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
            "turn_uid": turn_uid,
            "summary": final_result,
            "transcript": transcript,
            "actions": self.controller.last_executed_actions,
            "controller_assessment": dict(self.controller.last_assessment),
            "batch_instructions": list(self.controller.last_batch_instructions),
            "worker_reports_raw": worker_reports_raw,
            "decision_raw": self.controller.last_decision_raw,
            "parsed_actions": list(self.controller.last_parsed_actions),
            "execution_log": self.controller.last_execution_log,
            "retry_history": list(
                getattr(self.controller, "error_history", [])[error_history_start:]
            ),
            "action_cache": list(getattr(self.controller, "action_cache", [])),
            "plan_attempts_used": int(getattr(self.controller, "plan_attempt", 0)),
            "push_attempts_used": len(
                [
                    row
                    for row in getattr(self.controller, "action_cache", [])
                    if isinstance(row, dict) and row.get("stage") == "push"
                ]
            ),
            "narrative_raw_sentences": [
                str(item.content) for item in self.controller.last_executed_actions
            ],
            "ts_ms": int(time.time() * 1000),
        }
