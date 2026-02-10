"""Defines the ArgumentController, the strategic agent of a debate team.

This module contains the `ArgumentController` class, which acts as the "lead
lawyer" or "brain" for a debate team (plaintiff or defendant). It orchestrates
the team's turn by following an internal state machine (pipeline), assessing
the debate's state, delegating research tasks to worker agents, synthesizing
their findings, and ultimately deciding on the concrete actions to take on the
debate graph.
"""

import asyncio
import json
import time
from enum import Enum, auto
from typing import Any, Dict, List

from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message

from actions.controller_actions import (
    AssessFactNeeds,
    AssessLawNeeds,
    AssessRecallNeeds,
    VerifyAndDecide,
)
from mas.core.schemas import (
    AgentAction,
    ResourceRequirement,
    WorkerInstruction,
)
from tools.action_parser import parse_agent_action_output
from tools.graph_tool import GraphTool
from tools.initializer import AgentPersona
from tools.json_utils import extract_json_from_text


class ControllerPipelineStep(Enum):
    """Enumeration for the internal states of the controller's turn-based pipeline."""

    IDLE = auto()
    ASSESS_NEEDS = auto()
    WAIT_FOR_WORKERS = auto()
    DECIDE = auto()
    DONE = auto()


class ArgumentController(Role):
    """The strategic core of a debate team, responsible for orchestrating a turn.

    The controller manages a pipeline for its turn:
    1.  `ASSESS_NEEDS`: It evaluates the current debate graph to determine what
        information is needed (facts, laws, or historical strategies).
    2.  It dispatches instructions to the appropriate worker agents.
    3.  `WAIT_FOR_WORKERS`: It waits for the workers to return their findings.
    4.  It ingests the workers' reports into a consolidated summary.
    5.  `DECIDE`: It uses this summary and the graph context to decide on a final
        set of `AgentAction`s to modify the graph.
    6.  It handles feedback from failed actions and can retry its decision.

    Attributes:
        persona: The BDI (Belief, Desire, Intention) persona of the agent.
        graph_tool: The tool for interacting with the debate graph.
        insights: Initial strategic insights provided at the start of the case.
        pipeline_step: The current step in the controller's internal state machine.
        investigation_buffer: A dictionary to store results from worker agents.
        latest_summary: A consolidated text summary of the investigation buffer.
        recent_errors: A list of error messages from failed graph execution attempts.
        last_executed_actions: A list of the `AgentAction`s successfully
            executed in the last turn.
    """

    name: str = "Controller"
    profile: str = "Lead Lawyer"

    def __init__(
        self,
        name: str,
        persona: AgentPersona,
        graph_tool: GraphTool,
        insights: str = "",
    ):
        """Initialize the ArgumentController.

        Args:
            name: The specific name of the controller (e.g., "plaintiff_Controller").
            persona: The `AgentPersona` object defining the agent's goals.
            graph_tool: The `GraphTool` for executing actions on the debate graph.
            insights: A string containing pre-computed strategic insights relevant
                to the current case.
        """
        super().__init__(name=name, profile="Lead Lawyer")
        self.persona = persona
        self.graph_tool = graph_tool
        self.insights = insights

        self.set_actions(
            [AssessFactNeeds, AssessLawNeeds, AssessRecallNeeds, VerifyAndDecide]
        )

        self.pipeline_step = ControllerPipelineStep.IDLE
        self.investigation_buffer: Dict[str, str] = {}
        self.latest_summary: str = ""
        self.recent_errors: List[str] = []
        self.error_history: List[Dict[str, Any]] = []
        self.last_executed_actions: List[AgentAction] = []
        self.last_decision_raw: str = ""
        self.last_parsed_actions: List[Dict[str, Any]] = []
        self.last_execution_log: str = ""
        self.last_batch_instructions: List[Dict[str, Any]] = []
        self.last_assessment: Dict[str, Dict[str, Any]] = {}

    def reset_turn_state(self):
        """Reset the controller's state at the beginning of a new turn."""
        logger.info(f"[{self.name}] Resetting turn state (Buffer & Errors cleared).")
        self.pipeline_step = ControllerPipelineStep.IDLE
        self.investigation_buffer = {}
        self.latest_summary = ""
        self.recent_errors = []
        self.last_executed_actions = []
        self.last_decision_raw = ""
        self.last_parsed_actions = []
        self.last_execution_log = ""
        self.last_batch_instructions = []
        self.last_assessment = {}

    def ingest_results(self, results_list: List[Dict[str, str]]):
        """Process and store the results from worker agents.

        This method is called by the `DebateTeam` after the parallel worker
        tasks have completed. It parses the reports, updates the internal
        `investigation_buffer`, generates a summary, and advances the
        pipeline to the `DECIDE` step.

        Args:
            results_list: A list of dictionaries, where each dictionary contains
                the 'worker' name and 'content' of their report.
        """
        logger.info(
            f"[{self.name}] Ingesting {len(results_list)} worker results via direct call."
        )

        try:
            for item in results_list:
                worker_name = item.get("worker", "")
                raw_content = item.get("content", "")
                clean_content = raw_content

                if "{" in raw_content:
                    json_part = extract_json_from_text(raw_content)

                    if json_part and "content" in json_part:
                        clean_content = json_part["content"]
                        status = json_part.get("status", "UNKNOWN")

                        if status == "NOT_FOUND":
                            clean_content = f"（未找到）{clean_content}"

                if "FactWorker" in worker_name:
                    self.investigation_buffer["Fact"] = (
                        f"🔎 [事实检索报告]: {clean_content}"
                    )

                elif "LawWorker" in worker_name:
                    self.investigation_buffer["Law"] = (
                        f"⚖️ [法条检索报告]: {clean_content}"
                    )

                elif "RecallWorker" in worker_name:
                    self.investigation_buffer["Recall"] = (
                        f"🧠 [历史策略参考]: {clean_content}"
                    )

        except Exception as e:
            logger.error(f"[{self.name}] Failed to ingest results: {e}")
            self.investigation_buffer["Error"] = f"Ingestion error: {e}"

        finally:
            self._generate_and_memorize_summary()
            self.pipeline_step = ControllerPipelineStep.DECIDE
            logger.info(f"[{self.name}] State transitioned to DECIDE.")

    async def _act(self) -> Message:
        """Execute the main logic for the controller, driven by a state machine.

        This method is called repeatedly by the `DebateTeam`. Its behavior
        depends on the current `self.pipeline_step`.

        Returns:
            A `Message` object containing either instructions for workers, a
            status update (like "WAITING"), or the final result of the turn's
            actions.
        """
        memories = self.get_memories(k=1)
        last_msg = memories[-1] if memories else None
        last_content = str(last_msg.content) if last_msg else ""
        last_role = str(last_msg.role) if last_msg else ""

        if last_role == "System" and "SYSTEM_START" in last_content:
            if self.pipeline_step in [
                ControllerPipelineStep.IDLE,
                ControllerPipelineStep.DONE,
            ]:
                logger.info(
                    f"[{self.name}] Received START signal. Starting Assessment."
                )

                self.pipeline_step = ControllerPipelineStep.ASSESS_NEEDS

            elif self.pipeline_step != ControllerPipelineStep.ASSESS_NEEDS:
                logger.info(f"[{self.name}] Force Restart triggered.")
                self.pipeline_step = ControllerPipelineStep.ASSESS_NEEDS

        graph_context = self.graph_tool.current_graph.latest_context

        if self.pipeline_step == ControllerPipelineStep.ASSESS_NEEDS:
            logger.info(
                f"[{self.name}] Parallelly Assessing Needs (Fact, Law, Recall)..."
            )

            for key in ["Fact", "Law", "Recall"]:
                if key not in self.investigation_buffer:
                    self.investigation_buffer[key] = "（未评估）"

            fact_task = AssessFactNeeds(llm=self.llm).run(
                self.name, self.persona, graph_context
            )

            law_task = AssessLawNeeds(llm=self.llm).run(
                self.name, self.persona, graph_context
            )

            recall_task = AssessRecallNeeds(llm=self.llm).run(
                self.name, self.persona, graph_context
            )

            results = await asyncio.gather(fact_task, law_task, recall_task)
            raw_fact, raw_law, raw_recall = results
            req_fact = self._parse_requirement(raw_fact)
            req_law = self._parse_requirement(raw_law)
            req_recall = self._parse_requirement(raw_recall)
            instructions = []

            self.last_assessment = {
                "fact": req_fact.model_dump(exclude_none=True),
                "law": req_law.model_dump(exclude_none=True),
                "recall": req_recall.model_dump(exclude_none=True),
            }

            if req_fact.need:
                if not req_fact.intent:
                    logger.error(
                        f"[{self.name}] Fact Need is True but Intent is missing! Aborting instruction generation for FactWorker."
                    )

                    self.investigation_buffer["Fact"] = (
                        "⚠️ [指令错误]: AI 认为需要事实检索，但未提供意图描述。"
                    )

                else:
                    inst = WorkerInstruction(intent=req_fact.intent)

                    instructions.append(
                        {"target": "FactWorker", "instruction": inst.to_json()}
                    )

                    self.investigation_buffer["Fact"] = (
                        f"⏳ [正在检索事实]: {req_fact.intent}"
                    )

            else:
                reasoning = self._truncate(req_fact.reasoning, 500)
                self.investigation_buffer["Fact"] = f"✅ [无需事实检索]: {reasoning}"

            if req_law.need:
                if not req_law.intent:
                    logger.error(
                        f"[{self.name}] Law Need is True but Intent is missing! Aborting instruction generation for LawWorker."
                    )

                    self.investigation_buffer["Law"] = (
                        "⚠️ [指令错误]: AI 认为需要法条检索，但未提供意图描述。"
                    )

                else:
                    inst = WorkerInstruction(intent=req_law.intent)

                    instructions.append(
                        {"target": "LawWorker", "instruction": inst.to_json()}
                    )

                    self.investigation_buffer["Law"] = (
                        f"⏳ [正在检索法条]: {req_law.intent}"
                    )

            else:
                reasoning = self._truncate(req_law.reasoning, 500)
                self.investigation_buffer["Law"] = f"✅ [无法律检索]: {reasoning}"

            if req_recall.need:
                if not req_recall.intent:
                    logger.error(
                        f"[{self.name}] Recall Need is True but Intent is missing! Aborting instruction generation for RecallWorker."
                    )

                    self.investigation_buffer["Recall"] = (
                        "⚠️ [指令错误]: AI 认为需要案例策略，但未提供意图描述。"
                    )

                else:
                    inst = WorkerInstruction(intent=req_recall.intent)

                    instructions.append(
                        {"target": "RecallWorker", "instruction": inst.to_json()}
                    )

                    self.investigation_buffer["Recall"] = (
                        f"⏳ [正在检索案例]: {req_recall.intent}"
                    )

            else:
                reasoning = self._truncate(req_recall.reasoning, 500)
                self.investigation_buffer["Recall"] = f"✅ [无案例借鉴]: {reasoning}"

            if not instructions:
                logger.info(
                    f"[{self.name}] No workers needed. Generating summary and jumping to DECIDE."
                )

                self._generate_and_memorize_summary()
                self.pipeline_step = ControllerPipelineStep.DECIDE
                self.last_batch_instructions = []

                return Message(
                    content=json.dumps({"batch_instructions": []}), role=self.profile
                )

            self.pipeline_step = ControllerPipelineStep.WAIT_FOR_WORKERS
            self.last_batch_instructions = [dict(item) for item in instructions]
            batch_msg_content = json.dumps({"batch_instructions": instructions})
            return Message(content=batch_msg_content, role=self.profile)

        elif self.pipeline_step == ControllerPipelineStep.WAIT_FOR_WORKERS:
            return Message(content="WAITING_FOR_PARALLEL_WORKERS", role=self.profile)

        elif self.pipeline_step == ControllerPipelineStep.DECIDE:
            logger.info(
                f"[{self.name}] Finalizing Decision... (Error History: {len(self.recent_errors)})"
            )

            feedback_section = ""

            if self.recent_errors:
                error_log_str = "\n".join(self.recent_errors)

                feedback_section = (
                    f"【⚠️ 警告：之前的尝试执行失败】\n"
                    f"请仔细分析以下报错信息，并修正你的图谱操作 JSON（检查 ID 是否存在、类型是否匹配）：\n"
                    f"{error_log_str}\n"
                    f"请不要重复犯同样的错误。"
                )

            current_advice = self.latest_summary or "（本轮无额外信息）"
            action = VerifyAndDecide(llm=self.llm)
            id_list_str = self.graph_tool.current_graph.get_simple_id_list()

            decision_raw = await action.run(
                self.name,
                current_advice,
                graph_context,
                self.persona.initial_strategy,
                feedback=feedback_section,
                id_inventory=id_list_str,
            )

            self.last_decision_raw = decision_raw
            parsed_actions = parse_agent_action_output(decision_raw)

            if isinstance(parsed_actions, list) and all(
                isinstance(a, AgentAction) for a in parsed_actions
            ):
                self.last_parsed_actions = [
                    item.model_dump(exclude_none=True) for item in parsed_actions
                ]

                exec_msg = await self.graph_tool.process_intent(
                    self.name, parsed_actions
                )

                self.last_execution_log = exec_msg

                if "REJECT" in exec_msg or "Error" in exec_msg:
                    logger.warning(f"[{self.name}] Action Execution Failed: {exec_msg}")

                    detailed_error = (
                        f"[系统报错]: {exec_msg}\n[你的原始输出]: \n{decision_raw}\n"
                    )

                    self.recent_errors = [detailed_error]

                    self.error_history.append(
                        {
                            "kind": "execution_error",
                            "detail": detailed_error,
                            "ts_ms": int(time.time() * 1000),
                        }
                    )

                    return Message(
                        content=f"EXECUTION_FAILURE_RETRY: {exec_msg[:50]}...",
                        role=self.profile,
                    )

                else:
                    self.last_executed_actions = parsed_actions
                    self.pipeline_step = ControllerPipelineStep.DONE

                    return Message(
                        content=f"Action Completed: {exec_msg}", role=self.profile
                    )

            else:
                error_msg = f"JSON Parsing Failed: {parsed_actions}"
                logger.warning(f"[{self.name}] {error_msg}")
                self.last_parsed_actions = []
                self.last_execution_log = error_msg

                detailed_error = (
                    f"[格式错误]: {error_msg}\n[你的原始输出]: \n{decision_raw}\n"
                )

                self.recent_errors = [detailed_error]

                self.error_history.append(
                    {
                        "kind": "json_parse_error",
                        "detail": detailed_error,
                        "ts_ms": int(time.time() * 1000),
                    }
                )

                return Message(
                    content="EXECUTION_FAILURE_RETRY: JSON Error", role=self.profile
                )

        elif self.pipeline_step == ControllerPipelineStep.DONE:
            return Message(
                content="Action Completed: Pipeline Finished.", role=self.profile
            )

        else:
            return Message(content="Controller IDLE", role=self.profile)

    def _generate_and_memorize_summary(self):
        """Create a summary of worker findings and adds it to short-term memory."""
        try:
            step_info = "?"

            if (
                self.graph_tool
                and hasattr(self.graph_tool, "system")
                and hasattr(self.graph_tool.system, "step_counter")
            ):
                step_info = str(self.graph_tool.system.step_counter)

            summary_text = (
                f"=== 🕵️ 本轮调查综述 (Round {step_info}) ===\n"
                f"1. {self.investigation_buffer.get('Fact')}\n\n"
                f"2. {self.investigation_buffer.get('Law')}\n\n"
                f"3. {self.investigation_buffer.get('Recall')}\n"
                f"=============================="
            )

            self.latest_summary = summary_text
            self.rc.memory.add(Message(content=summary_text, role="System"))

            logger.info(
                f"[{self.name}] Memorized investigation summary (Length: {len(summary_text)})"
            )

        except Exception as e:
            logger.error(f"[{self.name}] Failed to memorize summary: {e}")

            fallback_msg = (
                f"=== 🕵️ 本轮调查综述 (Error) ===\nFailed to generate summary: {e}"
            )

            self.latest_summary = fallback_msg
            self.rc.memory.add(Message(content=fallback_msg, role="System"))

    def _parse_requirement(self, raw_text: str) -> ResourceRequirement:
        """Safely parse a JSON string into a ResourceRequirement object.

        Args:
            raw_text: The raw string output from a needs assessment LLM call.

        Returns:
            A `ResourceRequirement` object. Returns a default `need=False` object
            on parsing failure.
        """
        try:
            data = extract_json_from_text(raw_text)
            return ResourceRequirement.model_validate(data)

        except Exception as e:
            logger.warning(f"[{self.name}] Parse Error: {e}. Defaulting to NEED=False.")
            return ResourceRequirement(need=False, reasoning=f"Parse Error: {e}")

    def _truncate(self, text: str, length: int) -> str:
        """Truncate a string to a maximum length, adding an ellipsis."""
        if not text:
            return ""

        text = text.strip()

        if len(text) > length:
            return text[:length] + "..."

        return text

    def get_error_history(self, limit: int | None = None) -> List[Dict[str, Any]]:
        """Return append-only controller error history."""
        if limit is None:
            return [dict(item) for item in self.error_history]

        safe_limit = max(1, int(limit))
        return [dict(item) for item in self.error_history[-safe_limit:]]
