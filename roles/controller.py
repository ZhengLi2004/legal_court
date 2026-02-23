"""Defines the ArgumentController, the strategic agent of a debate team.

This module contains the `ArgumentController` class, which acts as the "lead
lawyer" or "brain" for a debate team (plaintiff or defendant). It orchestrates
the team's turn by following an internal state machine (pipeline), assessing
the debate's state, delegating research tasks to worker agents, synthesizing
their findings, and ultimately deciding on the concrete actions to take on the
debate graph.
"""

import json
import time
from typing import Any, Dict, List

from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from pydantic import ValidationError

from actions.controller_actions import (
    AssessFactNeeds,
    AssessLawNeeds,
    AssessRecallNeeds,
    ChoosePlanOrPush,
    PlanTool,
    PushTool,
    VerifyAndDecide,
)
from mas.core.controller_pipeline import ControllerPipelineStep
from mas.core.schemas import (
    AgentAction,
    ResourceRequirement,
    WorkerInstruction,
    WorkerReport,
    WorkerReportStatus,
)
from tools.graph_tool import GraphTool
from tools.initializer import AgentPersona
from tools.llm import ToolCallContractError


class ArgumentController(Role):
    """The strategic core of a debate team, responsible for orchestrating a turn.

    The controller manages a pipeline for its turn:
    1.  `ASSESS_NEEDS`: It evaluates the current debate graph to determine what
        information is needed (facts, laws, or historical strategies).
    2.  It dispatches instructions to the appropriate worker agents.
    3.  `WAIT_FOR_WORKERS`: It waits for the workers to return their findings.
    4.  It ingests the workers' reports into a consolidated summary.
    5.  `PLAN`: It generates and validates a JSON action plan.
    6.  `PUSH`: It executes actions only after plan validation passes.
    7.  It handles feedback from failed actions and can retry planning.

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
            [
                AssessFactNeeds,
                AssessLawNeeds,
                AssessRecallNeeds,
                VerifyAndDecide,
                ChoosePlanOrPush,
                PlanTool,
                PushTool,
            ]
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
        self.action_cache: List[Dict[str, Any]] = []
        self.max_plan_attempts: int = 3
        self.plan_attempt: int = 0
        self.push_ready: bool = False
        self._validated_actions_for_push: List[AgentAction] = []

    def reset_turn_state(self):
        """Reset the controller's state at the beginning of a new turn."""
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
        self.action_cache = []
        self.plan_attempt = 0
        self.push_ready = False
        self._validated_actions_for_push = []

    def ingest_results(self, results_list: List[Dict[str, str]]):
        """Process and store the results from worker agents.

        This method is called by the `DebateTeam` after the parallel worker
        tasks have completed. It parses the reports, updates the internal
            `investigation_buffer`, generates a summary, and advances the
            pipeline to the `PLAN` step.

        Args:
            results_list: A list of dictionaries, where each dictionary contains
                the 'worker' name and 'content' of their report.
        """
        for item in results_list:
            if not isinstance(item, dict):
                continue

            worker_name = str(item.get("worker", ""))
            raw_content = str(item.get("content", ""))
            clean_content = raw_content
            status = "UNKNOWN"

            try:
                parsed_report = WorkerReport.model_validate_json(raw_content)
                clean_content = parsed_report.content
                status = parsed_report.status.value

            except ValidationError:
                pass

            if status == WorkerReportStatus.NOT_FOUND.value:
                clean_content = f"（未找到）{clean_content}"

            if "FactWorker" in worker_name:
                self.investigation_buffer["Fact"] = (
                    f"🔎 [事实检索报告]: {clean_content}"
                )

            elif "LawWorker" in worker_name:
                self.investigation_buffer["Law"] = f"⚖️ [法条检索报告]: {clean_content}"

            elif "RecallWorker" in worker_name:
                self.investigation_buffer["Recall"] = (
                    f"🧠 [历史策略参考]: {clean_content}"
                )

        self._generate_and_memorize_summary()
        self.pipeline_step = ControllerPipelineStep.PLAN

    async def _act(self) -> Message:
        """Execute the main logic for the controller, driven by a state machine.

        This method is called repeatedly by the `DebateTeam`. Its behavior
        depends on the current `self.pipeline_step`. Strategic branches consume
        strict function-calling outputs from the LLM and record contract errors
        explicitly in `error_history`.

        Returns:
            A `Message` object containing either instructions for workers, a
            status update (like "WAITING"), or the final result of the turn's
            actions.
        """
        memories = self.get_memories(k=1)
        last_msg = memories[-1] if memories else None
        last_content = str(last_msg.content) if last_msg else ""
        last_role = str(last_msg.role) if last_msg else ""

        self._sync_pipeline_step_from_system_start(
            last_role=last_role,
            last_content=last_content,
        )

        graph_context = self.graph_tool.current_graph.latest_context

        if self.pipeline_step == ControllerPipelineStep.ASSESS_NEEDS:
            return await self._act_assess_needs(graph_context)

        if self.pipeline_step == ControllerPipelineStep.WAIT_FOR_WORKERS:
            return Message(content="WAITING_FOR_PARALLEL_WORKERS", role=self.profile)

        if self.pipeline_step == ControllerPipelineStep.PLAN:
            return await self._act_plan(graph_context)

        if self.pipeline_step == ControllerPipelineStep.PUSH:
            return await self._act_push()

        if self.pipeline_step == ControllerPipelineStep.DONE:
            return Message(
                content="Action Completed: Pipeline Finished.", role=self.profile
            )

        return Message(content="Controller IDLE", role=self.profile)

    def _sync_pipeline_step_from_system_start(
        self,
        *,
        last_role: str,
        last_content: str,
    ) -> None:
        """Sync pipeline step when a new system round starts."""
        if last_role == "System" and "SYSTEM_START" in last_content:
            if self.pipeline_step in [
                ControllerPipelineStep.IDLE,
                ControllerPipelineStep.DONE,
            ]:
                self.pipeline_step = ControllerPipelineStep.ASSESS_NEEDS

            elif self.pipeline_step != ControllerPipelineStep.ASSESS_NEEDS:
                self.pipeline_step = ControllerPipelineStep.ASSESS_NEEDS

    async def _act_assess_needs(self, graph_context: Any) -> Message:
        """Run ASSESS_NEEDS stage and emit worker instructions if needed."""
        for key in ["Fact", "Law", "Recall"]:
            if key not in self.investigation_buffer:
                self.investigation_buffer[key] = "（未评估）"

        plan_tool = PlanTool(llm=self.llm)

        try:
            (
                fact_payload,
                law_payload,
                recall_payload,
            ) = await plan_tool.assess_needs(self.name, self.persona, graph_context)

        except ToolCallContractError as e:
            detail = self._record_tool_call_error(stage="assess_needs", error=e)
            logger.warning(f"[{self.name}] Tool-call contract error: {detail}")

            return Message(
                content="EXECUTION_FAILURE_RETRY: TOOL_CALL_ERROR",
                role=self.profile,
            )

        except Exception as e:
            logger.error(f"[{self.name}] Assessment failed unexpectedly: {e}")

            self.error_history.append(
                {
                    "kind": "assessment_error",
                    "detail": str(e),
                    "ts_ms": int(time.time() * 1000),
                }
            )

            return Message(
                content="EXECUTION_FAILURE_RETRY: ASSESS_NEEDS_ERROR",
                role=self.profile,
            )

        try:
            req_fact = self._parse_requirement(fact_payload)
            req_law = self._parse_requirement(law_payload)
            req_recall = self._parse_requirement(recall_payload)

        except ValueError as e:
            self._record_tool_call_error(stage="assess_needs_parse", error=e)

            return Message(
                content="EXECUTION_FAILURE_RETRY: TOOL_CALL_ERROR",
                role=self.profile,
            )

        instructions = []

        self.last_assessment = {
            "fact": req_fact.model_dump(exclude_none=True),
            "law": req_law.model_dump(exclude_none=True),
            "recall": req_recall.model_dump(exclude_none=True),
        }

        if req_fact.need:
            if not req_fact.intent:
                logger.warning(
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
                logger.warning(
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
                logger.warning(
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
            self._generate_and_memorize_summary()
            self.pipeline_step = ControllerPipelineStep.PLAN
            self.last_batch_instructions = []

            return Message(
                content=json.dumps({"batch_instructions": []}), role=self.profile
            )

        self.pipeline_step = ControllerPipelineStep.WAIT_FOR_WORKERS
        self.last_batch_instructions = [dict(item) for item in instructions]
        batch_msg_content = json.dumps({"batch_instructions": instructions})
        return Message(content=batch_msg_content, role=self.profile)

    async def _act_plan(self, graph_context: Any) -> Message:
        """Run PLAN stage and keep validated actions for push."""
        current_advice = self.latest_summary or "（本轮无额外信息）"
        route_reason = ""
        route_step = "plan"
        route_tool = ChoosePlanOrPush(llm=self.llm)

        try:
            route_payload = await route_tool.run(
                role_name=self.name,
                worker_advice=current_advice,
                graph_context=graph_context,
                action_cache_context=self._build_action_cache_context(limit=5),
                has_validated_plan=(
                    self.push_ready and bool(self._validated_actions_for_push)
                ),
                plan_attempt=self.plan_attempt,
                max_plan_attempts=self.max_plan_attempts,
                recent_errors=(
                    "\n".join(self.recent_errors) if self.recent_errors else "（无）"
                ),
            )

            route_step = str(route_payload.get("next_step", "plan")).strip().lower()
            route_reason = str(route_payload.get("reason", "")).strip()

        except ToolCallContractError as e:
            detail = self._record_tool_call_error(stage="route_plan_or_push", error=e)
            self.last_execution_log = detail

            return Message(
                content="EXECUTION_FAILURE_RETRY: TOOL_CALL_ERROR",
                role=self.profile,
            )

        except ValueError as e:
            route_step = "plan"
            route_reason = f"Router payload invalid: {e}"

        if route_step == "push":
            if self.push_ready and self._validated_actions_for_push:
                self.last_execution_log = (
                    f"ROUTED_TO_PUSH: {route_reason or '模型选择直接执行已验证动作。'}"
                )

                self._log_plan_summary(
                    attempt=self.plan_attempt,
                    route_step="push",
                    route_reason=route_reason,
                    status="routed_to_push",
                    action_count=len(self._validated_actions_for_push),
                    error_count=0,
                )

                self.pipeline_step = ControllerPipelineStep.PUSH

                return Message(
                    content="EXECUTION_FAILURE_RETRY: ROUTED_TO_PUSH",
                    role=self.profile,
                )

            self.last_execution_log = (
                "ROUTED_TO_PUSH_BUT_NO_VALIDATED_PLAN: continue planning."
            )

        else:
            self.last_execution_log = (
                f"ROUTED_TO_PLAN: {route_reason or '模型选择继续规划。'}"
            )

        if self.plan_attempt >= self.max_plan_attempts:
            limit_msg = (
                f"PLAN_RETRY_LIMIT_EXCEEDED: max_plan_attempts={self.max_plan_attempts}"
            )

            self.last_execution_log = limit_msg
            self.pipeline_step = ControllerPipelineStep.DONE

            return Message(content=f"Action Completed: {limit_msg}", role=self.profile)

        self.plan_attempt += 1
        feedback_section = self._build_plan_feedback()
        id_list_str = self.graph_tool.current_graph.get_simple_id_list()
        plan_tool = PlanTool(llm=self.llm)

        try:
            plan_payload = await plan_tool.run(
                role_name=self.name,
                graph_tool=self.graph_tool,
                worker_advice=current_advice,
                graph_context=graph_context,
                focus=self.persona.intention,
                id_inventory=id_list_str,
                feedback=feedback_section,
            )

        except ToolCallContractError as e:
            detail = self._record_tool_call_error(stage="plan", error=e)
            logger.warning(f"[{self.name}] Tool-call contract error: {detail}")
            self.last_parsed_actions = []
            self.last_execution_log = detail

            self._append_action_cache(
                attempt=self.plan_attempt,
                stage="plan",
                status="validation_failed",
                decision_raw="",
                parsed_actions=[],
                validation_errors=[detail],
                push_error=None,
                sandbox_actions=[],
            )

            return Message(
                content="EXECUTION_FAILURE_RETRY: TOOL_CALL_ERROR",
                role=self.profile,
            )

        except ValueError as e:
            error_msg = f"Action Payload Failed: {e}"
            logger.warning(f"[{self.name}] {error_msg}")
            self.last_parsed_actions = []
            self.last_execution_log = error_msg
            self.last_decision_raw = ""
            detailed_error = f"[参数结构错误]: {error_msg}"
            self.recent_errors = [detailed_error]

            self.error_history.append(
                {
                    "kind": "action_payload_error",
                    "detail": detailed_error,
                    "ts_ms": int(time.time() * 1000),
                }
            )

            self._append_action_cache(
                attempt=self.plan_attempt,
                stage="plan",
                status="validation_failed",
                decision_raw="",
                parsed_actions=[],
                validation_errors=[detailed_error],
                push_error=None,
                sandbox_actions=[],
            )

            return Message(
                content="EXECUTION_FAILURE_RETRY: ACTION_PAYLOAD_ERROR",
                role=self.profile,
            )

        raw_actions = plan_payload.get("raw_actions", [])
        parsed_actions_payload = plan_payload.get("parsed_actions", [])
        sandbox_actions_payload = plan_payload.get("sandbox_actions", [])
        validation_errors = plan_payload.get("validation_errors", [])
        validated = bool(plan_payload.get("validated", False))

        if not isinstance(raw_actions, list):
            raw_actions = []

        if not isinstance(parsed_actions_payload, list):
            parsed_actions_payload = []

        if not isinstance(sandbox_actions_payload, list):
            sandbox_actions_payload = []

        if not isinstance(validation_errors, list):
            validation_errors = [str(validation_errors)]

        self.last_decision_raw = json.dumps(raw_actions, ensure_ascii=False)

        self.last_parsed_actions = [
            item for item in parsed_actions_payload if isinstance(item, dict)
        ]

        parsed_actions: List[AgentAction] = []

        if validated:
            try:
                parsed_actions = [
                    AgentAction.model_validate(item) for item in sandbox_actions_payload
                ]

            except (TypeError, ValueError, ValidationError) as e:
                validated = False
                validation_errors.append(f"Action Validation Failed: {e}")

        if not validated:
            error_text = "\n".join([str(item) for item in validation_errors]).strip()

            if not error_text:
                error_text = "Action Validation Failed: 未提供有效的校验错误详情。"

            self.last_execution_log = error_text

            detailed_error = (
                f"[参数校验错误]: {error_text}\n"
                f"[你的原始输出]: \n{self.last_decision_raw}\n"
            )

            self.recent_errors = [detailed_error]
            self.push_ready = False
            self._validated_actions_for_push = []

            self.error_history.append(
                {
                    "kind": "action_validation_error",
                    "detail": detailed_error,
                    "ts_ms": int(time.time() * 1000),
                }
            )

            self._append_action_cache(
                attempt=self.plan_attempt,
                stage="plan",
                status="validation_failed",
                decision_raw=self.last_decision_raw,
                parsed_actions=self.last_parsed_actions,
                validation_errors=[str(item) for item in validation_errors],
                push_error=None,
                sandbox_actions=[
                    item for item in sandbox_actions_payload if isinstance(item, dict)
                ],
            )

            self._log_plan_summary(
                attempt=self.plan_attempt,
                route_step=route_step,
                route_reason=route_reason,
                status="validation_failed",
                action_count=len(self.last_parsed_actions),
                error_count=len(validation_errors),
            )

            if self.plan_attempt >= self.max_plan_attempts:
                limit_msg = (
                    "PLAN_RETRY_LIMIT_EXCEEDED: validation failed after max attempts."
                )

                self.pipeline_step = ControllerPipelineStep.DONE

                return Message(
                    content=f"Action Completed: {limit_msg}",
                    role=self.profile,
                )

            self.pipeline_step = ControllerPipelineStep.PLAN

            return Message(
                content="EXECUTION_FAILURE_RETRY: ACTION_VALIDATION_ERROR",
                role=self.profile,
            )

        self.push_ready = True
        self._validated_actions_for_push = parsed_actions
        self.pipeline_step = ControllerPipelineStep.PLAN
        self.last_execution_log = f"PLAN_VALIDATED: attempt={self.plan_attempt}"

        self._append_action_cache(
            attempt=self.plan_attempt,
            stage="plan",
            status="validated",
            decision_raw=self.last_decision_raw,
            parsed_actions=self.last_parsed_actions,
            validation_errors=[],
            push_error=None,
            sandbox_actions=[
                item.model_dump(exclude_none=True) for item in parsed_actions
            ],
        )

        self._log_plan_summary(
            attempt=self.plan_attempt,
            route_step=route_step,
            route_reason=route_reason,
            status="validated",
            action_count=len(parsed_actions),
            error_count=0,
        )

        return Message(
            content="EXECUTION_FAILURE_RETRY: PLAN_VALIDATED",
            role=self.profile,
        )

    async def _act_push(self) -> Message:
        """Run PUSH stage and execute validated actions."""
        if not self.push_ready or not self._validated_actions_for_push:
            detail = "Push blocked: validation gate did not pass."
            self.recent_errors = [detail]
            self.last_execution_log = detail

            self.error_history.append(
                {
                    "kind": "push_permission_denied",
                    "detail": detail,
                    "ts_ms": int(time.time() * 1000),
                }
            )

            self._append_action_cache(
                attempt=self.plan_attempt,
                stage="push",
                status="push_failed",
                decision_raw=self.last_decision_raw,
                parsed_actions=self.last_parsed_actions,
                validation_errors=[],
                push_error=detail,
                sandbox_actions=[
                    item.model_dump(exclude_none=True)
                    for item in self._validated_actions_for_push
                ],
            )

            self.pipeline_step = ControllerPipelineStep.PLAN

            return Message(
                content="EXECUTION_FAILURE_RETRY: PUSH_PERMISSION_DENIED",
                role=self.profile,
            )

        push_tool = PushTool(llm=self.llm)

        exec_msg = await push_tool.run(
            role_name=self.name,
            graph_tool=self.graph_tool,
            actions=self._validated_actions_for_push,
        )

        self.last_execution_log = exec_msg

        if "REJECT" in exec_msg or "Error" in exec_msg:
            logger.warning(f"[{self.name}] Action Execution Failed: {exec_msg}")

            detailed_error = (
                f"[系统报错]: {exec_msg}\n[你的原始输出]: \n{self.last_decision_raw}\n"
            )

            self.recent_errors = [detailed_error]

            self.error_history.append(
                {
                    "kind": "execution_error",
                    "detail": detailed_error,
                    "ts_ms": int(time.time() * 1000),
                }
            )

            self._append_action_cache(
                attempt=self.plan_attempt,
                stage="push",
                status="push_failed",
                decision_raw=self.last_decision_raw,
                parsed_actions=self.last_parsed_actions,
                validation_errors=[],
                push_error=exec_msg,
                sandbox_actions=[
                    item.model_dump(exclude_none=True)
                    for item in self._validated_actions_for_push
                ],
            )

            self.push_ready = False
            self._validated_actions_for_push = []

            if self.plan_attempt >= self.max_plan_attempts:
                limit_msg = (
                    "PLAN_RETRY_LIMIT_EXCEEDED: push failed after max plan attempts."
                )

                self.pipeline_step = ControllerPipelineStep.DONE

                return Message(
                    content=f"Action Completed: {limit_msg}",
                    role=self.profile,
                )

            self.pipeline_step = ControllerPipelineStep.PLAN

            return Message(
                content=f"EXECUTION_FAILURE_RETRY: {exec_msg[:50]}...",
                role=self.profile,
            )

        self.last_executed_actions = list(self._validated_actions_for_push)

        self._append_action_cache(
            attempt=self.plan_attempt,
            stage="push",
            status="pushed",
            decision_raw=self.last_decision_raw,
            parsed_actions=self.last_parsed_actions,
            validation_errors=[],
            push_error=None,
            sandbox_actions=[
                item.model_dump(exclude_none=True)
                for item in self.last_executed_actions
            ],
        )

        self.pipeline_step = ControllerPipelineStep.DONE
        self.push_ready = False
        self._validated_actions_for_push = []
        return Message(content=f"Action Completed: {exec_msg}", role=self.profile)

    def _generate_and_memorize_summary(self):
        """Create a summary of worker findings and adds it to short-term memory."""
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

        try:
            self.rc.memory.add(Message(content=summary_text, role="System"))

        except (AttributeError, RuntimeError, TypeError) as e:
            logger.error(f"[{self.name}] Failed to memorize summary: {e}")

    def _parse_requirement(self, payload: Any) -> ResourceRequirement:
        """Safely parse a resource requirement payload from tool arguments.

        Args:
            payload: Parsed function arguments, typically a dict.

        Returns:
            A `ResourceRequirement` object.

        Raises:
            ValueError: If payload cannot be validated as ResourceRequirement.
        """
        try:
            if isinstance(payload, str):
                return ResourceRequirement.model_validate_json(payload)

            return ResourceRequirement.model_validate(payload)

        except (TypeError, ValueError, ValidationError) as e:
            raise ValueError(f"Invalid resource requirement payload: {e}") from e

    def _record_tool_call_error(self, stage: str, error: Exception) -> str:
        """Persist a tool-call routing error and update retry context.

        Args:
            stage: Pipeline stage where the failure happened.
            error: Exception instance raised during tool-call processing.

        Returns:
            Normalized error detail string stored in controller state.
        """
        detail = f"[tool_call_error][{stage}] {error}"
        self.recent_errors = [detail]

        self.error_history.append(
            {
                "kind": "tool_call_error",
                "detail": detail,
                "stage": stage,
                "ts_ms": int(time.time() * 1000),
            }
        )

        return detail

    def _build_plan_feedback(self) -> str:
        """Build retry feedback text from recent errors and action cache."""
        blocks: List[str] = []

        if self.recent_errors:
            blocks.append(
                "【上一轮执行反馈】\n"
                "请根据以下报错信息修正图谱操作函数参数（检查 ID 是否存在、类型是否匹配），"
                "并同步修正对应动作的 metadata.reason_brief：\n"
                + "\n".join(self.recent_errors)
            )

        if self.action_cache:
            latest_cache = self.action_cache[-3:]

            blocks.append(
                "【动作 Cache（最近3条）】\n"
                + json.dumps(latest_cache, ensure_ascii=False, indent=2)
            )

        return "\n\n".join(blocks)

    def _build_action_cache_context(self, limit: int = 5) -> str:
        """Build compact JSON context from recent action-cache records."""
        if not self.action_cache:
            return "[]"

        safe_limit = max(1, int(limit))
        recent = self.action_cache[-safe_limit:]
        return json.dumps(recent, ensure_ascii=False, indent=2)

    def _log_plan_summary(
        self,
        *,
        attempt: int,
        route_step: str,
        route_reason: str,
        status: str,
        action_count: int,
        error_count: int,
    ):
        """Emit one compact plan-summary log for each meaningful plan outcome."""
        route_text = (route_reason or "").strip() or "n/a"

        logger.info(
            "[{}] PLAN_SUMMARY attempt={} route={} status={} actions={} errors={} reason={}",
            self.name,
            int(attempt),
            str(route_step),
            str(status),
            int(action_count),
            int(error_count),
            route_text,
        )

    def _append_action_cache(
        self,
        *,
        attempt: int,
        stage: str,
        status: str,
        decision_raw: str,
        parsed_actions: List[Dict[str, Any]],
        validation_errors: List[str],
        push_error: str | None,
        sandbox_actions: List[Dict[str, Any]],
    ):
        """Append one action cache record for plan/push stage."""
        self.action_cache.append(
            {
                "attempt": int(attempt),
                "stage": str(stage),
                "status": str(status),
                "decision_raw": str(decision_raw or ""),
                "parsed_actions": [
                    dict(item) for item in parsed_actions if isinstance(item, dict)
                ],
                "validation_errors": [str(item) for item in validation_errors],
                "push_error": None if push_error is None else str(push_error),
                "sandbox_actions": [
                    dict(item) for item in sandbox_actions if isinstance(item, dict)
                ],
                "ts_ms": int(time.time() * 1000),
            }
        )

    def _truncate(self, text: str, length: int) -> str:
        """Truncate a string to a maximum length, adding an ellipsis."""
        if not text:
            return ""

        text = text.strip()

        if len(text) > length:
            return text[:length] + "..."

        return text
