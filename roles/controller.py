from enum import Enum, auto
from typing import List

from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message

from actions.controller_actions import (
    AssessFactNeeds,
    AssessLawNeeds,
    AssessRecallNeeds,
    VerifyAndDecide,
)
from mas.action_parser import parse_agent_action_output
from mas.schema import (
    AgentAction,
    ResourceRequirement,
    WorkerInstruction,
    WorkerReport,
)
from tools.graph_tool import GraphTool
from tools.initializer import AgentPersona
from tools.json_utils import extract_json_from_text


class ControllerPipelineStep(Enum):
    IDLE = auto()
    FACT_CHECK = auto()
    WAIT_FACT_REPORT = auto()
    LAW_CHECK = auto()
    WAIT_LAW_REPORT = auto()
    RECALL_CHECK = auto()
    WAIT_RECALL_REPORT = auto()
    DECIDE = auto()
    DONE = auto()


class ArgumentController(Role):
    name: str = "Controller"
    profile: str = "Lead Lawyer"

    def __init__(
        self,
        name: str,
        persona: AgentPersona,
        graph_tool: GraphTool,
        insights: str = "",
    ):
        super().__init__(name=name, profile="Lead Lawyer")
        self.persona = persona
        self.graph_tool = graph_tool
        self.insights = insights

        self.set_actions(
            [AssessFactNeeds, AssessLawNeeds, AssessRecallNeeds, VerifyAndDecide]
        )

        self.pipeline_step = ControllerPipelineStep.IDLE
        self.accumulated_reports: List[str] = []

    async def _act(self) -> Message:
        memories = self.get_memories(k=5)
        last_msg = memories[-1] if memories else None
        last_content = str(last_msg.content) if last_msg else ""
        last_role = str(last_msg.role) if last_msg else ""

        logger.debug(
            f"[{self.name}] Latest Memory: [{last_role}] {last_content[:30]}..."
        )

        if last_role == "System" and "SYSTEM_START" in last_content:
            if self.pipeline_step in [
                ControllerPipelineStep.IDLE,
                ControllerPipelineStep.DONE,
            ]:
                logger.info(f"[{self.name}] Received START signal. Resetting Pipeline.")
                self.pipeline_step = ControllerPipelineStep.FACT_CHECK
                self.accumulated_reports = []

            elif self.pipeline_step != ControllerPipelineStep.FACT_CHECK:
                logger.info(f"[{self.name}] Force Restart triggered.")
                self.pipeline_step = ControllerPipelineStep.FACT_CHECK
                self.accumulated_reports = []

        if self.pipeline_step == ControllerPipelineStep.WAIT_FACT_REPORT:
            if "REPORT" in last_content or "{" in last_content:
                self._handle_worker_report(last_msg, "FactWorker")
                logger.info(f"[{self.name}] Received Fact Report. Moving to LAW_CHECK.")
                self.pipeline_step = ControllerPipelineStep.LAW_CHECK

            else:
                logger.debug(
                    f"[{self.name}] Waiting for Fact Report... (Ignored: {last_content[:20]})"
                )

        elif self.pipeline_step == ControllerPipelineStep.WAIT_LAW_REPORT:
            if "REPORT" in last_content or "{" in last_content:
                self._handle_worker_report(last_msg, "LawWorker")

                logger.info(
                    f"[{self.name}] Received Law Report. Moving to RECALL_CHECK."
                )

                self.pipeline_step = ControllerPipelineStep.RECALL_CHECK

            else:
                logger.debug(
                    f"[{self.name}] Waiting for Law Report... (Ignored: {last_content[:20]})"
                )

        elif self.pipeline_step == ControllerPipelineStep.WAIT_RECALL_REPORT:
            if "REPORT" in last_content or "{" in last_content:
                self._handle_worker_report(last_msg, "RecallWorker")
                logger.info(f"[{self.name}] Received Recall Report. Moving to DECIDE.")
                self.pipeline_step = ControllerPipelineStep.DECIDE

            else:
                logger.debug(
                    f"[{self.name}] Waiting for Recall Report... (Ignored: {last_content[:20]})"
                )

        while True:
            graph_context = self.graph_tool.current_graph.latest_context

            if self.pipeline_step == ControllerPipelineStep.FACT_CHECK:
                logger.info(f"[{self.name}] Assessing Fact Needs...")
                action = AssessFactNeeds(llm=self.llm)
                raw_resp = await action.run(self.name, self.persona, graph_context)
                req = self._parse_requirement(raw_resp)

                if req.need:
                    logger.info(f"[{self.name}] Fact Check Needed: {req.query}")
                    self.pipeline_step = ControllerPipelineStep.WAIT_FACT_REPORT

                    return self._create_instruction(
                        req.query, graph_context, "FactWorker"
                    )

                else:
                    logger.info(f"[{self.name}] Fact Check Skipped: {req.reasoning}")

                    self.pipeline_step = ControllerPipelineStep.LAW_CHECK

                    continue

            elif self.pipeline_step == ControllerPipelineStep.LAW_CHECK:
                logger.info(f"[{self.name}] Assessing Law Needs...")
                action = AssessLawNeeds(llm=self.llm)
                raw_resp = await action.run(self.name, self.persona, graph_context)
                req = self._parse_requirement(raw_resp)

                if req.need:
                    logger.info(f"[{self.name}] Law Check Needed: {req.query}")
                    self.pipeline_step = ControllerPipelineStep.WAIT_LAW_REPORT

                    return self._create_instruction(
                        req.query, graph_context, "LawWorker"
                    )

                else:
                    logger.info(f"[{self.name}] Law Check Skipped: {req.reasoning}")

                    self.pipeline_step = ControllerPipelineStep.RECALL_CHECK

                    continue

            elif self.pipeline_step == ControllerPipelineStep.RECALL_CHECK:
                logger.info(f"[{self.name}] Assessing Recall Needs...")
                action = AssessRecallNeeds(llm=self.llm)
                raw_resp = await action.run(self.name, self.persona, graph_context)
                req = self._parse_requirement(raw_resp)

                if req.need:
                    logger.info(f"[{self.name}] Recall Check Needed: {req.query}")
                    self.pipeline_step = ControllerPipelineStep.WAIT_RECALL_REPORT

                    return self._create_instruction(
                        req.query, graph_context, "RecallWorker"
                    )

                else:
                    logger.info(f"[{self.name}] Recall Check Skipped: {req.reasoning}")
                    self.pipeline_step = ControllerPipelineStep.DECIDE  # -> DECIDE
                    continue

            elif self.pipeline_step == ControllerPipelineStep.DECIDE:
                logger.info(f"[{self.name}] Finalizing Decision...")
                feedback = ""

                if "SYSTEM_FEEDBACK" in last_content:
                    feedback = last_content

                    logger.warning(
                        f"[{self.name}] Retrying with feedback: {feedback[:50]}..."
                    )

                combined_advice = (
                    "\n".join(self.accumulated_reports)
                    if self.accumulated_reports
                    else "（本轮无额外检索报告）"
                )

                action = VerifyAndDecide(llm=self.llm)

                decision_raw = await action.run(
                    self.name,
                    combined_advice,
                    graph_context,
                    self.persona.initial_strategy,
                    feedback=feedback,
                )

                parsed_actions = parse_agent_action_output(decision_raw)

                if isinstance(parsed_actions, list) and all(
                    isinstance(a, AgentAction) for a in parsed_actions
                ):
                    exec_msg = await self.graph_tool.process_intent(
                        self.name, parsed_actions
                    )

                    if "REJECT" in exec_msg or "Error" in exec_msg:
                        logger.warning(
                            f"[{self.name}] Action Execution Failed: {exec_msg}"
                        )

                        return Message(
                            content=f"EXECUTION_FAILURE: {exec_msg}", role=self.profile
                        )

                    else:
                        self.pipeline_step = ControllerPipelineStep.DONE

                        return Message(
                            content=f"Action Completed: {exec_msg}", role=self.profile
                        )

                else:
                    error_msg = f"JSON Parsing Failed: {parsed_actions}"
                    logger.warning(f"[{self.name}] {error_msg}")

                    return Message(
                        content=f"EXECUTION_FAILURE: {error_msg}", role=self.profile
                    )

            elif self.pipeline_step == ControllerPipelineStep.DONE:
                return Message(
                    content="Action Completed: Pipeline Finished.", role=self.profile
                )

            elif self.pipeline_step in [
                ControllerPipelineStep.WAIT_FACT_REPORT,
                ControllerPipelineStep.WAIT_LAW_REPORT,
                ControllerPipelineStep.WAIT_RECALL_REPORT,
            ]:
                return Message(
                    content=f"Waiting for Report... (Current Step: {self.pipeline_step})",
                    role=self.profile,
                )

            else:
                return Message(content="Controller IDLE", role=self.profile)

    def _handle_worker_report(self, msg: Message, worker_name: str):
        try:
            content = str(msg.content)

            if "{" in content and "}" in content:
                report = WorkerReport.model_validate_json(content)
                summary = f"【{worker_name} Report】({report.status}): {report.content}"

            else:
                summary = f"【{worker_name} Raw Msg】: {content}"

            self.accumulated_reports.append(summary)
            logger.info(f"[{self.name}] Stored report from {worker_name}")

        except Exception:
            self.accumulated_reports.append(
                f"【{worker_name} Error】: {str(msg.content)}"
            )

    def _parse_requirement(self, raw_text: str) -> ResourceRequirement:
        try:
            data = extract_json_from_text(raw_text)
            return ResourceRequirement.model_validate(data)

        except Exception as e:
            logger.warning(f"[{self.name}] Parse Error: {e}. Defaulting to NEED=False.")
            return ResourceRequirement(need=False, reasoning=f"Parse Error: {e}")

    def _create_instruction(self, query: str, context: str, target: str) -> Message:
        instruction = WorkerInstruction(query=query, graph_context=context)
        return Message(content=instruction.to_json(), role=self.profile, send_to=target)
