import asyncio
import json
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
    ASSESS_NEEDS = auto()
    WAIT_FOR_WORKERS = auto()
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
            f"[{self.name}] State: {self.pipeline_step.name} | Latest Memory: [{last_role}] {last_content[:30]}..."
        )

        if last_role == "System" and "SYSTEM_START" in last_content:
            if self.pipeline_step in [
                ControllerPipelineStep.IDLE,
                ControllerPipelineStep.DONE,
            ]:
                logger.info(
                    f"[{self.name}] Received START signal. Starting Assessment."
                )

                self.pipeline_step = ControllerPipelineStep.ASSESS_NEEDS
                self.accumulated_reports = []

            elif self.pipeline_step != ControllerPipelineStep.ASSESS_NEEDS:
                logger.info(f"[{self.name}] Force Restart triggered.")
                self.pipeline_step = ControllerPipelineStep.ASSESS_NEEDS
                self.accumulated_reports = []

        if self.pipeline_step == ControllerPipelineStep.WAIT_FOR_WORKERS:
            completed = False

            for msg in reversed(memories):
                if "WORKERS_COMPLETED" in str(msg.content):
                    completed = True
                    break

            for msg in memories:
                if "Worker" in str(msg.role):
                    self._handle_worker_report(msg, str(msg.role))

            if completed:
                logger.info(
                    f"[{self.name}] Parallel Workers Completed signal detected. Moving to DECIDE."
                )

                self.pipeline_step = ControllerPipelineStep.DECIDE

        graph_context = self.graph_tool.current_graph.latest_context

        if self.pipeline_step == ControllerPipelineStep.ASSESS_NEEDS:
            logger.info(
                f"[{self.name}] Parallelly Assessing Needs (Fact, Law, Recall)..."
            )

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

            if req_fact.need:
                logger.info(f"[{self.name}] + Need Fact: {req_fact.query}")

                inst = WorkerInstruction(
                    query=req_fact.query, graph_context=graph_context
                )

                instructions.append(
                    {"target": "FactWorker", "instruction": inst.to_json()}
                )

            else:
                logger.info(f"[{self.name}] - Skip Fact: {req_fact.reasoning[:30]}...")

            if req_law.need:
                logger.info(f"[{self.name}] + Need Law: {req_law.query}")

                inst = WorkerInstruction(
                    query=req_law.query, graph_context=graph_context
                )

                instructions.append(
                    {"target": "LawWorker", "instruction": inst.to_json()}
                )

            else:
                logger.info(f"[{self.name}] - Skip Law: {req_law.reasoning[:30]}...")

            if req_recall.need:
                logger.info(f"[{self.name}] + Need Recall: {req_recall.query}")

                inst = WorkerInstruction(
                    query=req_recall.query, graph_context=graph_context
                )

                instructions.append(
                    {"target": "RecallWorker", "instruction": inst.to_json()}
                )

            else:
                logger.info(
                    f"[{self.name}] - Skip Recall: {req_recall.reasoning[:30]}..."
                )

            if not instructions:
                logger.info(
                    f"[{self.name}] No workers needed. Jumping directly to DECIDE."
                )

                self.pipeline_step = ControllerPipelineStep.DECIDE

                return Message(
                    content=json.dumps({"batch_instructions": []}), role=self.profile
                )

            self.pipeline_step = ControllerPipelineStep.WAIT_FOR_WORKERS

            batch_msg_content = json.dumps({"batch_instructions": instructions})
            return Message(content=batch_msg_content, role=self.profile)

        elif self.pipeline_step == ControllerPipelineStep.WAIT_FOR_WORKERS:
            return Message(content="WAITING_FOR_PARALLEL_WORKERS", role=self.profile)

        elif self.pipeline_step == ControllerPipelineStep.DECIDE:
            logger.info(
                f"[{self.name}] Finalizing Decision with {len(self.accumulated_reports)} reports..."
            )

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
                    logger.warning(f"[{self.name}] Action Execution Failed: {exec_msg}")

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

        else:
            return Message(content="Controller IDLE", role=self.profile)

    def _handle_worker_report(self, msg: Message, worker_name: str):
        content = str(msg.content)

        try:
            if "{" in content and "}" in content:
                json_part = extract_json_from_text(content)

                if json_part and "status" in json_part:
                    report = WorkerReport.model_validate(json_part)

                    summary = (
                        f"【{worker_name} Report】({report.status}): {report.content}"
                    )

                else:
                    summary = f"【{worker_name} Raw Msg】: {content}"

            else:
                summary = f"【{worker_name} Raw Msg】: {content}"

            self.accumulated_reports.append(summary)
            logger.info(f"[{self.name}] Stored report from {worker_name}")

        except Exception:
            if f"【{worker_name} Raw Msg】" not in str(self.accumulated_reports):
                self.accumulated_reports.append(
                    f"【{worker_name} Raw Msg (ParseError)】: {str(msg.content)[:100]}..."
                )

    def _parse_requirement(self, raw_text: str) -> ResourceRequirement:
        try:
            data = extract_json_from_text(raw_text)
            return ResourceRequirement.model_validate(data)

        except Exception as e:
            logger.warning(f"[{self.name}] Parse Error: {e}. Defaulting to NEED=False.")
            return ResourceRequirement(need=False, reasoning=f"Parse Error: {e}")
