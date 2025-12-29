from typing import List

from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message

from actions.controller_actions import PlanTactics, VerifyAndDecide
from mas.action_parser import parse_agent_action_output
from mas.schema import (
    AgentAction,
    ControllerIntent,
    TargetRole,
    WorkerInstruction,
    WorkerReport,
    WorkerReportStatus,
)
from tools.graph_tool import GraphTool
from tools.initializer import AgentPersona
from tools.json_utils import extract_json_from_text


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
        self.set_actions([PlanTactics, VerifyAndDecide])
        self.round_count = 0

    async def _act(self) -> Message:
        logger.info(f"{self.name} is thinking...")
        memories = self.get_memories(k=1)

        if not memories or "SYSTEM_START" in memories[0].content:
            return await self._plan_phase()

        last_msg = memories[0]

        if "SYSTEM_FEEDBACK" in str(last_msg.content):
            feedback = str(last_msg.content)
            logger.info(f"Controller received SYSTEM_FEEDBACK: {feedback[:50]}...")
            return await self._plan_phase(feedback=feedback)

        if "REPORT" in str(last_msg.content):
            try:
                report = WorkerReport.model_validate_json(last_msg.content)

                if report.message_type == "REPORT":
                    return await self._decide_phase(report)

            except Exception as e:
                logger.warning(
                    f"Invalid report format: {e}. Raw content: {last_msg.content}"
                )

                return await self._plan_phase(
                    feedback=f"接收到的报告格式无效，请严格按照 WorkerReport 的 JSON 格式输出。错误信息: {e}"
                )

        logger.warning(f"Unexpected message type in _act: {last_msg.content}")
        return await self._plan_phase()

    async def _plan_phase(self, feedback: str = "") -> Message:
        context = self.graph_tool.current_graph.latest_context
        feedback_text = ""

        if feedback:
            feedback_text = f"【⚠️ 上次尝试失败反馈】:\n{feedback}\n请分析失败原因，并重新规划。如果是指令错误，请修正格式；如果是逻辑错误，请调整策略。"

        memories = self.get_memories(k=5)
        history_text = "\n".join([f"[{m.role}]: {m.content}" for m in memories])
        action = PlanTactics(llm=self.llm)

        raw_intent = await action.run(
            self.name,
            self.persona,
            self.insights,
            context,
            feedback=feedback_text,
            history=history_text,
        )

        try:
            data = extract_json_from_text(raw_intent)
            intent = ControllerIntent.model_validate(data)

        except Exception as e:
            error_msg = f"Intent parsing failed for role {self.name}. Error: {e}. Raw LLM Output: '{raw_intent}'"
            logger.error(f"[{self.name}] {error_msg}. Preparing to re-plan...")
            return await self._plan_phase(feedback=error_msg)

        logger.info(
            f"[{self.name}] Planned Intent: {intent.target} -> {intent.content}"
        )

        if intent.target == TargetRole.SELF:
            fake_report = WorkerReport(
                status=WorkerReportStatus.FOUND,
                content=f"无需外部检索。战术思路: {intent.content}",
                max_score=1.0,
            )

            return await self._decide_phase(fake_report, feedback=feedback)

        else:
            instruction = WorkerInstruction(query=intent.content, graph_context=context)
            target_worker_name = "FactWorker"

            if intent.target == TargetRole.LAW_WORKER:
                target_worker_name = "LawWorker"

            return Message(
                content=instruction.to_json(),
                role=self.profile,
                cause_by=PlanTactics,
                send_to=target_worker_name,
            )

    async def _decide_phase(self, report: WorkerReport, feedback: str = "") -> Message:
        context = self.graph_tool.current_graph.latest_context

        if report.status == WorkerReportStatus.NOT_FOUND:
            logger.info("Worker found nothing. Re-planning...")
            return await self._plan_phase()

        if report.status == WorkerReportStatus.FOUND:
            action = VerifyAndDecide(llm=self.llm)

            decision_raw_output = await action.run(
                self.name,
                report.content,
                context,
                self.persona.initial_strategy,
                feedback=feedback,
            )

            if decision_raw_output.strip().startswith("REJECT:"):
                reject_reason = decision_raw_output.split("REJECT:", 1)[1].strip()
                logger.info(f"Controller rejected advice: {reject_reason}")

                return await self._plan_phase(
                    feedback=f"上次的决策被拒绝，理由是: {reject_reason}。请重新评估并提出新方案。"
                )

            parsed_actions = parse_agent_action_output(decision_raw_output)

            if isinstance(parsed_actions, List) and all(
                isinstance(a, AgentAction) for a in parsed_actions
            ):
                exec_result_message = await self.graph_tool.process_intent(
                    self.name, parsed_actions
                )

                if "REJECT:" in exec_result_message:
                    logger.warning(f"GraphTool rejected intent: {exec_result_message}")

                    return await self._plan_phase(
                        feedback=f"GraphTool 拒绝执行您的意图：{exec_result_message}。请根据反馈调整。"
                    )

                return Message(
                    content=f"动作已完成: {exec_result_message}", role=self.profile
                )

            else:
                logger.warning(f"Controller decision parsing failed: {parsed_actions}")

                return await self._plan_phase(
                    feedback=f"您的输出格式不正确。请严格按照 AgentAction 的 JSON 格式输出。错误信息：{parsed_actions}"
                )

        return Message(content="ERROR: 无法处理决策阶段。", role=self.profile)
