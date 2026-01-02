import asyncio
import json
from enum import Enum, auto
from typing import Dict, List

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
        self.investigation_buffer: Dict[str, str] = {}
        self.latest_summary: str = ""
        self.recent_errors: List[str] = []

    def reset_turn_state(self):
        logger.info(f"[{self.name}] Resetting turn state (Buffer & Errors cleared).")
        self.pipeline_step = ControllerPipelineStep.IDLE
        self.investigation_buffer = {}
        self.latest_summary = ""
        self.recent_errors = []

    def ingest_results(self, results_list: List[Dict[str, str]]):
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

            if req_fact.need:
                inst = WorkerInstruction(
                    query=req_fact.query, graph_context=graph_context
                )

                instructions.append(
                    {"target": "FactWorker", "instruction": inst.to_json()}
                )

                self.investigation_buffer["Fact"] = (
                    f"⏳ [正在检索事实]: {req_fact.query}"
                )

            else:
                reasoning = self._truncate(req_fact.reasoning, 500)
                self.investigation_buffer["Fact"] = f"✅ [无需事实检索]: {reasoning}"

            if req_law.need:
                inst = WorkerInstruction(
                    query=req_law.query, graph_context=graph_context
                )

                instructions.append(
                    {"target": "LawWorker", "instruction": inst.to_json()}
                )

                self.investigation_buffer["Law"] = f"⏳ [正在检索法条]: {req_law.query}"

            else:
                reasoning = self._truncate(req_law.reasoning, 500)
                self.investigation_buffer["Law"] = f"✅ [无法律检索]: {reasoning}"

            if req_recall.need:
                inst = WorkerInstruction(
                    query=req_recall.query, graph_context=graph_context
                )

                instructions.append(
                    {"target": "RecallWorker", "instruction": inst.to_json()}
                )

                self.investigation_buffer["Recall"] = (
                    f"⏳ [正在检索案例]: {req_recall.query}"
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

            decision_raw = await action.run(
                self.name,
                current_advice,
                graph_context,
                self.persona.initial_strategy,
                feedback=feedback_section,
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
                    error_entry = f"Attempt #{len(self.recent_errors) + 1}: {exec_msg}"
                    self.recent_errors.append(error_entry)

                    if len(self.recent_errors) > 5:
                        self.recent_errors.pop(0)

                    return Message(
                        content=f"EXECUTION_FAILURE_RETRY: {exec_msg[:50]}...",
                        role=self.profile,
                    )

                else:
                    self.pipeline_step = ControllerPipelineStep.DONE

                    return Message(
                        content=f"Action Completed: {exec_msg}", role=self.profile
                    )

            else:
                error_msg = f"JSON Parsing Failed: {parsed_actions}"
                logger.warning(f"[{self.name}] {error_msg}")

                self.recent_errors.append(
                    f"Attempt #{len(self.recent_errors) + 1}: {error_msg}"
                )

                if len(self.recent_errors) > 5:
                    self.recent_errors.pop(0)

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
        try:
            data = extract_json_from_text(raw_text)
            return ResourceRequirement.model_validate(data)

        except Exception as e:
            logger.warning(f"[{self.name}] Parse Error: {e}. Defaulting to NEED=False.")
            return ResourceRequirement(need=False, reasoning=f"Parse Error: {e}")

    def _truncate(self, text: str, length: int) -> str:
        if not text:
            return ""

        text = text.strip()

        if len(text) > length:
            return text[:length] + "..."

        return text
