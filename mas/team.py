from metagpt.schema import Message
from metagpt.logs import logger
from roles.controller import ArgumentController
from roles.worker import FactWorker, LawWorker
from tools.graph_tool import GraphTool
from tools.fact_es_tool import FactEsTool
from tools.law_es_tool import LawEsTool
from tools.initializer import AgentPersona
from .common import ShadowGraph
from .llm import GPTChat
from mas.schema import AgentAction
from mas.action_parser import parse_agent_action_output

class DebateTeam:
    def __init__(
        self,
        side: str,
        persona: AgentPersona,
        graph_tool: GraphTool,
        fact_es: FactEsTool,
        law_es: LawEsTool,
        llm: GPTChat,
        insights: str = "",
        verbose: bool = False
    ):
        self.side = side
        self.persona = persona
        self.graph_tool = graph_tool

        self.controller = ArgumentController(
            name=f"{side}_Controller",
            persona=persona,
            graph_tool=graph_tool,
            insights=insights
        )

        self.controller.llm = llm
        for action in self.controller.actions: action.llm = llm

        self.fact_worker = FactWorker(
            name=f"{side}_FactWorker",
            es_tool=fact_es,
            llm=llm
        )

        self.law_worker = LawWorker(
            name=f"{side}_LawWorker",
            es_tool=law_es,
            llm=llm
        )

        self.max_micro_loops = 3
        self.verbose = verbose

    async def run_turn(self, graph: ShadowGraph) -> str:
        logger.info(f"\n{'='*10} Team {self.side} Turn Start {'='*10}")
        self.graph_tool.set_current_graph(graph)
        self.controller.rc.memory.add(Message(content="SYSTEM_START", role="System"))
        transcript = []
        loop_count = 0
        final_result = None

        while loop_count < self.max_micro_loops:
            loop_count += 1
            logger.info(f"--- Micro Loop {loop_count}/{self.max_micro_loops} ---")
            ctrl_msg = await self.controller._act()
            
            if self.verbose:
                transcript.append({
                    "from": self.controller.name,
                    "to": ctrl_msg.send_to or "GraphTool", # Assuming GraphTool is implied if no send_to
                    "content": ctrl_msg.content
                })

            content = str(ctrl_msg.content)

            if "Action Completed" in content or "EXECUTED:" in content:
                if "ERROR" in content or "REJECT" in content:
                    logger.warning(f"Controller action failed: {content}")

                    feedback_msg = Message(
                        content=f"SYSTEM_FEEDBACK: 上次操作失败。{content}。请重新规划。",
                        role="System"
                    )
                    self.controller.rc.memory.add(feedback_msg)
                    if self.verbose: transcript.append({"from": "System", "to": self.controller.name, "content": feedback_msg.content})
                    continue
                
                else:
                    final_result = content
                    break
            
            elif "query" in content and "graph_context" in content and ctrl_msg.send_to:
                target_worker = self.fact_worker    # Default worker
                if "LawWorker" in ctrl_msg.send_to: target_worker = self.law_worker
                logger.info(f"Routing to {target_worker.name}")
                target_worker.rc.memory.add(ctrl_msg)
                worker_msg = await target_worker._act()
                
                if self.verbose:
                    transcript.append({
                        "from": target_worker.name,
                        "to": self.controller.name,
                        "content": worker_msg.content
                    })
                
                self.controller.rc.memory.add(worker_msg)
                continue
            
            else:
                final_result = f"Controller produced unroutable output: {content}"
                logger.warning(final_result)
                break

        if final_result is None:
            logger.warning(f"Team {self.side} loop exhausted. Entering FORCE ACTION phase.")
            max_forced_attempts = 3
            forced_count = 0
            last_error = ""

            while forced_count < max_forced_attempts:
                forced_count += 1

                force_prompt = f"""
                你已经消耗了所有思考轮次。现在进入【强制行动阶段】。
                
                【当前战局】:
                {graph.latest_context}
                
                【重要】你的任务是生成一个图谱操作意图。
                **请严格按照以下 AgentAction 的 JSON 格式输出，不包含任何额外文本或代码块标记**:
                ```json
                {{
                    "action_type": "add_claim" | "add_fact" | "cite_law" | "rebut_claim",
                    "content": "具体的行动内容",
                    "target_id": "可选，当为 cite_law 或 rebut_claim 时提供，目标节点的 UUID",
                    "source_id": "可选，当为 rebut_claim 时提供，源节点的 UUID",
                    "relation_type": "可选，当为 cite_law 或 rebut_claim 时提供，'SUPPORT' 或 'CONFLICT'"
                }}
                ```
                请确保 `action_type` 字段的值是预定义枚举中的一个。
                请确保 `content` 字段包含详细的文本内容。
                当 `action_type` 是 `cite_law` 或 `rebut_claim` 时，`target_id` 和 `relation_type` 字段必须存在。
                当 `action_type` 是 `rebut_claim` 时，如果 `source_id` 为空，系统会自动基于 `content` 创建一个隐式主张作为反驳来源。
                """

                if last_error: force_prompt += f"\n\n【⚠️ 上次强制尝试失败反馈】:\n{last_error}\n请修正你的意图或格式。"

                intent_res = await self.controller.llm.aask(
                    f"你是{self.controller.name}。{force_prompt}"
                )

                logger.info(f"LLM Response for Forced Action (Attempt {forced_count}): {intent_res}")

                if self.verbose:
                    transcript.append({
                        "from": "System (Force)", 
                        "to": self.controller.name, 
                        "content": f"Attempt {forced_count}: {intent_res}"
                    })

                parsed_forced_action = parse_agent_action_output(intent_res)

                if isinstance(parsed_forced_action, AgentAction):
                    exec_result = await self.graph_tool.process_intent(self.controller.name, parsed_forced_action) # Pass AgentAction
                    if "REJECT" in exec_result or "ERROR" in exec_result:
                        last_error = exec_result
                        logger.warning(f"Forced action failed ({forced_count}/{max_forced_attempts}): {last_error}")
                        continue

                    else:
                        final_result = f"Forced Action Completed: {exec_result}"
                        break
                
                else:
                    last_error = f"REJECT: 强制行动意图解析失败。LLM输出不是有效的 AgentAction JSON。错误信息: {parsed_forced_action}"
                    logger.warning(f"Forced action parsing failed ({forced_count}/{max_forced_attempts}): {last_error}")
                    continue
            
            if final_result is None: final_result = f"Turn Failed: Controller unable to produce valid action after {max_forced_attempts} forced attempts."

        return {
            "summary": final_result,
            "transcript": transcript
        }