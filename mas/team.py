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
                **请严格按照以下 JSON 格式输出，不包含任何额外文本或代码块标记**:
                你的输出必须是完整的、合法的 JSON 字符串，且必须是包含一个或多个 AgentAction 对象的 JSON 数组。数组中的每个对象必须符合 AgentAction 模型的定义。

                AgentAction模型定义（字段说明）：
                - action_type (enum): 动作类型，必须是 'add_claim', 'cite_law', 'rebut_claim' 之一。
                - content (string): 动作的具体内容，例如主张的详细文本，法条查询的关键词。
                - target_id (string, optional): 动作的目标节点 ID。例如，反驳某个主张时，这是被反驳的主张的 ID。
                - source_id (string, optional): 动作的来源节点 ID。例如，一个主张支持另一个主张，这是支持方的 ID。
                - relation_type (enum, optional): **【强制】** 仅当动作涉及创建关系（即 `action_type` 为 `cite_law` 或 `rebut_claim`）时使用，**必须且只能是 'SUPPORT' 或 'CONFLICT'**。**严禁使用其他任何值！**
                    当 `action_type` 为 `add_claim` 时，`relation_type` 必须为 `null`。

                【输出示例】:
                ```json
                [
                    {{
                        "action_type": "add_claim",
                        "content": "被告无证据证明其已履行还款义务",
                        "target_id": null,
                        "source_id": null,
                        "relation_type": null
                    }},
                    {{
                        "action_type": "cite_law",
                        "content": "中华人民共和国合同法 第二百零六条",
                        "target_id": "CLAIM_12345678",
                        "source_id": null,
                        "relation_type": "SUPPORT"
                    }}
                ]
                ```
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

                parsed_forced_actions = parse_agent_action_output(intent_res)

                if isinstance(parsed_forced_actions, list) and all(isinstance(a, AgentAction) for a in parsed_forced_actions):
                    exec_result = await self.graph_tool.process_intent(self.controller.name, parsed_forced_actions) # Pass List[AgentAction]
                    
                    if "REJECT" in exec_result or "ERROR" in exec_result:
                        last_error = exec_result
                        logger.warning(f"Forced action failed ({forced_count}/{max_forced_attempts}): {last_error}")
                        continue

                    else:
                        final_result = f"Forced Action Completed: {exec_result}"
                        break
                
                else:
                    last_error = f"REJECT: 强制行动意图解析失败。LLM输出不是有效的 AgentAction JSON。错误信息: {parsed_forced_actions}"
                    logger.warning(f"Forced action parsing failed ({forced_count}/{max_forced_attempts}): {last_error}")
                    continue
            
            if final_result is None: final_result = f"Turn Failed: Controller unable to produce valid action after {max_forced_attempts} forced attempts."

        return {
            "summary": final_result,
            "transcript": transcript
        }