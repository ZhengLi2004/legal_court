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
        final_result = f"Turn ended without action (Max loops {self.max_micro_loops} reached)."

        while loop_count < self.max_micro_loops:
            loop_count += 1
            logger.info(f"--- Micro Loop {loop_count}/{self.max_micro_loops} ---")
            ctrl_msg = await self.controller._act()
            
            if self.verbose:
                transcript.append({
                    "from": self.controller.name,
                    "to": ctrl_msg.send_to or "GraphTool",
                    "content": ctrl_msg.content
                })

            content = str(ctrl_msg.content)

            if "Action Completed" in content or "Executed:" in content:
                if "ERROR" in content or "REJECT" in content:
                    logger.warning(f"Controller action failed: {content}")

                    feedback_msg = Message(
                        content=f"SYSTEM_FEEDBACK: 上次操作失败。{content}。请重新规划。",
                        role="System"
                    )

                    self.controller.rc.memory.add(feedback_msg)

                    if self.verbose:
                        transcript.append({
                            "from": "System",
                            "to": self.controller.name,
                            "content": feedback_msg.content
                        })

                    continue

            if "query" in content and "graph_context" in content:
                target_worker = self.fact_worker    # 默认
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

            final_result = f"Controller produced unroutable output: {content}"
            break

        return {
            "summary": final_result,
            "transcript": transcript
        }