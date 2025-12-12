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
        insights: str = ""
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

    async def run_turn(self, graph: ShadowGraph) -> str:
        logger.info(f"\n{'='*10} Team {self.side} Turn Start {'='*10}")
        self.graph_tool.set_current_graph(graph)
        self.controller.rc.memory.add(Message(content="SYSTEM_START", role="System"))
        loop_count = 0

        while loop_count < self.max_micro_loops:
            loop_count += 1
            logger.info(f"--- Micro Loop {loop_count}/{self.max_micro_loops} ---")
            ctrl_msg = await self.controller._act()
            content = str(ctrl_msg.content)

            if "Action Completed" in content or "Executed:" in content:
                logger.info(f"Turn Finished: {content}")
                return content

            if "query" in content and "graph_context" in content:
                target_worker = self.fact_worker    # 默认
                if "LawWorker" in ctrl_msg.send_to: target_worker = self.law_worker
                elif "FactWorker" in ctrl_msg.send_to: target_worker = self.fact_worker
                logger.info(f"Controller -> {target_worker.name}")
                target_worker.rc.memory.add(ctrl_msg)
                worker_msg = await target_worker._act()
                logger.info(f"{target_worker.name} -> Controller")
                self.controller.rc.memory.add(worker_msg)
                continue

            logger.warning(f"Controller produced unexpected output: {content}")
            break

        return f"Turn ended without action (Max loops {self.max_micro_loops} reached)."