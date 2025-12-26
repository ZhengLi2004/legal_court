from typing import Tuple
from mas.legal_system import LegalSystem
from mas.common import ShadowGraph, NodeType, EdgeType # Added NodeType, EdgeType for execute_action
from mas.llm import LLMCallable, Message
from mas.schema import AgentAction, AgentActionType # Import AgentActionType too

class GraphTool:
    def __init__(self, legal_system: LegalSystem, llm: LLMCallable):
        self.system = legal_system
        self.llm = llm
        self.current_graph: ShadowGraph = None

    def set_current_graph(self, graph: ShadowGraph): self.current_graph = graph

    async def process_intent(self, agent_id: str, action: AgentAction) -> str:
        try:
            if not self.current_graph: return "REJECT: 当前图谱上下文未设置。"

            logs = self.system.execute_action(
                graph=self.current_graph,
                agent_id=agent_id,
                action=action
            )

            error_logs = [l for l in logs if "Error" in l or "Failed" in l or "Reject" in l]
            if error_logs: return f"REJECT: 执行操作时发生错误: {'; '.join(error_logs)}"
            return f"EXECUTED: 操作成功。\n日志:\n" + "\n".join(logs)

        except Exception as e: return f"REJECT: GraphTool 处理意图时发生异常: {str(e)}"