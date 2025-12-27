from typing import Tuple, Union, List
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

    async def process_intent(self, agent_id: str, actions: Union[AgentAction, List[AgentAction]]) -> str:
        try:
            if not self.current_graph: return "REJECT: 当前图谱上下文未设置。"
            
            actions_list: List[AgentAction] = []
            if isinstance(actions, AgentAction): actions_list.append(actions)
            elif isinstance(actions, list) and all(isinstance(a, AgentAction) for a in actions): actions_list = actions
            else: return "REJECT: 无效的动作格式。期望 AgentAction 对象或 AgentAction 对象的列表。"

            logs = self.system.execute_action(
                graph=self.current_graph,
                agent_id=agent_id,
                actions=actions_list
            )

            error_logs = [l for l in logs if "Error" in l or "Failed" in l or "Reject" in l]
            if error_logs: return f"REJECT: 执行操作时发生错误: {'; '.join(error_logs)}"
            return f"EXECUTED: 操作成功。\n日志:\n" + "\n".join(logs)

        except Exception as e: return f"REJECT: GraphTool 处理意图时发生异常: {str(e)}"
    # LawWorker 专用
    def inject_law_nodes(self, law_contents: List[str]) -> str:
        if not self.current_graph: return "Error: Graph not set."
        count = 0
        ids = []

        for content in law_contents:
            node_id, is_new = self.current_graph.add_node(
                content=content,
                node_type=NodeType.LAW,
                agent_id="LawWorker"
            )

            if is_new:
                count += 1
                ids.append(node_id)

        if ids:
            self.current_graph.touch_nodes(ids, step_index=self.system.step_counter) 
            self.current_graph.refresh_context(current_step=self.system.step_counter)

        return f"已底层注入 {count} 条法条到图谱中。"