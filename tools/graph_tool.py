from typing import Union, List
from mas.legal_system import LegalSystem
from mas.common import ShadowGraph, NodeType
from mas.llm import LLMCallable
from mas.schema import AgentAction
from mas.graph_ops import GraphExecutor

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
            else: return "REJECT: 无效的动作格式。"

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
        executor = GraphExecutor(self.current_graph, matcher=self.system.dedup_matcher)
        current_step = self.system.step_counter
        count = 0
        logs = []

        for content in law_contents:
            node_id, log = executor._apply_add_node(
                content=content,
                node_type=NodeType.LAW,
                agent_id="LawWorker",
                current_step=current_step
            )

            logs.append(log)
            if "✅" in log: count += 1

        if count > 0: self.current_graph.refresh_context(current_step=current_step)
        return f"已底层注入 {count} 条法条到图谱中。\n日志:\n" + "\n".join(logs)