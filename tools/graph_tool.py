from mas.legal_system import LegalSystem
from mas.common import ShadowGraph

class GraphTool:
    def __init__(self, legal_system: LegalSystem):
        self.system = legal_system
        self.current_graph = None

    def set_current_graph(self, graph: ShadowGraph): self.current_graph = graph

    def execute_actions(self, agent_id: str, actions_text: str) -> str:
        if not self.current_graph: return "错误：尚未设置当前操作的图谱。"

        logs = self.system.execute_action(
            graph=self.current_graph,
            agent_id=agent_id,
            action_text=actions_text
        )

        return "\n".join(logs)