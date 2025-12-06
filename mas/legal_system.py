from typing import Tuple, List, Any
from .llm import GPTChat
from .utils import EmbeddingFunc
from .common import ShadowGraph, LegalMessage, NodeStatus
from .legal_memory import LegalGMemory
from .insights_manager import InsightsManager
from .judge import LLMJudge
from .graph_ops import GraphExecutor
from .semantic_matcher import SemanticMatcher
from .projection import GraphProjector
from .backprop import BackPropagator
# Legal-G-Memory 的统一入口
class LegalSystem:
    def __init__(self, persist_dir: str = "./storage", recorder: Any = None):
        self.llm = GPTChat()
        self.ef = EmbeddingFunc(model_path="./bge-m3")
        self.matcher = SemanticMatcher(self.ef, threshold=0.90)
        self.memory = LegalGMemory(persist_dir=persist_dir)
        self.insights = InsightsManager(persist_dir, self.llm, self.matcher)
        self.judge = LLMJudge(self.llm)
        self.projector = GraphProjector(self.matcher)
        self.backprop = BackPropagator()
        self.recorder = recorder

    def new_case(self, context: str) -> ShadowGraph:
        sg = ShadowGraph()
        sg.add_node(context, "FACT", "system", matcher=self.matcher)
        msgs, _ = self.memory.retrieve_memory(context, top_k=2)
        history_graphs = [m.shadow_graph for m in msgs]
        self.projector.project(sg, history_graphs)
        relevant_strategies = self.insights.get_relevant_insights(context, top_k=3)
        
        if self.recorder:
            self.recorder.log_event(
                step_name="New Case Initialization",
                shadow_graph=sg,
                message=f"Context: {context[:30]}...\nProjected {len(history_graphs)} historical cases."
            )
        
        return sg, relevant_strategies
    
    def execute_action(self, graph: ShadowGraph, agent_id: str, action_text: str) -> List[str]:
        executor = GraphExecutor(graph, self.matcher)
        logs = executor.execute_batch(action_text, agent_id)

        if self.recorder:
            self.recorder.log_event(
                step_name=f"Action: {agent_id}",
                shadow_graph=graph,
                message=f"Executed: {action_text[:50]}..."
            )

        return logs

    def adjudicate(self, context: str, graph: ShadowGraph) -> Tuple[bool, str]: return self.judge.evaluate(context, graph)
    # 学习: BackProp -> Store -> Extract Insights
    def learn(self, context: str, current_graph: ShadowGraph, winner: str, case_id: str):
        final_graph = self.backprop.propagate(current_graph, winner)

        if self.recorder:
            self.recorder.log_event(
                step_name="Learning & Verdict",
                shadow_graph=final_graph,
                message=f"Winner: {winner}. Graph status updated."
            )
        
        msg = LegalMessage(case_id=case_id, case_context=context, shadow_graph=final_graph)
        self.memory.add_memory(msg)
        win_nodes = [n for n, d in final_graph.graph.nodes(data=True) if d['status'] == NodeStatus.VALIDATED]
        win_graph = final_graph.get_subgraph(win_nodes)
        lose_nodes = [n for n, d in final_graph.graph.nodes(data=True) if d['status'] == NodeStatus.DEFEATED]
        lose_graph = final_graph.get_subgraph(lose_nodes)
        self.insights.extract_adversarial_insights(case_id, context, win_graph, lose_graph)