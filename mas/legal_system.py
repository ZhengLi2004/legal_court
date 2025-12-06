from typing import Tuple, List, Any
import re
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
        self.matcher = SemanticMatcher(self.ef, threshold=0.5)
        self.memory = LegalGMemory(persist_dir=persist_dir)
        self.insights = InsightsManager(persist_dir, self.llm, self.matcher)
        self.judge = LLMJudge(self.llm)
        self.projector = GraphProjector(self.matcher)
        self.backprop = BackPropagator()
        self.recorder = recorder

    def new_case(self, context: str) -> ShadowGraph:
        sg = ShadowGraph()
        sg.add_node(context, "FACT", "system", matcher=None)
        relevant_strategies = self.insights.get_relevant_insights(context, top_k=3)
        
        if self.recorder:
            self.recorder.log_event(
                step_name="New Case Initialization",
                shadow_graph=sg,
                message="Context: {context[:30]}..."
            )
        
        return sg, relevant_strategies
    
    def execute_action(self, graph: ShadowGraph, agent_id: str, action_text: str) -> List[str]:
        executor = GraphExecutor(graph, self.matcher)
        logs = executor.execute_batch(action_text, agent_id)

        if self.recorder:
            self.recorder.log_event(
                step_name=f"Action: {agent_id} (Executed)",
                shadow_graph=graph,
                message=f"Agent added nodes. Preparing for projection."
            )

        if "ADD_" in action_text.upper():
            projected_count = 0
            
            try:
                context_node_candidates = [n for n, d in graph.graph.nodes(data=True) if d.get('agent_id') == 'system']
                
                if not context_node_candidates:
                    all_facts = [d['content'] for n, d in graph.graph.nodes(data=True) if str(d.get('type')) == 'FACT']
                    context = " ".join(all_facts)

                else:
                    context_node = context_node_candidates[0]
                    context = graph.graph.nodes[context_node]['content']

                msgs, _ = self.memory.retrieve_memory(context, top_k=2)
                history_graphs = [m.shadow_graph for m in msgs]
                nodes_before = graph.graph.number_of_nodes()
                self.projector.project(graph, history_graphs)
                nodes_after = graph.graph.number_of_nodes()
                projected_count = nodes_after - nodes_before

                if projected_count > 0:
                    logs.append(f"Projection triggered: {projected_count} nodes imported.")

                    if self.recorder:
                        self.recorder.log_event(
                            step_name=f"Projection on ADD Action",
                            shadow_graph=graph,
                            message=f"{projected_count} nodes were projected into the graph."
                        )
                    
            except Exception as e: logs.append(f"Error during projection: {e}")
    
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