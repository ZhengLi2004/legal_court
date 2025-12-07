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
from .config import SystemConfig
# Legal-G-Memory 的统一入口
class LegalSystem:
    def __init__(self, persist_dir: str = "./storage", recorder: Any = None, config: SystemConfig = None):
        self.cfg = config or SystemConfig()
        self.llm = GPTChat(model_name=self.cfg.llm.model_name)
        self.ef = EmbeddingFunc(model_path="./bge-m3")
        self.projection_matcher = SemanticMatcher(self.ef, threshold=self.cfg.matcher.projection_threshold)
        self.insight_matcher = SemanticMatcher(self.ef, threshold=self.cfg.matcher.insight_threshold)
        self.dedup_matcher = SemanticMatcher(self.ef)
        self.memory = LegalGMemory(persist_dir=persist_dir, config=self.cfg)
        self.insights = InsightsManager(persist_dir, self.llm, self.insight_matcher, self.cfg)
        self.judge = LLMJudge(self.llm)
        self.projector = GraphProjector(self.projection_matcher, config=self.cfg)
        self.backprop = BackPropagator()
        self.recorder = recorder
        self._current_case_insights: List[str] = []

    def new_case(self, context: str) -> ShadowGraph:
        sg = ShadowGraph()
        sg.id_alias["FACT_1"] = sg.add_node(context, "FACT", self.cfg.agent.system_id, matcher=None)
        relevant_strategies = self.insights.get_relevant_insights(context, top_k=self.cfg.retrieval.insight_top_k)
        self._current_case_insights = relevant_strategies
        initial_msgs, _ = self.memory.retrieve_memory(context, top_k=self.cfg.retrieval.initial_top_k)
        corrective_msgs = []

        if relevant_strategies:
            top_insight = relevant_strategies[0]
            case_ids = self.insights.find_cases_by_insight(top_insight, top_k=self.cfg.retrieval.corrective_top_k)

            if case_ids:
                raw_data = self.memory.collection.get(ids=case_ids, include=["metadatas", "documents"])

                if raw_data['ids']:
                    for i in range(len(raw_data['ids'])):
                        msg = LegalMessage.from_dict({
                            "case_id": raw_data['ids'][i],
                            "case_context": raw_data['documents'][i],
                            "graph_json": raw_data['metadatas'][i]['graph_json']
                        })
                        
                        corrective_msgs.append(msg)

        all_msgs = {m.case_id: m for m in initial_msgs + corrective_msgs}.values()
        history_graphs = [m.shadow_graph for m in all_msgs]
        self.projector.project(sg, history_graphs)

        if self.recorder:
            self.recorder.log_event(
                step_name="New Case Initialization",
                shadow_graph=sg,
                message="Context: {context[:30]}..."
            )
        
        return sg, relevant_strategies
    
    def execute_action(self, graph: ShadowGraph, agent_id: str, action_text: str) -> List[str]:
        executor = GraphExecutor(graph, matcher=self.dedup_matcher)
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
                context_node_candidates = [n for n, d in graph.graph.nodes(data=True) if d.get('agent_id') == self.cfg.agent.system_id]
                
                if not context_node_candidates:
                    all_facts = [d['content'] for n, d in graph.graph.nodes(data=True) if str(d.get('type')) == 'FACT']
                    context = " ".join(all_facts)

                else:
                    context_node = context_node_candidates[0]
                    context = graph.graph.nodes[context_node]['content']

                msgs, _ = self.memory.retrieve_memory(context, top_k=self.cfg.retrieval.initial_top_k)
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
        was_successful = (winner == "plaintiff")
        
        self.insights.update_scores_from_verdict(
            case_id=case_id,
            used_insights=self._current_case_insights,
            was_successful=was_successful
        )

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