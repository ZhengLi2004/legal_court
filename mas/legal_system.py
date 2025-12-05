from typing import Tuple, List
from dataclasses import dataclass
from .llm import GPTChat
from .utils import EmbeddingFunc
from .common import ShadowGraph, LegalMessage
from .legal_memory import LegalGMemory
from .insights_manager import InsightsManager
from .judge import LLMJudge
from .graph_ops import GraphExecutor
from .semantic_matcher import SemanticMatcher
from .projection import GraphProjector
# Legal-G-Memory 的统一入口
class LegalSystem:
    def __init__(self, persist_dir: str = "./storage"):
        self.llm = GPTChat()
        self.ef = EmbeddingFunc(model_path="./bge-m3")
        self.matcher = SemanticMatcher(self.ef, threshold=0.90)
        self.memory = LegalGMemory(persist_dir=persist_dir)
        self.insights = InsightsManager(persist_dir, self.llm, self.matcher)
        self.judge = LLMJudge(self.llm)
        self.projector = GraphProjector(self.matcher)

    def new_case(self, context: str) -> ShadowGraph:
        sg = ShadowGraph()
        sg.add_node(context, "FACT", "system", matcher=self.matcher)
        msgs, _ = self.memory.retrieve_memory(context, top_k=2)
        history_graphs = [m.shadow_graph for m in msgs]
        self.projector.project(sg, history_graphs)
        return sg
    
    def execute_action(self, graph: ShadowGraph, agent_id: str, action_text: str) -> List[str]:
        executor = GraphExecutor(graph, self.matcher)
        return executor.execute_batch(action_text, agent_id)

    def adjudicate(self, context: str, graph: ShadowGraph) -> Tuple[bool, str]: return self.judge.evaluate(context, graph)

    def learn(self, context: str, win_graph: ShadowGraph, lose_graph: ShadowGraph, case_id: str):
        msg = LegalMessage(case_id=case_id, case_context=context, shadow_graph=win_graph)
        self.memory.add_memory(msg)
        self.insights.extract_adversarial_insights(case_id, context, win_graph, lose_graph)