from typing import Dict, List, Tuple

from mas.schema import AgentAction

from .backprop import BackPropagator
from .common import LegalMessage, NodeStatus, ShadowGraph
from .config import SystemConfig
from .graph_ops import GraphExecutor
from .insights_manager import InsightsManager
from .judge import LLMJudge
from .legal_memory import LegalGMemory
from .llm import GPTChat
from .projection import GraphProjector
from .semantic_matcher import SemanticMatcher
from .utils import EmbeddingFunc


class LegalSystem:
    def __init__(self, persist_dir: str = None, config: SystemConfig = None):
        self.cfg = config or SystemConfig()
        self.llm = GPTChat(model_name=self.cfg.llm.model_name)  # This is the agent LLM
        self.ef = EmbeddingFunc(model_path=self.cfg.path.embedding_model_path)

        self.projection_matcher = SemanticMatcher(
            self.ef, threshold=self.cfg.matcher.projection_threshold
        )

        self.insight_matcher = SemanticMatcher(
            self.ef, threshold=self.cfg.matcher.insight_threshold
        )

        self.dedup_matcher = SemanticMatcher(self.ef)

        self.memory = LegalGMemory(
            persist_dir=self.cfg.path.storage_root_dir, config=self.cfg
        )

        self.insights = InsightsManager(
            self.cfg.path.storage_root_dir, self.llm, self.insight_matcher, self.cfg
        )

        judge_llm = GPTChat(
            model_name=self.cfg.judge.model_name,
            base_url=self.cfg.judge.base_url,
            api_key=self.cfg.judge.api_key,
        )

        extraction_llm = GPTChat(
            model_name=self.cfg.llm.model_name,
            base_url=self.cfg.llm.base_url,
            api_key=self.cfg.llm.api_key,
        )

        self.judge = LLMJudge(judge_llm=judge_llm, extraction_llm=extraction_llm)
        self.projector = GraphProjector(self.projection_matcher, config=self.cfg)
        self.backprop = BackPropagator()
        self._current_case_insights: List[str] = []
        self.step_counter = 0

    def new_case(self, context: str) -> ShadowGraph:
        self.step_counter = 0
        sg = ShadowGraph()

        relevant_strategies = self.insights.get_relevant_insights(
            context, top_k=self.cfg.retrieval.insight_top_k
        )

        self._current_case_insights = relevant_strategies

        initial_msgs, _ = self.memory.retrieve_memory(
            context, top_k=self.cfg.retrieval.initial_top_k
        )

        corrective_msgs = []

        if relevant_strategies:
            top_insight = relevant_strategies[0]

            case_ids = self.insights.find_cases_by_insight(
                top_insight, top_k=self.cfg.retrieval.corrective_top_k
            )

            if case_ids:
                raw_data = self.memory.collection.get(
                    ids=case_ids, include=["metadatas", "documents"]
                )

                if raw_data["ids"]:
                    for i in range(len(raw_data["ids"])):
                        msg = LegalMessage.from_dict(
                            {
                                "case_id": raw_data["ids"][i],
                                "case_context": raw_data["documents"][i],
                                "graph_json": raw_data["metadatas"][i]["graph_json"],
                            }
                        )

                        corrective_msgs.append(msg)

        all_msgs = list({m.case_id: m for m in initial_msgs + corrective_msgs}.values())
        self.projector.project(sg, all_msgs)
        sg.refresh_context(0)
        return sg, relevant_strategies

    def advance_step(self):
        self.step_counter += 1

    def execute_action(
        self, graph: ShadowGraph, agent_id: str, actions: List[AgentAction]
    ) -> List[str]:
        current_step = self.step_counter
        executor = GraphExecutor(graph, matcher=self.dedup_matcher)
        logs = executor.execute_batch(actions, agent_id, current_step=self.step_counter)
        focus_nodes = graph.get_nodes_by_step(self.step_counter)
        query_context = ""
        retrieval_mode = ""

        if focus_nodes:
            query_context = graph.to_tactical_text(focus_nodes)
            retrieval_mode = f"Tactical (Focus on {len(focus_nodes)} new nodes)"

        else:
            context_node_candidates = [
                n
                for n, d in graph.graph.nodes(data=True)
                if d.get("agent_id") == self.cfg.agent.system_id
            ]

            if not context_node_candidates:
                all_facts = [
                    d["content"]
                    for n, d in graph.graph.nodes(data=True)
                    if str(d.get("type")) == "FACT"
                ]

                query_context = " ".join(all_facts)

            else:
                context_node = context_node_candidates[0]
                query_context = graph.graph.nodes[context_node]["content"]

            retrieval_mode = "Global Context"

        if query_context:
            try:
                msgs, _ = self.memory.retrieve_memory(
                    query_context, top_k=self.cfg.retrieval.initial_top_k
                )

                nodes_before = graph.graph.number_of_nodes()
                self.projector.project(graph, msgs)
                nodes_after = graph.graph.number_of_nodes()
                added = nodes_after - nodes_before

                if added > 0:
                    logs.append(
                        f"System auto-projected {added} nodes ({retrieval_mode})."
                    )

            except Exception as e:
                logs.append(f"Error during projection: {e}")

        graph.refresh_context(current_step)
        return logs

    async def adjudicate(
        self, context: str, graph: ShadowGraph
    ) -> Tuple[str, Dict[str, NodeStatus]]:
        judgment_document = self.judge.evaluate(context, graph)
        root_claims_status = await self.judge.extract_verdict(judgment_document, graph)
        return judgment_document, root_claims_status

    def learn(
        self, context: str, current_graph: ShadowGraph, winner: str, case_id: str
    ):
        was_successful = winner == "plaintiff"

        self.insights.update_scores_from_verdict(
            case_id=case_id,
            used_insights=self._current_case_insights,
            was_successful=was_successful,
        )

        final_graph = self.backprop.propagate(current_graph, winner)

        msg = LegalMessage(
            case_id=case_id, case_context=context, shadow_graph=final_graph
        )

        self.memory.add_memory(msg)

        win_nodes = [
            n
            for n, d in final_graph.graph.nodes(data=True)
            if d["status"] == NodeStatus.VALIDATED
        ]

        win_graph = final_graph.get_subgraph(win_nodes)

        lose_nodes = [
            n
            for n, d in final_graph.graph.nodes(data=True)
            if d["status"] == NodeStatus.DEFEATED
        ]

        lose_graph = final_graph.get_subgraph(lose_nodes)

        self.insights.extract_adversarial_insights(
            case_id, context, win_graph, lose_graph
        )
