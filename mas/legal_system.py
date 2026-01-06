from typing import Dict, List, Tuple

from metagpt.logs import logger

from mas.schema import AgentAction

from .backprop import BackPropagator
from .common import LegalMessage, NodeStatus, NodeType, ShadowGraph
from .config import SystemConfig
from .graph_ops import GraphExecutor
from .insights_manager import InsightsManager
from .judge import LLMJudge
from .legal_memory import LegalGMemory
from .llm import GPTChat
from .projection import GraphProjector
from .semantic_matcher import SemanticMatcher
from .utils import EmbeddingFunc, cosine_similarity


class LegalSystem:
    def __init__(self, persist_dir: str = None, config: SystemConfig = None):
        self.cfg = config or SystemConfig()
        self.llm = GPTChat(model_name=self.cfg.llm.model_name)
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
        self.step_counter = 0
        self._static_history_cases: List[LegalMessage] = []
        self._dynamic_law_cases: List[LegalMessage] = []

    @property
    def active_history_cases(self) -> List[LegalMessage]:
        unique_map = {}

        for c in self._static_history_cases:
            unique_map[c.case_id] = c

        for c in self._dynamic_law_cases:
            if c.case_id not in unique_map:
                unique_map[c.case_id] = c

        return list(unique_map.values())

    def _merge_static_cases(self, candidates: List[LegalMessage]):
        if not candidates:
            return

        existing_ids = {c.case_id for c in self._static_history_cases}

        for case in candidates:
            if case.case_id not in existing_ids:
                self._static_history_cases.append(case)
                existing_ids.add(case.case_id)

    def new_case(self, context: str) -> Tuple[ShadowGraph, Tuple[List[str], List[str]]]:
        self.step_counter = 0
        self._static_history_cases = []
        self._dynamic_law_cases = []
        sg = ShadowGraph()

        initial_msgs, _ = self.memory.retrieve_memory(
            context, top_k=self.cfg.retrieval.initial_top_k
        )

        self._merge_static_cases(initial_msgs)

        p_insights, d_insights = self.insights.get_relevant_insights_by_side(
            context, top_k=self.cfg.retrieval.insight_top_k
        )

        all_strategies = list(set(p_insights + d_insights))
        candidates_from_strategy = []

        if all_strategies:
            representatives = []

            for strat in all_strategies:
                reps = self.insights.find_cases_by_insight(
                    strat, top_k=self.cfg.retrieval.corrective_top_k
                )

                representatives.extend(reps)

            components = self.memory.task_layer.get_subgraph_components(representatives)
            candidate_ids = set()

            for comp in components:
                candidate_ids.update(comp)

            if candidate_ids:
                candidate_msgs = self.memory._fetch_messages_by_ids(list(candidate_ids))

                if candidate_msgs:
                    query_vec = self.ef.embed_query(context)
                    scored_candidates = []

                    for msg in candidate_msgs:
                        msg_vec = self.ef.embed_query(msg.case_context)
                        sim = cosine_similarity(query_vec, msg_vec)
                        scored_candidates.append((sim, msg))

                    scored_candidates.sort(key=lambda x: x[0], reverse=True)

                    candidates_from_strategy = [
                        item[1]
                        for item in scored_candidates[
                            : self.cfg.retrieval.corrective_top_k
                        ]
                    ]

        self._merge_static_cases(candidates_from_strategy)
        sg.refresh_context(0)
        return sg, (p_insights, d_insights)

    def learn(
        self,
        context: str,
        current_graph: ShadowGraph,
        root_claims_status: Dict[str, NodeStatus],
        case_id: str,
        transcript: List[str],
    ):
        validated_ids = [
            nid
            for nid, status in root_claims_status.items()
            if status == NodeStatus.VALIDATED
        ]

        final_graph = self.backprop.propagate(
            current_graph, explicit_validated_ids=validated_ids
        )

        msg = LegalMessage(
            case_id=case_id, case_context=context, shadow_graph=final_graph
        )

        self.memory.add_memory(msg)
        status_str_dict = {k: v.value for k, v in root_claims_status.items()}

        target_insight = self.insights.extract_adversarial_insights(
            case_id=case_id,
            case_context=context,
            transcript=transcript,
            root_claims_status=status_str_dict,
        )

        if target_insight:
            self.memory.task_layer.add_node(case_id)
            active_ids = {c.case_id for c in self.active_history_cases}
            insight_peers = set(target_insight.cases)

            if case_id in insight_peers:
                insight_peers.remove(case_id)

            links_to_create = active_ids.intersection(insight_peers)

            if links_to_create:
                logger.info(
                    f"[Topology] Linking new case {case_id} to {len(links_to_create)} existing cases in TaskLayer."
                )

                for old_id in links_to_create:
                    self.memory.task_layer.add_link(case_id, old_id)

            else:
                logger.info(
                    f"[Topology] New case {case_id} added as isolated node in strategy cluster."
                )

            self.insights.update_insight_topology(
                target_insight.content, self.memory.task_layer
            )

    def inject_jurisprudential_context(self, current_graph: ShadowGraph):
        if not current_graph:
            return

        current_laws = set()

        for _, data in current_graph.graph.nodes(data=True):
            if data.get("type") == NodeType.LAW:
                content = data.get("content", "").strip()

                if content:
                    current_laws.add(content)

        if not current_laws:
            if self._dynamic_law_cases:
                self._dynamic_law_cases = []

            return

        law_list = list(current_laws)
        retrieved_cases = self.memory.retrieve_cases_by_law_codes(law_list)
        static_ids = {c.case_id for c in self._static_history_cases}
        new_dynamic_cases = []

        for case in retrieved_cases:
            if case.case_id not in static_ids:
                new_dynamic_cases.append(case)

        self._dynamic_law_cases = new_dynamic_cases

    def advance_step(self):
        self.step_counter += 1

    def execute_action(
        self, graph: ShadowGraph, agent_id: str, actions: List[AgentAction]
    ) -> List[str]:
        current_step = self.step_counter
        executor = GraphExecutor(graph, matcher=self.dedup_matcher)
        logs = executor.execute_batch(actions, agent_id, current_step=self.step_counter)
        graph.refresh_context(current_step)
        return logs

    async def adjudicate(
        self, context: str, graph: ShadowGraph, transcript: List[str]
    ) -> Tuple[str, Dict[str, NodeStatus]]:
        judgment_document = self.judge.evaluate(context, graph, transcript)
        root_claims_status = await self.judge.extract_verdict(judgment_document, graph)
        return judgment_document, root_claims_status
