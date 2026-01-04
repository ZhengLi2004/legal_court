import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from prompts.common_prompts import EXTRACT_ADVERSARIAL_INSIGHTS_PROMPT

from .config import SystemConfig
from .llm import GPTChat, Message
from .semantic_matcher import SemanticMatcher
from .utils import cosine_similarity, file_lock


@dataclass
class Insight:
    content: str
    cases: List[str] = field(default_factory=list)
    representatives: List[str] = field(default_factory=list)


class InsightsManager:
    def __init__(
        self,
        working_dir: str,
        llm: GPTChat,
        matcher: SemanticMatcher,
        config: SystemConfig = None,
    ):
        self.working_dir = working_dir
        self.llm = llm
        self.matcher = matcher
        self.cfg = config or SystemConfig()
        self.file_path = os.path.join(working_dir, self.cfg.path.file_insight_graph)
        self.insights: List[Insight] = self._load_insights()
        self._insight_index = []
        self._rebuild_index()

    def _load_insights(self) -> List[Insight]:
        if not os.path.exists(self.file_path):
            return []

        with open(self.file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                loaded_insights = []

                for item in data:
                    cases = list(
                        set(
                            item.get("cases", [])
                            + item.get("positive_cases", [])
                            + item.get("negative_cases", [])
                        )
                    )

                    reps = item.get("representatives", [])

                    if cases and not reps:
                        reps = cases

                    loaded_insights.append(
                        Insight(
                            content=item["content"], cases=cases, representatives=reps
                        )
                    )

                return loaded_insights

            except (json.JSONDecodeError, TypeError):
                return []

    def _save_insights(self):
        lock_file = self.file_path + ".lock"

        with file_lock(lock_file):
            data = [inst.__dict__ for inst in self.insights]

            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._rebuild_index()

    def _rebuild_index(self):
        self._insight_index = []

        for inst in self.insights:
            emb = self.matcher.embedding_func.embed_query(inst.content)
            self._insight_index.append((emb, inst))

    def extract_adversarial_insights(
        self,
        case_id: str,
        case_context: str,
        transcript: List[str],
        root_claims_status: Dict[str, str],
    ) -> "Insight":
        status_desc = "\n".join(
            [f"- {cid}: {status}" for cid, status in root_claims_status.items()]
        )

        transcript_text = "\n".join(transcript) if transcript else "（无庭审记录）"

        prompt = EXTRACT_ADVERSARIAL_INSIGHTS_PROMPT.format(
            case_context=case_context,
            claims_status=status_desc,
            transcript=transcript_text[:15000],
        )

        response = self.llm([Message(role="user", content=prompt)])
        content = response.replace("策略：", "").replace("Insight:", "").strip()
        candidates = [(str(i), inst.content) for i, inst in enumerate(self.insights)]
        match_idx_str = self.matcher.find_match(content, candidates)
        target_insight = None

        if match_idx_str:
            idx = int(match_idx_str)
            target_insight = self.insights[idx]

            if case_id not in target_insight.cases:
                target_insight.cases.append(case_id)

        else:
            target_insight = Insight(
                content=content, cases=[case_id], representatives=[case_id]
            )

            self.insights.append(target_insight)

        self._save_insights()
        return target_insight

    def update_insight_topology(self, insight_content: str, task_layer: Any):
        target_insight = None

        for inst in self.insights:
            if inst.content == insight_content:
                target_insight = inst
                break

        if not target_insight:
            return

        components = task_layer.get_subgraph_components(target_insight.cases)
        new_reps = []

        for comp in components:
            rep = task_layer.get_central_node(comp)

            if rep:
                new_reps.append(rep)

        target_insight.representatives = new_reps
        self._save_insights()

    def get_relevant_insights(self, context: str, top_k: int = 3) -> List[str]:
        if not self._insight_index:
            return []

        query_emb = self.matcher.embedding_func.embed_query(context)
        candidates = []

        for emb, inst in self._insight_index:
            sim = cosine_similarity(query_emb, emb)
            candidates.append((sim, inst))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1].content for c in candidates[:top_k]]

    def find_cases_by_insight(
        self, insight_content: str, memory_retriever: Any = None, top_k: int = 3
    ) -> List[str]:
        target_insight = None

        for inst in self.insights:
            if inst.content == insight_content:
                target_insight = inst
                break

        if not target_insight:
            query_emb = self.matcher.embedding_func.embed_query(insight_content)
            best_score = -1

            for emb, inst in self._insight_index:
                sim = cosine_similarity(query_emb, emb)

                if sim > best_score:
                    best_score = sim
                    target_insight = inst

            if best_score < 0.7:
                return []

        if target_insight:
            return target_insight.representatives

        return []
