"""Manages the creation, storage, and retrieval of strategic legal insights.

This module provides the `InsightsManager` class, which is responsible for the
system's learning capabilities. It extracts high-level, reusable strategic
insights from completed debates, stores them persistently, and provides them
to agents at the start of new, similar cases.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from prompts.common_prompts import (
    EXTRACT_ADVERSARIAL_INSIGHTS_PROMPT,
    SYSTEM_PROMPT_INSIGHT_EXTRACTOR,
)
from tools.embedding import cosine_similarity, file_lock
from tools.llm import GPTChat, Message
from tools.matcher import SemanticMatcher

from ..config import SystemConfig

_INSIGHT_EXTRACTION_SCHEMA = {
    "name": "adversarial_insight",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "side": {"type": "string", "enum": ["PLAINTIFF", "DEFENDANT", "COMMON"]},
            "content": {"type": "string"},
        },
        "required": ["side", "content"],
        "additionalProperties": False,
    },
}


class InsightSide(str, Enum):
    """Enumeration for the party an insight is most relevant to."""

    PLAINTIFF = "PLAINTIFF"
    DEFENDANT = "DEFENDANT"
    COMMON = "COMMON"


@dataclass
class Insight:
    """Represents a single, reusable strategic insight.

    Attributes:
        content: The textual description of the insight/strategy.
        side: The party (`InsightSide`) this insight is primarily for.
        cases: A list of case IDs where this insight was observed.
        representatives: A list of case IDs that are central representatives
            of this insight in the task topology graph.
    """

    content: str
    side: InsightSide = InsightSide.COMMON
    cases: List[str] = field(default_factory=list)
    representatives: List[str] = field(default_factory=list)


class InsightsManager:
    """Handles the lifecycle of legal insights: extraction, storage, and retrieval.

    This manager maintains a persistent JSON file of all learned insights.
    It uses an LLM to extract new insights from completed cases and a semantic
    matcher to merge new insights with existing similar ones. It also provides
    methods to retrieve relevant insights for a new case based on context.

    Attributes:
        insights: A list of all loaded `Insight` objects.
    """

    def __init__(
        self,
        working_dir: str,
        llm: GPTChat,
        matcher: SemanticMatcher,
        config: SystemConfig = None,
    ):
        """Initialize the InsightsManager.

        Args:
            working_dir: The root directory for storing persistent data.
            llm: The `GPTChat` instance for extracting insights.
            matcher: The `SemanticMatcher` for finding similar insights.
            config: The system configuration object.
        """
        self.working_dir = working_dir
        self.llm = llm
        self.matcher = matcher
        self.cfg = config or SystemConfig()
        self.file_path = os.path.join(working_dir, self.cfg.path.file_insight_graph)
        self.insights: List[Insight] = self._load_insights()
        self._insight_index = []
        self._rebuild_index()

    def _load_insights(self) -> List[Insight]:
        """Load insights from the persistent JSON file."""
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

                    side_str = item.get("side", "COMMON")

                    try:
                        side = InsightSide(side_str)

                    except ValueError:
                        side = InsightSide.COMMON

                    loaded_insights.append(
                        Insight(
                            content=item["content"],
                            side=side,
                            cases=cases,
                            representatives=reps,
                        )
                    )

                return loaded_insights

            except (json.JSONDecodeError, TypeError):
                return []

    def _save_insights(self):
        """Save the current list of insights to the persistent JSON file."""
        lock_file = self.file_path + ".lock"

        with file_lock(lock_file):
            data = []

            for inst in self.insights:
                d = inst.__dict__.copy()
                d["side"] = inst.side.value
                data.append(d)

            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild the in-memory semantic index of insights for fast retrieval."""
        self._insight_index = []

        for inst in self.insights:
            emb = self.matcher.embedding_func.embed_query(inst.content)
            self._insight_index.append((emb, inst))

    @staticmethod
    def _parse_legacy_insight_text(raw_text: str) -> Tuple["InsightSide", str]:
        """Fallback parser for historical free-text insight outputs."""
        content = (raw_text or "").strip()
        side = InsightSide.COMMON
        upper_text = content.upper()

        if "SIDE: PLAINTIFF" in upper_text:
            side = InsightSide.PLAINTIFF

            content = content.replace("SIDE: PLAINTIFF", "").replace(
                "SIDE: plaintiff", ""
            )

        elif "SIDE: DEFENDANT" in upper_text:
            side = InsightSide.DEFENDANT

            content = content.replace("SIDE: DEFENDANT", "").replace(
                "SIDE: defendant", ""
            )

        content = (
            content.replace("CONTENT:", "")
            .replace("策略：", "")
            .replace("Insight:", "")
            .strip()
        )

        return side, content

    def extract_adversarial_insights(
        self,
        case_id: str,
        case_context: str,
        transcript: List[str],
        root_claims_status: Dict[str, str],
    ) -> "Insight":
        """Extract a new insight from a completed case using an LLM.

        It prompts the LLM with the case outcome and transcript to generate a
        high-level strategic takeaway. It then checks if a similar insight
        already exists; if so, it merges them. Otherwise, it adds the new
        insight to the knowledge base.

        Args:
            case_id: The ID of the completed case.
            case_context: A summary of the case facts.
            transcript: The narrated debate transcript.
            root_claims_status: A dictionary mapping root claim IDs to their
                final status ('VALIDATED', 'DEFEATED').

        Returns:
            The newly created or updated `Insight` object.
        """
        status_desc = "\n".join(
            [f"- {cid}: {status}" for cid, status in root_claims_status.items()]
        )

        transcript_text = "\n".join(transcript) if transcript else "（无庭审记录）"

        prompt = EXTRACT_ADVERSARIAL_INSIGHTS_PROMPT.format(
            case_context=case_context,
            claims_status=status_desc,
            transcript=transcript_text[:15000],  # Truncate for context window
        )

        side = InsightSide.COMMON
        content = ""

        try:
            data = self.llm.ask_json_schema(
                messages=[
                    Message(role="system", content=SYSTEM_PROMPT_INSIGHT_EXTRACTOR),
                    Message(role="user", content=prompt),
                ],
                schema=_INSIGHT_EXTRACTION_SCHEMA,
            )

            side_raw = str(data.get("side", "COMMON")).upper().strip()
            content = str(data.get("content", "")).strip()

            try:
                side = InsightSide(side_raw)

            except ValueError:
                side = InsightSide.COMMON

        except Exception:
            legacy_response = self.llm(
                [
                    Message(role="system", content=SYSTEM_PROMPT_INSIGHT_EXTRACTOR),
                    Message(role="user", content=prompt),
                ]
            )

            side, content = self._parse_legacy_insight_text(legacy_response)

        if not content:
            content = "围绕证据链完整性构建攻防，并针对关键薄弱点形成可执行反制。"

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
                content=content, side=side, cases=[case_id], representatives=[case_id]
            )

            self.insights.append(target_insight)

        self._save_insights()
        return target_insight

    def update_insight_topology(self, insight_content: str, task_layer: Any):
        """Update the representative cases for an insight based on the TaskLayer graph.

        After the task layer (case topology graph) is updated, this method
        re-calculates the most central nodes for an insight's case cluster to
        serve as its representatives.

        Args:
            insight_content: The content of the insight to update.
            task_layer: The `TaskLayer` instance containing the case graph.
        """
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

    def get_relevant_insights_by_side(
        self, context: str, top_k: int = 3
    ) -> Tuple[List[str], List[str]]:
        """Retrieve the most relevant insights for a given context.

        It performs a semantic search over the insight index and returns separate
        lists of insights for the plaintiff and defendant, including common insights
        in both.

        Args:
            context: The case context string to search by.
            top_k: The maximum number of insights to return per side.

        Returns:
            A tuple containing two lists of strings: (plaintiff_insights,
            defendant_insights).
        """
        if not self._insight_index:
            return [], []

        query_emb = self.matcher.embedding_func.embed_query(context)
        candidates = []

        for emb, inst in self._insight_index:
            sim = cosine_similarity(query_emb, emb)
            candidates.append((sim, inst))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[: top_k * 2]
        p_insights = []
        d_insights = []

        for _, inst in top_candidates:
            if inst.side == InsightSide.PLAINTIFF or inst.side == InsightSide.COMMON:
                if len(p_insights) < top_k:
                    p_insights.append(inst.content)

            if inst.side == InsightSide.DEFENDANT or inst.side == InsightSide.COMMON:
                if len(d_insights) < top_k:
                    d_insights.append(inst.content)

        return p_insights, d_insights

    def find_cases_by_insight(
        self, insight_content: str, memory_retriever: Any = None, top_k: int = 3
    ) -> List[str]:
        """Find the representative case IDs associated with a given insight.

        Args:
            insight_content: The text of the insight to look up.
            memory_retriever: (Not currently used) For potential future use.
            top_k: (Not currently used) For potential future use.

        Returns:
            A list of representative case IDs for that insight.
        """
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
