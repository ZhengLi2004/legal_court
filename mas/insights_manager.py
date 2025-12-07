import os
import json
from typing import List, Any
from dataclasses import dataclass, field
from .llm import GPTChat, Message
from .common import ShadowGraph
from .semantic_matcher import SemanticMatcher
from .utils import file_lock, cosine_similarity
from .config import SystemConfig

@dataclass
class Insight:
    content: str
    score: float = 1.0
    positive_cases: List[str] = field(default_factory=list)
    negative_cases: List[str] = field(default_factory=list)
# 法理策略 -> Insight Graph
class InsightsManager:
    def __init__(self, working_dir: str, llm: GPTChat, matcher: SemanticMatcher, config: SystemConfig = None):
        self.working_dir = working_dir
        self.llm = llm
        self.matcher = matcher
        self.cfg = config or SystemConfig()
        self.file_path = os.path.join(working_dir, self.cfg.path.file_insight_graph)
        self.insights: List[Insight] = self._load_insights()
        self._insight_index = []
        self._rebuild_index()

    def _load_insights(self) -> List[Insight]:
        if not os.path.exists(self.file_path): return []

        with open(self.file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)

                return [
                    Insight(
                        content=item['content'],
                        score=item.get('score', 1.0),
                        positive_cases=list(set(item.get('positive_cases', []))),
                        negative_cases=list(set(item.get('negative_cases', [])))
                    ) for item in data
                ]
            
            except (json.JSONDecodeError, TypeError): return []
    
    def _save_insights(self):
        lock_file = self.file_path + ".lock"
        
        with file_lock(lock_file):
            self.insights = [inst for inst in self.insights if inst.score > 0]
            data = [inst.__dict__ for inst in self.insights]
            with open(self.file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
            self._rebuild_index()
    
    def _rebuild_index(self):
        self._insight_index = []
        
        for inst in self.insights:
            emb = self.matcher.embedding_func.embed_query(inst.content)
            self._insight_index.append((emb, inst))
    # 通过对比同一案件的胜诉/败诉途径，提取策略
    def extract_adversarial_insights(self, 
                                     case_id: str,
                                     case_context: str,
                                     winning_graph: ShadowGraph, 
                                     losing_graph: ShadowGraph):
        prompt = f"""
        作为法理专家，请分析以下案件的辩论过程。
        
        【案情摘要】
        {case_context}
        
        【被判决采纳的论证 (Winning Path)】
        {winning_graph.to_json()}
        
        【被驳回的论证 (Defeated Path)】
        {losing_graph.to_json()}
        
        请分析胜方为何成功（如：引用了更有力的法条、指出了证据漏洞等），并总结出一条通用的辩护策略。
        只输出策略内容，不要解释。
        格式: STRATEGY: <策略内容>
        """

        response = self.llm([Message(role="user", content=prompt)])

        if "STRATEGY:" in response:
            content = response.split("STRATEGY:")[1].strip()
            candidates = [(str(i), inst.content) for i, inst in enumerate(self.insights)]
            match_idx_str = self.matcher.find_match(content, candidates)
                
            if match_idx_str:
                idx = int(match_idx_str)
                self.insights[idx].score += self.cfg.insight.reward_merge
                if case_id not in self.insights[idx].positive_cases: self.insights[idx].positive_cases.append(case_id)
            
            else:

                new_insight = Insight(content=content, positive_cases=[case_id])
                self.insights.append(new_insight)
            
            self._save_insights()

    def update_scores_from_verdict(self, case_id: str, used_insights: List[str], was_successful: bool):
        for content in used_insights:
            for inst in self.insights:
                if inst.content == content:
                    if was_successful:
                        inst.score += self.cfg.insight.reward_win
                        if case_id not in inst.positive_cases: inst.positive_cases.append(case_id)
                        if case_id in inst.negative_cases: inst.negative_cases.remove(case_id)

                    else:
                        inst.score -= self.cfg.insight.penalty_lose # 惩罚不宜过重，可能是用法错误
                        if case_id not in inst.negative_cases: inst.negative_cases.append(case_id)

        self._save_insights()
    
    def get_relevant_insights(self, context: str, top_k: int = 3) -> List[str]:
        if not self._insight_index: return []
        query_emb = self.matcher.embedding_func.embed_query(context)
        candidates = []
        cfg = SystemConfig().insight
        
        for emb, inst in self._insight_index:
            sim = cosine_similarity(query_emb, emb)
            final_score = sim * (1.0 + cfg.score_weight * inst.score)
            candidates.append((final_score, inst))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1].content for c in candidates[:top_k]]
    # 根据策略反向查找成功应用策略的案件
    def find_cases_by_insight(self, insight_content: str, memory_retriever: Any = None, top_k: int = 5) -> List[str]:
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

            if best_score < 0.7: return []

        if target_insight: return target_insight.positive_cases[-top_k:]
        return []