import os
import json
from typing import List
from dataclasses import dataclass
from .llm import GPTChat, Message
from .common import ShadowGraph
from .semantic_matcher import SemanticMatcher
from .utils import simple_file_lock, cosine_similarity

@dataclass
class Insight:
    content: str
    score: float = 1.0
    source_case_id: str = ""
# 法理策略 -> Insight Graph
class InsightsManager:
    def __init__(self, working_dir: str, llm: GPTChat, matcher: SemanticMatcher):
        self.working_dir = working_dir
        self.llm = llm
        self.matcher = matcher
        self.file_path = os.path.join(working_dir, "legal_insights.json")
        self.insights: List[Insight] = self._load_insights()
        self._insight_index = []
        self._rebuild_index()

    def _load_insights(self) -> List[Insight]:
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [Insight(**item) for item in data]
        
        return []
    
    def _save_insights(self):
        lock_file = self.file_path + ".lock"
        
        with simple_file_lock(lock_file):
            data = [inst.__dict__ for inst in self.insights]
            with open(self.file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
    
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
                self.insights[idx].score += 1.0
                print(f"[Insights] Merged similar strategy: '{content}' -> '{self.insights[idx].content}'")
            
            else:

                new_insight = Insight(content=content, source_case_id=case_id)
                self.insights.append(new_insight)
                print(f"[Insights] Added new strategy: '{content}'")
            
            self._save_insights()
            self._rebuild_index()

    def get_relevant_insights(self, context: str, top_k: int = 3) -> List[str]:
        if not self._insight_index: return []
        query_emb = self.matcher.embedding_func.embed_query(context)
        candidates = []

        for emb, inst in self._insight_index:
            sim = cosine_similarity(query_emb, emb)
            final_score = sim * (1.0 + 0.05 * inst.score)
            candidates.append((final_score, inst))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1].content for c in candidates[:top_k]]