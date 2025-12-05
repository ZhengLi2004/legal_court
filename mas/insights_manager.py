import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from .llm import GPTChat, Message
from .common import ShadowGraph

@dataclass
class Insight:
    content: str
    score: float = 1.0
    source_case_id: str = ""
# 法理策略 -> Insight Graph
class InsightsManager:
    def __init__(self, working_dir: str, llm: GPTChat):
        self.working_dir = working_dir
        self.llm = llm
        self.file_path = os.path.join(working_dir, "legal_insights.json")
        self.insights: List[Insight] = self._load_insights()

    def _load_insights(self) -> List[Insight]:
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [Insight(**item) for item in data]
        
        return []
    
    def _save_insights(self):
        data = [inst.__dict__ for inst in self.insights]
        with open(self.file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
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
            
            for inst in self.insights:
                if content == inst.content:
                    inst.score += 1.0
                    self._save_insights()
                    return
                
            new_insight = Insight(content=content, source_case_id=case_id)
            self.insights.append(new_insight)
            self._save_insights()

    def get_relevant_insights(self, top_k: int = 3) -> List[str]:
        sorted_insights = sorted(self.insights, key=lambda x: x.score, reverse=True)
        return [inst.content for inst in sorted_insights[:top_k]]