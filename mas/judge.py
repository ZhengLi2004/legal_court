from abc import ABC, abstractmethod
from typing import Tuple, Dict
from .common import ShadowGraph
from .llm import GPTChat, Message

class BaseJudge(ABC):
    def evaluate(self, context: str, graph: ShadowGraph) -> Tuple[bool, str]: pass
# LLM 模拟法官进行判决
class LLMJudge(BaseJudge):
    def __init__(self, llm: GPTChat): self.llm = llm

    def evaluate(self, context: str, graph: ShadowGraph) -> Tuple[bool, str]:
        prompt = f"""
        请作为法官审理以下案件。
        案情: {context}
        当前论辩图谱:
        {graph.to_json()}
        
        请判断当前证据链是否足以支持原告的主张？
        如果是，输出: VERDICT: PLAINTIFF_WINS
        如果证据被有效反驳，且原告无新证据，输出: VERDICT: DEFENDANT_WINS
        如果尚不明确，需要更多辩论，输出: VERDICT: UNSETTLED
        """

        response = self.llm([Message(role="user", content=prompt)])
        if "PLAINTIFF_WINS" in response: return True, "plaintiff"
        elif "DEFENDANT_WINS" in response: return True, "defendant"
        else: return False, None