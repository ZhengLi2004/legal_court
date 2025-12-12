from typing import Tuple
from mas.legal_system import LegalSystem
from mas.common import ShadowGraph
from mas.llm import LLMCallable, Message
from prompts.shared import get_shared_prompt, OutputMode

class GraphTool:
    def __init__(self, legal_system: LegalSystem, llm: LLMCallable):
        self.system = legal_system
        self.llm = llm
        self.current_graph: ShadowGraph = None

    def set_current_graph(self, graph: ShadowGraph): self.current_graph = graph

    async def process_intent(self, agent_id: str, intent_text: str) -> str:
        if not self.current_graph: return "ERROR: Current graph context is not set."
        is_valid, reason = await self._validate_intent(intent_text)
        if not is_valid: return f"REJECT: {reason}"
        actions_script = await self._translate_intent(intent_text)
        print(f"[GraphTool] Intent: '{intent_text}' -> Script: '{actions_script}'")

        logs = self.system.execute_action(
            graph=self.current_graph,
            agent_id=agent_id,
            action_text=actions_script
        )

        return f"EXECUTED:\nScript: {actions_script}\nLogs:\n" + "\n".join(logs)
    
    async def _validate_intent(self, intent: str) -> Tuple[bool, str]:
        context_text = self.current_graph.latest_context

        prompt = f"""
        你是一个辩论图谱的语义审查官。你的职责是确保每一个进入图谱的操作都是高质量、强相关且合乎逻辑的。
        
        【当前辩论局势】：
        {context_text}

        【待审查的意图】：
        {intent}

        【审查标准】（必须全部满足）：
        1. **相关性（Relevance）**：该意图是否直接回应了上述局势中的争议点？拒绝无关的闲聊或偏题的论述。
        2. **强度（Strength）**：该意图是否提供了实质性的信息增量？拒绝“我认为他是错的”这类没有证据支撑的空洞反驳。
        3. **逻辑（Logic）**：是否存在逻辑谬误？（如用‘没有证据’去支撑‘事实存在’）。

        请判断该意图是否应被执行：
        如果通过，仅输出：VALID
        如果决绝，输出格式：REJECT: <拒绝具体理由，指出是相关性、强度还是逻辑问题>
        """

        response = self.llm([Message(role="user", content=prompt)], temperature=0.0)

        if "REJECT:" in response:
            reason = response.split("REJECT:", 1)[1].strip()
            return False, reason
        
        if "VALID" in response: return True, ""
        return False, f"Validation failed: Ambiguous response '{response}'"

    async def _translate_intent(self, intent: str) -> str:
        rules = get_shared_prompt(mode=OutputMode.DSL_STRICT)
        
        prompt = f"""
        你是一个指令翻译官。将自然语言的辩论意图转化为标准的图操作指令。

        {rules}

        【待转化意图】："{intent}"

        请生成指令序列（一行一条，分号分隔）：
        """

        response = self.llm([Message(role="user", content=prompt)], temperature=0.0)
        clean_script = response.replace("```", "").strip()
        return clean_script