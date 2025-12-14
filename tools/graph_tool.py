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
        try:
            if not self.current_graph: return "ERROR: Current graph context is not set."
            is_valid, reason = await self._validate_intent(intent_text)
            if not is_valid: return f"REJECT: {reason}"
            max_retries = 2
            last_error = ""
            current_script = ""
            
            for attempt in range(max_retries + 1):
                if attempt == 0: current_script = await self._translate_intent(intent_text)

                else:
                    print(f"[GraphTool] Self-correcting (Attempt {attempt})...")
                    current_script = await self._fix_script(intent_text, current_script, last_error)
            
                try:
                    logs = self.system.execute_action(
                        graph=self.current_graph,
                        agent_id=agent_id,
                        action_text=current_script
                    )

                    error_logs = [l for l in logs if l.startswith("Error") or "Failed" in l]
                    if not error_logs: return f"EXECUTED:\nScript: {current_script}\nLogs:\n" + "\n".join(logs)
                    last_error = "; ".join(error_logs)
                
                except Exception as e: last_error = f"System Exception: {str(e)}"

            return f"ERROR: GraphTool failed after {max_retries} retries. Last Error: {last_error}"

        except Exception as e: return f"ERROR in GraphTool: {str(e)}"

    async def _fix_script(self, original_intent: str, bad_script: str, error_msg: str) -> str:
        rules = get_shared_prompt(mode=OutputMode.DSL_STRICT)
        id_inventory = "（图谱为空）"
        if self.current_graph: id_inventory = self.current_graph.get_id_inventory()

        prompt = f"""
        你是一个 DSL 修复专家。上次生成的指令执行失败了。

        {rules}
        
        【原始意图】: "{original_intent}"
        【生成的指令】: "{bad_script}"
        【执行报错】: "{error_msg}"

        【当前图谱中的有效 ID 清单 (Valid IDs)】:
        {id_inventory}
        
        【修复指南】:
        1. **ID 修正**: 报错如果是 "Source not found"，请在上面的清单中找到内容匹配的**正确 UUID** 并替换。
        2. **幻觉去除**: 如果意图引用了一个根本不存在的节点，请删除该指令。
        3. **格式修正**: 确保移除所有方括号 `[]`。

        请输出修正后的指令序列（不要解释，只输出代码）。
        """

        response = self.llm([Message(role="user", content=prompt)], temperature=0.0)
        return response.replace("```","").strip()

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
        id_inventory = "（图谱为空）"
        if self.current_graph: id_inventory = self.current_graph.get_id_inventory()

        prompt = f"""
        你是一个指令翻译官。将自然语言的辩论意图转化为标准的图操作指令。

        {rules}

        【当前图谱中的有效 ID 清单 (Valid IDs)】:
        {id_inventory}

        【待转化意图】："{intent}"

        【翻译指南】:
        1. **查表引用**: 当意图提到某个已有的事实或观点时，**必须**在上面的清单中找到对应的 UUID 并填入。严禁编造 ID。
        2. **新节点**: 只有当意图明确表示要“添加”新内容时，才使用 `ADD_CLAIM/LAW` 和临时别名。
        3. **格式**: 严格遵守 DSL，不保留方括号。

        请生成指令序列（一行一条，分号分隔）：
        """

        response = self.llm([Message(role="user", content=prompt)], temperature=0.0)
        clean_script = response.replace("```", "").strip()
        return clean_script