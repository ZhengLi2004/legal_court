import json
from dataclasses import dataclass, asdict
from typing import List, Tuple
from mas.llm import GPTChat, Message

@dataclass
class AgentPersona:
    role_name: str
    belief: str     # B: 坚信的事实/认知
    desire: str     # D: 想要达成的法律后果
    intention: str  # I: 具体的行动策略/风格
    initial_strategy: str

@dataclass
class InitializationResult:
    plaintiff_persona: AgentPersona
    defendant_persona: AgentPersona
    fact_actions: List[str]
    root_claim_actions: List[str]

class CaseInitializer:
    def __init__(self, llm: GPTChat): self.llm = llm
    # 拆解事实 -> 生成诉求 -> 生成 Persona
    async def initialize(self, fact_finding: str, cause: str) -> InitializationResult:
        fact_actions = await self._decompose_facts(fact_finding)
        root_actions = await self._generate_root_claim(fact_finding, cause)
        p_persona = await self._generate_persona(fact_finding, cause, "plaintiff")
        d_persona = await self._generate_persona(fact_finding, cause, "defendant")

        return InitializationResult(
            plaintiff_persona=p_persona,
            defendant_persona=d_persona,
            fact_actions=fact_actions,
            root_claim_actions=root_actions
        )
    
    async def _decompose_facts(self, text: str) -> List[str]:
        prompt = f"""
        你是一个数据预处理助手。请将以下【审理查明】的法律事实文本，拆解为多个独立的、原子的事实描述。
        每个事实应该包含时间、地点、人物或关键行为。

        【审理查明】：
        {text}

        请直接输出 ADD_FACT 指令列表，每行一条。
        格式: ADD_FACT("2023年5月1日，张三与李四签订合同")
        不要输出任何其他内容。
        """

        response = self.llm([Message(role="user", content=prompt)], temperature=0.0)
        return self._parse_lines(response, "ADD_FACT")

    async def _generate_root_claim(self, facts: str, cause: str) -> List[str]:
        prompt = f"""
        基于案由【{cause}】和以下事实，请提炼出原告的所有核心法律诉求。
        诉求应具体明确（如，要求被告偿还本金XX元）。

        【事实】：
        {facts}

        请直接输出 ADD_CLAIM 指令列表。
        格式: ADD_CLAIM("判令被告偿还借款本金10万元")
        """

        response = self.llm([Message(role="user", content=prompt)], temperature=0.0)
        return self._parse_lines(response, "ADD_CLAIM")
    
    async def _generate_persona(self, facts: str, cause: str, role: str) -> AgentPersona:
        role_cn = "原告" if role == "plaintiff" else "被告"

        prompt = f"""
        基于案由【{cause}】和事实，请为【{role_cn}】构建 BDI 画像和初始策略。

        【事实】：
        {facts}

        请严格按照以下 JSON 格式输出（不要输出 Markdown 标记）：
        {{
            "belief": "简述信念...",
            "desire": "简述愿望...",
            "intention": "简述风格/意图...",
            "strategy": "简述初始冷启动策略..."
        }}
        """

        response = self.llm([Message(role="user", content=prompt)], temperature=0.7)    # 稍微增加创造性

        try:
            clean_json = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)

            return AgentPersona(
                role_name=role,
                belief=data.get("belief", "N/A"),
                desire=data.get("desire", "N/A"),
                intention=data.get("intention", "N/A"),
                initial_strategy=data.get("strategy", "N/A")
            )
        
        except Exception as e:
            print(f"Error parsing persona for {role}: {e}\nResponse: {response}")
            return AgentPersona(role, "Default Belief", "Default Desire", "Default Intention", "Default Strategy")

    def _parse_lines(self, text: str, prefix: str) -> List[str]:
        lines = []

        for line in text.strip().split('\n'):
            clean = line.strip()
            if clean.startswith(prefix): lines.append(clean)
        
        return lines