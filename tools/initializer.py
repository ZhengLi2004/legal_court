import json
import re
from dataclasses import dataclass
from typing import List
from mas.llm import GPTChat, Message
from mas.schema import AgentAction, AgentActionType
from metagpt.logs import logger

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
    fact_statements: List[str]
    root_claim_actions: List[AgentAction]

class CaseInitializer:
    def __init__(self, llm: GPTChat): self.llm = llm
    
    async def initialize(self, fact_finding: str, cause: str) -> InitializationResult:
        fact_statements = await self._decompose_facts(fact_finding)
        root_actions = await self._generate_root_claim(fact_finding, cause)
        p_persona = await self._generate_persona(fact_finding, cause, "plaintiff")
        d_persona = await self._generate_persona(fact_finding, cause, "defendant")

        return InitializationResult(
            plaintiff_persona=p_persona,
            defendant_persona=d_persona,
            fact_statements=fact_statements,
            root_claim_actions=root_actions
        )
    
    async def _parse_numbered_list_to_agent_actions(self, text: str) -> List[str]:
        facts = []
        lines = text.strip().split('\n')
        current_fact_content = []
    
        for line in lines:
            line = line.strip()
            if not line: continue
            match = re.match(r"^\d+\.\s*(.*)", line)
    
            if match:
                if current_fact_content: facts.append(" ".join(current_fact_content).strip())
                current_fact_content = [match.group(1)]
    
            elif current_fact_content: current_fact_content.append(line)
    
        if current_fact_content: facts.append(" ".join(current_fact_content).strip())
        return facts
    
    async def _decompose_facts(self, text: str) -> List[str]:
        prompt = f"""
        你是一个数据预处理助手。请将以下【审理查明】的法律事实文本，拆解为多个独立的、原子的事实描述。
    
        每个事实应该包含时间、地点、人物或关键行为。
    
        【审理查明】：
        {text}
    
        请直接输出一个编号列表，每个条目是一个独立的事实描述。

        例如：
        1. 2023年5月1日，张三与李四签订合同
        2. ...

        不要输出任何其他内容或 Markdown 标记外的文字。
        """

        response = self.llm([Message(role="user", content=prompt)], temperature=0.0)

        try: return await self._parse_numbered_list_to_agent_actions(response)

        except Exception as e:
            logger.error(f"Error parsing decomposed facts from numbered list: {e}\nResponse: {response}")
            return []

    async def _generate_root_claim(self, facts: str, cause: str) -> List[AgentAction]:
        prompt = f"""
        基于案由【{cause}】和以下事实，请提炼出原告的所有核心法律诉求。
        诉求应具体明确（如，要求被告偿还本金XX元）。

        【事实】：
        {facts}

        请直接输出一个 JSON 数组，每个元素是一个 AgentAction 对象，表示一个 ADD_CLAIM 操作。
        **AgentAction 对象的 action_type 必须是 "add_claim"**。
        AgentAction 模型定义（仅限 action_type="add_claim"）：
        {{
            "action_type": "add_claim",
            "content": "主张的详细描述，例如：判令被告偿还借款本金10万元",
            "target_id": null,
            "source_id": null,
            "relation_type": null
        }}
        不要输出任何其他内容或 Markdown 标记外的文字。
        """

        response = self.llm([Message(role="user", content=prompt)], temperature=0.0)
        
        try:
            clean_json = response
            match = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", response, re.DOTALL)
            if match: clean_json = match.group(1)
            else: clean_json = response.replace("```json", "").replace("```", "").strip()
            actions_data = json.loads(clean_json)
            if not isinstance(actions_data, list): raise ValueError("LLM did not return a JSON array for ADD_CLAIM.")
            actions = []
            
            for data in actions_data:
                action = AgentAction(**data)
                action.metadata["is_root_claim"] = True
                actions.append(action)
            
            return actions
        
        except Exception as e:
            logger.error(f"Error parsing root claims into AgentAction: {e}\nResponse: {response}")
            return []
    
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
            clean_json = response
            match = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", response, re.DOTALL)
            if match:
                clean_json = match.group(1)
            else:
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
            logger.error(f"Error parsing persona for {role}: {e}\nResponse: {response}")
            return AgentPersona(role, "Default Belief", "Default Desire", "Default Intention", "Default Strategy")