import re
from dataclasses import dataclass
from typing import List
from mas.llm import GPTChat, Message
from metagpt.logs import logger
from tools.json_utils import extract_json_from_text
from prompts.common_prompts import DECOMPOSE_FACTS_PROMPT, GENERATE_ROOT_CLAIM_PROMPT, GENERATE_PERSONA_PROMPT

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
    root_claim_actions: List[str]

class CaseInitializer:
    def __init__(self, llm: GPTChat): self.llm = llm
    
    async def initialize(self, fact_finding: str, cause: str) -> InitializationResult:
        fact_statements = await self._decompose_facts(fact_finding)
        root_claim_texts = await self._generate_root_claim(fact_finding, cause)
        p_persona = await self._generate_persona(fact_finding, cause, "plaintiff")
        d_persona = await self._generate_persona(fact_finding, cause, "defendant")

        return InitializationResult(
            plaintiff_persona=p_persona,
            defendant_persona=d_persona,
            fact_statements=fact_statements,
            root_claim_actions=root_claim_texts
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
        prompt = DECOMPOSE_FACTS_PROMPT.format(text=text)
        response = self.llm([Message(role="user", content=prompt)], temperature=0.0)

        try: return await self._parse_numbered_list_to_agent_actions(response)

        except Exception as e:
            logger.error(f"Error parsing decomposed facts from numbered list: {e}\nResponse: {response}")
            return []

    async def _generate_root_claim(self, facts: str, cause: str) -> List[str]:
        prompt = GENERATE_ROOT_CLAIM_PROMPT.format(cause=cause, facts=facts)
        response = self.llm([Message(role="user", content=prompt)], temperature=0.0)
        
        try:
            claims_list = extract_json_from_text(response)
            if not isinstance(claims_list, list) or not all(isinstance(item, str) for item in claims_list): raise ValueError("LLM did not return a JSON array of strings.")
            return claims_list
        
        except Exception as e:
            logger.error(f"Error parsing root claim texts: {e}\nResponse: {response}")
            return []
    
    async def _generate_persona(self, facts: str, cause: str, role: str) -> AgentPersona:
        role_cn = "原告" if role == "plaintiff" else "被告"

        prompt = GENERATE_PERSONA_PROMPT.format(cause=cause, role_cn=role_cn, facts=facts)
        response = self.llm([Message(role="user", content=prompt)], temperature=0.7)

        try:
            data = extract_json_from_text(response)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict): data = data[0]

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