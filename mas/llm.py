import time
import asyncio
from typing import Protocol, Literal, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from openai import OpenAI
from .config import SystemConfig
_CONFIG = SystemConfig().llm
completion_tokens, prompt_tokens = 0, 0

@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

class LLMCallable(Protocol):
    def __call__(
        self,
        messages: List[Message],
        temperature: float = None,
        max_tokens: int = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = 1
    ) -> str: pass

class LLM(ABC):
    def __init__(self, model_name: str = None, base_url: str = None, api_key: str = None):
        self.model_name = model_name
        self._base_url = base_url if base_url else _CONFIG.base_url
        self._api_key = api_key if api_key else _CONFIG.api_key
        self._model_name = model_name or _CONFIG.model_name

        self.client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key
        )

    @abstractmethod
    def __call__(self, *args, **kwargs) -> str: pass

class GPTChat(LLM):
    def __init__(self, model_name: str = None, base_url: str = None, api_key: str = None): super().__init__(model_name=model_name, base_url=base_url, api_key=api_key) 

    def __call__(
        self,
        messages: List[Message],
        temperature: float = None,
        max_tokens: int = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = 1
    ) -> str:
        global prompt_tokens, completion_tokens
        final_temp = temperature if temperature is not None else _CONFIG.temperature
        final_max_tokens = max_tokens if max_tokens is not None else _CONFIG.max_tokens
        openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        max_retries = 3
        wait_time = 2

        for attempt in range(max_retries):
            try:                
                response = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=openai_messages,
                    max_tokens=final_max_tokens,
                    temperature=final_temp,
                    n=num_comps,
                    stop=stop_strs
                )

                if not response.choices: continue

                answer = response.choices[0].message.content
                
                if response.usage:
                    prompt_tokens += response.usage.prompt_tokens
                    completion_tokens += response.usage.completion_tokens
                
                if answer is None:
                    print("Error: LLM returned None")
                    continue
                
                return answer

            except Exception as e:
                import traceback
                error_msg = str(e)
                print(f"⚠️ [LLM Error] Attempt {attempt+1}/{max_retries}: {error_msg}")
                traceback.print_exc()
                if "rate limit" in error_msg.lower() or "429" in error_msg: time.sleep(wait_time * (attempt + 1))
                else: break 

        return ""
    
    async def aask(self, prompt:str, system_msgs: Optional[List[str]] = None) -> str:
        messages = []

        if system_msgs:
            for sm in system_msgs: messages.append(Message(role="system", content=sm))
        
        messages.append(Message(role="user", content=prompt))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.__call__, messages)
    
def get_price(): return completion_tokens, prompt_tokens