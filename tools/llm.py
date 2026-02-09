"""Provides a wrapper for interacting with OpenAI-compatible language models.

This module defines the `GPTChat` class, which simplifies making API calls to
language models. It handles client initialization, request formatting, error
handling with retries, and asynchronous execution. It also includes a simple
token counter for tracking usage.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional, Protocol

from openai import OpenAI

from mas.config import SystemConfig

_CONFIG = SystemConfig().llm
completion_tokens, prompt_tokens = 0, 0


@dataclass(frozen=True)
class Message:
    """Represents a single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class LLMCallable(Protocol):
    """A protocol defining the signature for a callable LLM object."""

    def __call__(
        self,
        messages: List[Message],
        temperature: float = None,
        max_tokens: int = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = 1,
    ) -> str:
        """Define the standard call signature for a language model.

        Any class implementing this protocol can be used as a drop-in
        replacement for an LLM client within the system.

        Args:
            messages: A list of `Message` objects representing the conversation history.
            temperature: The sampling temperature to control randomness.
            max_tokens: The maximum number of tokens to generate in the response.
            stop_strs: An optional list of strings at which to stop generation.
            num_comps: The number of completions to generate (typically, only the
                first one is used).

        Returns:
            The string content of the language model's response.
        """
        ...


class LLM(ABC):
    """Abstract base class for a language model client."""

    def __init__(
        self, model_name: str = None, base_url: str = None, api_key: str = None
    ):
        """Initialize the LLM client.

        Args:
            model_name: The name of the model to use.
            base_url: The base URL of the API endpoint.
            api_key: The API key for authentication.
        """
        self.model_name = model_name
        self._base_url = base_url if base_url else _CONFIG.base_url
        self._api_key = api_key if api_key else _CONFIG.api_key
        self._model_name = model_name or _CONFIG.model_name
        self.client = OpenAI(base_url=self._base_url, api_key=self._api_key)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> str:
        """Make the LLM instance callable.

        This is an abstract method that must be implemented by concrete subclasses.
        It should contain the core logic for making a request to the language
        model's API.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The string content of the language model's response.
        """
        pass


class GPTChat(LLM):
    """A concrete implementation for chat-based OpenAI-compatible models.

    This class provides synchronous and asynchronous methods to interact with
    chat models. It includes a retry mechanism for handling transient API errors
    like rate limiting.
    """

    def __init__(
        self, model_name: str = None, base_url: str = None, api_key: str = None
    ):
        """Initialize the GPTChat client."""
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key)

    def __call__(
        self,
        messages: List[Message],
        temperature: float = None,
        max_tokens: int = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = 1,
    ) -> str:
        """Make a synchronous chat completion request.

        Args:
            messages: A list of `Message` objects representing the conversation history.
            temperature: The sampling temperature. Defaults to system config.
            max_tokens: The maximum number of tokens to generate. Defaults to system config.
            stop_strs: A list of strings to stop generation at.
            num_comps: The number of completions to generate (always uses the first).

        Returns:
            The content of the assistant's response as a string, or an empty
            string if the request fails after all retries.
        """
        global prompt_tokens, completion_tokens
        final_temp = temperature if temperature is not None else _CONFIG.temperature
        final_max_tokens = max_tokens if max_tokens is not None else _CONFIG.max_tokens

        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

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
                    stop=stop_strs,
                )

                if not response.choices:
                    continue

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
                print(f"⚠️ [LLM Error] Attempt {attempt + 1}/{max_retries}: {error_msg}")
                traceback.print_exc()

                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    time.sleep(wait_time * (attempt + 1))

                else:
                    break

        return ""

    async def aask(
        self,
        prompt: str,
        system_msgs: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Make an asynchronous chat completion request with a simple prompt.

        This is a convenience wrapper around the synchronous `__call__` method,
        running it in a separate thread to avoid blocking the asyncio event loop.

        Args:
            prompt: The user's prompt string.
            system_msgs: An optional list of system message strings.
            max_tokens: The maximum number of tokens to generate.
            temperature: The sampling temperature.

        Returns:
            The content of the assistant's response as a string.
        """
        messages = []

        if system_msgs:
            for sm in system_msgs:
                messages.append(Message(role="system", content=sm))

        messages.append(Message(role="user", content=prompt))

        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            None, self.__call__, messages, temperature, max_tokens
        )


def get_price():
    """Return the total token counts since the application started."""
    return completion_tokens, prompt_tokens
