"""Provide wrappers for OpenAI-compatible chat models.

The module exposes synchronous/asynchronous helpers for plain text, strict JSON
responses, and strict function-calling responses. Function-calling helpers use
contract validation and never fall back to plain JSON prompting.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol

from openai import OpenAI

from mas.config import SystemConfig

_CONFIG = SystemConfig().llm
completion_tokens, prompt_tokens = 0, 0


@dataclass(frozen=True)
class Message:
    """Represents a single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class ToolCallContractError(ValueError):
    """Raised when the model response violates the expected tool-call contract."""


@dataclass(frozen=True)
class ToolCallResult:
    """Store one validated tool-call result from the model.

    Attributes:
        tool_name: The function name emitted by the model.
        arguments: Parsed JSON object from `function.arguments`.
        raw_message: The raw first choice message object from the OpenAI SDK.
    """

    tool_name: str
    arguments: Dict[str, Any]
    raw_message: Any


class LLMCallable(Protocol):
    """Define callable interfaces expected by MAS LLM clients.

    Implementations must support plain chat generation and may expose strict
    function-calling helpers used by routing-sensitive code paths.
    """

    def __call__(
        self,
        messages: List[Message],
        temperature: float = None,
        max_tokens: int = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = 1,
        response_format: Optional[Dict[str, Any]] = None,
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
            response_format: Optional structured output constraint for compatible
                OpenAI-style APIs.

        Returns:
            The string content of the language model's response.
        """
        ...

    def ask_tool_call(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        tool_choice: str,
        temperature: float = None,
        max_tokens: int = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = 1,
    ) -> ToolCallResult:
        """Request one strict function call and parse structured arguments.

        Args:
            messages: Conversation history.
            tools: OpenAI-compatible tools payload.
            tool_choice: Required function name for the completion.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            stop_strs: Optional stop strings.
            num_comps: Number of completions requested.

        Returns:
            One validated tool-call result.
        """
        ...

    async def aask_tool_call(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        tool_choice: str,
        system_msgs: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ToolCallResult:
        """Asynchronously request one strict function call.

        Args:
            prompt: User prompt content.
            tools: OpenAI-compatible tools payload.
            tool_choice: Required function name for the completion.
            system_msgs: Optional system prompts.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            One validated tool-call result.
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
    def __call__(self, *_args, **_kwargs) -> str:
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
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Make a synchronous chat completion request.

        Args:
            messages: A list of `Message` objects representing the conversation history.
            temperature: The sampling temperature. Defaults to system config.
            max_tokens: The maximum number of tokens to generate. Defaults to system config.
            stop_strs: A list of strings to stop generation at.
            num_comps: The number of completions to generate (always uses the first).
            response_format: Optional OpenAI-style response format constraint,
                e.g. {"type": "json_object"} or {"type": "json_schema", ...}.

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
                request_kwargs: Dict[str, Any] = {}

                if response_format is not None:
                    request_kwargs["response_format"] = response_format

                response = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=openai_messages,
                    max_tokens=final_max_tokens,
                    temperature=final_temp,
                    n=num_comps,
                    stop=stop_strs,
                    **request_kwargs,
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

    @staticmethod
    def _build_forced_tool_choice(tool_name: str) -> Dict[str, Any]:
        """Build OpenAI `tool_choice` payload for one forced function.

        Args:
            tool_name: Function name that the model must call.

        Returns:
            OpenAI-compatible `tool_choice` object.
        """
        return {"type": "function", "function": {"name": tool_name}}

    @staticmethod
    def _extract_single_tool_call(
        message: Any,
        expected_tool_name: str,
    ) -> ToolCallResult:
        """Extract and validate one tool call from a chat-completion message.

        Args:
            message: OpenAI SDK message object from the first completion choice.
            expected_tool_name: Function name required by caller-side contract.

        Returns:
            A validated `ToolCallResult` object.

        Raises:
            ToolCallContractError: If the model does not return exactly one tool
                call, returns a mismatched function name, or emits invalid JSON
                function arguments.
        """
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            raise ToolCallContractError("LLM returned no tool_calls.")

        if len(tool_calls) != 1:
            raise ToolCallContractError(
                f"LLM must return exactly one tool call, got {len(tool_calls)}."
            )

        tool_call = tool_calls[0]
        function_payload = getattr(tool_call, "function", None)

        if function_payload is None:
            raise ToolCallContractError("LLM tool_call missing function payload.")

        tool_name = getattr(function_payload, "name", "")

        if tool_name != expected_tool_name:
            raise ToolCallContractError(
                f"LLM returned unexpected tool '{tool_name}', expected "
                f"'{expected_tool_name}'."
            )

        raw_arguments = getattr(function_payload, "arguments", None)

        if not isinstance(raw_arguments, str) or not raw_arguments.strip():
            raise ToolCallContractError("LLM returned empty tool_call arguments.")

        try:
            arguments = json.loads(raw_arguments)

        except json.JSONDecodeError as exc:
            raise ToolCallContractError(
                f"LLM tool_call arguments JSON decode failed: {exc}"
            ) from exc

        if not isinstance(arguments, dict):
            raise ToolCallContractError(
                "LLM tool_call arguments must be a JSON object."
            )

        return ToolCallResult(
            tool_name=tool_name,
            arguments=arguments,
            raw_message=message,
        )

    @staticmethod
    def _parse_json_output(raw: str) -> Any:
        """Parse a JSON string and raise clear errors for invalid outputs."""
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("LLM returned empty or non-string JSON content.")

        try:
            return json.loads(raw)

        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM JSON decode failed: {exc}") from exc

    def ask_tool_call(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        tool_choice: str,
        temperature: float = None,
        max_tokens: int = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = 1,
    ) -> ToolCallResult:
        """Request one strict function call and return parsed arguments.

        Args:
            messages: Conversation history.
            tools: OpenAI-compatible `tools` payload. Expected to contain the
                function schema referenced by `tool_choice`.
            tool_choice: Required function name that the model must call.
            temperature: Sampling temperature. Defaults to system config.
            max_tokens: Maximum output tokens. Defaults to system config.
            stop_strs: Optional stop strings passed to completion API.
            num_comps: Number of completions requested; only first choice is used.

        Returns:
            A validated `ToolCallResult` with parsed object arguments.

        Raises:
            ToolCallContractError: If API call fails after retries or response
                violates the strict single-tool-call contract.
        """
        global prompt_tokens, completion_tokens
        final_temp = temperature if temperature is not None else _CONFIG.temperature
        final_max_tokens = max_tokens if max_tokens is not None else _CONFIG.max_tokens

        if not tools:
            raise ToolCallContractError("`tools` must contain at least one function.")

        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        max_retries = 3
        wait_time = 2
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=openai_messages,
                    max_tokens=final_max_tokens,
                    temperature=final_temp,
                    n=num_comps,
                    stop=stop_strs,
                    tools=tools,
                    tool_choice=self._build_forced_tool_choice(tool_choice),
                )

                if response.usage:
                    prompt_tokens += response.usage.prompt_tokens
                    completion_tokens += response.usage.completion_tokens

                if not response.choices:
                    last_error = ToolCallContractError("LLM returned no choices.")
                    continue

                first_message = response.choices[0].message

                return self._extract_single_tool_call(
                    message=first_message,
                    expected_tool_name=tool_choice,
                )

            except ToolCallContractError:
                raise

            except Exception as e:
                import traceback

                last_error = e
                error_msg = str(e)
                print(f"⚠️ [LLM Error] Attempt {attempt + 1}/{max_retries}: {error_msg}")
                traceback.print_exc()

                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    time.sleep(wait_time * (attempt + 1))

                else:
                    break

        if last_error is None:
            raise ToolCallContractError("LLM failed to return a valid tool call.")

        raise ToolCallContractError(
            f"LLM failed to return a valid tool call after retries: {last_error}"
        ) from last_error

    async def aask(
        self,
        prompt: str,
        system_msgs: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Make an asynchronous chat completion request with a simple prompt.

        This is a convenience wrapper around the synchronous `__call__` method,
        running it in a separate thread to avoid blocking the asyncio event loop.

        Args:
            prompt: The user's prompt string.
            system_msgs: An optional list of system message strings.
            max_tokens: The maximum number of tokens to generate.
            temperature: The sampling temperature.
            response_format: Optional OpenAI-style response format constraint.

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
            None,
            lambda: self.__call__(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            ),
        )

    async def aask_tool_call(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        tool_choice: str,
        system_msgs: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ToolCallResult:
        """Asynchronously request one strict function call.

        Args:
            prompt: User prompt content.
            tools: OpenAI-compatible `tools` payload.
            tool_choice: Required function name emitted by the model.
            system_msgs: Optional system prompts.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            A validated `ToolCallResult`.

        Raises:
            ToolCallContractError: If the tool-call contract is not satisfied.
        """
        messages = []

        if system_msgs:
            for sm in system_msgs:
                messages.append(Message(role="system", content=sm))

        messages.append(Message(role="user", content=prompt))
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            None,
            lambda: self.ask_tool_call(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )

    def ask_json_object(
        self,
        messages: List[Message],
        temperature: float = None,
        max_tokens: int = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = 1,
    ) -> Dict[str, Any]:
        """Request strict JSON-object output and return parsed dict."""
        raw = self.__call__(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_strs=stop_strs,
            num_comps=num_comps,
            response_format={"type": "json_object"},
        )

        data = self._parse_json_output(raw)

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}.")

        return data

    def ask_json_schema(
        self,
        messages: List[Message],
        schema: Dict[str, Any],
        temperature: float = None,
        max_tokens: int = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = 1,
    ) -> Any:
        """Request strict JSON-schema output and return parsed JSON value."""
        raw = self.__call__(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_strs=stop_strs,
            num_comps=num_comps,
            response_format={"type": "json_schema", "json_schema": schema},
        )

        return self._parse_json_output(raw)

    async def aask_json_object(
        self,
        prompt: str,
        system_msgs: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Async helper for strict JSON-object outputs."""
        raw = await self.aask(
            prompt=prompt,
            system_msgs=system_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        data = self._parse_json_output(raw)

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}.")

        return data

    async def aask_json_schema(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_msgs: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Any:
        """Async helper for strict JSON-schema outputs."""
        raw = await self.aask(
            prompt=prompt,
            system_msgs=system_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_schema", "json_schema": schema},
        )

        return self._parse_json_output(raw)


def get_price():
    """Return the total token counts since the application started."""
    return completion_tokens, prompt_tokens
