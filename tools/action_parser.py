"""Parses and validates raw language model output into structured AgentAction objects.

This module provides a robust parsing function that takes the string output
from a language model, which is expected to be a JSON object or array, and
converts it into a list of `AgentAction` Pydantic models. It handles potential
JSON decoding errors and Pydantic validation errors gracefully, returning
a descriptive error string in case of failure.
"""

import json
from typing import List, Union

from pydantic import ValidationError

from mas.core.schemas import AgentAction
from tools.json_utils import extract_json_from_text


def parse_agent_action_output(raw_output: str) -> Union[List[AgentAction], str]:
    """Parse a raw string from an LLM into a list of AgentAction objects.

    The function first attempts to extract a JSON block from the raw text. It then
    parses the JSON and validates it against the `AgentAction` schema. It can
    handle both a single JSON object and a JSON array of objects.

    Args:
        raw_output: The raw string output from the language model.

    Returns:
        A list of validated `AgentAction` objects if parsing and validation
        are successful.
        A descriptive error string if any part of the process fails.
    """
    try:
        data = extract_json_from_text(raw_output)

        if isinstance(data, list):
            return [AgentAction.model_validate(item) for item in data]

        return [AgentAction.model_validate(data)]

    except json.JSONDecodeError as e:
        return f"解析失败：JSON格式错误，请确保输出严格符合JSON规范。错误信息: {e}。原始输出: {raw_output}"

    except ValidationError as e:
        return f"解析失败：数据校验不通过，请检查JSON字段名、类型和值是否符合AgentAction模型。错误信息: {e}。原始输出: {raw_output}"

    except Exception as e:
        return f"解析失败：发生未知错误。错误信息: {e}。原始输出: {raw_output}"
