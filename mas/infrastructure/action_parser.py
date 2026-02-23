"""Parses strict JSON action output into structured AgentAction objects."""

import json
from typing import Any, Dict, List, Union

from pydantic import ValidationError

from mas.core.schemas import AgentAction


def _load_action_payload(raw_output: Union[str, Dict[str, Any], List[Any]]) -> Any:
    """Load action payload from strict JSON string or already-decoded data."""
    if isinstance(raw_output, (dict, list)):
        return raw_output

    if not isinstance(raw_output, str):
        raise TypeError(
            f"Unsupported raw_output type: {type(raw_output).__name__}. Expected str|dict|list."
        )

    return json.loads(raw_output)


def parse_agent_action_output(
    raw_output: Union[str, Dict[str, Any], List[Any]],
) -> Union[List[AgentAction], str]:
    """Parse strict JSON action payload into `AgentAction` objects.

    Args:
        raw_output: Raw model output as JSON string, object, or object list.

    Returns:
        List of parsed `AgentAction` objects on success, otherwise an error
        message string in Chinese.

    Raises:
        Assumption/Unverified: The function catches and stringifies all errors,
            so no parsing exception is propagated to callers.
    """
    try:
        data = _load_action_payload(raw_output)

        if isinstance(data, list):
            return [AgentAction.model_validate(item) for item in data]

        if isinstance(data, dict):
            return [AgentAction.model_validate(data)]

        raise ValueError(
            f"解析失败：JSON根结构必须是对象或数组，当前为 {type(data).__name__}。"
        )

    except json.JSONDecodeError as e:
        return f"解析失败：JSON格式错误，请确保输出严格符合JSON规范。错误信息: {e}。原始输出: {raw_output}"

    except ValidationError as e:
        return f"解析失败：数据校验不通过，请检查JSON字段名、类型和值是否符合AgentAction模型。错误信息: {e}。原始输出: {raw_output}"

    except Exception as e:
        return f"解析失败：发生未知错误。错误信息: {e}。原始输出: {raw_output}"
