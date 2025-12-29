import json
from typing import Union, List
from pydantic import ValidationError
from mas.schema import AgentAction
from tools.json_utils import extract_json_from_text

def parse_agent_action_output(raw_output: str) -> Union[List[AgentAction], str]:
    try:
        data = extract_json_from_text(raw_output)
        if isinstance(data, list): return [AgentAction.model_validate(item) for item in data]
        else: return [AgentAction.model_validate(data)]
            
    except json.JSONDecodeError as e: return f"解析失败：JSON格式错误，请确保输出严格符合JSON规范。错误信息: {e}。原始输出: {raw_output}"
    except ValidationError as e: return f"解析失败：数据校验不通过，请检查JSON字段名、类型和值是否符合AgentAction模型。错误信息: {e}。原始输出: {raw_output}"
    except Exception as e: return f"解析失败：发生未知错误。错误信息: {e}。原始输出: {raw_output}"