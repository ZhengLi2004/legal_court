import json
from typing import Union, List
from pydantic import ValidationError
from mas.schema import AgentAction

def parse_agent_action_output(raw_output: str) -> Union[List[AgentAction], str]:
    try:
        clean_output = raw_output.strip()
        if clean_output.startswith("```json"): clean_output = clean_output[len("```json"):].strip()
        if clean_output.endswith("```"): clean_output = clean_output[:-len("```")].strip()
        if not clean_output: return "解析失败：输入为空或只包含代码块标记。原始输出: " + raw_output
        
        try:
            actions_data = json.loads(clean_output)
            if not isinstance(actions_data, list): raise ValueError("Expected a JSON array or a single JSON object.")
            parsed_actions = [AgentAction.model_validate(item) for item in actions_data]
            return parsed_actions
        
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            action_data = json.loads(clean_output)
            parsed_action = AgentAction.model_validate(action_data)
            return [parsed_action]
            
    except json.JSONDecodeError as e: return f"解析失败：JSON格式错误，请确保输出严格符合JSON规范。错误信息: {e}。原始输出: {raw_output}"
    except ValidationError as e: return f"解析失败：数据校验不通过，请检查JSON字段名、类型和值是否符合AgentAction模型。错误信息: {e}。原始输出: {raw_output}"
    except Exception as e: return f"解析失败：发生未知错误。错误信息: {e}。原始输出: {raw_output}"