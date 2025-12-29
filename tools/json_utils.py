import json
import re
from typing import Any


def extract_json_from_text(text: str) -> Any:
    text = text.strip()
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)

    if match:
        json_str = match.group(1)

    else:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)

        if match:
            json_str = match.group(1)

        else:
            json_str = text

    try:
        return json.loads(json_str)

    except json.JSONDecodeError:
        p1 = json_str.find("{")
        p2 = json_str.rfind("}")

        if p1 != -1 and p2 != -1 and p2 > p1:
            candidate = json_str[p1 : p2 + 1]

            try:
                return json.loads(candidate)

            except json.JSONDecodeError:
                pass

        p3 = json_str.find("[")
        p4 = json_str.rfind("]")

        if p3 != -1 and p4 != -1 and p4 > p3:
            candidate = json_str[p3 : p4 + 1]

            try:
                return json.loads(candidate)

            except json.JSONDecodeError:
                pass

        raise
