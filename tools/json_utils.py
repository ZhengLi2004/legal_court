"""Provides a utility function for robustly extracting JSON from text.

This module contains the `extract_json_from_text` function, which is designed
to find and parse a JSON object or array that may be embedded within a larger
block of text, often with markdown code fences (```json ... ```).
"""

import json
import re
from typing import Any


def _fix_unescaped_quotes(json_str: str) -> str:
    """Fix unescaped ASCII double quotes inside JSON string values.

    This function uses a state machine to track JSON structure and
    identify quotes that need to be escaped. A quote inside a string value
    is identified by looking ahead to see if the next non-whitespace character
    is JSON punctuation (',', '}', ']', ':'). If not, the quote is inside
    the string value and needs to be escaped.

    Args:
        json_str: JSON string with potential unescaped quotes.

    Returns:
        Fixed JSON string with unescaped quotes properly escaped.
    """
    result = []
    i = 0
    in_string = False
    escape_next = False

    while i < len(json_str):
        c = json_str[i]

        if escape_next:
            result.append(c)
            escape_next = False
            i += 1
            continue

        if c == "\\":
            result.append(c)
            escape_next = True
            i += 1
            continue

        if c == '"':
            if in_string:
                j = i + 1

                while j < len(json_str) and json_str[j].isspace():
                    j += 1

                if j < len(json_str):
                    next_char = json_str[j]

                    if next_char in ",}]":
                        in_string = False
                        result.append(c)

                    elif next_char == ":":
                        in_string = False
                        result.append(c)

                    else:
                        result.append('\\"')

                else:
                    in_string = False
                    result.append(c)

            else:
                in_string = True
                result.append(c)

        else:
            result.append(c)

        i += 1

    return "".join(result)


def extract_json_from_text(text: str) -> Any:
    """Extract and parses a JSON object or array from a string.

    This function is designed to be resilient to common ways LLMs format JSON
    in their responses. It searches for JSON enclosed in markdown code fences
    (with or without the 'json' language identifier). If fences are not found,
    it assumes the whole string is JSON. If initial parsing fails, it attempts
    to fix unescaped quotes and retries. If that also fails, it attempts
    to find the outermost curly or square brackets and parse the content
    between them as a fallback.

    Args:
        text: The input string potentially containing an embedded JSON object or array.

    Returns:
        The parsed Python object (typically a `dict` or `list`).

    Raises:
        json.JSONDecodeError: If a JSON object or array cannot be successfully
            parsed from the text after all fallbacks have been attempted.
    """
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
        try:
            fixed_json_str = _fix_unescaped_quotes(json_str)
            return json.loads(fixed_json_str)

        except json.JSONDecodeError:
            pass

    p1 = json_str.find("{")
    p2 = json_str.rfind("}")

    if p1 != -1 and p2 != -1 and p2 > p1:
        candidate = json_str[p1 : p2 + 1]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            try:
                fixed_candidate = _fix_unescaped_quotes(candidate)
                return json.loads(fixed_candidate)

            except json.JSONDecodeError:
                pass

    p3 = json_str.find("[")
    p4 = json_str.rfind("]")

    if p3 != -1 and p4 != -1 and p4 > p3:
        candidate = json_str[p3 : p4 + 1]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            try:
                fixed_candidate = _fix_unescaped_quotes(candidate)
                return json.loads(fixed_candidate)

            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError("Failed to extract JSON from text", text, 0)
