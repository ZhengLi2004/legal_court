"""Provides a utility function for robustly extracting JSON from text.

This module contains the `extract_json_from_text` function, which is designed
to find and parse a JSON object or array that may be embedded within a larger
block of text, often with markdown code fences (```json ... ```).
"""

import json
import re
from typing import Any


def extract_json_from_text(text: str) -> Any:
    """Extract and parses a JSON object or array from a string.

    This function is designed to be resilient to common ways LLMs format JSON
    in their responses. It searches for JSON enclosed in markdown code fences
    (with or without the 'json' language identifier). If fences are not found,
    it assumes the whole string is JSON. If initial parsing fails, it attempts
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
