"""Data loading helpers for experiment pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_cases_from_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of case dictionaries.

    Args:
        path: JSONL file path containing one case object per line.

    Returns:
        Parsed case dictionaries in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If one line is not valid JSON or not a JSON object.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    cases: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} in {file_path}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no} in {file_path} is not a JSON object")

            cases.append(record)

    return cases
