"""Loads and parses legal case data from files into structured objects.

This module provides the `CaseDataLoader` class, which is responsible for
reading raw legal case data from a directory of JSON or JSONL files. It handles
different data schemas, performs cleaning and validation, and yields structured
`CaseData` objects for use in the application.
"""

import json
import os
from typing import Generator, List, Optional

from pydantic import ValidationError

from .schema import CaseData


class CaseDataLoader:
    """Loads, parses, and validates legal case data from a specified directory.

    This class iterates through `.json` and `.jsonl` files in a given directory,
    parsing each line as a potential case data entry. It is robust to parsing
    errors and schema variations, skipping invalid lines and logging warnings.

    Attributes:
        data_dir: The path to the directory containing the case data files.
    """

    def __init__(self, data_dir: str):
        """Initialize the CaseDataLoader.

        Args:
            data_dir: The path to the directory where the case data files
                (e.g., cleaned_samples.jsonl) are stored.
        """
        self.data_dir = data_dir

    def _clean_text(self, text: str) -> str:
        """Perform basic cleaning on a string field.

        Args:
            text: The input string to clean.

        Returns:
            The stripped string, or an empty string if the input is None.
        """
        if not text:
            return ""

        return text.strip()

    def _is_valid_field(self, field_value: any) -> bool:
        """Check if a field contains a meaningful value.

        Validation fails if the value is falsy (e.g., None, empty list/string)
        or if it is a string containing "unknown" (case-insensitive).

        Args:
            field_value: The value of the field to validate.

        Returns:
            True if the field is considered valid, False otherwise.
        """
        if not field_value:
            return False

        if isinstance(field_value, str) and "unknown" in field_value.lower():
            return False

        return True

    def _parse_single_line(self, line: str) -> Optional[CaseData]:
        """Parse and validate a single line of JSON data into a CaseData object.

        This method supports two primary JSON structures:
        1. A nested format with "metaInfo" and "content" keys.
        2. A flat format that maps directly to the `CaseData` schema.

        It performs validation on critical fields. If parsing or validation fails,
        it prints a warning and returns None.

        Args:
            line: A string containing a single JSON object.

        Returns:
            A `CaseData` object if parsing and validation are successful,
            otherwise None.
        """
        try:
            raw = json.loads(line)

            if "metaInfo" in raw and "content" in raw:
                meta = raw.get("metaInfo", {})
                content = raw.get("content", {})
                uid = meta.get("uid", "")
                title = meta.get("案件名称")
                cause = meta.get("案由", [])

                plaintiffs = [
                    p.get("pname")
                    for p in meta.get("人物信息", [])
                    if "原告" in p.get("ptypes", [])
                ]

                defendants = [
                    p.get("pname")
                    for p in meta.get("人物信息", [])
                    if "被告" in p.get("ptypes", [])
                ]

                p_claim = self._clean_text(content.get("原告诉称", ""))
                d_arg = self._clean_text(content.get("被告辩称", ""))
                fact_finding = self._clean_text(content.get("审理查明", ""))
                court_opinion = self._clean_text(content.get("法院观点", ""))
                verdict_result = self._clean_text(content.get("裁判结果", ""))
                cited_laws = meta.get("法律条款", [])

                core_fields_to_check = [
                    uid,
                    title,
                    cause,
                    plaintiffs,
                    defendants,
                    p_claim,
                    d_arg,
                    fact_finding,
                    court_opinion,
                    verdict_result,
                    cited_laws,
                ]

                if "辩称" not in d_arg or not all(
                    self._is_valid_field(field) for field in core_fields_to_check
                ):
                    return None

                return CaseData(
                    uid=uid,
                    title=title,
                    cause=cause,
                    plaintiffs=plaintiffs,
                    defendants=defendants,
                    plaintiff_claim=p_claim,
                    defendant_argument=d_arg,
                    fact_finding=fact_finding,
                    court_opinion=court_opinion,
                    verdict_result=verdict_result,
                    cited_laws=cited_laws,
                )

            else:
                return CaseData(**raw)

        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Warning: Skipping line due to parsing error: {e}")
            return None

        except Exception as e:
            print(f"Warning: Skipping line due to unexpected error: {e}")
            return None

    def load_all(self, limit: int = None) -> List[CaseData]:
        """Load all valid case data from the directory into a list.

        This method reads all `.json` and `.jsonl` files in the data directory,
        parses each line, and collects the valid `CaseData` objects.

        Args:
            limit: An optional integer to limit the number of cases loaded.

        Returns:
            A list of `CaseData` objects.

        Raises:
            FileNotFoundError: If the specified data directory does not exist.
        """
        results = []
        count = 0

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        for filename in os.listdir(self.data_dir):
            if not (filename.endswith(".json") or filename.endswith(".jsonl")):
                continue

            filepath = os.path.join(self.data_dir, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    case = self._parse_single_line(line)

                    if case:
                        results.append(case)
                        count += 1

                        if limit and count >= limit:
                            return results

        return results

    def stream(self) -> Generator[CaseData, None, None]:
        """Streams valid case data from the directory one by one.

        This method is memory-efficient for large datasets, as it yields
        cases instead of loading them all into memory at once.

        Yields:
            A `CaseData` object for each valid line in the data files.

        Raises:
            FileNotFoundError: If the specified data directory does not exist.
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        for filename in os.listdir(self.data_dir):
            if not (filename.endswith(".json") or filename.endswith(".jsonl")):
                continue

            filepath = os.path.join(self.data_dir, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    case = self._parse_single_line(line)

                    if case:
                        yield case
