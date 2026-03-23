"""Defines judge abstractions used for final adjudication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from mas.infrastructure.llm import GPTChat
from prompts.common_prompts import (
    JUDGE_DIRECT_VERDICT_PROMPT,
    SYSTEM_PROMPT_DIRECT_JUDGE,
)

from ..core.graph import NodeStatus, ShadowGraph

_DIRECT_VERDICT_SCHEMA = {
    "name": "direct_claim_adjudication",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["ACCEPTED", "REJECTED", "UNMENTIONED"],
                        },
                    },
                    "required": ["claim_id", "status"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    },
}


class BaseJudge(ABC):
    """Abstract interface for direct root-claim adjudication."""

    @abstractmethod
    async def adjudicate(self, graph: ShadowGraph) -> Dict[str, NodeStatus]:
        """Return one final verdict status for every root claim in the graph."""


class LLMJudge(BaseJudge):
    """Judge implementation backed by one strict JSON-schema completion."""

    def __init__(self, llm: GPTChat):
        self.llm = llm

    def _build_adjudication_input(self, graph: ShadowGraph) -> Dict[str, Any]:
        root_claims: Dict[str, str] = {}

        for node_id, data in graph.graph.nodes(data=True):
            metadata = data.get("metadata", {})

            if metadata.get("is_root_claim", False):
                root_claims[str(node_id)] = str(data.get("content", "未知诉求"))

        if not root_claims:
            raise ValueError("No root claims found in graph metadata.")

        issue_list = "\n".join(
            [
                f"{idx + 1}. [{claim_id}] {content}"
                for idx, (claim_id, content) in enumerate(
                    sorted(root_claims.items(), key=lambda row: row[0])
                )
            ]
        )

        return {
            "root_claims": root_claims,
            "issue_list": issue_list,
            "graph_context": graph.to_recursive_text(),
        }

    @staticmethod
    def _normalize_status(raw_status: str) -> NodeStatus:
        status = str(raw_status or "").upper().strip()

        if status == "ACCEPTED":
            return NodeStatus.VALIDATED

        if status == "REJECTED":
            return NodeStatus.DEFEATED

        if status == "UNMENTIONED":
            return NodeStatus.HYPOTHETICAL

        raise ValueError(f"Invalid adjudication status: {raw_status}")

    async def adjudicate(self, graph: ShadowGraph) -> Dict[str, NodeStatus]:
        payload = self._build_adjudication_input(graph)

        prompt = JUDGE_DIRECT_VERDICT_PROMPT.format(
            issue_list=payload["issue_list"],
            graph_context=payload["graph_context"],
        )

        response = await self.llm.aask_json_schema(
            prompt,
            schema=_DIRECT_VERDICT_SCHEMA,
            system_msgs=[SYSTEM_PROMPT_DIRECT_JUDGE],
            temperature=0,
        )

        if not isinstance(response, dict):
            raise ValueError("Judge response must be a JSON object.")

        claim_rows = response.get("claims")

        if not isinstance(claim_rows, list):
            raise ValueError("Judge response `claims` must be a list.")

        expected_ids = set(payload["root_claims"].keys())
        seen_ids: set[str] = set()
        verdicts: Dict[str, NodeStatus] = {}

        for row in claim_rows:
            if not isinstance(row, dict):
                raise ValueError("Each adjudication row must be a JSON object.")

            claim_id = str(row.get("claim_id", "") or "").strip()

            if not claim_id:
                raise ValueError("Adjudication row missing non-empty `claim_id`.")

            if claim_id in seen_ids:
                raise ValueError(f"Duplicate adjudication claim_id: {claim_id}")

            seen_ids.add(claim_id)

            if claim_id not in expected_ids:
                raise ValueError(f"Unexpected adjudication claim_id: {claim_id}")

            verdicts[claim_id] = self._normalize_status(str(row.get("status", "")))

        missing_ids = sorted(expected_ids - seen_ids)

        if missing_ids:
            raise ValueError(f"Missing adjudication claim_ids: {missing_ids}")

        return verdicts
