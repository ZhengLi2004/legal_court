"""External naive MAD baseline for Claim 1 experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.experiments.methods._external_common import (
    SYSTEM_PROMPT_WORKER_ANALYST,
    _adjudicate_claim_statuses,
    _build_adjudicator_prompt,
    _build_claim_inventory,
    _build_claim_result_rows,
    _build_debate_turn_prompt,
    _build_result,
    _chat_text,
    prepare_external_method_context,
)


def run_external_naive_mad(
    *,
    case: Mapping[str, Any],
    storage_root_dir: str | None = None,
    budget: Mapping[str, Any] | None = None,
    seed: int | None = None,
    retrieval_config: Mapping[str, Any] | None = None,
    verbose: bool = False,
    test_mode_no_learning: bool = False,
) -> dict[str, Any]:
    """Run one independent experiment-side MAD baseline without memory or retrieval."""
    del verbose, test_mode_no_learning

    (
        _cfg,
        llm,
        prepared_case,
        root_claim_rows,
        warnings,
        started_at,
        start_completion,
        start_prompt,
    ) = prepare_external_method_context(
        case=case,
        storage_root_dir=storage_root_dir,
        seed=seed,
    )

    case_uid_value = str(prepared_case["uid"])
    fact_finding = str(prepared_case.get("fact_finding", "") or "")
    cause_value = prepared_case.get("cause", ["未知案由"])

    if isinstance(cause_value, list):
        cause = str(cause_value[0] if cause_value else "未知案由")

    else:
        cause = str(cause_value or "未知案由")

    claim_inventory = _build_claim_inventory(root_claim_rows)
    expected_claim_ids = [row["claim_id"] for row in claim_inventory]
    max_turns = max(1, int((budget or {}).get("max_turns", 10)))
    transcript: list[str] = []

    for turn_index in range(max_turns):
        is_plaintiff = turn_index % 2 == 0
        speaker = "Plaintiff Advocate" if is_plaintiff else "Defendant Advocate"

        turn_prompt = _build_debate_turn_prompt(
            side_label=speaker,
            fact_finding=fact_finding,
            claims=claim_inventory,
            transcript=transcript,
        )

        turn_text = _chat_text(
            llm=llm,
            system_prompt=SYSTEM_PROMPT_WORKER_ANALYST,
            prompt=turn_prompt,
        )

        transcript.append(f"{speaker}: {turn_text}")

    adjudication_prompt = _build_adjudicator_prompt(
        case_uid_value=case_uid_value,
        cause=cause,
        fact_finding=fact_finding,
        claims=claim_inventory,
        transcript=transcript,
    )

    adjudication_rows = _adjudicate_claim_statuses(
        llm=llm,
        prompt=adjudication_prompt,
        expected_claim_ids=expected_claim_ids,
    )

    claim_rows, status_rows = _build_claim_result_rows(
        case_uid_value=case_uid_value,
        root_claim_rows=root_claim_rows,
        adjudication_rows=adjudication_rows,
    )

    return _build_result(
        method_name="external_naive_mad",
        case_uid_value=case_uid_value,
        claim_rows=claim_rows,
        status_rows=status_rows,
        trace={
            "baseline_family": "external",
            "baseline_mode": "naive_mad_external",
            "debate_turn_count": len(transcript),
            "adjudication_mode": "direct_status_json",
            "retrieved_case_hits": [],
            "retrieved_law_hits": [],
            "full_transcript": transcript,
            "turn_artifacts": [],
            "graph_stats": {},
            "requested_budget": dict(budget or {}),
            "requested_retrieval_config": dict(retrieval_config or {}),
        },
        warnings=warnings,
        started_at=started_at,
        start_completion=start_completion,
        start_prompt=start_prompt,
    )
