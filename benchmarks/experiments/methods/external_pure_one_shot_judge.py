"""External pure one-shot adjudication baseline for Claim 1 experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.experiments.methods._external_common import (
    _adjudicate_claim_statuses,
    _build_adjudicator_prompt,
    _build_claim_inventory,
    _build_claim_result_rows,
    _build_result,
    prepare_external_method_context,
)


def run_external_pure_one_shot_judge(
    *,
    case: Mapping[str, Any],
    storage_root_dir: str | None = None,
    budget: Mapping[str, Any] | None = None,
    seed: int | None = None,
    retrieval_config: Mapping[str, Any] | None = None,
    verbose: bool = False,
    test_mode_no_learning: bool = False,
) -> dict[str, Any]:
    """Run one direct no-retrieval, no-debate adjudication baseline."""
    del budget, verbose, test_mode_no_learning

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

    prompt = _build_adjudicator_prompt(
        case_uid_value=case_uid_value,
        cause=cause,
        fact_finding=fact_finding,
        claims=claim_inventory,
    )

    adjudication_rows = _adjudicate_claim_statuses(
        llm=llm,
        prompt=prompt,
        expected_claim_ids=expected_claim_ids,
    )

    claim_rows, status_rows = _build_claim_result_rows(
        case_uid_value=case_uid_value,
        root_claim_rows=root_claim_rows,
        adjudication_rows=adjudication_rows,
    )

    return _build_result(
        method_name="external_pure_one_shot_judge",
        case_uid_value=case_uid_value,
        claim_rows=claim_rows,
        status_rows=status_rows,
        trace={
            "baseline_family": "external",
            "baseline_mode": "pure_one_shot_direct_judge",
            "adjudication_mode": "direct_status_json",
            "retrieved_case_hits": [],
            "retrieved_law_hits": [],
            "full_transcript": [],
            "turn_artifacts": [],
            "graph_stats": {},
            "requested_budget": {},
            "requested_retrieval_config": dict(retrieval_config or {}),
        },
        warnings=warnings,
        started_at=started_at,
        start_completion=start_completion,
        start_prompt=start_prompt,
    )
