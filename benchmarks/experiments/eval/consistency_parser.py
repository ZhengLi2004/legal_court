"""Offline consistency parsing helpers for Step 11.

This module intentionally stays detached from orchestrators and runtime paths.
It parses one method result at a time and produces structured diagnostics that
Step 11 can consume later.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.experiments.eval._metric_utils import normalize_status_eval

REQUIRED_TOP_LEVEL_KEYS = ("method", "case_uid", "claims", "status")


@dataclass(frozen=True)
class ParsedClaim:
    claim_id: str
    uid: str
    claim_text_raw: str
    claim_text_norm: str
    action: str = ""
    target: str = ""
    amount: str = ""
    stage_role: str = ""
    canonical_signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParseFailure:
    reason: str
    detail: str
    claim_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConflictEdge:
    canonical_signature: str
    claim_ids: tuple[str, ...]
    statuses: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConsistencyParseResult:
    method: str
    case_uid: str
    parsed_claims: tuple[ParsedClaim, ...]
    claim_status_sets: dict[str, tuple[str, ...]]
    conflict_edges: tuple[ConflictEdge, ...]
    coverage_passed: bool
    parse_failures: tuple[ParseFailure, ...]
    input_claim_count: int
    input_status_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "case_uid": self.case_uid,
            "parsed_claims": [item.to_dict() for item in self.parsed_claims],
            "claim_status_sets": dict(self.claim_status_sets),
            "conflict_edges": [item.to_dict() for item in self.conflict_edges],
            "coverage_passed": self.coverage_passed,
            "parse_failures": [item.to_dict() for item in self.parse_failures],
            "input_claim_count": self.input_claim_count,
            "input_status_count": self.input_status_count,
        }


@dataclass(frozen=True)
class ParseCoverageSummary:
    case_count: int
    fully_parsed_case_count: int
    claim_count_total: int
    claim_count_parsed: int
    parse_failure_count: int
    parse_failure_reason_breakdown: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _failure(reason: str, detail: str, *, claim_id: str = "") -> ParseFailure:
    return ParseFailure(reason=reason, detail=detail, claim_id=claim_id)


def _canonical_signature(row: Mapping[str, Any]) -> str:
    action = str(row.get("action", "") or "").strip()
    target = str(row.get("target", "") or "").strip()
    amount = str(row.get("amount", "") or "").strip()
    stage_role = str(row.get("stage_role", "") or "").strip()
    claim_text_norm = str(row.get("claim_text_norm", "") or "").strip()

    if any((action, target, amount, stage_role)):
        return "sig:" + "|".join((action, target, amount, stage_role))

    return "sig:" + claim_text_norm


def _build_result_with_top_level_failures(
    payload: Mapping[str, Any],
    failures: list[ParseFailure],
) -> ConsistencyParseResult:
    method = str(payload.get("method", "") or "")
    case_uid = str(payload.get("case_uid", "") or "")
    claims = payload.get("claims", [])
    status_rows = payload.get("status", [])
    input_claim_count = len(claims) if isinstance(claims, list) else 0
    input_status_count = len(status_rows) if isinstance(status_rows, list) else 0

    return ConsistencyParseResult(
        method=method,
        case_uid=case_uid,
        parsed_claims=(),
        claim_status_sets={},
        conflict_edges=(),
        coverage_passed=False,
        parse_failures=tuple(failures),
        input_claim_count=input_claim_count,
        input_status_count=input_status_count,
    )


def parse_method_result(result: Mapping[str, Any]) -> ConsistencyParseResult:
    """Parse one Step 10 method result into Step 11 consistency diagnostics."""
    payload = dict(result)
    top_level_failures: list[ParseFailure] = []

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in payload:
            top_level_failures.append(
                _failure(
                    "missing_top_level_key",
                    f"Missing required top-level key: {key}",
                )
            )

    if top_level_failures:
        return _build_result_with_top_level_failures(payload, top_level_failures)

    claims_raw = payload["claims"]
    status_raw = payload["status"]

    if not isinstance(claims_raw, list):
        return _build_result_with_top_level_failures(
            payload,
            [_failure("malformed_row", "`claims` must be a list.")],
        )

    if not isinstance(status_raw, list):
        return _build_result_with_top_level_failures(
            payload,
            [_failure("malformed_row", "`status` must be a list.")],
        )

    method = str(payload.get("method", "") or "")
    case_uid = str(payload.get("case_uid", "") or "")
    parse_failures: list[ParseFailure] = []
    parsed_claims: dict[str, ParsedClaim] = {}

    for index, row in enumerate(claims_raw):
        if not isinstance(row, Mapping):
            parse_failures.append(
                _failure(
                    "malformed_row",
                    f"claims[{index}] is not a mapping.",
                )
            )

            continue

        claim_id = str(row.get("claim_id", "") or "").strip()
        uid = str(row.get("uid", "") or case_uid).strip() or case_uid
        claim_text_raw = str(row.get("claim_text_raw", "") or "").strip()
        claim_text_norm = str(row.get("claim_text_norm", "") or claim_text_raw).strip()
        has_row_error = False

        if not claim_id:
            parse_failures.append(
                _failure(
                    "empty_claim_id",
                    f"claims[{index}] missing non-empty claim_id.",
                )
            )

            has_row_error = True

        if not claim_text_raw:
            parse_failures.append(
                _failure(
                    "empty_claim_text",
                    f"claims[{index}] missing non-empty claim_text_raw.",
                    claim_id=claim_id,
                )
            )

            has_row_error = True

        if has_row_error:
            continue

        if claim_id in parsed_claims:
            parse_failures.append(
                _failure(
                    "duplicate_claim_id",
                    f"Duplicate claim row detected for claim_id={claim_id}.",
                    claim_id=claim_id,
                )
            )

            continue

        claim_row = dict(row)

        parsed_claims[claim_id] = ParsedClaim(
            claim_id=claim_id,
            uid=uid,
            claim_text_raw=claim_text_raw,
            claim_text_norm=claim_text_norm,
            action=str(claim_row.get("action", "") or "").strip(),
            target=str(claim_row.get("target", "") or "").strip(),
            amount=str(claim_row.get("amount", "") or "").strip(),
            stage_role=str(claim_row.get("stage_role", "") or "").strip(),
            canonical_signature=_canonical_signature(claim_row),
        )

    status_by_claim: dict[str, set[str]] = defaultdict(set)

    for index, row in enumerate(status_raw):
        if not isinstance(row, Mapping):
            parse_failures.append(
                _failure(
                    "malformed_row",
                    f"status[{index}] is not a mapping.",
                )
            )

            continue

        claim_id = str(row.get("claim_id", "") or "").strip()
        status_eval_raw = str(row.get("status_eval", "") or "").strip()

        if not claim_id:
            parse_failures.append(
                _failure(
                    "empty_claim_id",
                    f"status[{index}] missing non-empty claim_id.",
                )
            )

            continue

        try:
            status_eval = normalize_status_eval(status_eval_raw)

        except ValueError:
            parse_failures.append(
                _failure(
                    "invalid_status_label",
                    f"Unsupported status_eval for claim_id={claim_id}: {status_eval_raw!r}",
                    claim_id=claim_id,
                )
            )

            continue

        status_by_claim[claim_id].add(status_eval)

    for claim_id in sorted(parsed_claims):
        if claim_id not in status_by_claim:
            parse_failures.append(
                _failure(
                    "claim_without_status",
                    f"Claim row has no matching status row for claim_id={claim_id}.",
                    claim_id=claim_id,
                )
            )

    for claim_id in sorted(status_by_claim):
        if claim_id not in parsed_claims:
            parse_failures.append(
                _failure(
                    "status_without_claim",
                    f"Status row has no matching claim row for claim_id={claim_id}.",
                    claim_id=claim_id,
                )
            )

    claim_status_sets = {
        claim_id: tuple(sorted(status_by_claim[claim_id]))
        for claim_id in sorted(parsed_claims)
        if claim_id in status_by_claim
    }

    conflicts: list[ConflictEdge] = []
    grouped_claims: dict[str, list[str]] = defaultdict(list)
    grouped_statuses: dict[str, set[str]] = defaultdict(set)

    for claim_id, claim in parsed_claims.items():
        if claim_id not in status_by_claim:
            continue

        grouped_claims[claim.canonical_signature].append(claim_id)
        grouped_statuses[claim.canonical_signature].update(status_by_claim[claim_id])

    for signature, claim_ids in sorted(grouped_claims.items()):
        statuses = grouped_statuses.get(signature, set())

        if len(statuses) > 1:
            conflicts.append(
                ConflictEdge(
                    canonical_signature=signature,
                    claim_ids=tuple(sorted(claim_ids)),
                    statuses=tuple(sorted(statuses)),
                )
            )

    coverage_passed = not parse_failures

    return ConsistencyParseResult(
        method=method,
        case_uid=case_uid,
        parsed_claims=tuple(
            parsed_claims[claim_id] for claim_id in sorted(parsed_claims)
        ),
        claim_status_sets=claim_status_sets,
        conflict_edges=tuple(conflicts),
        coverage_passed=coverage_passed,
        parse_failures=tuple(parse_failures),
        input_claim_count=len(claims_raw),
        input_status_count=len(status_raw),
    )


def parse_method_result_file(path: str | Path) -> ConsistencyParseResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_method_result(payload)


def summarize_parse_results(
    results: list[ConsistencyParseResult] | tuple[ConsistencyParseResult, ...],
) -> ParseCoverageSummary:
    if not results:
        raise ValueError("results must not be empty")

    parse_failure_reason_breakdown: Counter[str] = Counter()

    for result in results:
        for failure in result.parse_failures:
            parse_failure_reason_breakdown[failure.reason] += 1

    return ParseCoverageSummary(
        case_count=len(results),
        fully_parsed_case_count=sum(1 for result in results if result.coverage_passed),
        claim_count_total=sum(result.input_claim_count for result in results),
        claim_count_parsed=sum(len(result.parsed_claims) for result in results),
        parse_failure_count=sum(len(result.parse_failures) for result in results),
        parse_failure_reason_breakdown=dict(
            sorted(parse_failure_reason_breakdown.items())
        ),
    )
