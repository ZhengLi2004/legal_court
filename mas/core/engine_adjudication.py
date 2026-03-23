"""Adjudication flow for DebateEngine."""

from __future__ import annotations

from typing import Any

from .engine_convergence import normalize_engine_convergence_payload
from .engine_post_learning import run_post_adjudication_learning


async def run_engine_adjudication(engine: Any) -> None:
    """Execute adjudication flow and finalize engine state.

    Args:
        engine: Debate engine instance.

    Returns:
        None.

    Side Effects:
        Performs graph garbage collection, calls judge adjudication, updates
        convergence/adjudication logs, triggers post-learning, and marks engine
        as finished.
    """
    removed_count = engine.graph.garbage_collect()
    raw_convergence = {}

    if isinstance(getattr(engine, "last_step_log", None), dict):
        raw_convergence = engine.last_step_log.get("convergence", {})

    convergence_payload = normalize_engine_convergence_payload(
        engine,
        raw_convergence,
        fallback_is_converged=bool(getattr(engine, "is_ready_for_adjudication", False)),
    )

    convergence_payload["gc_removed"] = max(
        int(convergence_payload.get("gc_removed", 0)),
        int(removed_count),
    )

    if not isinstance(getattr(engine, "last_step_log", None), dict):
        engine.last_step_log = {}

    engine.last_step_log["convergence"] = convergence_payload
    engine._notify_state_change("adjudication_start", None)
    engine.root_claims_status = await engine.legal_sys.adjudicate(engine.graph)

    engine.last_step_log["adjudication_result"] = {
        "claims_status": {
            key: value.value if hasattr(value, "value") else str(value)
            for key, value in engine.root_claims_status.items()
        },
        "claim_count": len(engine.root_claims_status),
        "adjudication_mode": "direct_status_json",
    }

    run_post_adjudication_learning(engine)
    engine.is_finished = True
    engine.is_ready_for_adjudication = False

    engine._notify_state_change(
        "adjudication_complete",
        {
            "root_claim_count": len(engine.root_claims_status),
            "adjudication_mode": "direct_status_json",
        },
    )
