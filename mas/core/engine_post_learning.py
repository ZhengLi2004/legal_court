"""Post-adjudication learning helpers for DebateEngine."""

from __future__ import annotations

import time
from typing import Any, Dict

from metagpt.logs import logger


def run_post_adjudication_learning(engine: Any) -> Dict[str, Any]:
    """Run post-adjudication learning and emit completion callback.

    Args:
        engine: Debate engine instance.

    Returns:
        Learning result dict containing `case_id`, `success`, and optional error.

    Side Effects:
        Calls `engine.legal_sys.learn`, writes `learning_result` into
        `engine.last_step_log`, and emits `learning_complete` state event.
    """
    learning_case_id = str(getattr(engine, "current_case_id", "")).strip()

    if not learning_case_id:
        learning_case_id = f"case_{int(time.time() * 1000)}"

    learning_result: Dict[str, Any] = {
        "case_id": learning_case_id,
        "success": False,
        "error": "",
    }

    try:
        engine.legal_sys.learn(
            context=engine.raw_facts,
            current_graph=engine.graph,
            root_claims_status=engine.root_claims_status,
            case_id=learning_case_id,
            transcript=list(engine.transcript),
        )

        learning_result["success"] = True

    except Exception as exc:
        learning_result["error"] = str(exc)

        logger.warning(
            "[Engine] post-adjudication learning failed for case {}: {}",
            learning_case_id,
            exc,
        )

    engine.last_step_log["learning_result"] = learning_result
    engine._notify_state_change("learning_complete", learning_result)
    return learning_result
