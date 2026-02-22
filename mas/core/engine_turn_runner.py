"""Turn execution flow for DebateEngine."""

from __future__ import annotations

from typing import Any

from metagpt.logs import logger

from .engine_convergence import advance_engine_convergence


async def run_engine_step(
    engine: Any,
    *,
    turn_enum: Any,
    controller_idle_step: Any,
    persist_snapshot: bool = True,
) -> None:
    """Execute a single turn of the debate."""
    if engine.is_finished:
        return

    if engine._is_running:
        logger.warning("[Engine] Step already in progress, skipping.")
        return

    engine._is_running = True

    try:
        await engine.open_resources()
        engine.legal_sys.advance_step()
        is_plaintiff_turn = engine.current_turn == turn_enum.PLAINTIFF
        turn_name_str = "plaintiff" if is_plaintiff_turn else "defendant"

        engine._notify_state_change(
            "turn_start",
            {
                "turn": turn_name_str,
                "round": engine.round_idx,
            },
        )

        if is_plaintiff_turn:
            if engine.round_idx == 0:
                engine.round_idx = 1

            team_to_run = engine.p_team
            next_turn = turn_enum.DEFENDANT

        else:
            team_to_run = engine.d_team
            next_turn = turn_enum.PLAINTIFF

        turn_result = await team_to_run.run_turn(engine.graph)
        executed_actions = turn_result.get("actions", [])
        narrative_text = ""

        if executed_actions:
            narrative_text = await engine.narrator.generate_narrative(
                actions=executed_actions,
                graph=engine.graph,
                turn=turn_name_str,
            )

            if narrative_text:
                engine.transcript.append(narrative_text)

                engine._notify_state_change(
                    "transcript_update",
                    {
                        "text": narrative_text,
                        "turn": turn_name_str,
                    },
                )

        convergence = advance_engine_convergence(engine)
        delta_phi = float(convergence["delta_phi"])
        sma = float(convergence["sma"])
        epsilon = float(convergence["epsilon"])

        logger.info(
            f"[Convergence] Round {engine.round_idx} | "
            f"ΔΦ: {delta_phi:.4f} | SMA: {sma:.4f} | ε: {epsilon:.4f}"
        )

        engine.last_step_log = {
            "turn": engine.current_turn.value,
            "round": engine.round_idx,
            "action": turn_result["summary"],
            "dialogue": turn_result["transcript"],
            "narrative": narrative_text,
            "convergence": convergence,
        }

        turn_artifact = engine._build_turn_artifact(
            turn=turn_name_str,
            turn_result=turn_result,
            narrative_text=narrative_text,
        )

        engine.turn_artifacts.append(turn_artifact)
        engine.latest_turn_uid = str(turn_artifact.get("turn_uid", ""))

        engine._notify_state_change(
            "turn_complete",
            {
                "turn": turn_name_str,
                "round": engine.round_idx,
                "delta_phi": delta_phi,
                "sma": sma,
                "turn_uid": engine.latest_turn_uid,
            },
        )

        should_adjudicate = bool(convergence.get("is_converged", False))

        if should_adjudicate:
            reason = "Convergence Reached"
            logger.info(f">>> [Engine] Adjudication ready. Reason: {reason}")
            engine.is_ready_for_adjudication = True
            engine._notify_state_change("adjudication_ready", {"reason": reason})

        if not engine.is_finished:
            engine.current_turn = next_turn

            if not is_plaintiff_turn:
                engine.round_idx += 1

        if engine.p_team and engine.p_team.controller:
            engine.p_team.controller.pipeline_step = controller_idle_step

        if engine.d_team and engine.d_team.controller:
            engine.d_team.controller.pipeline_step = controller_idle_step

        if persist_snapshot:
            turn_snapshot = engine._create_snapshot(engine.round_idx, turn_name_str)
            engine.round_snapshots.append(turn_snapshot)

            engine._notify_state_change(
                "snapshot_saved",
                {
                    "round": engine.round_idx,
                    "turn": turn_name_str,
                    "total": len(engine.round_snapshots),
                    "turn_uid": engine.latest_turn_uid,
                },
            )

    finally:
        engine._is_running = False
        await engine.close_resources()
