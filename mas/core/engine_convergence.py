"""Convergence helpers for DebateEngine."""

from __future__ import annotations

from typing import Any, Dict, List

from .graph import EdgeType


def build_engine_convergence_config(engine: Any) -> Dict[str, Any]:
    """Return normalized convergence config values."""
    return {
        "epsilon": float(engine.cfg.convergence.epsilon),
        "min_rounds": int(engine.cfg.convergence.min_rounds),
        "window_size": int(engine.cfg.convergence.window_size),
    }


def calculate_engine_convergence_delta(engine: Any) -> float:
    """Calculate delta-phi for current turn and update previous stats."""
    current_claim_nodes = engine._count_claim_nodes()
    current_conflicts = 0

    for _, _, data in engine.graph.graph.edges(data=True):
        edge_type = data.get("type")

        if str(edge_type) == "CONFLICT" or edge_type == EdgeType.CONFLICT:
            current_conflicts += 1

    delta_v = max(0, current_claim_nodes - engine.prev_stats["claim_nodes"])
    delta_e = max(0, current_conflicts - engine.prev_stats["conflict_edges"])

    engine.prev_stats = {
        "claim_nodes": current_claim_nodes,
        "conflict_edges": current_conflicts,
    }

    alpha = engine.cfg.convergence.alpha
    return (1 - alpha) * delta_v + alpha * delta_e


def determine_engine_converged(
    *,
    round_idx: int,
    sma: float,
    epsilon: float,
    min_rounds: int,
) -> bool:
    """Return whether convergence threshold is reached."""
    return round_idx >= min_rounds and sma < epsilon


def advance_engine_convergence(engine: Any) -> Dict[str, Any]:
    """Advance convergence state after one executed turn."""
    delta_phi = calculate_engine_convergence_delta(engine)
    engine.convergence_history.append(delta_phi)
    cfg = build_engine_convergence_config(engine)
    recent_history = engine.convergence_history[-cfg["window_size"] :]
    sma = sum(recent_history) / len(recent_history) if recent_history else 0.0

    is_converged = determine_engine_converged(
        round_idx=engine.round_idx,
        sma=sma,
        epsilon=cfg["epsilon"],
        min_rounds=cfg["min_rounds"],
    )

    return {
        "delta_phi": delta_phi,
        "sma": sma,
        "is_converged": is_converged,
        "gc_removed": 0,
        "epsilon": cfg["epsilon"],
        "min_rounds": cfg["min_rounds"],
        "window_size": cfg["window_size"],
        "history": list(engine.convergence_history),
    }


def normalize_engine_convergence_payload(
    engine: Any,
    raw_convergence: Any = None,
    *,
    fallback_is_converged: bool = False,
) -> Dict[str, Any]:
    """Normalize any convergence dict into one canonical payload."""
    cfg = build_engine_convergence_config(engine)
    row = raw_convergence if isinstance(raw_convergence, dict) else {}
    raw_history = row.get("history")
    history: List[float]

    if isinstance(raw_history, list):
        history = list(raw_history)

    else:
        history = list(engine.convergence_history)

    delta_default = history[-1] if history else 0.0
    delta_phi = _to_float(row.get("delta_phi"), delta_default)
    sma = _to_float(row.get("sma"), delta_phi)
    epsilon = _to_float(row.get("epsilon"), cfg["epsilon"])
    min_rounds = _to_int(row.get("min_rounds"), cfg["min_rounds"])
    window_size = max(1, _to_int(row.get("window_size"), cfg["window_size"]))
    gc_removed = max(0, _to_int(row.get("gc_removed"), 0))
    raw_is_converged = row.get("is_converged")

    is_converged = (
        bool(raw_is_converged)
        if raw_is_converged is not None
        else fallback_is_converged
    )

    return {
        "delta_phi": delta_phi,
        "sma": sma,
        "is_converged": is_converged,
        "gc_removed": gc_removed,
        "epsilon": epsilon,
        "min_rounds": min_rounds,
        "window_size": window_size,
        "history": history,
    }


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)

    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)

    except (TypeError, ValueError):
        return int(default)
