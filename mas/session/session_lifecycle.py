"""Helpers for session lifecycle and status derivation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

from .session_status import SessionStatus


def default_case_path() -> Path:
    """Return the default bundled case JSONL path."""
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "sampling"
        / "cleaned_samples.jsonl"
    )


def default_frontend_snapshots_dir() -> Path:
    """Return the directory used for persisted frontend snapshot files."""
    return Path(__file__).resolve().parents[2] / "tmp" / "frontend_snapshots"


def default_engine_factory() -> Any:
    """Build a default `DebateEngine` instance for API sessions."""
    from mas.config import SystemConfig
    from mas.core.engine import DebateEngine

    return DebateEngine(config=SystemConfig(), judge_config={})


def load_case_from_jsonl(path: Path) -> Dict[str, Any]:
    """Load the first valid case row from a JSONL file."""
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = line.strip()

            if not row:
                continue

            payload = json.loads(row)

            if isinstance(payload, dict):
                return payload

    raise ValueError(f"No valid case row found in {path}")


def resolve_memory_storage_dir(storage_root: str) -> Path:
    """Resolve and validate the long-term memory storage directory."""
    cleaned = str(storage_root or "").strip()

    if not cleaned:
        raise ValueError(
            "Missing MAS_STORAGE_DIR. Cannot locate long-term memory storage."
        )

    resolved = Path(cleaned).expanduser().resolve()

    if str(resolved) in {"", "/", "."}:
        raise ValueError(f"Refuse to clear unsafe storage path: {resolved}")

    return resolved


def reset_memory_storage_dir(storage_root: str) -> str:
    """Reset long-term memory storage directory and return absolute path."""
    storage_dir = resolve_memory_storage_dir(storage_root)

    if storage_dir.exists():
        shutil.rmtree(storage_dir)

    storage_dir.mkdir(parents=True, exist_ok=True)
    return str(storage_dir)


def derive_status(engine: Any) -> SessionStatus:
    """Derive API session status from engine runtime fields."""
    if getattr(engine, "is_finished", False):
        return SessionStatus.FINISHED

    if getattr(engine, "is_ready_for_adjudication", False):
        return SessionStatus.READY_FOR_ADJUDICATION

    if getattr(engine, "round_idx", 0) > 0:
        return SessionStatus.DEBATING

    if getattr(engine, "graph", None) is not None:
        return SessionStatus.SETUP_DONE

    return SessionStatus.CREATED
