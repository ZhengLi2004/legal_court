"""Helpers for session lifecycle and status derivation."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from .session_status import SessionStatus


def default_case_path() -> Path:
    """Resolve default case JSONL path for session setup.

    The environment variable `MAS_CASE_DATA_FILE` takes precedence. If unset,
    the function falls back to the repository sample file.

    Returns:
        Resolved absolute path to the default case JSONL file.
    """
    configured_path = str(os.getenv("MAS_CASE_DATA_FILE", "")).strip()

    if configured_path:
        return Path(configured_path).expanduser().resolve()

    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "sampling"
        / "cleaned_samples.jsonl"
    )


def default_frontend_snapshots_dir() -> Path:
    """Resolve default directory for persisted frontend snapshots.

    The environment variable `MAS_FRONTEND_SNAPSHOTS_DIR` takes precedence.

    Returns:
        Absolute path used by snapshot persistence services.
    """
    configured_dir = str(os.getenv("MAS_FRONTEND_SNAPSHOTS_DIR", "")).strip()

    if configured_dir:
        return Path(configured_dir).expanduser().resolve()

    return Path(__file__).resolve().parents[2] / "tmp" / "frontend_snapshots"


def default_engine_factory() -> Any:
    """Build a default `DebateEngine` instance for API sessions.

    Returns:
        A new `DebateEngine` configured from process-level settings.
    """
    from mas.core.engine import DebateEngine
    from mas.infrastructure.settings_provider import build_system_config

    return DebateEngine(config=build_system_config())


def load_case_from_jsonl(path: Path) -> Dict[str, Any]:
    """Load the first non-empty JSON object row from a JSONL file.

    Args:
        path: JSONL file path.

    Returns:
        First parsed JSON object in the file.

    Raises:
        ValueError: If no valid JSON object row is found.
        OSError: If the file cannot be opened.
        json.JSONDecodeError: If a non-empty row contains invalid JSON.
    """
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
    """Resolve and validate long-term memory storage directory path.

    Args:
        storage_root: Raw directory string, usually from `MAS_STORAGE_DIR`.

    Returns:
        Expanded absolute directory path.

    Raises:
        ValueError: If the input is empty or resolves to an unsafe root-like path.
    """
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
    """Reset long-term memory storage directory and return absolute path.

    Args:
        storage_root: Raw storage root string.

    Returns:
        Absolute path string of recreated storage directory.

    Raises:
        ValueError: If `storage_root` is empty or unsafe.
        OSError: If deleting or recreating directories fails.

    Side Effects:
        Recursively deletes and recreates the storage directory on disk.
    """
    storage_dir = resolve_memory_storage_dir(storage_root)

    if storage_dir.exists():
        shutil.rmtree(storage_dir)

    storage_dir.mkdir(parents=True, exist_ok=True)
    return str(storage_dir)


def derive_status(engine: Any) -> SessionStatus:
    """Derive API session status from engine runtime fields.

    Args:
        engine: Debate engine instance (or compatible object).

    Returns:
        Derived `SessionStatus` based on finished/converged/round/graph state.
    """
    if getattr(engine, "is_finished", False):
        return SessionStatus.FINISHED

    if getattr(engine, "is_ready_for_adjudication", False):
        return SessionStatus.READY_FOR_ADJUDICATION

    if getattr(engine, "round_idx", 0) > 0:
        return SessionStatus.DEBATING

    if getattr(engine, "graph", None) is not None:
        return SessionStatus.SETUP_DONE

    return SessionStatus.CREATED
