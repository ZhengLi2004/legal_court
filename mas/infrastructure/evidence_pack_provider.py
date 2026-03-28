"""Utilities for fixed evidence-pack generation and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mas.core.graph import LegalMessage


@dataclass(frozen=True)
class FixedEvidencePack:
    """Serializable snapshot of memory-driven evidence for one case."""

    plaintiff_insights: list[str]
    defendant_insights: list[str]
    active_history_cases: list[LegalMessage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plaintiff_insights": list(self.plaintiff_insights),
            "defendant_insights": list(self.defendant_insights),
            "active_history_cases": [
                LegalMessage.to_dict(item) for item in self.active_history_cases
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FixedEvidencePack":
        plaintiff = [str(item) for item in list(payload.get("plaintiff_insights", []))]
        defendant = [str(item) for item in list(payload.get("defendant_insights", []))]
        history_rows = list(payload.get("active_history_cases", []))

        return cls(
            plaintiff_insights=plaintiff,
            defendant_insights=defendant,
            active_history_cases=[
                LegalMessage.from_dict(dict(row)) for row in history_rows
            ],
        )


def coerce_fixed_evidence_pack(payload: Any) -> FixedEvidencePack:
    """Validate one fixed evidence-pack payload."""
    if isinstance(payload, FixedEvidencePack):
        return payload

    if not isinstance(payload, dict):
        raise ValueError("fixed evidence pack must be a dict-like payload")

    return FixedEvidencePack.from_dict(dict(payload))


def build_fixed_evidence_pack_for_context(
    *,
    storage_root_dir: str,
    context: str,
) -> dict[str, Any]:
    """Generate one fixed evidence pack from a frozen memory snapshot."""
    from mas.infrastructure.legal_system_factory import build_legal_system
    from mas.infrastructure.settings_provider import build_system_config

    cfg = build_system_config(storage_root_dir=storage_root_dir)
    legal_system = build_legal_system(cfg)
    _, (plaintiff_insights, defendant_insights) = legal_system.new_case(context)
    pack = FixedEvidencePack(
        plaintiff_insights=list(plaintiff_insights),
        defendant_insights=list(defendant_insights),
        active_history_cases=list(legal_system.active_history_cases),
    )
    return pack.to_dict()


def build_fixed_evidence_pack_map_for_contexts(
    *,
    storage_root_dir: str,
    contexts_by_uid: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Generate fixed evidence packs for multiple cases with one system instance."""
    from mas.infrastructure.legal_system_factory import build_legal_system
    from mas.infrastructure.settings_provider import build_system_config

    cfg = build_system_config(storage_root_dir=storage_root_dir)
    legal_system = build_legal_system(cfg)
    packs: dict[str, dict[str, Any]] = {}

    for uid, context in contexts_by_uid.items():
        _, (plaintiff_insights, defendant_insights) = legal_system.new_case(context)
        packs[str(uid)] = FixedEvidencePack(
            plaintiff_insights=list(plaintiff_insights),
            defendant_insights=list(defendant_insights),
            active_history_cases=list(legal_system.active_history_cases),
        ).to_dict()

    return packs


def write_fixed_evidence_pack_map(
    path: str | Path,
    rows: dict[str, dict[str, Any]],
) -> None:
    """Persist one uid -> fixed evidence pack mapping."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_fixed_evidence_pack_map(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load one uid -> fixed evidence pack mapping and validate rows."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(payload, dict) or not payload:
        raise ValueError("fixed evidence pack map must be a non-empty object")

    normalized: dict[str, dict[str, Any]] = {}

    for uid, row in payload.items():
        normalized[str(uid)] = coerce_fixed_evidence_pack(row).to_dict()

    return normalized


def extract_fixed_evidence_pack_for_uid(
    mapping: dict[str, dict[str, Any]],
    uid: str,
) -> dict[str, Any]:
    """Return one validated fixed evidence pack for the requested uid."""
    key = str(uid)

    if key not in mapping:
        raise KeyError(f"Missing fixed evidence pack for uid={key}")

    return coerce_fixed_evidence_pack(mapping[key]).to_dict()


def list_history_case_ids(
    history_cases: Iterable[LegalMessage],
) -> list[str]:
    """Return stable case ids for one history-case collection."""
    return [str(item.case_id) for item in history_cases]
