"""Centralized runtime settings provider for backend bootstrapping."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from mas.config import SystemConfig


@dataclass(frozen=True)
class LLMConfigView:
    """Expose immutable LLM defaults for runtime client construction.

    Attributes:
        temperature: Default sampling temperature.
        max_tokens: Default maximum completion length.
        model_name: Default model identifier.
        base_url: OpenAI-compatible API base URL.
        api_key: Credential used by runtime LLM clients.
    """

    temperature: float
    max_tokens: int
    model_name: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class DedupThresholds:
    """Store node-deduplication thresholds used by graph execution.

    Attributes:
        fact_threshold: Deduplication threshold for fact nodes.
        other_threshold: Deduplication threshold for non-fact nodes.
    """

    fact_threshold: float
    other_threshold: float


def apply_storage_root_override(
    config: SystemConfig,
    storage_root_dir: str | None = None,
) -> SystemConfig:
    """Apply one explicit storage-root override to a config instance.

    Args:
        config: Base system configuration object.
        storage_root_dir: Optional override path for runtime storage.

    Returns:
        Configuration instance with the resolved storage root applied.
    """
    if not storage_root_dir:
        return config

    resolved = str(Path(storage_root_dir).expanduser().resolve())
    config.path = replace(config.path, storage_root_dir=resolved)
    Path(resolved).mkdir(parents=True, exist_ok=True)
    return config


def build_system_config(storage_root_dir: str | None = None) -> SystemConfig:
    """Create one process-level system configuration object.

    Returns:
        New `SystemConfig` instance loaded from environment/defaults.
    """
    return apply_storage_root_override(SystemConfig(), storage_root_dir)


def build_llm_config_view(config: SystemConfig) -> LLMConfigView:
    """Create immutable LLM defaults view from root configuration.

    Args:
        config: Root system configuration.

    Returns:
        `LLMConfigView` used by OpenAI-compatible clients.
    """
    return LLMConfigView(
        temperature=float(config.llm.temperature),
        max_tokens=int(config.llm.max_tokens),
        model_name=str(config.llm.model_name),
        base_url=str(config.llm.base_url),
        api_key=str(config.llm.api_key),
    )


def build_dedup_thresholds(config: SystemConfig) -> DedupThresholds:
    """Create dedup-threshold view from root system configuration.

    Args:
        config: Root system configuration.

    Returns:
        Deduplication threshold view for fact/non-fact nodes.
    """
    return DedupThresholds(
        fact_threshold=float(config.dedup.fact_threshold),
        other_threshold=float(config.dedup.other_threshold),
    )


def build_embedding_model_path(config: SystemConfig) -> str:
    """Return embedding model path from root system configuration.

    Args:
        config: Root system configuration.

    Returns:
        Embedding model path string.
    """
    return str(config.path.embedding_model_path)
