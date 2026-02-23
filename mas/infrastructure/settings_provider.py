"""Centralized runtime settings provider for backend bootstrapping."""

from __future__ import annotations

from dataclasses import dataclass

from mas.config import SystemConfig


@dataclass(frozen=True)
class LLMConfigView:
    """Expose immutable LLM defaults for runtime client construction."""

    temperature: float
    max_tokens: int
    model_name: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class DedupThresholds:
    """Store node-deduplication thresholds used by graph execution."""

    fact_threshold: float
    other_threshold: float


def build_system_config() -> SystemConfig:
    """Create one process-level system configuration object.

    Returns:
        New `SystemConfig` instance loaded from environment/defaults.
    """
    return SystemConfig()


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
