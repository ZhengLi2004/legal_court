"""Centralized runtime settings provider for backend bootstrapping."""

from __future__ import annotations

from dataclasses import dataclass

from mas.config import SystemConfig


@dataclass(frozen=True)
class LLMConfigView:
    """Read-only defaults required by OpenAI-compatible LLM clients."""

    temperature: float
    max_tokens: int
    model_name: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class DedupThresholds:
    """Deduplication thresholds used by graph execution."""

    fact_threshold: float
    other_threshold: float


def build_system_config() -> SystemConfig:
    """Build one process-level system config object."""
    return SystemConfig()


def build_llm_config_view(config: SystemConfig) -> LLMConfigView:
    """Create LLM defaults view from root system configuration."""
    return LLMConfigView(
        temperature=float(config.llm.temperature),
        max_tokens=int(config.llm.max_tokens),
        model_name=str(config.llm.model_name),
        base_url=str(config.llm.base_url),
        api_key=str(config.llm.api_key),
    )


def build_dedup_thresholds(config: SystemConfig) -> DedupThresholds:
    """Create dedup-threshold view from root system configuration."""
    return DedupThresholds(
        fact_threshold=float(config.dedup.fact_threshold),
        other_threshold=float(config.dedup.other_threshold),
    )


def build_embedding_model_path(config: SystemConfig) -> str:
    """Return embedding model path from root system configuration."""
    return str(config.path.embedding_model_path)
