"""Factory for wiring `LegalSystem` with concrete infrastructure adapters."""

from mas.agents.judge import LLMJudge
from mas.config import SystemConfig
from mas.core.system import LegalSystem, LegalSystemDependencies
from mas.infrastructure.settings_provider import (
    build_dedup_thresholds,
    build_embedding_model_path,
    build_llm_config_view,
)
from tools.embedding import EmbeddingFunc
from tools.llm import GPTChat
from tools.matcher import SemanticMatcher


def build_legal_system(config: SystemConfig) -> LegalSystem:
    """Build a fully wired `LegalSystem` instance from concrete tool adapters."""
    llm_defaults = build_llm_config_view(config)
    embedding_model_path = build_embedding_model_path(config)
    dedup_thresholds = build_dedup_thresholds(config)
    embedding_func = EmbeddingFunc(model_path=embedding_model_path)
    llm = GPTChat(defaults=llm_defaults, model_name=config.llm.model_name)

    projection_matcher = SemanticMatcher(
        embedding_func,
        threshold=config.matcher.projection_threshold,
    )

    insight_matcher = SemanticMatcher(
        embedding_func,
        threshold=config.matcher.insight_threshold,
    )

    dedup_matcher = SemanticMatcher(embedding_func)

    judge_llm = GPTChat(
        defaults=llm_defaults,
        model_name=config.judge.model_name,
        base_url=config.judge.base_url,
        api_key=config.judge.api_key,
    )

    extraction_llm = GPTChat(
        defaults=llm_defaults,
        model_name=config.llm.model_name,
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
    )

    judge = LLMJudge(judge_llm=judge_llm, extraction_llm=extraction_llm)

    dependencies = LegalSystemDependencies(
        llm=llm,
        embedding_func=embedding_func,
        projection_matcher=projection_matcher,
        insight_matcher=insight_matcher,
        dedup_matcher=dedup_matcher,
        dedup_thresholds=dedup_thresholds,
        judge=judge,
    )

    return LegalSystem(config=config, dependencies=dependencies)
