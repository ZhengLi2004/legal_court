"""Defines configuration classes for the entire MAS system.

This module uses dataclasses to define a hierarchical configuration structure,
loading sensitive values (like API keys and paths) from environment variables
via `python-dotenv`. It provides a centralized point of control for all system
parameters, from LLM settings to file paths and algorithm thresholds.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)


def get_env_strict(key: str) -> str:
    """Retrieve an environment variable, raising an error if it's not set.

    Args:
        key: The name of the environment variable.

    Returns:
        The value of the environment variable.

    Raises:
        ValueError: If the environment variable is not found.
    """
    value = os.getenv(key)

    if value is None:
        raise ValueError(
            f"❌ [Config Error] Environment variable '{key}' not found! Please check .env document."
        )

    return value


@dataclass
class ESConfig:
    """Configuration for Elasticsearch connection."""

    host: str = get_env_strict("ES_HOST")


@dataclass
class AgentConfig:
    """Configuration for agent identifiers."""

    system_id: str = "system"
    projection_id: str = "projection"
    default_commander_id: str = "commander"


@dataclass
class PathConfig:
    """Configuration for file system paths."""

    embedding_model_path: str = get_env_strict("EMBEDDING_MODEL_PATH")
    storage_root_dir: str = get_env_strict("MAS_STORAGE_DIR")
    storage_subdir_chroma: str = "chroma_db"
    file_query_graph: str = "case_graph.pkl"
    file_insight_graph: str = "legal_insights.json"


@dataclass
class LLMConfig:
    """Configuration for the primary language model used by agents."""

    temperature: float = 0.1
    max_tokens: int = 8192
    model_name: str = "Kimi-K2.5-Instruct"
    api_key: str = get_env_strict("LEGAL_LLM_KEY")
    base_url: str = get_env_strict("LEGAL_LLM_URL")


@dataclass
class JudgeLLMConfig:
    """Configuration for the language model used by the Judge agent."""

    temperature: float = 0.1
    model_name: str = "qwen3"
    base_url: str = get_env_strict("JUDGE_API_BASE")
    api_key: str = get_env_strict("JUDGE_API_KEY")


@dataclass
class ConvergenceConfig:
    """Parameters for the debate convergence detection algorithm."""

    alpha: float = 0.6
    epsilon: float = 2
    window_size: int = 4
    min_rounds: int = 2


@dataclass
class MatcherConfig:
    """Thresholds for semantic matching."""

    projection_threshold: float = 0.58
    insight_threshold: float = 0.70


@dataclass
class DeduplicationConfig:
    """Thresholds for semantic deduplication of graph nodes."""

    fact_threshold: float = 0.95
    other_threshold: float = 0.90


@dataclass
class RetrievalConfig:
    """Parameters for information retrieval across the three long-memory paths.

    Long-term case recall follows three paths:
    1. Semantic path (fact similarity).
    2. Jurisprudence path (law overlap).
    3. Strategy path (insight-guided strategy recall).
    """

    semantic_path_top_k: int = 3
    jurisprudence_path_top_k: int = 3
    strategy_path_top_k: int = 9
    insight_top_k: int = 3
    chroma_n_results: int = 3
    hop: int = 1
    projection_anchor_top_k: int = 3
    projection_case_top_k: int = 3


@dataclass
class TopologyConfig:
    """Configuration for the case topology graph (TaskLayer)."""

    task_layer_threshold: float = 0.50


@dataclass
class SystemConfig:
    """The root configuration class that aggregates all other configs.

    This class provides a single object through which all system parameters can
    be accessed. It also performs initial setup, such as creating the storage
    directory if it doesn't exist.
    """

    agent: AgentConfig = AgentConfig()
    path: PathConfig = PathConfig()
    llm: LLMConfig = LLMConfig()
    judge: JudgeLLMConfig = JudgeLLMConfig()
    matcher: MatcherConfig = MatcherConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    topology: TopologyConfig = TopologyConfig()
    dedup: DeduplicationConfig = DeduplicationConfig()
    es: ESConfig = ESConfig()
    convergence: ConvergenceConfig = ConvergenceConfig()

    def __post_init__(self):
        """Create the root storage directory after initialization if it's missing."""
        if not os.path.exists(self.path.storage_root_dir):
            os.makedirs(self.path.storage_root_dir, exist_ok=True)
