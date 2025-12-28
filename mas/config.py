from dataclasses import dataclass
import os
from dotenv import load_dotenv
load_dotenv(override=True)

def get_env_strict(key: str) -> str:
    value = os.getenv(key)
    if value is None: raise ValueError(f"❌ [Config Error] Environment variable '{key}' not found! Please check .env document.")
    return value

@dataclass
class ESConfig:
    host: str = get_env_strict("ES_HOST")

@dataclass
class AgentConfig:
    system_id: str = "system"
    projection_id: str = "projection"
    default_commander_id: str = "commander"

@dataclass
class PathConfig:
    embedding_model_path: str = get_env_strict("EMBEDDING_MODEL_PATH")
    storage_root_dir: str = get_env_strict("MAS_STORAGE_DIR")
    storage_subdir_chroma: str = "chroma_db"
    file_query_graph: str = "case_graph.pkl"
    file_insight_graph: str = "legal_insights.json"

@dataclass
class LLMConfig:
    temperature: float = 0.1
    max_tokens: int = 1024
    model_name: str = "DeepSeek-V3.2"
    api_key: str = get_env_strict("LEGAL_LLM_KEY")
    base_url: str = get_env_strict("LEGAL_LLM_URL")

@dataclass
class JudgeLLMConfig:
    temperature: float = 0.1
    model_name: str = "qwen3"
    base_url: str = get_env_strict("JUDGE_API_BASE")
    api_key: str = get_env_strict("JUDGE_API_KEY")

@dataclass
class ConvergenceConfig:
    alpha: float = 0.3
    epsilon: float = 2
    window_size: int = 4
    min_rounds: int = 2

@dataclass
class MatcherConfig:
    projection_threshold: float = 0.60
    insight_threshold: float = 0.70

@dataclass
class DeduplicationConfig:
    fact_threshold: float = 0.95
    other_threshold: float = 0.90

@dataclass
class RetrievalConfig:
    initial_top_k: int = 2
    corrective_top_k: int = 2
    insight_top_k: int = 3
    chroma_n_results: int = 5
    hop: int = 1
    max_neighbors_per_anchor: int = 10

@dataclass
class TopologyConfig:
    task_layer_threshold: float = 0.50

@dataclass
class InsightConfig:
    weight_relevance: float = 1.0
    weight_utility: float = 0.1
    reward_win: float = 1.0
    penalty_lose: float = 0.5
    reward_merge: float = 0.5

@dataclass
class SystemConfig:
    agent: AgentConfig = AgentConfig()
    path: PathConfig = PathConfig()
    llm: LLMConfig = LLMConfig()
    judge: JudgeLLMConfig = JudgeLLMConfig()
    matcher: MatcherConfig = MatcherConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    topology: TopologyConfig = TopologyConfig()
    insight: InsightConfig = InsightConfig()
    dedup: DeduplicationConfig = DeduplicationConfig()
    es: ESConfig = ESConfig()
    convergence: ConvergenceConfig = ConvergenceConfig()

    def __post_init__(self):
        if not os.path.exists(self.path.storage_root_dir): os.makedirs(self.path.storage_root_dir, exist_ok=True)