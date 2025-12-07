from dataclasses import dataclass
import os

@dataclass
class AgentConfig:
    system_id: str = "system"
    projection_id: str = "projection"
    default_commander_id: str = "commander"

@dataclass
class PathConfig:
    embedding_model_path: str = os.getenv("EMBEDDING_MODEL_PATH", "./bge-m3")
    storage_subdir_chroma: str = "chroma_db"
    file_query_graph: str = "case_graph.pkl"
    file_insight_graph: str = "legal_insights.json"

@dataclass
class LLMConfig:
    temperature: float = 0.1
    max_tokens: int = 1024
    model_name: str = "法衡"

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
    score_weight: float = 0.05
    reward_win: float = 1.0
    penalty_lose: float = 0.5
    reward_merge: float = 0.5

@dataclass
class SystemConfig:
    agent: AgentConfig = AgentConfig()
    path: PathConfig = PathConfig()
    llm: LLMConfig = LLMConfig()
    matcher: MatcherConfig = MatcherConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    topology: TopologyConfig = TopologyConfig()
    insight: InsightConfig = InsightConfig()
    dedup: DeduplicationConfig = DeduplicationConfig()