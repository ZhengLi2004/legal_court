from dataclasses import dataclass

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

@dataclass
class TopologyConfig:
    task_layer_threshold: float = 0.50

@dataclass
class InsightConfig:
    score_weight: float = 0.05

@dataclass
class SystemConfig:
    llm: LLMConfig = LLMConfig()
    matcher: MatcherConfig = MatcherConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    topology: TopologyConfig = TopologyConfig()
    insight: InsightConfig = InsightConfig()
    dedup: DeduplicationConfig = DeduplicationConfig()