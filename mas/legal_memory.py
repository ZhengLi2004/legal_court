import os
import json
import chromadb
from dataclasses import dataclass, field
from typing import List, Tuple
from chromadb.utils import embedding_functions
from .memory_base import MASMemoryBase
from .common import LegalMessage, ShadowGraph
from .task_layer import TaskLayer
from .config import SystemConfig
# 法律辩论系统的核心记忆模块
@dataclass
class LegalGMemory(MASMemoryBase):
    config: SystemConfig = field(default_factory=SystemConfig)
    collection_name: str = "legal_cases"

    def __post_init__(self):
        super().__post_init__()
        chroma_path = os.path.join(self.persist_dir, self.config.path.storage_subdir_chroma)
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        self.chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_path
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.chroma_ef,
            metadata={"hnsw:space": "cosine"}
        )

        self.task_layer = TaskLayer(working_dir=self.persist_dir, similarity_threshold=self.config.topology.task_layer_threshold)
    # 存入新案件
    def add_memory(self, message: LegalMessage) -> None:
        self.collection.add(
            documents=[message.case_context],
            metadatas=[{"graph_json": json.dumps(ShadowGraph.to_dict(message.shadow_graph))}],
            ids=[message.case_id]
        )

        results = self.collection.query(query_texts=[message.case_context], n_results=self.config.retrieval.chroma_n_results)
        
        neighbors: List[Tuple[str, float]] = []
        
        if results['ids'] and results['distances']:
            retrieved_ids = results['ids'][0]
            retrieved_distances = results['distances'][0]
            
            for rid, dist in zip(retrieved_ids, retrieved_distances):
                if rid == message.case_id: continue
                sim = max(0.0, 1.0 - dist)
                neighbors.append((rid, sim))
            
        self.task_layer.update_topology(message.case_id, neighbors)

    def retrieve_memory(self, query_context: str, top_k: int = 3) -> Tuple[List[LegalMessage], List[str]]:
        results = self.collection.query(query_texts=[query_context], n_results=top_k)
        anchor_ids = results['ids'][0] if results['ids'] else []
        expanded_ids = self.task_layer.get_k_hop_neighbors(anchor_ids, hop=self.config.retrieval.hop)
        if not expanded_ids: return [], []
        final_results = self.collection.get(ids=expanded_ids, include=["metadatas", "documents"])
        messages = []
        
        for i in range(len(final_results['ids'])):
            case_id = final_results['ids'][i]
            context = final_results['documents'][i]
            graph_json_str = final_results['metadatas'][i]['graph_json']
            sg = ShadowGraph.from_dict(json.loads(graph_json_str))
            msg = LegalMessage(case_id=case_id, case_context=context, shadow_graph=sg)
            messages.append(msg)
            
        return messages, []