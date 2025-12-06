import os
import chromadb
from dataclasses import dataclass
from typing import List, Tuple
from chromadb.utils import embedding_functions
from .memory_base import MASMemoryBase
from .common import LegalMessage
from .task_layer import TaskLayer
# 法律辩论系统的核心记忆模块
@dataclass
class LegalGMemory(MASMemoryBase):
    collection_name: str = "legal_cases"

    def __post_init__(self):
        super().__post_init__()
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(self.persist_dir, "chroma_db"))

        self.chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_path
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.chroma_ef,
            metadata={"hnsw:space": "cosine"}
        )

        self.task_layer = TaskLayer(working_dir=self.persist_dir)
    # 存入新案件
    def add_memory(self, message: LegalMessage) -> None:
        self.collection.add(
            documents=[message.case_context],
            metadatas=[{"graph_json": message.shadow_graph.to_json()}], # 存图结构
            ids=[message.case_id]
        )

        results = self.collection.query(
            query_texts=[message.case_context],
            n_results=5
        )

        neighbors: List[Tuple[str, float]] = []
        retrieved_ids = results['ids'][0]
        retrieved_distances = results['distances'][0]

        for rid, dist in zip(retrieved_ids, retrieved_distances):
            if rid == message.case_id: continue
            sim = max(0.0, 1.0 - dist)
            neighbors.append((rid, sim))

        self.task_layer.update_topology(message.case_id, neighbors)

    def retrieve_memory(self, query_context: str, top_k: int = 3) -> Tuple[List[LegalMessage], List[str]]:
        results = self.collection.query(
            query_texts=[query_context],
            n_results=top_k
        )

        anchor_ids = results['ids'][0]
        expanded_ids = self.task_layer.get_k_hop_neighbors(anchor_ids, hop=1)
        if not expanded_ids: return [], []

        final_results = self.collection.get(
            ids=expanded_ids,
            include=["metadatas", "documents"]
        )
        
        messages = []

        for i in range(len(final_results['ids'])):
            case_id = final_results['ids'][i]
            context = final_results['documents'][i]
            graph_json = final_results['metadatas'][i]['graph_json']
            
            msg = LegalMessage.from_dict({
                "case_id": case_id,
                "case_context": context,
                "graph_json": graph_json
            })
            
            messages.append(msg)
            
        return messages, []