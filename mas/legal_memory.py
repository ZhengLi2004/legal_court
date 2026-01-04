import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import chromadb
from chromadb.utils import embedding_functions

from .common import LegalMessage, NodeStatus, NodeType, ShadowGraph
from .config import SystemConfig
from .memory_base import MASMemoryBase
from .task_layer import TaskLayer
from .utils import file_lock


@dataclass
class LegalGMemory(MASMemoryBase):
    config: SystemConfig = field(default_factory=SystemConfig)
    collection_name: str = "legal_cases"

    def __post_init__(self):
        super().__post_init__()

        chroma_path = os.path.join(
            self.persist_dir, self.config.path.storage_subdir_chroma
        )
        self.index_path = os.path.join(self.persist_dir, "legal_indices.json")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        self.chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_path
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.chroma_ef,
            metadata={"hnsw:space": "cosine"},
        )

        self.task_layer = TaskLayer(working_dir=self.persist_dir)
        self.law_inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self.case_law_index: Dict[str, Set[str]] = defaultdict(set)
        self._load_indices()

    def _load_indices(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    for k, v in data.get("law_inverted_index", {}).items():
                        self.law_inverted_index[k] = set(v)

                    for k, v in data.get("case_law_index", {}).items():
                        self.case_law_index[k] = set(v)

            except Exception as e:
                print(f"[Memory] Failed to load indices: {e}")

    def _save_indices(self):
        data = {
            "law_inverted_index": {
                k: list(v) for k, v in self.law_inverted_index.items()
            },
            "case_law_index": {k: list(v) for k, v in self.case_law_index.items()},
        }

        with file_lock(self.index_path + ".lock"):
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def _extract_laws_from_graph(self, graph: ShadowGraph) -> Set[str]:
        laws = set()

        if graph and graph.graph:
            for _, data in graph.graph.nodes(data=True):
                if (
                    data.get("type") == NodeType.LAW
                    and data.get("status") == NodeStatus.VALIDATED
                ):
                    content = data.get("content", "").strip()

                    if content:
                        laws.add(content)

        return laws

    def _compute_jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        union_len = len(set_a.union(set_b))

        if union_len == 0:
            return 0.0

        return len(set_a.intersection(set_b)) / union_len

    def add_memory(self, message: LegalMessage) -> None:
        current_laws = self._extract_laws_from_graph(message.shadow_graph)

        if current_laws:
            self.case_law_index[message.case_id] = current_laws

            for law in current_laws:
                self.law_inverted_index[law].add(message.case_id)

            self._save_indices()

        cited_laws_str = "|||".join(current_laws)

        self.collection.add(
            documents=[message.case_context],
            metadatas=[
                {
                    "graph_json": json.dumps(ShadowGraph.to_dict(message.shadow_graph)),
                    "cited_laws": cited_laws_str,
                }
            ],
            ids=[message.case_id],
        )

    def retrieve_cases_by_law_codes(
        self, law_contents: List[str]
    ) -> List[LegalMessage]:
        if not law_contents:
            return []

        target_set = set(law_contents)
        candidate_case_ids = set()

        for law in target_set:
            if law in self.law_inverted_index:
                candidate_case_ids.update(self.law_inverted_index[law])

        if not candidate_case_ids:
            return []

        scores: List[Tuple[str, float]] = []

        for case_id in candidate_case_ids:
            case_laws = self.case_law_index.get(case_id, set())
            sim = self._compute_jaccard(target_set, case_laws)

            if sim > 0:
                scores.append((case_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_k_ids = [s[0] for s in scores[:3]]
        return self._fetch_messages_by_ids(top_k_ids)

    def retrieve_memory(
        self, query_context: str, top_k: int = 3
    ) -> Tuple[List[LegalMessage], List[str]]:
        count = self.collection.count()

        if count == 0:
            return [], []

        results = self.collection.query(
            query_texts=[query_context], n_results=min(top_k, count)
        )

        found_ids = results["ids"][0] if results["ids"] else []
        return self._fetch_messages_by_ids(found_ids), []

    def _fetch_messages_by_ids(self, ids: List[str]) -> List[LegalMessage]:
        if not ids:
            return []

        final_results = self.collection.get(ids=ids, include=["metadatas", "documents"])
        messages = []

        if not final_results["ids"]:
            return []

        for i in range(len(final_results["ids"])):
            try:
                case_id = final_results["ids"][i]
                context = final_results["documents"][i]
                graph_json_str = final_results["metadatas"][i]["graph_json"]
                sg = ShadowGraph.from_dict(json.loads(graph_json_str))

                msg = LegalMessage(
                    case_id=case_id, case_context=context, shadow_graph=sg
                )

                messages.append(msg)

            except Exception as e:
                print(f"Error loading memory {final_results['ids'][i]}: {e}")
                continue

        return messages
