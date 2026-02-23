"""Provides the long-term memory storage for the legal MAS.

This module defines `LegalGMemory`, a class responsible for persistently
storing and retrieving completed legal cases. It uses ChromaDB for semantic
vector-based retrieval of case contexts and maintains its own inverted indices
for structured retrieval based on cited laws.
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from metagpt.logs import logger

from mas.infrastructure.embedding import file_lock

from ..config import SystemConfig
from ..core.graph import LegalMessage, NodeStatus, NodeType, ShadowGraph
from .base import MASMemoryBase
from .topology import TaskLayer


@dataclass
class LegalGMemory(MASMemoryBase):
    """Manages long-term storage and retrieval of legal cases.

    This class extends `MASMemoryBase` and implements a hybrid memory system.
    It stores case data, including the final argument graph, in a ChromaDB
    collection for semantic search. It also builds and maintains inverted
    indices mapping legal statutes to the cases that cite them, allowing for
    more structured, jurisprudence-based retrieval.

    Attributes:
        config: The system configuration object.
        collection_name: The name of the ChromaDB collection.
        chroma_client: The ChromaDB client instance.
        collection: The ChromaDB collection object.
        task_layer: A `TaskLayer` instance to manage the topology of related cases.
        law_inverted_index: Maps law content to a set of case IDs.
        case_law_index: Maps case IDs to a set of cited law contents.
    """

    config: SystemConfig
    collection_name: str = "legal_cases"

    def __post_init__(self):
        """Initialize the memory system, ChromaDB, and loads indices."""
        super().__post_init__()

        chroma_path = os.path.join(
            self.persist_dir, self.config.path.storage_subdir_chroma
        )
        self.index_path = os.path.join(self.persist_dir, "legal_indices.json")

        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

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
        """Load the law-to-case indices from a JSON file."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    for k, v in data.get("law_inverted_index", {}).items():
                        self.law_inverted_index[k] = set(v)

                    for k, v in data.get("case_law_index", {}).items():
                        self.case_law_index[k] = set(v)

            except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.warning(f"[Memory] Failed to load indices: {e}")

    def _save_indices(self):
        """Save the law-to-case indices to a JSON file."""
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
        """Extract all validated LAW node contents from a graph."""
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
        """Compute the Jaccard similarity between two sets."""
        union_len = len(set_a.union(set_b))

        if union_len == 0:
            return 0.0

        return len(set_a.intersection(set_b)) / union_len

    def add_memory(self, message: LegalMessage) -> None:
        """Add a completed case to the long-term memory.

        This method extracts validated laws from the case's graph, updates the
        law indices, and adds the case document and metadata to the ChromaDB
        collection.

        Args:
            message: The `LegalMessage` object representing the completed case.
        """
        current_laws = self._extract_laws_from_graph(message.shadow_graph)

        if current_laws:
            self.case_law_index[message.case_id] = current_laws

            for law in current_laws:
                self.law_inverted_index[law].add(message.case_id)

            self._save_indices()

        cited_laws_str = "|||".join(current_laws)

        payload = {
            "documents": [message.case_context],
            "metadatas": [
                {
                    "graph_json": json.dumps(ShadowGraph.to_dict(message.shadow_graph)),
                    "cited_laws": cited_laws_str,
                }
            ],
            "ids": [message.case_id],
        }

        if hasattr(self.collection, "upsert"):
            self.collection.upsert(**payload)

        else:
            raise AttributeError(
                "Chroma collection is missing `upsert`; current runtime requires it."
            )

    def retrieve_cases_by_law_codes(
        self, law_contents: List[str], top_k: int = 3
    ) -> List[LegalMessage]:
        """Retrieve cases that cite a given set of laws.

        It uses the inverted index to find candidate cases and then ranks them
        by the Jaccard similarity of their cited laws to the query laws.

        Args:
            law_contents: A list of law content strings to search for.
            top_k: The maximum number of retrieved cases.

        Returns:
            A list of the most relevant `LegalMessage` objects.
        """
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

            min_sim = float(
                getattr(self.config.retrieval, "law_jaccard_min_similarity", 0.0)
            )

            if sim > min_sim:
                scores.append((case_id, sim))

        try:
            limit = int(top_k)

        except (TypeError, ValueError):
            limit = 3

        limit = max(1, limit)
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k_ids = [s[0] for s in scores[:limit]]
        return self._fetch_messages_by_ids(top_k_ids)

    def retrieve_memory(
        self, query_context: str, top_k: int = 3
    ) -> Tuple[List[LegalMessage], List[float]]:
        """Retrieve semantically similar cases from ChromaDB.

        Args:
            query_context: The natural language description of the current case.
            top_k: The maximum number of cases to retrieve.

        Returns:
            A tuple containing a list of retrieved `LegalMessage` objects and
            their cosine similarities.
        """
        count = self.collection.count()

        if count == 0:
            return [], []

        try:
            limit = int(top_k)

        except (TypeError, ValueError):
            limit = 3

        limit = max(1, limit)
        candidate_k = min(max(limit * 5, limit), count)

        results = self.collection.query(
            query_texts=[query_context],
            n_results=candidate_k,
            include=["distances"],
        )

        found_ids = results["ids"][0] if results["ids"] else []

        if not found_ids:
            return [], []

        distances = []

        if results.get("distances") and results["distances"]:
            distances = results["distances"][0]

        min_similarity = float(
            getattr(self.config.retrieval, "semantic_min_similarity", 0.0)
        )

        filtered_pairs: List[Tuple[str, float]] = []

        for idx, case_id in enumerate(found_ids):
            if idx >= len(distances):
                continue

            distance = distances[idx]

            if distance is None:
                continue

            similarity = 1.0 - float(distance)
            similarity = max(-1.0, min(1.0, similarity))

            if similarity >= min_similarity:
                filtered_pairs.append((str(case_id), similarity))

        selected_pairs = filtered_pairs[:limit]

        if not selected_pairs:
            return [], []

        selected_ids = [case_id for case_id, _ in selected_pairs]
        fetched_messages = self._fetch_messages_by_ids(selected_ids)
        message_by_id = {msg.case_id: msg for msg in fetched_messages}

        ordered_messages: List[LegalMessage] = []
        ordered_scores: List[float] = []

        for case_id, score in selected_pairs:
            msg = message_by_id.get(case_id)

            if msg is not None:
                ordered_messages.append(msg)
                ordered_scores.append(score)

        return ordered_messages, ordered_scores

    def _fetch_messages_by_ids(self, ids: List[str]) -> List[LegalMessage]:
        """Fetch full LegalMessage objects from ChromaDB using their IDs."""
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

            except (
                IndexError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as e:
                logger.warning(f"Error loading memory {final_results['ids'][i]}: {e}")
                continue

        return messages
