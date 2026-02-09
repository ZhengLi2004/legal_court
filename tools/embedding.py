"""Provides miscellaneous utility functions and classes for the MAS.

This module contains a collection of helper functions and classes used across
the application, including file locking, configuration loading, mathematical
operations like cosine similarity, and a singleton-like pattern for managing
the embedding model to avoid loading it multiple times.
"""

import contextlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Union

import numpy as np
import portalocker
import yaml

from mas.config import SystemConfig


@contextlib.contextmanager
def file_lock(lock_path: str, timeout: int = 10):
    """Create a file-based lock as a context manager.

    This is useful for preventing race conditions when multiple processes might
    try to write to the same file (e.g., the insights or task layer files).

    Args:
        lock_path: The path to the file to use for the lock.
        timeout: The number of seconds to wait to acquire the lock.

    Yields:
        None.

    Raises:
        TimeoutError: If the lock cannot be acquired within the timeout period.
    """
    lock = portalocker.Lock(lock_path, mode="a", timeout=timeout)

    try:
        with lock:
            yield

    except portalocker.exceptions.LockException:
        raise TimeoutError(
            f"Could not acquire lock on {lock_path} within {timeout} seconds."
        )


def load_config(config_path: str = "configs/configs.yaml"):
    """Load a YAML configuration file.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A dictionary containing the loaded configuration, or an empty dict
        if the file does not exist.

    Raises:
        yaml.YAMLError: If the YAML file is malformed.
    """
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(file_name: str) -> Union[list, dict]:
    """Load a JSON file."""
    if not os.path.exists(file_name):
        return None

    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def random_divide_list(lst: list[Any], k: int) -> list[list]:
    """Randomly divide a list into chunks of a maximum size k."""
    if len(lst) == 0:
        return []

    lst_copy = list(lst)
    random.shuffle(lst_copy)

    if len(lst_copy) <= k:
        return [lst_copy]

    else:
        num_chunks = math.ceil(len(lst_copy) / k)
        chunk_size = math.ceil(len(lst_copy) / num_chunks)

        return [
            lst_copy[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)
        ]


def cosine_similarity(
    vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]
) -> float:
    """Calculate the cosine similarity between two vectors.

    Args:
        vec1: The first vector.
        vec2: The second vector.

    Returns:
        The cosine similarity score, a float between -1.0 and 1.0.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def deduplicate_and_rerank(
    all_hits_list: List[List[Dict[str, Any]]],
    key_extractor: Callable[[Dict[str, Any], Dict[str, Any]], str],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Deduplicate and reranks search results from multiple queries.

    This function is used when a high-level intent is broken down into multiple
    search queries. It takes the lists of results from each query, merges them,
    removes duplicates based on a provided key, and returns the top_k results
    based on their original search score.

    Args:
        all_hits_list: A list where each element is a list of search hits from
            one query.
        key_extractor: A function that takes a search hit dictionary and its
            _source field and returns a unique key for deduplication.
        top_k: The final number of results to return.

    Returns:
        A single, deduplicated, and reranked list of the top_k search hits.
    """
    unique_map = {}

    for hits in all_hits_list:
        for hit in hits:
            source = hit.get("_source", {})
            key = key_extractor(hit, source)
            score = hit.get("_score", 0.0)

            if key not in unique_map:
                unique_map[key] = hit

            else:
                if score > unique_map[key].get("_score", 0.0):
                    unique_map[key] = hit

    all_unique_hits = list(unique_map.values())
    all_unique_hits.sort(key=lambda x: x.get("_score", 0.0), reverse=True)

    return all_unique_hits[:top_k]


_EMBEDDING_MODEL_CACHE = {}


@dataclass
class EmbeddingFunc:
    """A wrapper for the sentence transformer embedding model.

    This class ensures that the potentially large embedding model is loaded
    into memory only once and is shared across all components that need it.
    It uses a global cache keyed by the model path.

    Attributes:
        model_path: The file path to the sentence transformer model directory.
        func: The loaded embedding function object from `chromadb.utils`.
    """

    model_path: str = None

    def __post_init__(self):
        """Load the embedding model or retrieves it from the cache."""
        if self.model_path is None:
            self.model_path = SystemConfig().path.embedding_model_path

        from chromadb.utils import embedding_functions

        if self.model_path not in _EMBEDDING_MODEL_CACHE:
            print(f"[EmbeddingFunc] Loading model from: {self.model_path}")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Embedding model not found at {self.model_path}"
                )

            _EMBEDDING_MODEL_CACHE[self.model_path] = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.model_path
                )
            )

        self.func = _EMBEDDING_MODEL_CACHE[self.model_path]

    def embed_documents(self, texts: list[str]) -> list[list]:
        """Generate embeddings for a list of documents.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors.
        """
        return self.func(texts)

    def embed_query(self, query: str) -> list:
        """Generate an embedding for a single query string.

        Args:
            query: The string to embed.

        Returns:
            A single embedding vector.
        """
        return self.func([query])[0]
