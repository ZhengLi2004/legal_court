"""Provides a tool for semantic similarity matching between texts.

This module defines the `SemanticMatcher` class, which uses an embedding
function to compare the semantic meaning of a query string against a list of
candidate strings. It's used for tasks like node deduplication and finding
analogous arguments in historical cases.
"""

from typing import List, Optional, Tuple

from .utils import EmbeddingFunc, cosine_similarity


class SemanticMatcher:
    """Compares and finds the best semantic match for a query from a list of candidates.

    This class encapsulates the logic for embedding texts and calculating cosine
    similarity. It finds the candidate with the highest similarity score above a
    configurable threshold. It also includes a simple cache to avoid re-embedding
    the same text multiple times within its lifecycle.

    Attributes:
        embedding_func: An instance of `EmbeddingFunc` to generate text vectors.
        threshold: The minimum cosine similarity score required to be considered a match.
    """

    def __init__(self, embedding_func: EmbeddingFunc, threshold: float = 0.85):
        """Initialize the SemanticMatcher.

        Args:
            embedding_func: The embedding function to use for vectorization.
            threshold: The similarity threshold for a match.
        """
        self.embedding_func = embedding_func
        self.threshold = threshold
        self._cache = {}

    def _get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a text, using a cache to avoid redundant computation."""
        if text not in self._cache:
            self._cache[text] = self.embedding_func.embed_query(text)

        return self._cache[text]

    def find_match(
        self, query: str, candidates: List[Tuple[str, str]]
    ) -> Optional[str]:
        """Find the best matching candidate for a given query.

        It iterates through all candidates, calculates the cosine similarity
        between the query and each candidate's content, and returns the ID of
        the candidate with the highest score, provided it exceeds the threshold.

        Args:
            query: The text string to find a match for.
            candidates: A list of tuples, where each tuple is (candidate_id,
                candidate_content).

        Returns:
            The ID of the best matching candidate, or None if no candidate
            meets the similarity threshold.
        """
        query_emb = self._get_embedding(query)
        best_id = None
        best_score = -1.0

        for cid, content in candidates:
            cand_emb = self._get_embedding(content)
            score = cosine_similarity(query_emb, cand_emb)

            if score > best_score:
                best_score = score
                best_id = cid

        if best_score >= self.threshold:
            return best_id

        return None
