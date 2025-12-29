from typing import List, Optional, Tuple

from .utils import EmbeddingFunc, cosine_similarity


class SemanticMatcher:
    def __init__(self, embedding_func: EmbeddingFunc, threshold: float = 0.85):
        self.embedding_func = embedding_func
        self.threshold = threshold
        self._cache = {}

    def _get_embedding(self, text: str) -> List[float]:
        if text not in self._cache:
            self._cache[text] = self.embedding_func.embed_query(text)

        return self._cache[text]

    def find_match(
        self, query: str, candidates: List[Tuple[str, str]]
    ) -> Optional[str]:
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
