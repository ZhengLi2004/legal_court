from typing import Any, Dict, List

from mas.utils import EmbeddingFunc

from .base_es_tool import BaseEsTool


class LawEsTool(BaseEsTool):
    INDEX_NAME = "rag_legal_laws"
    VECTOR_FIELD = "vector"
    SOURCE_FIELDS = ["law_name", "article_id", "content"]

    def __init__(self, es_host: str, embedding_func: EmbeddingFunc):
        super().__init__(es_host, embedding_func)

    async def search_laws_raw(
        self, query_text: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        query_vector = self.embedding_func.embed_query(query_text)

        return await self._search(
            index_name=self.INDEX_NAME,
            query_vector=query_vector,
            vector_field=self.VECTOR_FIELD,
            source_fields=self.SOURCE_FIELDS,
            top_k=top_k,
        )
