"""Provides a specialized tool for searching a legal statutes index in Elasticsearch.

This module defines `LawEsTool`, a concrete implementation of `BaseEsTool`
configured specifically for querying an index of legal articles and statutes.
"""

from typing import Any, Dict, List

from tools.embedding import EmbeddingFunc

from .base_es_tool import BaseEsTool


class LawEsTool(BaseEsTool):
    """An Elasticsearch tool for searching legal statutes.

    This class inherits from `BaseEsTool` and presets the index name, vector
    field, and source fields for querying a pre-defined legal statutes index.
    """

    INDEX_NAME = "rag_legal_laws"
    VECTOR_FIELD = "vector"
    SOURCE_FIELDS = ["law_name", "article_id", "content"]

    def __init__(self, es_host: str, embedding_func: EmbeddingFunc):
        """Initialize the LawEsTool.

        Args:
            es_host: The URL of the Elasticsearch host.
            embedding_func: The function/object used to generate query embeddings.
        """
        super().__init__(es_host, embedding_func)

    async def search_laws_raw(
        self, query_text: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Search the legal statutes index with a natural language query.

        This method converts the query text to an embedding vector and then calls
        the underlying `_search` method to find the most relevant statutes.

        Args:
            query_text: The natural language search query.
            top_k: The number of top statutes to return.

        Returns:
            A list of hit dictionaries from the Elasticsearch response.
        """
        query_vector = self.embedding_func.embed_query(query_text)

        return await self._search(
            index_name=self.INDEX_NAME,
            query_vector=query_vector,
            vector_field=self.VECTOR_FIELD,
            source_fields=self.SOURCE_FIELDS,
            top_k=top_k,
        )
