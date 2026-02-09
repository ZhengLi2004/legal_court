"""Provides a specialized tool for searching a legal cases index in Elasticsearch.

This module defines `FactEsTool`, a concrete implementation of `BaseEsTool`
configured specifically for querying an index of legal case documents.
"""

from typing import Any, Dict, List

from tools.embedding import EmbeddingFunc

from .base_es_tool import BaseEsTool


class FactEsTool(BaseEsTool):
    """An Elasticsearch tool for searching factual information in legal cases.

    This class inherits from `BaseEsTool` and presets the index name, vector
    field, and source fields for querying a pre-defined legal cases index.
    """

    INDEX_NAME = "rag_legal_cases"
    VECTOR_FIELD = "combined_vector"
    SOURCE_FIELDS = ["case_title", "analysis", "fact_finding"]

    def __init__(self, es_host: str, embedding_func: EmbeddingFunc):
        """Initialize the FactEsTool.

        Args:
            es_host: The URL of the Elasticsearch host.
            embedding_func: The function/object used to generate query embeddings.
        """
        super().__init__(es_host, embedding_func)

    async def search_cases_raw(
        self, query_text: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Search the legal cases index with a natural language query.

        This method converts the query text to an embedding vector and then calls
        the underlying `_search` method to find the most relevant case documents.

        Args:
            query_text: The natural language search query.
            top_k: The number of top cases to return.

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
