"""Provides a base class for interacting with Elasticsearch.

This module defines `BaseEsTool`, an abstract base class that encapsulates the
common logic for connecting to and performing vector similarity searches on an
Elasticsearch index. Concrete tool classes for specific indices (like facts or
laws) should inherit from this class.
"""

from typing import Any, Dict, List

from elasticsearch import (
    ApiError,
    AsyncElasticsearch,
    ConnectionError,
    SerializationError,
    TransportError,
)
from metagpt.logs import logger

from tools.embedding import EmbeddingFunc


class BaseEsTool:
    """A base class for asynchronous Elasticsearch vector search tools.

    This class handles the asynchronous client connection management (`open`,
    `close`) and provides a generic `_search` method for performing a cosine
    similarity search against a specified vector field.

    Attributes:
        es_host: The hostname and port of the Elasticsearch instance.
        embedding_func: An `EmbeddingFunc` instance to convert query text to vectors.
        client: The `AsyncElasticsearch` client instance, initialized on first use.
    """

    def __init__(self, es_host: str, embedding_func: EmbeddingFunc):
        """Initialize the BaseEsTool.

        Args:
            es_host: The URL of the Elasticsearch host (e.g., "http://localhost:9200").
            embedding_func: The function/object used to generate query embeddings.
        """
        self.es_host = es_host
        self.embedding_func = embedding_func
        self.client = None

    async def open(self):
        """Initialize the asynchronous Elasticsearch client if not already connected."""
        if self.client is None:
            self.client = AsyncElasticsearch(self.es_host)

    async def _search(
        self,
        index_name: str,
        query_vector: List[float],
        vector_field: str,
        source_fields: List[str],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Perform a generic k-NN vector search on an Elasticsearch index.

        It uses a script_score query with cosineSimilarity to find the top_k
        most similar documents to the given query_vector.

        Args:
            index_name: The name of the Elasticsearch index to search.
            query_vector: The embedding vector of the search query.
            vector_field: The name of the field in the index that contains the document vectors.
            source_fields: A list of field names to be returned in the search results.
            top_k: The number of top results to return.

        Returns:
            A list of hit dictionaries from the Elasticsearch response, or an
            empty list if an error occurs.
        """
        if self.client is None:
            await self.open()

        search_body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{vector_field}') + 1.0",
                        "params": {"query_vector": query_vector},
                    },
                }
            },
            "_source": source_fields,
        }

        try:
            response = await self.client.search(
                index=index_name, body=search_body, request_timeout=30
            )

            return response["hits"]["hits"]

        except (ApiError, ConnectionError, SerializationError, TransportError) as e:
            logger.warning(f"Error during ES search in index '{index_name}': {e}")
            return []

    async def close(self):
        """Close the asynchronous Elasticsearch client connection if it exists."""
        if self.client:
            await self.client.close()
            self.client = None
