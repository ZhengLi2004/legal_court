from typing import Any, Dict, List

from elasticsearch import AsyncElasticsearch

from mas.utils import EmbeddingFunc


class BaseEsTool:
    def __init__(self, es_host: str, embedding_func: EmbeddingFunc):
        self.es_host = es_host
        self.embedding_func = embedding_func
        self.client = None

    async def open(self):
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

        except Exception as e:
            print(f"Error during ES search in index '{index_name}': {e}")
            return []

    async def close(self):
        if self.client:
            await self.client.close()
            self.client = None
