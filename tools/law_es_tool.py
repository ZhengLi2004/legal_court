from typing import List, Dict, Any
from .base_es_tool import BaseEsTool
from mas.utils import EmbeddingFunc

class LawEsTool(BaseEsTool):
    INDEX_NAME = "rag_legal_laws"
    VECTOR_FIELD = "vector"
    SOURCE_FIELDS = ["law_name", "article_id", "content"]

    def __init__(self, es_host: str, embedding_func: EmbeddingFunc): super().__init__(es_host, embedding_func)

    async def search_laws_raw(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vector = self.embedding_func.embed_query(query_text)
        
        return await self._search(
            index_name=self.INDEX_NAME,
            query_vector=query_vector,
            vector_field=self.VECTOR_FIELD,
            source_fields=self.SOURCE_FIELDS,
            top_k=top_k
        )

    async def search_laws(self, query_text: str, top_k: int = 3) -> str:
        results = await self.search_laws_raw(query_text, top_k)
        if not results: return "未找到相关法律条款。"
        formatted_results = ["### 相关法律条款参考："]

        for i, hit in enumerate(results):
            source = hit['_source']
            score = hit['_score'] - 1.0
            law_name = source.get('law_name', '')
            article_id = source.get('article_id', '')
            content = source.get('content', '').strip().replace('\n', ' ')[:150]

            formatted_results.append(
                f"{i+1}. [相似度: {score:.4f}] {law_name} - {article_id}\n"
                f"   [内容]: {content}..."
            )

        return "\n".join(formatted_results)