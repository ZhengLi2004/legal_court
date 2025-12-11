from .base_es_tool import BaseEsTool
from mas.utils import EmbeddingFunc

class FactEsTool(BaseEsTool):
    INDEX_NAME = "rag_legal_cases"
    VECTOR_FIELD = "combined_vector"
    SOURCE_FIELDS = ["case_title", "analysis", "fact_finding"]

    def __init__(self, es_host: str, embedding_func: EmbeddingFunc): super().__init__(es_host, embedding_func)

    async def search_cases(self, query_text: str, top_k: int = 3) -> str:
        query_vector = self.embedding_func.embed_query(query_text)

        results = await self._search(
            index_name=self.INDEX_NAME,
            query_vector=query_vector,
            vector_field=self.VECTOR_FIELD,
            source_fields=self.SOURCE_FIELDS,
            top_k=top_k
        )

        if not results: return "未找到相关历史案例。"
        formatted_results = ["### 相关历史案例参考："]

        for i, hit in enumerate(results):
            source = hit['_source']
            score = hit['_score'] - 1.0
            title = source.get('case_title', '无标题')
            analysis = source.get('analysis', '').strip().replace('\n', ' ')[:100]

            formatted_results.append(
                f"{i+1}. [相似度: {score:.4f}] 标题: {title}\n"
                f"   [案情摘要]: {analysis}..."
            )

        return "\n".join(formatted_results)