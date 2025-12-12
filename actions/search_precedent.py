from typing import Any
from metagpt.actions import Action
from tools.fact_es_tool import FactEsTool

class SearchPrecedent(Action):
    name: str = "SearchPrecedent"
    tool: Any = None

    async def run(self, query: str) -> str:
        if not self.tool: return "Error: FactEsTool not initialized."
        search_result = await self.tool.search_cases(query, top_k=3)
        return search_result