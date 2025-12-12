from metagpt.roles import Role
from metagpt.schema import Message
from actions.search_precedent import SearchPrecedent
from tools.fact_es_tool import FactEsTool

class PrecedentWorker(Role):
    name: str = "PrecedentSearcher"
    profile: str = "Senior Legal Researcher"
    tool: FactEsTool = None

    def __init__(self, tool: FactEsTool, **kwargs):
        super().__init__(**kwargs)
        self.tool = tool
        action_instance = SearchPrecedent()
        action_instance.tool = self.tool
        self.set_actions([action_instance])
        self._set_react_mode(react_mode="by_order")

    async def _act(self) -> Message:
        memories = self.get_memories(k=1)
        if not memories: return Message(content="No instruction received.", role=self.profile)
        msg = memories[0]
        query = msg.content
        todo = self.actions[0]
        result = await todo.run(query)

        response_msg = Message(
            content=f"【先例检索报告】\n基于您的指令 '{query}'，检索结果如下：\n{result}",
            role=self.profile,
            cause_by=SearchPrecedent
        )

        self.rc.memory.add(response_msg)

        return response_msg