from enum import Enum
from pydantic import BaseModel, Field

class MessageType(str, Enum):
    INSTRUCTION = "INSTRUCTION"
    REPORT = "REPORT"
# 指挥者发给工作者的指令包
class WorkerInstruction(BaseModel):
    message_type: MessageType = MessageType.INSTRUCTION
    query: str = Field(..., description="自然语言查询指令，例如：'查询关于借条未写明还款日期的先例'")
    graph_context: str = Field(..., description="当前的 ShadowGraph.latest_context 文本")
    def to_json(self) -> str: return self.model_dump_json(exclude_none=True)

class WorkerReportStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    ERROR = "ERROR"
# 工作者发给指挥者的报告
class WorkerReport(BaseModel):
    message_type: MessageType = MessageType.REPORT
    status: WorkerReportStatus = Field(..., description="搜索结果的状态")
    content: str = Field(..., description="给指挥者的最终建议文本")
    max_score: float = Field(default=0.0, description="本次检索的最高相似度得分")
    def to_json(self) -> str: return self.model_dump_json(exclude_none=True)