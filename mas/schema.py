from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Literal, Union, Dict, Any
from mas.common import EdgeType

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
# 新增 AgentAction 模型
class AgentActionType(str, Enum):
    ADD_CLAIM = "add_claim"
    CITE_LAW = "cite_law"
    REBUT_CLAIM = "rebut_claim"
    ADD_FACT = "add_fact"

class AgentAction(BaseModel):
    action_type: AgentActionType = Field(..., description="智能体执行的动作类型")
    content: str = Field(..., description="动作的主要内容，例如主张文本、法条查询内容或事实描述")
    target_id: Optional[str] = Field(None, description="动作目标节点的ID（例如，反驳某个主张时的主张ID）")
    source_id: Optional[str] = Field(None, description="动作源节点的ID（例如，某个主张支持另一个主张时的发起主张ID）")
    relation_type: Optional[Literal[EdgeType.SUPPORT, EdgeType.CONFLICT]] = Field(None, description="当操作涉及建立关系时，关系的类型")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="额外元数据")
    def to_json(self) -> str: return self.model_dump_json(exclude_none=True, indent=2)