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
    CITE_FACT = "cite_fact"
    CITE_LAW = "cite_law"
    SUPPORT_CLAIM = "support_claim"
    REBUT_CLAIM = "rebut_claim"

class AgentAction(BaseModel):
    action_type: AgentActionType = Field(..., description="智能体执行的动作类型")
    content: str = Field(..., description="当创建新节点时必填，仅连线时做备注。")
    target_id: Optional[str] = Field(None, description="cite类若为空则创建新Target；logic类必填。")
    source_id: Optional[str] = Field(None, description="cite类必填；logic类若为空则创建新Source。")
    relation_type: Optional[Literal[EdgeType.SUPPORT, EdgeType.CONFLICT]] = Field(None, description="当操作涉及建立关系时，关系的类型")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="额外元数据")
    def to_json(self) -> str: return self.model_dump_json(exclude_none=True, indent=2)

class TargetRole(str, Enum):
    LAW_WORKER = "LawWorker"
    FACT_WORKER = "FactWorker"
    SELF = "Self"

class ControllerIntent(BaseModel):
    target: TargetRole = Field(..., description="指令接收方")
    content: str = Field(..., description="具体的查询指令（给Worker）或 思考备注（给Self）")
    def to_json(self) -> str: return self.model_dump_json(exclude_none=True, indent=2)

AGENT_ACTION_SCHEMA_DESC = """
【AgentAction JSON 输出规范】

你的输出必须是一个 JSON 数组 `[...]`，其中每个对象代表一个动作。

1. **核心拓扑规则**:
   - **严禁造物**: 你不能创建 FACT 或 LAW 节点。
   - **连接即创造**: 当你需要提出新观点时，请通过【留空】一端ID的方式，由系统自动创建。

2. **字段定义**:
   - `action_type` (必填):
     - "cite_fact": 引用事实（source_id=FACT_xxx）。若 target_id 为 null，则创建新观点。
     - "cite_law": 引用法条（source_id=LAW_xxx）。若 target_id 为 null，则创建新观点。
     - "support_claim": 观点支持（target_id=CLAIM_xxx）。【注意】若你要提出一个新的支持理由，务必将 source_id 设为 null！
     - "rebut_claim": 观点反驳（target_id=CLAIM_xxx）。【注意】若你要提出一个新的反驳理由，务必将 source_id 设为 null！
   - `content` (必填): 动作的简短描述或新观点的内容。
   - `target_id`: 被支持或被攻击的目标节点ID。
   - `source_id`: 证据/法条/前提的节点ID。如果是创建新节点，请务必保持 null，切勿填入 target_id（否则会导致自环错误）。
   - `relation_type` (必填): "SUPPORT" 或 "CONFLICT"。

3. **标准示例**:
```json
[
    {
        "action_type": "cite_fact",
        "content": "基于事实提出新观点...",
        "source_id": "FACT_9e8c2d", 
        "target_id": null,
        "relation_type": "SUPPORT"
    },
    {
        "action_type": "cite_law",
        "content": "依据民法典支持旧观点...",
        "source_id": "LAW_34da21",
        "target_id": "CLAIM_ff562a",
        "relation_type": "SUPPORT"
    },
    {
        "action_type": "rebut_claim",
        "content": "提出新观点反驳...",
        "source_id": null, 
        "target_id": "CLAIM_4eda56",
        "relation_type": "CONFLICT"
    }
]
```
"""