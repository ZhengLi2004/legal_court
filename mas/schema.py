from enum import Enum
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from mas.common import EdgeType


class MessageType(str, Enum):
    INSTRUCTION = "INSTRUCTION"
    REPORT = "REPORT"


class WorkerInstruction(BaseModel):
    message_type: MessageType = MessageType.INSTRUCTION
    query: str = Field(...)
    graph_context: str = Field(...)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True)


class WorkerReportStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    ERROR = "ERROR"


class WorkerReport(BaseModel):
    message_type: MessageType = MessageType.REPORT
    status: WorkerReportStatus = Field(...)
    content: str = Field(...)
    max_score: float = Field(default=0.0)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True)


class AgentActionType(str, Enum):
    CITE_FACT = "cite_fact"
    CITE_LAW = "cite_law"
    SUPPORT_CLAIM = "support_claim"
    REBUT_CLAIM = "rebut_claim"


class AgentAction(BaseModel):
    action_type: AgentActionType = Field(...)
    content: str = Field(...)
    target_id: Optional[str] = Field(None)
    source_id: Optional[str] = Field(None)
    relation_type: Optional[Literal[EdgeType.SUPPORT, EdgeType.CONFLICT]] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True, indent=2)


class ResourceRequirement(BaseModel):
    need: bool = Field(...)
    reasoning: str = Field(...)
    query: Optional[str] = Field(None)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True, indent=2)


AGENT_ACTION_SCHEMA_DESC = """
【AgentAction JSON 输出规范】

你的输出必须是一个 JSON 数组 `[...]`，其中每个对象代表一个动作。

1. **核心拓扑规则**:
   - **严禁造物**: 你不能创建 FACT 或 LAW 节点。
   - **连接即创造**: 当你需要提出新观点时，请通过【留空】一端ID的方式，由系统自动创建。

2. **字段定义**:
   - `action_type` (必填):
     - "cite_fact": 引用事实（source_id=FACT_xxx）。【警告】source_id 必须填入图谱中存在的 ID，严禁为 null！
     - "cite_law": 引用法条（source_id=LAW_xxx）。【警告】source_id 必须填入图谱中存在的 ID，严禁为 null！
     - "support_claim": 观点支持（target_id=CLAIM_xxx）。【警告】target_id 必须填入图谱中存在的 ID，严禁为 null！
     - "rebut_claim": 观点反驳（target_id=CLAIM_xxx）。【警告】target_id 必须填入图谱中存在的 ID，严禁为 null！
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
