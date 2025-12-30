from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


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
   - **严禁造物**: 你不能创建 FACT 或 LAW 节点，只能引用它们。
   - **连接即创造**: 当你需要基于现有证据提出新观点时，请将 `target_id` (如果引用证据) 或 `source_id` (如果支持/反驳现有观点) 设为 `null`，系统将为你自动创建新节点。

2. **字段定义**:
   - `action_type` (必填): 动作类型。
     - `"cite_fact"`: 引用一个事实来支撑一个新观点或现有观点。
     - `"cite_law"`: 引用一条法条来支撑一个新观点或现有观点。
     - `"support_claim"`: 提出一个新观点来支撑另一个现有观点。
     - `"rebut_claim"`: 提出一个新观点来反驳另一个现有观点。
   - `content` (必填): 动作的简短描述，或你新提出的观点内容。
   - `target_id` (`string` 或 `null`): 动作的目标节点 ID。对于 `support_claim` 和 `rebut_claim`，这是被支持或被反驳的观点。对于 `cite_fact` 和 `cite_law`，如果想支持一个现有观点，请填入该观点 ID。
   - `source_id` (`string` 或 `null`): 动作的源头节点 ID。对于 `cite_fact` 和 `cite_law`，这是你引用的事实或法条的 ID (【警告】必须填入图谱中已存在的 ID)。对于 `support_claim` 和 `rebut_claim`，如果你想用一个新观点去操作，请保持为 `null`。

3. **标准示例**:
```json
[
    {
        "action_type": "cite_fact",
        "content": "基于借条事实，我认为被告有还款义务。",
        "source_id": "FACT_9e8c2d", 
        "target_id": null
    },
    {
        "action_type": "cite_law",
        "content": "依据民法典支持我方关于利息的观点。",
        "source_id": "LAW_34da21",
        "target_id": "CLAIM_ff562a"
    },
    {
        "action_type": "rebut_claim",
        "content": "对方的观点忽略了合同中的附加条款，因此不成立。",
        "source_id": null, 
        "target_id": "CLAIM_4eda56"
    }
]
```
"""
