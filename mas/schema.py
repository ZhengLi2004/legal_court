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

    你的输出必须是一个 JSON 数组 `[...]`，其中每个对象代表一个图谱操作。

    **核心字段**:
    - `action_type` (string): 你的意图。必须是 "cite_fact", "cite_law", "support_claim", "rebut_claim" 之一。
    - `content` (string): 你的观点内容。
    - `source_id` (string | null): 动作的“来源”或“依据”。
    - `target_id` (string | null): 动作的“目标”。

    **【重要】规则按 `action_type` 定义**:

    1.  **当 `action_type` 是 "cite_fact" 或 "cite_law" (引用证据/法条)**:
        -   `source_id`: **必须**是图谱中一个已存在的 `FACT` 或 `LAW` 节点的ID。
        -   `target_id`:
            -   如果你想用这个证据支持一个**已存在的观点**，这里就填那个观点的ID。
            -   如果你想用这个证据提出一个**新观点**，这里必须是 `null`。
        -   `content`: 描述你基于该证据提出的观点。

    2.  **当 `action_type` 是 `"support_claim"` 或 `"rebut_claim"` (支持/反驳观点)**:
        -   `target_id`: **必须**是图谱中一个已存在的 `CLAIM` 节点的ID (即被你支持或反驳的目标)。
        -   `source_id`:
            -   如果你想用一个**已存在的观点**去支持/反驳目标，这里就填那个观点的ID。
            -   如果你想提出一个**新观点**去支持/反驳目标，这里必须是 `null`。
        -   `content`: 描述你的新观点。

    **示例**:
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
