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
    CITE_FACT = "cite_fact"
    CITE_LAW = "cite_law"
    SUPPORT_CLAIM = "support_claim"
    REBUT_CLAIM = "rebut_claim"

class AgentAction(BaseModel):
    action_type: AgentActionType = Field(..., description="智能体执行的动作类型")
    content: str = Field(..., description="动作的主要内容，例如主张文本、法条查询内容或事实描述")
    target_id: Optional[str] = Field(None, description="动作目标节点的ID（例如，反驳某个主张时的主张ID）")
    source_id: Optional[str] = Field(None, description="动作源节点的ID（例如，某个主张支持另一个主张时的发起主张ID）")
    relation_type: Optional[Literal[EdgeType.SUPPORT, EdgeType.CONFLICT]] = Field(None, description="当操作涉及建立关系时，关系的类型")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="额外元数据")
    def to_json(self) -> str: return self.model_dump_json(exclude_none=True, indent=2)

AGENT_ACTION_SCHEMA_DESC = """
【AgentAction JSON 输出规范】

你的输出必须是一个 JSON 数组 `[...]`，其中每个对象代表一个动作。

1. **核心拓扑规则**:
   - **严禁造物**: 你不能创建 FACT 或 LAW 节点。
   - **引用必连**: 使用 `cite_fact` / `cite_law` 时，`source_id` 必须填入图谱中【已存在】的节点ID。
   - **逻辑链**: 使用 `support_claim` / `rebut_claim` 连接观点。

2. **字段定义**:
   - `action_type` (必填):
     - "add_claim": 提出新观点（仅当无法连接现有节点时）。
     - "cite_fact": 引用事实（source_id=FACT_xxx -> target_id=CLAIM_yyy）。
     - "cite_law": 引用法条（source_id=LAW_xxx -> target_id=CLAIM_yyy）。
     - "support_claim": 观点支持（source_id=CLAIM_A -> target_id=CLAIM_B）。
     - "rebut_claim": 观点反驳（source_id=CLAIM_A -> target_id=CLAIM_B）。
   - `content` (必填): 动作的简短描述或新观点的内容。
   - `target_id` (必填): 被支持或被攻击的目标节点ID。
   - `source_id` (引用类必填): 证据/法条/前提的节点ID。
   - `relation_type` (连线类必填): "SUPPORT" 或 "CONFLICT"。

3. **标准示例**:
```json
[
    {
        "action_type": "cite_fact",
        "content": "证据显示...",
        "source_id": "FACT_9e8c2d", 
        "target_id": "CLAIM_14d2e6",
        "relation_type": "SUPPORT"
    },
    {
        "action_type": "cite_law",
        "content": "依据民法典...",
        "source_id": "LAW_34da21",
        "target_id": "CLAIM_ff562a",
        "relation_type": "SUPPORT"
    },
    {
        "action_type": "rebut_claim",
        "content": "反驳对方观点...",
        "source_id": "CLAIM_3e2a21", 
        "target_id": "CLAIM_4eda56",
        "relation_type": "CONFLICT"
    }
]
```
"""