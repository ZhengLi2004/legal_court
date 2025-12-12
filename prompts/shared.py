from enum import Enum

class OutputMode(Enum):
    LOGIC_ONLY = "logic"
    DSL_STRICT = "dsl"

GRAPH_DSL_VOCAB = """
【核心概念词汇表】:
1. 观点 (CLAIM): 辩论方提出的主张。
2. 事实 (FACT): 客观存在的证据或案情。
3. 法条 (LAW): 法律规定的条文。
4. 支持 (SUPPORT): 逻辑上的支撑关系。
5. 反驳 (CHALLENGE): 逻辑上的攻击/否定关系。
"""

TOPOLOGY_RULES = """
【逻辑拓扑公理 (必须遵守)】:
1. **法条神圣性**: 法条 (LAW) 是辩论的基石。严禁对法条进行 SUPPORT 或 CHALLENGE。法条只能作为 Source 去支持或反驳观点。
2. **事实客观性**: 事实 (FACT) 被视为客观发生。严禁对事实进行 SUPPORT 或 CHALLENGE。事实只能作为 Source 去支持或反驳观点。
3. **同类互斥**: 
   - 不存在 FACT vs FACT 的直接反驳（那是证据冲突，应体现为观点冲突）。
   - 不存在 LAW vs LAW 的直接反驳（那是法律适用冲突，应体现为观点冲突）。
   - 唯一合法的反驳目标通常是 **观点 (CLAIM)**。
"""

ALIAS_RULES = """
【引用与别名规则】:
1. **引用旧节点**: 必须使用图谱上下文中显示的真实 UUID (如 `[FACT_e8a1]`)。
2. **引用新节点**: 当你需要提出**新观点**或引入**新法条**时:
   - 使用临时别名: CLAIM_1, CLAIM_2, LAW_1... (计数器独立)。
"""

MODE_INSTRUCTION = {
    OutputMode.LOGIC_ONLY: """
    【输出要求 - 思路模式】:
    请使用清晰的自然语言描述你的建议。
    关键：在提及节点时，**必须**附带其 ID 或 临时别名。
    示例: "建议引入新观点 [CLAIM_1] (认为被告违约)，并使用案卷中的 [FACT_u8a9] 去支持它。"
    """,
    OutputMode.DSL_STRICT: """
    【输出要求 - 指令模式】:
    请输出严格的 DSL 代码块，不包含任何解释。
    可用指令: ADD_CLAIM, ADD_LAW, SUPPORT, CHALLENGE。
    示例:
    ADD_CLAIM("被告违约");
    SUPPORT(FACT_u8a9, CLAIM_1);
    """
}

def get_shared_prompt(mode: OutputMode = OutputMode.LOGIC_ONLY) -> str:
    sections = {
        GRAPH_DSL_VOCAB,
        TOPOLOGY_RULES,
        ALIAS_RULES,
        MODE_INSTRUCTION[mode]
    }

    return "\n".join(sections)