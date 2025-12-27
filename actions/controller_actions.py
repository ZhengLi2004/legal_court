from metagpt.actions import Action

class PlanTactics(Action):
    name: str = "PlanTactics"
    
    PROMPT_TEMPLATE: str = """
    你是【{role_name}】的辩护律师。
    
    【你的核心人设 (BDI)】:
    - 风格: {style}
    - 信念: {belief}
    - 战略重心: {strategic_focus}
    
    【历史策略锦囊】:
    {insights}
    
    【当前战局】:
    {graph_context}

    {feedback_section}
    
    你的任务：为了推进我方战略，你需要指示哪位参谋去寻找什么情报？
    
    【可用参谋】:
    - **事实参谋 (FactWorker)**: 负责检索相似的历史判例，提供战术参考。
    - **法律参谋 (LawWorker)**: 负责检索具体的法律条款。
    
    请输出一个简练的查询指令，并明确指出由谁执行。
    
    【输出格式】:
    指示 [FactWorker/LawWorker] 查询 [具体内容]
    
    【示例】:
    指示 FactWorker 查询关于'借条备注'的先例
    指示 LawWorker 查询《民法典》中关于'债务加入'的规定
    """

    async def run(self, role_name: str, persona: object, insights: str, graph_context: str, feedback: str = ""):
        feedback_text = ""
        if feedback: feedback_text = f"【⚠️ 上次尝试失败反馈】:\n{feedback}\n请分析失败原因，并重新规划。如果是指令错误，请修正格式；如果是逻辑错误，请调整策略。"
        
        prompt = self.PROMPT_TEMPLATE.format(
            role_name=role_name,
            style=persona.intention,
            belief=persona.belief,
            strategic_focus=persona.initial_strategy,
            insights=insights,
            graph_context=graph_context,
            feedback_section=feedback_text
        )
        
        return await self.llm.aask(prompt)
    
class VerifyAndDecide(Action):
    name: str = "VerifyAndDecide"

    PROMPT_TEMPLATE: str = """
    你是【{role_name}】。你的参谋提供了一条建议。

    【参谋建议】:
    "{worker_advice}"

    【当前战局】:
    {graph_context}

    请判断：这条建议是否有利于你的【战略重心】({focus})？
    
    如果采纳这条建议，请严格按照以下 JSON 格式生成一个表示你决策的 AgentAction 对象。
    你的输出必须是完整的、合法的 JSON 字符串，且必须是包含一个或多个 AgentAction 对象的 JSON 数组。数组中的每个对象必须符合 AgentAction 模型的定义。

    AgentAction模型定义（字段说明）：
    - action_type (enum): 动作类型，必须是 'add_claim', 'cite_law', 'rebut_claim' 之一。
    - content (string): 动作的具体内容，例如主张的详细文本，法条查询的关键词，或事实描述。
    - target_id (string, optional): 动作的目标节点 ID。例如，反驳某个主张时，这是被反驳的主张的 ID。
    - source_id (string, optional): 动作的来源节点 ID。例如，一个主张支持另一个主张，这是支持方的 ID。
    - relation_type (enum, optional): **【强制】** 仅当动作涉及创建关系（即 `action_type` 为 `cite_law` 或 `rebut_claim`）时使用，**必须且只能是 'SUPPORT' 或 'CONFLICT'**。**严禁使用其他任何值！**
        当 `action_type` 为 `add_claim` 时，`relation_type` 必须为 `null`。

    【输出示例】:
    ```json
    [
        {{
            "action_type": "add_claim",
            "content": "被告无证据证明其已履行还款义务",
            "target_id": null,
            "source_id": null,
            "relation_type": null
        }},
        {{
            "action_type": "cite_law",
            "content": "中华人民共和国合同法 第二百零六条",
            "target_id": "CLAIM_12345678",
            "source_id": null,
            "relation_type": "SUPPORT"
        }},
        {{
            "action_type": "rebut_claim",
            "content": "原告提交的证据链存在断裂，无法证明资金流向被告",
            "target_id": "CLAIM_98765432",
            "source_id": null,
            "relation_type": "CONFLICT"
        }}
    ]
    ```
    如果认为不应采纳建议，请直接输出 "REJECT:" 加上拒绝理由（例如 "REJECT: 建议与我方战略不符"）。
    """

    async def run(self, role_name: str, worker_advice: str, graph_context: str, focus: str):
        prompt = self.PROMPT_TEMPLATE.format(
            role_name=role_name,
            worker_advice=worker_advice,
            graph_context=graph_context,
            focus=focus
        )
        
        return await self.llm.aask(prompt, max_tokens=8192)