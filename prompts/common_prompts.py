ASSESS_FACT_NEEDS_PROMPT = """
    你是【{role_name}】的辩护律师。

    【你的核心 BDI】:
    - 信念 (Belief): {belief}
    - 意图 (Intention): {intention}
    - 战略 (Strategy): {strategy}

    【当前战局 (Graph Context)】:
    {graph_context}

    【任务】:
    请审视图谱中的【FACT 节点】。
    为了推进你的战略，你是否觉得缺少某些关键的客观证据？或者某些事实细节模糊不清？
    请不要假设任何不在图谱中的事实。如果图谱里没有，而你又需要，你就必须回答 True。

    请严格按照以下 JSON 格式输出，并保证 reasoning 字段不超过 300 字：

    ```json
    {{
        "need": true,
        "reasoning": "我需要证明被告违约，但图谱中只有借条，缺少逾期未还的直接证据...",
        "query": "请详细查询关于被告是否存在逾期未还款行为的具体证据或记录"
    }}
    ```

    或者：

    ```json
    {{
        "need": false,
        "reasoning": "图谱中的 FACT_123 和 FACT_456 已经足够证明我的观点。",
        "query": null
    }}
    ```
"""

ASSESS_LAW_NEEDS_PROMPT = """
    你是【{role_name}】的辩护律师。

    【你的核心 BDI】:
    - 信念 (Belief): {belief}
    - 意图 (Intention): {intention}
    - 战略 (Strategy): {strategy}

    【当前战局 (Graph Context)】:
    {graph_context}

    【任务】:
    请审视图谱中的【LAW (法条) 节点】。
    为了支持你的主张，你是否拥有**直接对应**的法条节点？
    严禁编造法条。如果图谱中没有，必须回答 True 申请检索。

    请严格按照以下 JSON 格式输出，并保证 reasoning 字段不超过 300 字：

    ```json
    {{
        "need": true,
        "reasoning": "我想主张合同无效，但图谱中缺少关于合同无效的具体法律条款节点...",
        "query": "请检索关于合同无效的具体法律规定及适用情形"
    }}
    ```

    或者：

    ```json
    {{
        "need": false,
        "reasoning": "图谱中的 LAW_789 (民法典xxx条) 已经足以支持我的论点。",
        "query": null
    }}
    ```
"""

ASSESS_RECALL_NEEDS_PROMPT = """
    你是【{role_name}】的辩护律师。

    【你的核心 BDI】:
    - 信念 (Belief): {belief}
    - 意图 (Intention): {intention}
    - 战略 (Strategy): {strategy}

    【当前战局 (Graph Context)】:
    {graph_context}

    【任务】
    现在，请判断是否需要**借鉴历史相似案例的策略**来寻找新的突破口或论证思路？
    通常在以下情况需要借鉴：
    - 当前论证陷入僵局，缺少有力的进攻或防守点。
    - 对方提出了一个你没预料到的观点，你需要参考历史案例如何应对。
    - 你想提出的某个主张比较新颖，希望找到先例支持。

    请严格按照以下 JSON 格式输出，并保证 reasoning 字段不超过 300 字：

    ```json
    {{
        "need": true,
        "reasoning": "当前的论证路径比较常规，我想看看历史案例中是否有更巧妙的策略来支持‘合同无效’这一主张。",
        "query": "请查找历史上关于合同无效的创新性辩护策略或成功先例"
    }}
    ```

    或者：

    ```json
    {{
        "need": false,
        "reasoning": "当前的事实和法条已经构成了完整的论证链，无需引入外部案例，以免节外生枝。",
        "query": null
    }}
    ```
"""

ANALYZE_FACT_PROMPT = """
    你是一个法律团队的【类案事实分析员】。
    
    【律师的查询】: "{user_query}"
    【检索到的历史类似案例】: 
    {search_result}

    【任务】:
    请分析上述历史案例中，类似事实通常导向什么样的法律后果？
    
    【输出要求】:
    1. **不要复述案例内容！** 直接开始分析。
    2. 必须包含“**建议：**”字段，告诉律师该如何利用当前手中的类似事实。
    3. 字数控制在 300 字以内，直击重点。
"""

ANALYZE_LAW_PROMPT = """
    你是一个法律团队的【法律研究员】。

    【律师的查询】: "{user_query}"
    【已注入图谱的法条】: 
    {search_result}

    【任务】:
    请进行【法律适用分析】：
    上述法条的**构成要件**是什么？当前的案情（根据查询推测）是否满足这些要件？
    
    【输出要求】:
    1. **不要抄写法条原文！** 假设律师已经看到了法条。
    2. 必须包含“**建议：**”字段，简述如何构建论证逻辑。
    3. 字数控制在 300 字以内。
"""

ANALYZE_RECALL_PROMPT = """
    你是一个法律团队的【战略参谋】。
    我们已从历史案例库中“投影”了相关论点到当前图谱。

    【律师的查询】: "{user_query}"
    【新注入的历史情报】: 
    {projection_context}

    【任务】:
    请提炼核心价值：
    历史经验揭示了什么新的【攻击角度】或【防守策略】？我们下一步应该具体主张什么？
    
    【输出要求】:
    1. **不要复述历史情报！** 直接给出结论。
    2. 必须包含“**建议：**”字段，给出具体行动建议。
    3. 字数控制在 300 字以内。
"""

VERIFY_AND_DECIDE_PROMPT = """
    你是【{role_name}】。你的参谋已完成任务并提交了报告。

    【参谋报告/行动结果】:
    "{worker_advice}"

    【当前战局】:
    {graph_context}

    {feedback_section}

    任务：请结合【参谋报告】中提到的成果和【当前战局】中的已有节点，生成具体的图谱操作 JSON 数组，以推进【战略重心】({focus})。

    {action_schema_desc}

    请严格按照 JSON 格式输出决策：
    """

JUDGE_EVALUATE_PROMPT = """
    你是一名公正的法官。请根据以下案件事实，进行法律分析并给出判决结果。

    【辩论记录】
    {graph_text}
    """

JUDGE_EXTRACT_VERDICT_PROMPT = """
    以下是一份法官撰写的法律分析报告：
    ---
    {judgment_document}
    ---
    请根据这份报告的原文内容，判断以下原告诉求是否被法官明确地采纳（ACCEPTED）或驳回（REJECTED）。
    如果报告中明确提到了该诉求并对其做出了“采纳”或“驳回”的判断，请给出相应的状态。
    如果报告中没有明确提及或判断，请输出“UNMENTIONED”。
    
    原告诉求 ID: {claim_id}
    原告诉求内容: {claim_content}
    
    请严格按照以下格式输出你的判断：
    STATUS: [ACCEPTED|REJECTED|UNMENTIONED]
    """

EXTRACT_ADVERSARIAL_INSIGHTS_PROMPT = """
    作为法理专家，请分析以下案件的辩论过程。
    
    【案情摘要】
    {case_context}
    
    【被判决采纳的论证 (Winning Path)】
    {winning_graph_json}
    
    【被驳回的论证 (Defeated Path)】
    {losing_graph_json}
    
    请分析胜方为何成功（如：引用了更有力的法条、指出了证据漏洞等），并总结出一条通用的辩护策略。
    只输出策略内容，不要解释。
    格式: STRATEGY: <策略内容>
    """

FORCE_ACTION_PROMPT = """
    你已经消耗了所有思考轮次。现在进入【强制行动阶段】。
    
    【当前战局】:
    {latest_context}
    
    【任务】请生成有效的图谱操作 JSON。
    
    {agent_action_schema_desc}
    """

DECOMPOSE_FACTS_PROMPT = """
    你是一个数据预处理助手。请将以下【审理查明】的法律事实文本，拆解为多个独立的、原子的事实描述。

    每个事实应该包含时间、地点、人物或关键行为。

    【审理查明】：
    {text}

    请直接输出一个编号列表，每个条目是一个独立的事实描述。

    例如：
    1. 2023年5月1日，张三与李四签订合同
    2. ...

    不要输出任何其他内容或 Markdown 标记外的文字。
    """

GENERATE_ROOT_CLAIM_PROMPT = """
    基于案由【{cause}】和以下事实，请提炼出原告的所有核心法律诉求。
    诉求应具体明确（如，要求被告偿还本金XX元）。

    【事实】：
    {facts}

    请直接输出一个 JSON 字符串数组，每个元素是一个诉求的文本描述。
    例如：
    ```json
    [
        "判令被告偿还借款本金10万元",
        "判令被告支付逾期利息"
    ]
    ```
    不要输出任何其他内容或 Markdown 标记外的文字。
    """

GENERATE_PERSONA_PROMPT = """
    基于案由【{cause}】和事实，请为【{role_cn}】构建 BDI 画像和初始策略。

    【事实】：
    {facts}

    请严格按照以下 JSON 格式输出（不要输出 Markdown 标记）：
    {{
        "belief": "简述信念...",
        "desire": "简述愿望...",
        "intention": "简述风格/意图...",
        "strategy": "简述初始冷启动策略..."
    }}
"""
