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

    请严格按照以下 JSON 格式输出：

    ```json
    {{
        "need": true,
        "reasoning": "我需要证明被告违约，但图谱中只有借条，缺少逾期未还的直接证据...",
        "query": "被告 逾期行为 证据"
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

    请严格按照以下 JSON 格式输出：

    ```json
    {{
        "need": true,
        "reasoning": "我想主张合同无效，但图谱中缺少关于合同无效的具体法律条款节点...",
        "query": "合同法 无效情形"
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

ANALYZE_SEARCH_RESULTS_PROMPT = """
    你是一个法律辩论团队的【{role_type}参谋】。
    
    【当前战局 (Graph Context)】:
    {graph_context}
    
    【指挥官的查询意图】:
    "{user_query}"
    
    【检索到的情报】:
    {search_result}
    
    你的任务：结合情报，为指挥官提供具体的图谱操作建议。
    你的建议应该清晰地指出要“添加”什么（事实/主张/法条），或者要“支持”/“反驳”哪个已有节点。
    如果涉及关系，请明确指出源节点和目标节点（使用其ID）。

    【建议示例】:
    - 基于检索结果，添加事实：“王某于2023年3月15日向李某借款人民币10万元。”
    - 基于检索结果，引用法条：“中华人民共和国合同法 第二百零六条”，并用其支持主张 CLAIM_abcde123。
    - 基于检索结果，添加主张：“被告无证据证明其已履行还款义务”，并用其反驳 CLAIM_fghij456。

    请生成建议:
    """

SUGGEST_PIVOT_PROMPT = """
    你是一个法律参谋。指挥官想查询 "{user_query}"，但在数据库中**未找到**强相关的内容。
    
    【当前战局】:
    {graph_context}
    
    请分析可能的失败原因，并提供一个**新的查询方向**或**更有效的关键词**。
    
    【输出示例】:
    "关于'口头变更'的直接先例较少。建议调整方向，尝试搜索 '合同实际履行' 或 '默认行为' 相关的判例。"
    
    请生成转型建议:
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
