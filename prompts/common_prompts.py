"""A centralized repository for all LLM prompt templates used in the system.

This module contains string constants that serve as f-string templates for
interacting with the Large Language Models (LLMs). Centralizing prompts here
makes them easier to manage, version, and refine.

The prompts are categorized by their function:
-   **Controller Prompts**: Used by the ArgumentController for strategic
    assessment and decision-making (e.g., ASSESS_FACT_NEEDS_PROMPT,
    VERIFY_AND_DECIDE_PROMPT).
-   **Worker Prompts**: Used by worker agents for tactical tasks like
    analyzing search results or decomposing search intents (e.g.,
    ANALYZE_FACT_PROMPT, DECOMPOSE_SEARCH_INTENT_PROMPT).
-   **Initialization Prompts**: Used by the CaseInitializer to preprocess
    case data (e.g., DECOMPOSE_FACTS_PROMPT, GENERATE_ROOT_CLAIM_PROMPT).
-   **Judge Prompts**: Used by the Judge agent to evaluate the debate and
    extract a final verdict (e.g., JUDGE_EVALUATE_PROMPT).
-   **Learning and Narration Prompts**: Used for post-case analysis and
    generating human-readable transcripts (e.g.,
    EXTRACT_ADVERSARIAL_INSIGHTS_PROMPT, NARRATOR_POLISH_PROMPT).

A key component, `GRAPH_READING_GUIDE`, is a reusable block of text
injected into several prompts to provide the LLM with consistent instructions
on how to interpret the adversarial debate graph.
"""

GRAPH_READING_GUIDE = """
    【图谱阅读指南】:
    1. **节点标签含义**:
       - `[核心诉求]`: 案件初始的诉讼请求。
       - `[原告观点]`: 原告方提出的论点。
       - `[被告观点]`: 被告方提出的论点。
    2. **你的立场 ({role_name}) 行动准则**:
       - **如果你是 plaintiff (原告)**:
         - 对影响裁判结果的 `[被告观点]` 进行有依据的 **CONFLICT (反驳)**。
         - 持续用客观事实 (`FACT`) 与法条 (`LAW`) 补强 `[原告观点]` 的说服力。
       - **如果你是 defendant (被告)**:
         - 对影响裁判结果的 `[核心诉求]` 与 `[原告观点]` 进行有依据的 **CONFLICT (反驳)**。
         - 提出可验证的独立抗辩观点，并补充证据支撑。
    3. **行动质量要求**:
       - 每次反驳都要明确：反驳哪个关键争点、依据是什么、为什么该动作有效。
       - 禁止无依据、重复、低信息量攻击；禁止情绪化或人身化表达。
       - 对抗的目标是提升裁判说服力，而不是制造无效冲突。
"""

ASSESS_FACT_NEEDS_PROMPT = (
    """
    你是【{role_name}】的辩护律师。

    【你的核心 BDI】:
    - 信念 (Belief): {belief}
    - 意图 (Intention): {intention}
    - 战略 (Strategy): {strategy}

    【当前战局 (Graph Context)】:
    {graph_context}
    
    """
    + GRAPH_READING_GUIDE
    + """

    【任务】:
    请审视图谱中的【事实节点】。
    基于你的**立场**，你是否觉得缺少某些关键证据来支持己方观点或反驳对方？
    
    【重要】：如果需要检索，请描述你的**高层意图**。例如，作为被告，你可能需要“查找原告是否存在过错的事实细节”。
    检索目标应服务于关键争点澄清与证据补强，不要为了攻击而检索。

    【输出方式】:
    你必须调用函数 `assess_fact_needs` 并在参数中填写：
    - need: 布尔值
    - reasoning: 字符串（不超过 300 字）
    - intent: 字符串或 null
    禁止输出普通文本或 Markdown 代码块。
"""
)

ASSESS_LAW_NEEDS_PROMPT = (
    """
    你是【{role_name}】的辩护律师。

    【你的核心 BDI】:
    - 信念 (Belief): {belief}
    - 意图 (Intention): {intention}
    - 战略 (Strategy): {strategy}

    【当前战局 (Graph Context)】:
    {graph_context}

    """
    + GRAPH_READING_GUIDE
    + """

    【任务】:
    请审视图谱中的【LAW (法条) 节点】。
    基于你的**立场**，你是否拥有支持己方主张的法条？或者是否需要查找法条来解释对方引用的法条并不适用？

    【重要】：请描述你的**法律调研意图**。
    法律检索应服务于关键争点与法律适用分析，不要为了攻击而检索。

    【输出方式】:
    你必须调用函数 `assess_law_needs` 并在参数中填写：
    - need: 布尔值
    - reasoning: 字符串（不超过 300 字）
    - intent: 字符串或 null
    禁止输出普通文本或 Markdown 代码块。
"""
)

ASSESS_RECALL_NEEDS_PROMPT = (
    """
    你是【{role_name}】的辩护律师。

    【你的核心 BDI】:
    - 信念 (Belief): {belief}
    - 意图 (Intention): {intention}
    - 战略 (Strategy): {strategy}

    【当前战局 (Graph Context)】:
    {graph_context}

    """
    + GRAPH_READING_GUIDE
    + """

    【任务】
    现在，请判断是否需要**借鉴历史相似案例的策略**？
    
    【重要】：请描述你希望从历史案例中获得的**策略意图**。作为{role_name}，你可能想看历史上类似案件中，胜诉方是如何构建逻辑链的，或者是败诉方犯了什么错误。
    历史借鉴应服务于关键争点策略优化，不要为了攻击而借鉴。

    【输出方式】:
    你必须调用函数 `assess_recall_needs` 并在参数中填写：
    - need: 布尔值
    - reasoning: 字符串（不超过 300 字）
    - intent: 字符串或 null
    禁止输出普通文本或 Markdown 代码块。
"""
)

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
    你是一个法律团队的【战略参谋】。你目前服务于 **{my_role}** 方。
    我们已从历史案例库中“投影”了相关论点到当前图谱。

    【律师的原始意图】: "{user_query}"
    
    【历史情报（参考资料）】: 
    {projection_context}

    【任务】:
    请分析历史经验对 **{my_role}** 有何启示？
    
    【核心思维链】:
    1. 观察历史案例中的 `[原告观点]` 和 `[被告观点]`。
    2. 关注历史判决结果（如果有标记【已采信】/【已驳回】）。
    3. **如果历史上的 {my_role} 输了**：警告我方不要重蹈覆辙，指出那个导致失败的关键弱点。
    4. **如果历史上的 {my_role} 赢了**：提炼那个导致胜利的关键策略。
    
    【⚠️ 严重警告】：
    1. **区分事实**：历史情报中的 [FACT] 节点是**过去案件**的事实，**绝对不是**本案事实。
    2. **对比差异**：请仔细对比本案意图与历史事实的差异。

    【输出要求】:
    1. **不要复述历史情报！** 直接给出针对 {my_role} 的结论。
    2. 必须包含“**建议：**”字段，给出具体行动建议。
    3. 字数控制在 300 字以内。
"""

VERIFY_AND_DECIDE_PROMPT = (
    """
    你是【{role_name}】。你的参谋已完成任务并提交了报告。

    【参谋报告/行动结果】:
    "{worker_advice}"

    【当前战局 (Graph Context)】:
    {graph_context}
    
    """
    + GRAPH_READING_GUIDE
    + """
    
    【🛡️ ID 防幻觉安全校验】:
    **你只能引用以下列表中真实存在的 ID**。严禁捏造 "FACT_1", "CLAIM_new" 等不存在的 ID！
    若你想提出新观点，`source_id` 或 `target_id` 必须设为 `null` (取决于具体动作类型)。
    
    **有效 ID 清单**:
    {id_inventory}

    {feedback_section}

    任务：请结合【参谋报告】和【当前战局】，生成具体的图谱操作函数参数，以推进【战略重心】({focus})。
    
    【对抗与质量要求】:
    - 对核心冲突点必须正面回应，保持有效对抗。
    - 每个动作都必须有清晰依据，禁止无依据、重复、低信息量攻击。
    - 输出要简洁，避免冗长展开，防止 token 膨胀。

    【简要推理链（CoT）要求】:
    - 每个 action 的 `metadata` 必须包含 `reason_brief`。
    - `reason_brief` 使用简要格式：`争点:...;依据:...;动作理由:...`。
    - 只写简要理由，不要输出冗长思维过程。

    {action_schema_desc}

    你必须调用函数 `verify_and_decide`，并将动作数组放在参数字段 `actions` 中。
    禁止输出普通文本或 Markdown 代码块。
"""
)

JUDGE_EVALUATE_PROMPT = """
    你是一名公正、专业的法官。请根据以下【核心诉求】和【图谱证据上下文】，撰写一份民事判决书。

    【核心诉求】:
    {issue_list}

    【图谱证据上下文（主要裁判依据）】:
    {graph_context}

    【裁判要求】:
    1. 必须逐项回应核心诉求，并给出采纳或驳回结论。
    2. 每项结论都要引用图谱中的关键证据链条（支持与冲突关系）。
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
    
    请输出一个 JSON 对象，字段如下：
    {{
        "status": "ACCEPTED" | "REJECTED" | "UNMENTIONED"
    }}
    """

EXTRACT_ADVERSARIAL_INSIGHTS_PROMPT = """
    你是一个法律复盘专家。请根据以下的【庭审笔录】和【判决结果】，提炼出导致胜诉（或败诉方致死原因）的关键辩论策略。

    【判决结果 (诉求采信状态)】:
    {claims_status}

    【庭审笔录】:
    {transcript}

    【任务】:
    1. 判断哪一方（原告/被告）在辩论中占据了上风或取得了关键胜利。
    2. 提炼其核心策略（如引用了特定司法解释、切断了因果关系链等）。

    【输出格式要求 (严格遵守)】:
    SIDE: [PLAINTIFF | DEFENDANT]
    CONTENT: [一句话的高度抽象策略]

    示例：
    SIDE: DEFENDANT
    CONTENT: 针对民间借贷纠纷，若无直接转账凭证，应重点攻击对方证据链的完整性。
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
    不要输出任何其他内容。
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

NARRATOR_POLISH_PROMPT = """
    你是一个法律文书润色专家。
    以下是本轮辩论中【{turn}】方的**原始逻辑操作记录**。这些句子是程序根据图谱操作自动生成的，虽然逻辑准确但略显生硬。

    请将其润色为一段**流畅、自然、有力**的辩论陈述。

    【原始逻辑记录】:
    {raw_sentences}

    【润色要求】:
    1. **角色带入**：请以客观叙述视角（如“原告指出...”、“被告反驳称...”）或第一人称视角（“我方认为...”）进行重写，保持统一。
    2. **逻辑衔接**：使用“首先”、“其次”、“鉴于”、“因此”、“对此”等连接词，将离散的句子串联成严密的逻辑流。
    3. **绝对忠实**：**严禁**编造原始记录中不存在的证据或观点，**严禁**修改引用的原文内容。
    4. **输出格式**：直接输出润色后的段落，不要包含任何前言后语。
"""

DECOMPOSE_SEARCH_INTENT_PROMPT = """
    你是一个专业的法律检索专家。
    【任务】：将律师的高层意图（Intent）拆解为 2-3 个具体的自然语言查询句，用于在案例库中进行语义检索。

    【律师意图】："{intent}"

    【要求】：
    1. **必须是完整的自然语言问句或陈述句**，严禁使用关键词堆砌（如 "合同 违约 赔偿" ❌）。
    2. **多角度拆解**：
       - 角度1：侧重核心事实相似性（如：“类似[具体事实]的判决案例”）
       - 角度2：侧重法律争议焦点（如：“关于[争议点]的法律适用情形”）
    3. 你必须调用函数 `formulate_search_queries`，并将查询数组填入参数字段 `queries`。
    4. 禁止输出普通文本或 Markdown 代码块。
"""

DECOMPOSE_LAW_INTENT_PROMPT = """
    你是一个专业的法律检索专家。
    【任务】：将律师的高层意图（Intent）拆解为 2-3 个具体的自然语言查询句，用于在**法条数据库**（Statutes Library）中检索。

    【律师意图】："{intent}"

    【要求】：
    1. **查询句应侧重于法律规定、构成要件、司法解释**。
    2. 避免使用“......的案例”这种表述，而是用“......的法律规定”或“......的认定标准”。
    3. 多角度拆解：
       - 角度1：核心法律概念定义（如：“不可抗力的法律定义”）
       - 角度2：具体适用情形与后果（如：“因天气原因导致违约的免责条款规定”）

    【输出方式】：
    你必须调用函数 `formulate_search_queries`，并将查询数组填入参数字段 `queries`。
    禁止输出普通文本或 Markdown 代码块。
"""
