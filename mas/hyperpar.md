# G-Memory (Legal Adapter) Hyperparameter Reference

该文档列举了系统中硬编码或配置的默认超参数。在进行消融实验（Ablation Study）或针对不同数据集调优时，请重点关注这些参数。

## 1. 语义匹配与去重 (Semantic Matching)

控制系统判定两个文本是否“意指同一事物”的严格程度。

| Parameter | Default Value | Location | Description | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **`SemanticMatcher.threshold`** | **0.75** | `mas/legal_system.py` (`__init__`) | 语义去重和关联投影的相似度门槛。基于 BGE-M3 余弦相似度。 | **High**. <br>太高 -> 图谱分裂，投影失效 (Recall低)。<br>太低 -> 错误合并，引入噪声 (Precision低)。 |

## 2. 记忆检索 (Memory Retrieval)

控制从历史库中召回多少案例，以及如何在 Query Graph 中游走。

| Parameter | Default Value | Location | Description | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **`retrieve_memory.top_k`** | **2** | `mas/legal_system.py` (`new_case`) | 初始检索的历史案件数量。 | **Medium**. 影响投影的丰富度。 |
| **`retrieve_memory.hop`** | **1** | `mas/legal_memory.py` (`retrieve_memory`) | 在 Query Graph 中扩展搜索的跳数。 | **Low**. 除非图非常稠密，否则 1 跳足矣。 |
| **`Chroma.n_results`** | **5** | `mas/legal_memory.py` (`add_memory`) | 新案件入库时，检索多少个邻居来构建 Query Graph 的边。 | **Medium**. 决定 Query Graph 的连通性。 |

## 3. 图谱拓扑构建 (Graph Topology)

控制 Query Graph (TaskLayer) 中连边的密度。

| Parameter | Default Value | Location | Description | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **`TaskLayer.similarity_threshold`** | **0.50** | `mas/legal_memory.py` (`__post_init__`) | 只有相似度高于此值的案例之间才会建立边。 | **High**. <br>太高 -> 图充满孤岛，Hop 检索失效。<br>太低 -> 形成全连接图，丧失聚类意义。 |

## 4. 投影逻辑 (Projection Logic)

控制从历史案件中引入多少知识到新案件。

| Parameter | Default Value | Location | Description | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Projection Scope** | **1-Hop** | `mas/projection.py` (`_project_single_graph`) | 代码硬编码。只投影匹配节点的直接邻居（包括上游和下游）。 | **High**. 决定投影的深度。 |
| **Node Type Filter** | **LAW, CLAIM** | `mas/projection.py` (`_project_single_graph`) | 代码硬编码。投影时会过滤掉历史案件中的 `FACT` 节点，避免污染当前案情。 | **Critical**. 防止引入错误事实。 |

## 5. 策略检索 (Insight Retrieval)

控制 Upward Traversal 的行为。

| Parameter | Default Value | Location | Description | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **`get_relevant_insights.top_k`** | **3** | `mas/legal_system.py` (`new_case`) | 返回给 Agent 的高层策略数量。 | **Low**. Agent 上下文窗口限制。 |
| **Score Weight** | **0.05** | `mas/insights_manager.py` (`get_relevant_insights`) | 混合检索时，策略分数（出现次数）的权重。<br>Formula: `Sim * (1 + 0.05 * Score)` | **Low**. 调节“热门策略”与“精准策略”的平衡。 |

## 6. LLM Generation

控制 LLM 生成的随机性和长度。

| Parameter | Default Value | Location | Description | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **`temperature`** | **0.1** | `mas/llm.py` | 生成温度。 | **Medium**. 低温保证指令格式稳定 (Regex 能解析)。 |
| **`max_tokens`** | **1024** | `mas/llm.py` | 最大生成长度。 | **Low**. |