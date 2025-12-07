# G-Memory (法律适配版) 架构与实现指南

本文档阐述了 **G-Memory (法律适配版)** 的系统架构，旨在将 **对抗性螺旋协议 (Adversarial Spiral Protocol, ASP)** 方法论中的理论概念，映射到 `mas/` 包中的具体 Python 实现。

---

## 1. 核心数据结构: 影图 ($\mathcal{G}_t$, Shadow Graph)

**方法论**:
**影图**是法律论辩过程的结构化、语义化表示。它追踪以下要素间的逻辑依赖关系：
*   **事实 ($F$)**: 证据或事件。
*   **法条 ($L$)**: 法律法规。
*   **观点 ($C$)**: 基于事实或法条的论点。
*   **边 ($E$)**: 支持 ($\rightarrow$) 或 冲突 ($\dashv$) 关系。

**代码实现** (`mas/common.py`):
```python
@dataclass
class ShadowGraph:
    """
    维护法律论证的有向图。
    支持语义去重、序列化以及会话级 ID 别名 (id_alias) 管理。
    """
    graph: nx.DiGraph
    id_alias: Dict[str, str] # 维护 Agent 临时 ID (如 FACT_1) 到真实 Graph ID 的映射
```

---

## 2. 记忆架构 (G-Memory 三层结构)

本系统为法律推理场景适配了 G-Memory 的三层记忆架构。

### A. 交互图 (Interaction Graph - 案例历史)
**方法论**: 具体的历史案例，用于 **关联投影 (Associative Projection)**。
**代码实现** (`mas/legal_memory.py`):
*   **物理存储**: 以 JSON 字符串的形式，作为元数据存储在 **ChromaDB** 中。
*   **检索**: 通过 `retrieve_memory` 接口检索，并反序列化为 `ShadowGraph` 对象。

### B. 查询图 (Query Graph - 案例拓扑)
**方法论**: 一个导航层，用于发现语义相似的案件 (K-Hop 检索)。
**代码实现** (`mas/task_layer.py`):
*   **内存中**: 使用 `networkx` 存储案件之间的连通性（拓扑）。
*   **磁盘上**: 持久化为 `case_graph.pkl` 文件。
*   **逻辑**: 基于 ChromaDB 的 ANN (近似最近邻) 检索结果动态更新拓扑。

### C. 洞察图 (Insight Graph - 法律策略)
**方法论**: 从历史案件的胜负模式中抽象出的法律策略。
**代码实现** (`mas/insights_manager.py`):
*   **内存中**: 维护一个策略的向量索引。
*   **磁盘上**: 持久化为 `legal_insights.json` 文件。
*   **逻辑**:
    *   **提取**: `extract_adversarial_insights` 对比胜诉与败诉子图。
    *   **奖惩**: `update_scores_from_verdict` 根据实战结果调整策略分数。
    *   **反查**: `find_cases_by_insight` 根据策略召回历史案例 (Corrective Retrieval)。

---

## 3. 工作流与算法

### 算法 1: 执行 ($\text{Execute}(a_t, \mathcal{G}_t)$)
**方法论**: 智能体输出 `ADD/LINK` 等动作来修改图谱。
**代码实现** (`mas/graph_ops.py`):
*   **组件**: `GraphExecutor`
*   **逻辑**:
    1.  **解析**: 基于增强正则表达式解析 LLM 的输出。
    2.  **别名管理**: 自动维护 `FACT_1 -> Node_123` 的映射，确保持久化会话一致性。
    3.  **动态投影**: 当检测到 `ADD` 指令时，触发即时投影。

### 算法 3: 关联投影 ($\mathcal{P}(\mathcal{G}_0, \mathcal{M})$)
**方法论**: 通过从历史库中投影相关的子图来丰富当前案件。
**代码实现** (`mas/projection.py`):
*   **组件**: `GraphProjector`
*   **策略**: **三阶段子图同构复制 (Three-Stage Subgraph Copy)**。
    1.  **Gather**: 收集锚点及其 K-Hop 邻居 (过滤掉历史 Fact)。
    2.  **Copy Nodes**: 复制节点到新图，建立新旧 ID 映射。
    3.  **Copy Edges**: 复制子图内部的所有连接，保持拓扑完整。

### 算法 4: 判决反向传播
**方法论**: 根据法官的判决结果，更新图中节点的状态。
**代码实现** (`mas/backprop.py`):
*   **组件**: `BackPropagator`
*   **逻辑**: 标记 Winner 节点为 `VALIDATED`，Challenged Loser 节点为 `DEFEATED`。

---

## 4. 系统接口 (The Facade)

**代码实现** (`mas/legal_system.py`):
`LegalSystem` 类封装了所有组件，并实现了 **三阶段检索 (Three-Stage Retrieval)**。

| 阶段 | 方法 | 描述 |
| :--- | :--- | :--- |
| **Stage 1: Downward** | `retrieve_memory` | 基于 Context 相似度召回形似案例。 |
| **Stage 2: Upward** | `get_relevant_insights` | 检索高层策略指导。 |
| **Stage 3: Corrective** | `find_cases_by_insight` | 基于 Insight 反查，召回神似案例 (Recall Correction)。 |

---

## 5. 配置管理

所有超参数（阈值、路径、Limit）均通过 `mas/config.py` 中的 `SystemConfig` 统一管理，支持梯度化阈值设计：
*   **Query Graph**: 0.60 (宽松召回)
*   **Projection**: 0.68 (中等过滤)
*   **Insight Merge**: 0.80 (严格聚类)
*   **Deduplication**: 0.95+ (极严格去重)

---

## 6. 局限性与方法论对齐

#### **A. 饱和度 vs 语义判决 ($\Delta \Phi$ 的近似)**
*   **理论**: 论文提出基于**信息增益饱和度 ($\Delta \Phi \approx 0$)** 来终止对抗螺旋。
*   **现状**: 系统目前依赖**法官的语义判决** (`LLMJudge.evaluate`) 来决定终止。
*   **偏差说明**: 理论上的“信息饱和”被 LLM 的主观评估所替代。

#### **B. 缺失的驱动器 (算法 2)**
*   **现状**: `mas` 包是**Engine**。
*   **集成计划**: **ASP 循环 (算法 2)** 将被实现为一个 **MetaGPT SOP**。`mas` 包作为 Library 被 MetaGPT 调用。