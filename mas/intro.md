# G-Memory (法律适配版) 架构与实现指南

本文档阐述了 **G-Memory (法律适配版)** 的系统架构，旨在将 **对抗性螺旋协议 (Adversarial Spiral Protocol, ASP)** 方法论中的理论概念，映射到 `mas/` 包中的具体 Python 实现。

---

## 1. 核心数据结构: 影图 ($\mathcal{G}_t$, Shadow Graph)

**方法论**:
**影图**是法律论辩过程的结构化、语义化表示。与原始对话日志不同，它追踪以下要素间的逻辑依赖关系：
*   **事实 ($F$)**: 证据或事件。
*   **法条 ($L$)**: 法律法规。
*   **观点 ($C$)**: 基于事实或法条的论点。
*   **边 ($E$)**: 支持 ($\rightarrow$) 或 冲突 ($\dashv$) 关系。

每个节点都拥有一个状态: $S \in \{\text{假设的}, \text{已验证的}, \text{已驳斥的}\}$。

**代码实现** (`mas/common.py`):
```python
@dataclass
class ShadowGraph:
    """
    维护法律论证的有向图。
    支持语义去重和序列化。
    """
    graph: nx.DiGraph
    
    def add_node(self, content, node_type, agent_id, matcher=None):
        # 通过语义检查添加节点 (对应公式3: ShouldMerge)
        ...
```

---

## 2. 记忆架构 (G-Memory 三层结构)

本系统为法律推理场景适配了 G-Memory 的三层记忆架构。

### A. 交互图 (Interaction Graph - 案例历史)
**方法论**: 具体的历史案例，用于 **关联投影 (Associative Projection)**。
**代码实现** (`mas/legal_memory.py`):
*   以 JSON 字符串的形式，作为元数据存储在 **ChromaDB** 中。
*   通过 `retrieve_memory` 接口检索，并反序列化为 `ShadowGraph` 对象。

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
*   **逻辑**: `extract_adversarial_insights` 方法通过对比胜诉与败诉的子图来提炼新策略。

---

## 3. 工作流与算法

### 算法 1: 执行 ($\text{Execute}(a_t, \mathcal{G}_t)$)
**方法论**: 智能体输出 `ADD/LINK` 等动作来修改图谱。
**代码实现** (`mas/graph_ops.py`):
*   **组件**: `GraphExecutor`
*   **逻辑**: 基于正则表达式解析 LLM 的输出，将其转化为原子的图操作指令。
    *   `ADD_FACT("...")` $\rightarrow$ `graph.add_node(...)`
    *   `LINK(A, B, SUPPORT)` $\rightarrow$ `graph.add_edge(...)`

### 算法 3: 关联投影 ($\mathcal{P}(\mathcal{G}_0, \mathcal{M})$)
**方法论**: 通过从历史库中投影相关的子图来初始化当前案件的图谱。
**代码实现** (`mas/projection.py`):
*   **组件**: `GraphProjector`
*   **逻辑**:
    1.  使用 `SemanticMatcher` 匹配当前节点与历史节点。
    2.  将被匹配节点的 **1-Hop 邻居子图** (包括上游和下游) 复制到当前图中。
    3.  **过滤器**: 投影时会忽略历史案件中的 `FACT` 节点，以避免污染当前案情。

### 算法 4: 判决反向传播
**方法论**: 根据法官的判决结果，更新图中节点的状态。
**代码实现** (`mas/backprop.py`):
*   **组件**: `BackPropagator`
*   **逻辑**:
    1.  将胜诉方提出的节点标记为 `VALIDATED`。
    2.  将被反驳的败诉方节点标记为 `DEFEATED`。

---

## 4. 系统接口 (The Facade)

**代码实现** (`mas/legal_system.py`):
`LegalSystem` 类封装并调度了所有组件，作为 MetaGPT 智能体的“大脑”。

| 方法论步骤 | API 接口 | 描述 |
| :--- | :--- | :--- |
| **初始化** | `new_case(context)` | 执行 **向下** (投影) 和 **向上** (策略检索) 的双向遍历。 |
| **行动** | `execute_action(graph, agent, text)` | 解析智能体的文本输出，并更新图谱。 |
| **判决** | `adjudicate(context, graph)` | 调用法官 (LLM/模型) 获取判决。 |
| **学习** | `learn(context, graph, winner)` | 执行**反向传播**，存储案例，并提取新策略。 |

---

## 5. 使用示例

```python
from mas.legal_system import LegalSystem

# 1. 初始化系统
sys = LegalSystem("./storage")

# 2. 开启新案件 (触发投影和策略检索)
context = "被告人偷了钱包，但辩称是借用。"
graph, strategies = sys.new_case(context)

# 3. 智能体行动 (原告)
sys.execute_action(graph, "plaintiff", 'ADD_CLAIM("具有非法占有目的")')

# 4. 智能体行动 (被告)
sys.execute_action(graph, "defendant", 'CHALLENGE(CLAIM_1, "借用不等于盗窃")')

# 5. 判决与学习
settled, winner = sys.adjudicate(context, graph)
if settled:
    sys.learn(context, graph, winner, "case_001")
```

---

## 6. 局限性与方法论对齐

#### **A. 饱和度 vs 语义判决 ($\Delta \Phi$ 的近似)**
*   **理论**: 论文提出基于**信息增益饱和度 ($\Delta \Phi \approx 0$)** 来终止对抗螺旋，这是一个衡量新信息不再产生的结构化指标。
*   **现状**: 系统目前依赖**法官的语义判决** (`LLMJudge.evaluate`) 来决定终止。
*   **偏差说明**: 理论上的“信息饱和”被 LLM 对案件清晰度的主观评估（“证据是否充分？”）所近似替代。`Judge` 模块中尚未实现对图熵或节点饱和率的显式计算。

#### **B. 缺失的驱动器 (算法 2)**
*   **理论**: **算法 2 (ASP)** 定义了回合制的编排流程: `原告` $\rightarrow$ `被告` $\rightarrow$ `检查饱和度`。
*   **现状**: `mas` 包作为**记忆与推理引擎**，提供了所有原子能力 (`execute`, `project`, `learn`)，但**不包含**辩论循环本身。
*   **集成计划**: **ASP 循环 (算法 2)** 将被实现为一个 **MetaGPT 标准操作流程 (SOP)**。届时，MetaGPT 的角色 (`Plaintiff`, `Defendant`) 将轮流调用 `LegalSystem` 的 API。`mas` 包是*库*，MetaGPT 是*应用程序*。