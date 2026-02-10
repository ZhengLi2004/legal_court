# TODO：基于 React 的系统演示前端方案（结合仓库代码与 `main [5-13].pdf`）

## 0. 文档目标
本方案用于指导你在当前仓库上搭建一个“可演示、可解释、可扩展”的前端演示系统。

核心要求：
1. 忠实反映论文第 3 章方法论（论证图环境、分层智能体架构、多阶段生命周期）。
2. 忠实反映当前代码真实能力边界（不是“想象中的系统”）。
3. 强调系统行为可追溯、可验证、可解释。
4. 不陷入过度工程细节（先做演示闭环，再做平台化）。

---

## 1. 先对齐“代码现实”与“论文目标”

### 1.1 论文第 3 章要演示的主干（来自 `main [5-13].pdf`）
1. 论证图环境：
- 图结构：`FACT` / `LAW` / `CLAIM` 节点，`SUPPORT` / `CONFLICT` 边。
- 逻辑公理：支持目标必须是 `CLAIM`；冲突双方必须是 `CLAIM`；不允许有向环。
- BAF 语义：直接攻击、支持型攻击、间接攻击；可采纳集与首选扩展。

2. 分层智能体架构：
- 指挥者（Controller）：评估 -> 委派 -> 决策。
- 工作者（Fact/Law/Recall Worker）：并行检索与报告。
- 法官（Judge）：最终裁判，不参与博弈主循环。

3. 多阶段生命周期：
- 结构化初始化。
- 对抗图博弈。
- 自适应终止与裁决（收敛分数/滑窗）。
- 演化学习（长期记忆与策略更新）。

### 1.2 当前仓库已实现能力（必须前端可见）
1. 引擎与阶段推进：`mas/core/engine.py`
- `setup()` 初始化图和团队。
- `step()` 单回合推进。
- `adjudicate()` 裁判 + BAF 计算。
- `get_snapshot()` 与 `round_snapshots` 用于回放。
- `set_state_callback()` 可向前端推流事件。

2. 图谱与逻辑约束：
- 图模型：`mas/core/graph.py`
- 动作验证与事务执行：`mas/analysis/executor.py`
  - 静态校验失败会拒绝。
  - 执行异常会回滚。
  - 会阻止有向环（无环公理）。

3. 分层智能体协作：
- Team 编排：`mas/agents/team.py`
- Controller 状态机：`roles/controller.py`
- 三类 Worker：`roles/worker.py`

4. 判决与 BAF：
- Judge：`mas/agents/judge.py`
- BAF 语义：`mas/analysis/baf.py`

5. 长期记忆与演化：
- 记忆库（Chroma + 法条倒排）：`mas/memory/legal_memory.py`
- 洞察策略：`mas/memory/insights.py`
- 拓扑层：`mas/memory/topology.py`
- 投影检索：`mas/memory/projection.py`

### 1.3 需要明确给前端/演示者的现实约束
1. 当前仓库没有现成 Web API，只有 CLI（`run.py`）。
2. 引擎 `step()` 只负责推进回合，不自动触发裁判；当 `is_ready_for_adjudication=true` 时，需要显式调用 `adjudicate()`。
3. 因依赖 ES / LLM / embedding，本地演示可能出现检索空结果、延迟、超时，需要前端可观测错误与降级提示。

---

## 2. 演示叙事主线（用户看什么）

面向演示的主线不是“代码结构”，而是“法庭流程”：
1. 输入案件 -> 系统结构化初始化。
2. 双方回合辩论 -> 图谱不断演化。
3. 收敛触发终止 -> 法官裁判 + BAF 对齐。
4. 结果沉淀进长期记忆 -> 下一案可借鉴。

前端所有页面都要围绕这个主线，避免碎片化页面堆叠。

---

## 3. React 前端信息架构（页面设计）

建议路由（6 页，覆盖完整闭环）：
1. `/` 总览与启动
2. `/session/:id/live` 实时辩论台（核心页）
3. `/session/:id/graph` 图谱与逻辑公理
4. `/session/:id/judgment` 裁判与 BAF 分析
5. `/session/:id/memory` 演化学习与长期记忆
6. `/session/:id/replay` 快照回放与对比

### 3.1 首页 `/`（总览与启动）
页面目标：3 分钟内让观众明白“系统是什么、怎么跑”。

模块设计：
1. 方法论概览卡：对应论文三大模块 + 四阶段生命周期。
2. 案件启动卡：
- 选内置样例（来自 `data/sampling/cleaned_samples.jsonl`）。
- 或粘贴自定义 `fact_finding` + `cause`。
3. 配置摘要卡：显示关键配置（LLM、ES、收敛参数、BAF 开关）。
4. 会话入口卡：创建会话后跳转到 `live` 页面。

前端交互：
- `创建会话` -> `setup` -> 显示初始图统计（事实节点数、根诉求数）。

### 3.2 实时辩论台 `/session/:id/live`（核心页）
页面目标：完整看见“评估-委派-决策-执行-叙事-收敛”。

布局建议（单页 5 区）：
1. 顶部控制条：
- `Setup`、`Step`、`Auto Run`、`Pause`、`Adjudicate`。
- 当前回合、当前轮次、当前行动方、是否 ready for adjudication。

2. 左侧“团队状态列”：
- 原告/被告 Controller 当前 pipeline step。
- Worker 派发状态（Fact/Law/Recall 是否被调用）。
- 最近一次 retry 原因（如果有）。

3. 中央“图谱主画布”：
- 实时渲染节点/边。
- 节点颜色：按 `NodeStatus`（HYPOTHETICAL/VALIDATED/DEFEATED）。
- 节点形状：按 `NodeType`（FACT/LAW/CLAIM）。
- 边样式：`SUPPORT` / `CONFLICT`。
- 每步增量高亮（新增/复用/拒绝）。

4. 右侧“指标面板”：
- `delta_phi`、`SMA`、冲突边数、claim 节点数。
- 收敛曲线图（每一步更新）。
- 快照计数。

5. 底部“叙事与操作日志”：
- Narrator 文本（turn narrative）。
- 执行日志（GraphExecutor 返回日志）。
- 系统事件流（setup_start、turn_complete 等）。

### 3.3 图谱与逻辑公理页 `/session/:id/graph`
页面目标：把“为何这个动作被接受/拒绝”讲清楚。

模块设计：
1. 图谱浏览器（与 live 共用组件）。
2. 逻辑公理检查面板：
- 支持公理（目标必须 CLAIM）。
- 冲突公理（源/目标必须 CLAIM）。
- 无环公理（产生环即阻止）。
3. 动作审计表：
- 每个 `AgentAction` 的校验结果。
- 若失败，展示具体错误信息与动作原文。
4. 递归序列化视图（`latest_context`）：
- 显示系统给 LLM 的上下文文本，解释“模型看到什么”。

### 3.4 裁判与 BAF 页 `/session/:id/judgment`
页面目标：体现“LLM 判决 + 形式语义校验”的双轨机制。

模块设计：
1. 判决书面板：展示 `judgment_document`。
2. 根诉求状态面板：每个 root claim 的状态。
3. BAF 面板：
- preferred extensions 数量。
- 选中扩展大小。
- 与 LLM 判决对齐率（alignment_rate）。
- matching_details（简化展示）。
4. 攻击关系面板：
- 直接/支持型/间接攻击统计。
- 关键攻击链可视化。

### 3.5 演化学习与长期记忆页 `/session/:id/memory`
页面目标：证明系统不是一次性辩论，而是可学习系统。

模块设计：
1. 三层记忆视图（对应论文）：
- 抽象层：insights。
- 拓扑层：case relation graph（TaskLayer）。
- 实例层：历史 case graph 快照。
2. 检索路径展示：
- 语义路径（case context 相似检索）。
- 法理路径（law 倒排 + Jaccard）。
- 策略路径（insight representative cases）。
3. 当前会话记忆写回结果：
- 新增 insight 内容。
- 关联的案例簇变化。

### 3.6 快照回放页 `/session/:id/replay`
页面目标：演示“可追溯性”。

模块设计：
1. 回合滑条：从 setup 到 final。
2. 双时刻对比：任意两个快照 diff（节点/边/状态变化）。
3. 回放联动：图谱 + transcript + convergence 同步回放。

### 3.7 Debug / 演示体验专项设计（本轮重点补充）
页面目标：让“看得懂系统为何这样运行”优先于“看起来炫”。

#### 3.7.1 图谱 Diff 高亮策略（逐步定位变化）
1. 节点变化高亮：
- 新增节点：绿色外环 + 600ms 呼吸动画。
- 复用节点（语义去重命中）：蓝色虚线外环 + “Reused”角标。
- 状态变化节点（如 HYPOTHETICAL -> VALIDATED）：状态颜色闪烁一次并记录变更历史。
- 在双快照模式中，不存在于目标快照的节点显示为半透明红色（仅回放页显示，不影响 live 主图）。

2. 边变化高亮：
- 新增 `SUPPORT` 边：蓝绿色高亮。
- 新增 `CONFLICT` 边：橙红色高亮并增强箭头。
- 若因规则被拒绝未加边，在“候选边层”显示灰色虚线 + reject 原因 tooltip。

3. Diff 摘要条（固定在图右上）：
- `+N Nodes` / `+M Edges` / `Status Changes` / `Rejected Ops`。
- 点击任意指标可自动过滤并聚焦到对应变化集合。

4. 局部聚焦模式（Debug 友好）：
- `只看变化节点`：隐藏未变更节点。
- `变化邻域 1-hop/2-hop`：展示变化节点的局部因果上下文。
- `锁定节点追踪`：跨回合跟踪某节点的状态、入边、出边变化。

#### 3.7.2 团队内部流程可视化（Controller-Worker 内部机制）
1. 双泳道流程图（原告 / 被告）：
- 每方展示 Controller 状态机：`ASSESS_NEEDS -> WAIT_FOR_WORKERS -> DECIDE -> DONE`。
- 当前状态高亮，已完成状态打勾，失败重试状态红框。

2. Worker 任务卡片墙：
- 每次派发生成卡片：`FactWorker` / `LawWorker` / `RecallWorker`。
- 卡片字段：`intent`、状态（RUNNING/FOUND/NOT_FOUND/ERROR）、耗时、max_score（若有）。
- 任务卡与对应回合绑定，便于按回合回看。

3. 指挥者决策链路面板：
- “评估结果 JSON” -> “batch_instructions” -> “worker 汇总摘要” -> “最终 AgentAction JSON” 四段串联展示。
- 每段可折叠，默认只显示摘要，点击展开原文（避免信息过载）。

4. Retry 可解释性：
- 单独的 `Retry Timeline` 展示：失败时间、失败类型（格式/校验/执行）、本轮反馈文本、下一次修正输出。

#### 3.7.3 事件时序与因果关联（用于定位问题）
1. 统一事件时间轴（全局底部抽屉）：
- 按时间串联 `turn_start`、`team_*`、`transcript_update`、`snapshot_saved` 等事件。
- 支持按 side、event type、round 过滤。

2. 事件-快照联动：
- 点击某事件，主视图自动跳转到该时刻最近快照。
- 在时间轴标注“关键事件点”：ready for adjudication、adjudication complete、retry。

3. 因果跳转：
- 从“失败日志”一键跳到对应 `AgentAction` 与图中目标节点。
- 从图中某条冲突边反向定位“是谁在何回合添加的”。

#### 3.7.4 Prompt / Context 可见化（模型可解释调试）
1. Context Inspector：
- 显示当回合 `latest_context`（即传给 LLM 的结构化上下文）。
- 显示 `id_inventory`（防幻觉约束输入）以及当前根主张列表。

2. Decision Inspector：
- 原始 `decision_raw`（LLM 输出）与解析后的 `AgentAction[]` 对照展示。
- 对解析失败时，直接展示 parser 错误与原始片段高亮。

3. Narrative Inspector：
- 同屏对比“动作原语句列表”与“润色后叙事文本”，便于观察叙事偏差。

#### 3.7.5 回放增强（面向演示讲解）
1. 关键帧书签：
- 自动打点：首轮、首次冲突边出现、首次 retry、收敛触发、裁判完成。
- 演示时可一键跳关键帧，避免手动拖动。

2. 对比视角：
- `Graph A` vs `Graph B` 并排。
- `Delta Panel` 列出节点和边变化清单，支持点击定位。

3. 自动讲解模式（Demo Mode）：
- 预设路径：初始化 -> 典型对抗回合 -> 收敛 -> 裁判 -> 记忆写回。
- 每个节点给一句“解说词模板”，辅助现场汇报。

#### 3.7.6 大图可用性与信息降噪
1. 图操作基础能力：
- 小地图（minimap）、框选缩放、自动布局重排、节点搜索（ID/关键词）。

2. 过滤器：
- 按 `NodeType`、`NodeStatus`、`agent_id`、`last_modified_step` 过滤。
- `仅看冲突链`、`仅看根主张相关路径` 快捷开关。

3. 日志降噪：
- 默认显示摘要日志；高级模式才展开原始 JSON 和全部事件。

#### 3.7.7 错误反馈体验（让失败“可调试”）
1. 错误分层显示：
- `Validation Error`（动作不合法）
- `Execution Error`（运行时异常）
- `Retrieval Error`（ES 检索问题）
- `LLM Error`（超时/空响应）

2. 一键复制 Debug Bundle：
- 自动打包：session_id、round、last_log、最近 20 条事件、当前快照摘要。
- 便于提交 issue 或团队内排障。

3. 非阻塞失败体验：
- 单 worker 失败不阻塞全局；UI 明示“降级继续运行”。
- 裁判前若存在关键数据缺失，给出黄色预警但允许继续。

---

## 4. 后端接口设计（前端所需，不做过度工程）

说明：当前代码是 Python 引擎，不是 API 服务。建议新增一个“轻量 API 适配层”，只封装 `DebateEngine`。

### 4.1 会话模型
建议后端维护 `DebateSession`（内存字典即可，演示优先）：
- `session_id`
- `engine`
- `status`（CREATED / SETUP_DONE / DEBATING / READY_FOR_ADJUDICATION / ADJUDICATING / FINISHED / ERROR）
- `created_at` / `updated_at`

### 4.2 REST 接口（最小可演示闭环）

1. `GET /api/v1/health`
- 用途：前端探活。

2. `GET /api/v1/cases?limit=20&offset=0`
- 用途：列出样例案件（标题、uid、cause 摘要）。

3. `GET /api/v1/cases/{uid}`
- 用途：读取样例案件详情。

4. `POST /api/v1/sessions`
- 用途：创建会话。
- 请求：`{ "case_uid": "..." }` 或 `{ "case_data": {...} }`
- 返回：`{ "session_id": "...", "status": "CREATED" }`

5. `POST /api/v1/sessions/{id}/setup`
- 用途：调用 `engine.setup()`。
- 返回：初始快照摘要（fact_count/claim_count/node_count/edge_count）。

6. `POST /api/v1/sessions/{id}/step`
- 用途：执行单步 `engine.step()`。
- 返回：`last_log + snapshot + ready_flag`。

7. `POST /api/v1/sessions/{id}/run`
- 用途：自动推进到“ready”或“finished”。
- 参数：`{ "max_steps": 20, "auto_adjudicate": true }`
- 备注：内部逻辑应在 `is_ready_for_adjudication` 时自动 `adjudicate()`（否则用户会误解“为什么不结束”）。

8. `POST /api/v1/sessions/{id}/adjudicate`
- 用途：显式裁判。
- 返回：`judgment_document`、`root_claims_status`、`baf_details`。

9. `GET /api/v1/sessions/{id}`
- 用途：会话总状态（`get_snapshot()` 轻量摘要版）。

10. `GET /api/v1/sessions/{id}/snapshot`
- 用途：当前快照（图 + 指标 + transcript）。

11. `GET /api/v1/sessions/{id}/snapshots`
- 用途：快照索引列表（round_idx、turn、timestamp）。

12. `GET /api/v1/sessions/{id}/snapshots/{round_idx}`
- 用途：读取指定历史快照用于回放。

13. `GET /api/v1/sessions/{id}/memory`
- 用途：返回当前会话可解释记忆信息：
- static_history_cases
- dynamic_law_cases
- insights（可见摘要）
- task_layer（节点/边）

14. `DELETE /api/v1/sessions/{id}`
- 用途：释放资源（关闭 ES 客户端、清理引擎对象）。

### 4.3 实时事件流接口（关键）

1. `WS /api/v1/sessions/{id}/events`（推荐）
- 用途：把 `engine.set_state_callback()` 的事件实时推给前端。
- 统一事件 envelope：
```json
{
  "event": "turn_complete",
  "session_id": "sess_xxx",
  "ts": 1730000000,
  "data": {
    "turn": "plaintiff",
    "round": 2,
    "delta_phi": 1.2,
    "sma": 0.9
  }
}
```

2. 事件类型（与代码对齐）：
- `setup_start`
- `setup_complete`
- `turn_start`
- `turn_complete`
- `transcript_update`
- `snapshot_saved`
- `adjudication_ready`
- `adjudication_start`
- `adjudication_complete`
- `team_plaintiff_turn_start`
- `team_plaintiff_internal_step`
- `team_plaintiff_retry`
- `team_plaintiff_turn_complete`
- `team_defendant_turn_start`
- `team_defendant_internal_step`
- `team_defendant_retry`
- `team_defendant_turn_complete`

### 4.4 前后端数据结构约定（建议）

1. GraphNode
```json
{
  "id": "CLAIM_xxx",
  "content": "...",
  "type": "CLAIM",
  "status": "HYPOTHETICAL",
  "agent_id": "plaintiff_Controller",
  "metadata": { "is_root_claim": false, "last_modified_step": 3 }
}
```

2. GraphEdge
```json
{
  "source": "FACT_xxx",
  "target": "CLAIM_xxx",
  "type": "SUPPORT"
}
```

3. Snapshot
```json
{
  "round_idx": 2,
  "turn": "defendant",
  "graph_data": { "nodes": [], "edges": [] },
  "convergence": { "delta_phi": 1.0, "sma": 1.4, "history": [2.0, 1.0] },
  "stats": { "node_count": 15, "edge_count": 18, "claim_nodes": 9, "conflict_edges": 4 },
  "action_summary": "Action Completed: ...",
  "is_finished": false
}
```

### 4.5 错误与降级策略（必须有）
1. ES 不可用：Worker 报告以 `NOT_FOUND/ERROR` 呈现，但主流程不中断。
2. LLM 超时：返回错误事件并允许 `retry`。
3. 校验失败：把 GraphExecutor 的详细错误透传到 UI “动作审计表”。
4. 会话中断：允许从最近快照恢复展示（即使不能继续运行，也能回放）。

### 4.6 代码审查后发现的可视化缺口（必须补齐）
以下缺口是基于当前仓库代码得出的“实现级问题”，不补会直接影响可视化与 Debug 体验：

1. 快照恢复功能缺失（`mas/core/engine.py`）：
- `restore_snapshot()` 当前仅做边界检查后直接 `return True`，没有实际恢复图状态。
- 影响：回放页无法真正“回到某一轮并继续分析”。
- 需补：按 snapshot 重建 `graph`、`transcript`、`convergence_history`、`last_step_log`、`round_idx/current_turn`。

2. `get_snapshot()` 包含不可直接 JSON 序列化对象（`mas/core/engine.py`）：
- 返回体含 `shadow_graph`、`insights_manager`、`task_layer` 对象。
- 影响：REST 直出时会出现序列化问题，前端调试数据难统一。
- 需补：新增 `get_serializable_snapshot()`，只返回 JSON-safe 数据结构。

3. 缺少“动作级产物”持久化：
- 当前只有 `last_step_log.action` 摘要，没有完整保留：`decision_raw`、`parsed_actions`、`executor logs`、worker 原始报告。
- 影响：Decision Inspector / Retry 分析只能看零散日志，不能完整复盘。
- 需补：按 turn 持久化 `turn_artifacts`（详见 4.7）。

4. 事件缺少统一关联字段：
- `set_state_callback` 推送事件无 `event_id`、`turn_id`、`correlation_id`。
- 影响：事件时间轴难稳定排序，事件和快照无法精确一一对应。
- 需补：事件 envelope 增加 `seq`、`turn_uid`、`source`、`session_id`、`ts_ms`。

5. 失败历史信息丢失风险（`roles/controller.py`）：
- `recent_errors` 每次被覆盖为单条，历史重试链不完整。
- 影响：Retry Timeline 难还原“第几次错、如何改”。
- 需补：新增 `error_history` 列表（append-only），保留每次失败详情。

6. Team 内部 transcript 默认不可见（`mas/agents/team.py`）：
- 只有 `verbose=True` 才积累详细内部消息。
- 影响：演示环境默认看不到 worker 内部过程，不利于讲解。
- 需补：提供 `debug_mode`，即使不 verbose 也保留精简内部事件摘要。

### 4.7 为可视化新增的数据结构与接口（建议）
1. 新增 `TurnArtifacts`（后端内存结构）：
- `turn_uid`
- `side`
- `round_idx`
- `controller_assessment`（Fact/Law/Recall needs 原始输出与解析结果）
- `batch_instructions`
- `worker_reports`（含 worker 名称、状态、耗时、max_score）
- `decision_raw`
- `parsed_actions`
- `execution_logs`
- `retry_history`
- `narrative_raw_sentences` 与 `narrative_polished`

2. 新增 REST：
- `GET /api/v1/sessions/{id}/turns/{turn_uid}/artifacts`
- `GET /api/v1/sessions/{id}/events/history?from_seq=&to_seq=`
- `GET /api/v1/sessions/{id}/diff?from_round=&to_round=`（后端计算结构化 diff）

3. 新增 WebSocket 事件字段：
```json
{
  "seq": 1024,
  "session_id": "sess_xxx",
  "turn_uid": "turn_0007_defendant",
  "source": "engine|team|controller|worker|judge",
  "event": "team_defendant_internal_step",
  "ts_ms": 1730000000123,
  "data": {}
}
```

4. 新增导出接口（面向演示素材）：
- `GET /api/v1/sessions/{id}/export/replay.json`（全量回放数据）
- `GET /api/v1/sessions/{id}/export/graph.gexf`（图分析工具兼容）

---

## 5. 前端状态管理建议（简洁、够用）

推荐最小组合：
1. 路由：React Router。
2. 服务器状态：TanStack Query（拉取 snapshot / session / memory）。
3. 实时流：WebSocket hook（统一事件总线）。
4. 本地 UI 状态：轻量 store（例如 Zustand）保存当前选中的节点、回放指针、过滤条件。

补充 UX 基础约束（建议在第一版就落实）：
1. 图布局稳定性：同一节点跨回合尽量保持坐标（减少视觉抖动，便于 diff 识别）。
2. 可访问性：冲突/支持不只靠颜色区分，需有线型和图例文本，兼容色弱用户。
3. 快捷键：`Space` 播放暂停、`←/→` 上下快照、`F` 聚焦搜索。
4. 大数据保护：日志面板与事件列表采用虚拟列表，避免长会话卡顿。

不建议第一版引入：
1. 复杂 DDD 分层。
2. 过度插件化图编辑器。
3. 多租户与权限系统。

---

## 6. 关键交互流程（端到端）

### 6.1 演示流程 A（推荐默认）
1. 首页选择样例案件。
2. 创建 session + setup。
3. 进入 live 页面，连续点 `Step` 观察双方博弈。
4. 触发 `adjudication_ready` 后执行 `Adjudicate`。
5. 跳转 judgment 页面查看判决与 BAF。
6. 跳转 memory 页面查看策略沉淀。
7. 跳转 replay 页面回放关键轮次。

### 6.2 演示流程 B（自动演示）
1. 首页创建会话。
2. 直接 `run(auto_adjudicate=true)`。
3. 在 live 页面实时观看事件流与图谱变化。
4. 自动结束后进入 judgment。

---

## 7. 与论文公式/概念的 UI 映射（必须显式）

1. 公式 (7) 收敛分数：
- UI 展示 `Δ|V_claim|`、`Δ|E_conflict|`、`lambda(alpha)`、`delta_phi`、`SMA`。
- 数据来源：`DebateEngine._calculate_convergence()` 与 snapshot convergence 字段。

2. 公理约束：
- UI 在 graph 页显示每次动作违反的是哪条公理（支持/冲突/无环）。
- 数据来源：`GraphExecutor._validate_action_static()`、`_check_would_create_cycle()`。

3. BAF 语义：
- UI 展示 direct/support_based/indirect attacks 与 preferred extensions。
- 数据来源：`BAFCalculator.collective_attacks`、`find_preferred_extensions()`、`match_with_llm_judgment()`。

4. 三层记忆：
- UI 显式分层展示 Insights / TaskLayer / Case snapshots。
- 数据来源：`InsightsManager`、`TaskLayer`、`LegalGMemory`。

---

## 8. 实施清单（按优先级）

### Phase 1：可运行闭环（必须先做）
- [x] 新增轻量后端 API 适配层（封装 `DebateEngine`）。
- [x] 打通会话生命周期：create -> setup -> step -> adjudicate -> snapshot。
- [x] 打通 WebSocket 事件推流（状态回调 -> 前端）。
- [x] 完成 live 页面（控制、图谱、日志、指标）。

### Phase 1.5：可视化数据底座补强（代码遗漏补齐）
- [x] 实现 `restore_snapshot()` 真实恢复逻辑。
- [x] 新增 `get_serializable_snapshot()`，避免对象直出。
- [x] 引入 `TurnArtifacts` 持久化（decision_raw、worker_reports、execution_logs）。
- [x] 事件 envelope 增加 `seq/turn_uid/source/ts_ms`。
- [x] Controller 增加 `error_history`（非覆盖式）。

进展备注（2026-02-10）：已完成 engine 快照恢复与 JSON-safe 快照导出，并补充对应单元测试（`tests/test_engine_snapshot.py`）。
进展备注（2026-02-10）：已新增 FastAPI 轻量会话层（`/api/v1`），并打通前端 `http` 模式所需的 REST 闭环接口（含 graph/diff/memory/events），补充 API 测试（`tests/test_api_sessions.py`）。WebSocket 事件推流保留在下一增量。
进展备注（2026-02-10）：已完成 Phase 1.5 数据底座三项，新增 turn artifacts API（`/api/v1/sessions/{id}/turns/artifacts`），并提供一键启动脚本（`scripts/start_dev.sh` / `scripts/stop_dev.sh`）。

### Phase 2：可解释性增强
- [ ] graph 页面：动作审计 + 公理映射。
- [ ] judgment 页面：判决文书 + BAF 面板。
- [x] replay 页面：快照回放 + diff（最小可演示版本，基于 snapshots 索引 + round diff）。

进展备注（2026-02-10）：已新增 `WS /api/v1/sessions/{id}/events` 实时推流与 `GET /api/v1/sessions/{id}/snapshots` 快照索引；前端接入“WS 优先 + 轮询兜底”事件流，并补充 replay 控件可按 round 加载快照与对比 diff。

### Phase 2.5：Debug / 演示体验专项（本次新增）
- [ ] 图谱 Diff 高亮体系（新增/复用/状态变化/拒绝操作）。
- [ ] 团队内部流程可视化（Controller 状态机 + Worker 卡片墙 + Retry 时间线）。
- [ ] 统一事件时间轴与事件-快照联动跳转。
- [ ] Prompt/Decision/Context Inspector 三件套。
- [ ] 一键复制 Debug Bundle 与错误分层提示。

### Phase 3：学习能力展示
- [ ] memory 页面：三层记忆可视化。
- [ ] 展示策略提炼结果与代表案例变化。

### Phase 4：演示打磨
- [ ] 一键自动演示脚本（demo mode）。
- [ ] 错误场景演示（ES/LLM 不可用时的降级）。
- [ ] 页面统一视觉规范与讲解文案。
- [ ] 导出回放包（JSON/GEXF）用于复盘汇报。

---

## 9. 验收标准（Demo Done Definition）

达到以下条件即可认为“前端详尽全面演示系统”完成：
1. 观众可在一个会话中完整看到四阶段生命周期。
2. 每一步图谱变化都可追溯到动作和日志。
3. 至少一次清晰展示“动作被拒绝”的逻辑原因。
4. 至少一次清晰展示“LLM 判决 vs BAF 对齐”的结果。
5. 能展示长期记忆中策略与案例关联，不只是最终判决文本。
6. 即使外部依赖波动（ES/LLM），界面也能解释失败并保持可回放。
7. 在 live/replay 中，Diff 变化可被非开发者在 30 秒内读懂（新增、复用、拒绝、状态迁移）。
8. 团队内部协作流程可视化完整（评估、派发、汇总、决策、重试）。
9. 出现失败时可在 1 分钟内从 UI 定位到“哪一步、哪个动作、为什么失败”。
10. 任意回合都可导出标准化 Debug Bundle，且可在另一台机器重放同一可视化过程。
11. 快照恢复后的图统计值（node/edge/claim/conflict）与原始快照一致。

---

## 10. 一句话总结
这套前端不是“做个漂亮可视化”，而是把论文中的**逻辑约束 + 分层决策 + 生命周期 + 演化学习**，通过当前仓库代码可真实运行的数据流，完整、可解释地演示出来。
