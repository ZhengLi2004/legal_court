# BTODO：功能审查与修复增强清单（基于 `main [5-13].pdf` + 当前代码）

## 审查范围与原则
- 审查范围：后端（`mas/`, `roles/`, `actions/`, `tools/`, `run.py`）、前端（`frontend/`）、测试（`tests/`）。
- 审查依据：
  - 论文方法论第 3 章（论证图、公理约束、BAF、分层架构、四阶段生命周期）。
  - 工程常识（可终止性、可观测性、序列化、测试可维护性、前后端解耦）。
- 前后端分离约束：
  - 前端只消费 JSON DTO 与事件流，不直接接触 Python 对象。
  - 后端只输出稳定契约，不泄露内部类实例。

---

## 总体结论
- 当前后端已具备“多智能体辩论+图谱操作+裁判基础流程”的雏形，但存在多处会影响正确性与演示稳定性的关键缺陷。
- 论文中的“四阶段生命周期”在实现中未闭环，尤其“自适应终止后裁判触发”“演化学习写回”两环存在断裂。
- BAF 语义实现与论文定义存在高风险偏差，需要优先校核。
- 前端目前是空壳，尚未形成可演示系统。

---

## A. 后端缺陷与不一致（P0，必须优先）
- [ ] `P0-BE-01` 终止逻辑断裂，CLI 可能陷入无限循环。
  - 证据：`mas/core/engine.py:562` 仅设置 `is_ready_for_adjudication`，不自动 `adjudicate()`；`run.py:45` 使用 `while not engine.is_finished`。
  - 风险：实验无法自动结束，影响复现实验与前端自动演示。
  - 修复：在 `step()` 到达终止条件时进入裁判状态机，或在 `run.py` 检测 `is_ready_for_adjudication` 后调用 `adjudicate()`。

- [ ] `P0-BE-02` 生命周期第 4 阶段“演化学习”未接入主流程。
  - 证据：`mas/core/system.py:192` 定义 `learn()`，但仓库无调用（全局检索仅定义无引用）。
  - 风险：长期记忆无法随会话增长，和论文“知识反馈回路”不一致。
  - 修复：在 `adjudicate()` 完成后调用 `learn()`，并显式落库与拓扑更新。

- [ ] `P0-BE-03` `restore_snapshot()` 未实现实际恢复。
  - 证据：`mas/core/engine.py:414`~`mas/core/engine.py:427` 仅返回 `True`。
  - 风险：回放页无法“恢复并继续分析”，Debug 不可用。
  - 修复：恢复 `graph`、`transcript`、`last_step_log`、`convergence_history`、`round_idx/current_turn`。

- [ ] `P0-BE-04` BAF 集体攻击计算疑似方向错误（support-based/indirect）。
  - 证据：`mas/analysis/baf.py:84`~`mas/analysis/baf.py:88` 与 `mas/analysis/baf.py:93`~`mas/analysis/baf.py:102` 的索引方向与注释“`A supports B, B attacks C -> A attacks C`”不一致。
  - 风险：preferred extension 计算偏离语义定义，裁判逻辑失真。
  - 修复：重写攻击传播单元测试（最小图真值表）并修正方向。

- [ ] `P0-BE-05` BAF 结果未真正作用于最终判定状态。
  - 证据：`mas/core/engine.py:621` 先写入 `root_claims_status`（LLM提取），后续仅计算 `baf_details` 与 `preferred_extension`，未回写根诉求状态。
  - 风险：界面显示“有 BAF 分析”但结果仍是 LLM-only，与论文“首选扩展采纳”不一致。
  - 修复：明确策略：
    - 方案A：使用 `extract_verdict_with_baf`。
    - 方案B：保留双轨并明确字段 `llm_verdict` 与 `baf_fused_verdict`。

- [ ] `P0-BE-06` `get_snapshot()` 返回不可序列化对象，破坏前后端边界。
  - 证据：`mas/core/engine.py:717`~`mas/core/engine.py:721` 返回 `shadow_graph/insights_manager/task_layer` 实例。
  - 风险：API 序列化失败、前端强耦合内部对象。
  - 修复：新增 `get_serializable_snapshot()`，仅输出 JSON DTO。

- [ ] `P0-BE-07` `winner` 从未更新。
  - 证据：`mas/core/engine.py:69` 初始化 `Unsettled`，`mas/core/engine.py:723` 返回，但无任何写入逻辑。
  - 风险：业务字段长期失真，影响演示可信度。
  - 修复：根据根诉求状态与判决结论推导赢家并赋值。

---

## B. 后端缺陷与不一致（P1，高优先）
- [ ] `P1-BE-01` 快照轮次标记可能错位。
  - 证据：`mas/core/engine.py:577`~`mas/core/engine.py:578` 先在被告回合结束后 `round_idx += 1`，`mas/core/engine.py:586` 再写入本回合快照。
  - 风险：回放界面出现“回合号与行动方不一致”。
  - 修复：以 `turn_start` 时的回合号作为快照主键，或使用 `turn_uid`。

- [ ] `P1-BE-02` 快照体积过大，内存增长快。
  - 证据：`mas/core/engine.py:379` 每次快照携带全量 transcript；`mas/core/engine.py:347`~`mas/core/engine.py:359` 每次全图序列化。
  - 风险：长会话卡顿、导出回放慢。
  - 修复：增量快照或压缩策略；transcript 改为引用链。

- [ ] `P1-BE-03` 节点“修改时间”会被无效边操作污染。
  - 证据：`mas/analysis/executor.py:262` 后无论 CREATED/DUPLICATE/TYPE_CLASH 都执行 `touch_nodes`（`mas/analysis/executor.py:263`）。
  - 风险：上下文焦点计算偏移，Diff 噪音增加。
  - 修复：仅在 `EdgeAddResult.CREATED` 时 touch。

- [ ] `P1-BE-04` BAF 配置项定义与使用脱节。
  - 证据：`mas/config.py:132`~`mas/config.py:154` 定义了多项开关，但全局未检索到实际读取。
  - 风险：配置表面可调、实际无效。
  - 修复：把 `enabled/use_for_*` 接入裁判与传播路径。

- [ ] `P1-BE-05` `judge_config` 参数未使用。
  - 证据：`mas/core/engine.py:58` 赋值后无引用。
  - 风险：接口语义混乱。
  - 修复：删除该参数或实际接入 Judge 配置覆盖。

- [ ] `P1-BE-06` TaskLayer 文件名配置不一致。
  - 证据：`mas/config.py:62` 是 `case_graph.pkl`，`mas/memory/topology.py:35` 默认 `case_reference_graph.pkl`。
  - 风险：持久化文件路径混乱，迁移/备份困难。
  - 修复：统一由 `SystemConfig.path.file_query_graph` 驱动。

- [ ] `P1-BE-07` `TaskLayer.get_central_node()` 存在重复赋值和死代码。
  - 证据：`mas/memory/topology.py:139`~`mas/memory/topology.py:144` 两次 `central_node=`，前者被后者覆盖。
  - 风险：行为不透明，维护困难。
  - 修复：保留单一排序规则并写清 tie-break。

- [ ] `P1-BE-08` 配置 dataclass 使用实例默认值，存在共享状态风险。
  - 证据：`mas/config.py:166`~`mas/config.py:176`。
  - 风险：多实例场景下潜在污染。
  - 修复：改为 `field(default_factory=...)`。

- [ ] `P1-BE-09` 关键依赖未在 `requirements.txt` 显式声明。
  - 证据：`tools/llm.py:15` 用 `openai`，`tools/base_es_tool.py:11` 用 `elasticsearch`，`tools/embedding.py:18` 用 `portalocker`；`requirements.txt:1`~`requirements.txt:15` 未列出。
  - 风险：新环境安装后运行失败。
  - 修复：补充直接依赖并锁定兼容版本。

- [ ] `P1-BE-10` ES 连接每步开关一次，吞吐和稳定性不佳。
  - 证据：`mas/core/engine.py:459` 开资源，`mas/core/engine.py:605` 关资源（每 `step`）。
  - 风险：连接抖动、性能退化。
  - 修复：会话级连接池，`setup` 打开，`session close` 关闭。

- [ ] `P1-BE-11` 控制器错误历史覆盖式存储，不利于重试可视化。
  - 证据：`roles/controller.py:372` 与 `roles/controller.py:395` 每次覆盖 `recent_errors`。
  - 风险：无法完整呈现 retry 演化链。
  - 修复：新增 append-only `error_history`。

- [ ] `P1-BE-12` 默认模式下团队内部消息可观测性不足。
  - 证据：`mas/agents/team.py:181`~`mas/agents/team.py:188` 仅 `verbose` 才记录。
  - 风险：演示与排障信息不足。
  - 修复：引入 `debug_mode`，保留精简事件摘要。

---

## C. 与论文方法论不一致项（P1-P2）
- [ ] `P1-METHOD-01` “自适应终止与裁决”未闭环触发。
  - 论文点：第 3.4.3 强调自动终止后进入裁决。
  - 代码现状：仅置 `is_ready_for_adjudication`。

- [ ] `P1-METHOD-02` “演化学习反馈回路”未在主流程执行。
  - 论文点：第 3.4.4 强调裁判后知识提炼与拓扑更新。
  - 代码现状：`learn()` 未调用。

- [ ] `P1-METHOD-03` BAF 语义与“首选扩展采纳”未落地为最终状态。
  - 论文点：首选扩展应影响判定与状态传播。
  - 代码现状：仅记录 `baf_details`。

- [ ] `P2-METHOD-04` 上下文采样窗口与论文公式参数未配置化。
  - 论文点：`w` 为可调窗口。
  - 代码现状：`ShadowGraph._calculate_focus_nodes` 固定回看 4 步。

- [ ] `P2-METHOD-05` 投影算法实现偏“语义匹配启发式”，未显式包含结构一致性加权项。
  - 论文点：公式(3) 同时考虑语义相似与拓扑保持。
  - 代码现状：主要依赖 embedding 匹配与邻域展开。

---

## D. 前端缺陷与增强点（独立于后端实现）

### D1. 当前缺陷（P0）
- [ ] `P0-FE-01` 页面为空，无法承载任何演示任务。
  - 证据：`frontend/src/App.tsx:2` `return null`。

- [ ] `P0-FE-02` 无路由、无数据层、无事件流订阅。
  - 证据：仅 `main.tsx` 挂载空 `App`。

- [ ] `P0-FE-03` 无 API 客户端与 DTO 校验层。
  - 风险：后端结构变更时前端易崩溃。

### D2. 体验增强（P1）
- [ ] `P1-FE-01` 图谱 Diff 高亮与变化摘要条。
- [ ] `P1-FE-02` Controller/Worker 内部流程泳道图。
- [ ] `P1-FE-03` 事件时间轴 + 快照联动定位。
- [ ] `P1-FE-04` Prompt/Decision/Context Inspector。
- [ ] `P1-FE-05` 大会话性能优化（虚拟列表、图布局稳定、增量渲染）。

### D3. 可维护性（P2）
- [ ] `P2-FE-01` 增加状态图例、键盘快捷键、色弱友好样式。
- [ ] `P2-FE-02` 增加统一错误边界和空态页面。
- [ ] `P2-FE-03` 导出回放包查看器（离线重放）。

---

## E. 前后端分离整改项（契约优先）
- [ ] `P0-API-01` 统一 DTO：`SessionDTO`、`SnapshotDTO`、`NodeDTO`、`EdgeDTO`、`TurnArtifactsDTO`。
- [ ] `P0-API-02` 禁止接口返回 Python 内部对象（Graph/Manager/TaskLayer 实例）。
- [ ] `P0-API-03` WebSocket 事件统一 envelope：`seq/session_id/turn_uid/source/event/ts_ms/data`。
- [ ] `P1-API-04` API 版本化：`/api/v1` 固定字段，新增字段走向后兼容。
- [ ] `P1-API-05` 提供后端计算 diff 接口，前端只做展示。
- [ ] `P1-API-06` 提供 `export/replay.json` 标准化导出接口。

---

## G. 工程卫生与仓库管理
- [ ] `P1-ENG-01` `.gitignore` 规则过宽，屏蔽 `tests` 与全部 `*.md`，不利于协作。
  - 证据：`.gitignore:12`、`.gitignore:17`。
  - 建议：保留必要忽略项，不要全局屏蔽测试与文档。

- [ ] `P1-ENG-02` 前端构建产物 `dist/` 和 `node_modules/` 存在仓库目录中，需明确是否纳管。
  - 建议：默认不纳管，加入忽略并在 CI 构建。

- [ ] `P2-ENG-03` 将实验脚本、演示脚本、生产代码分层目录，减少职责混杂。

---

## H. 建议执行顺序（保持前后端分离）
1. 后端先修 `P0-BE` 与 `P0-API`：先保证“可终止、可裁判、可序列化、可回放”。
2. 后端补 `TurnArtifacts + Event Envelope`：为前端可视化提供稳定数据。
3. 前端再实现 live/graph/replay：只基于契约，不依赖后端内部类。
4. 最后补测试与导出能力：确保演示可复现、可排障。

---

## I. 完成标准
- [ ] 后端能自动完成一次完整生命周期并输出稳定 JSON 数据。
- [ ] 前端可在不访问 Python 内部对象的前提下完整演示流程。
- [ ] BAF 语义结果可被解释且与最小真值测试一致。
- [ ] 测试套件在离线/无外部密钥场景下可稳定通过核心用例。

