# 实验章节实现总 TODO（面向当前仓库）

> 目标：把 `TODO. 实验章节.md` 中的实验方案完整落地为“可运行、可复现、可审计”的工程实现。
> 范围：后端评测管线、基线与消融运行器、统计检验、附录产物、冻结归档与测试。

## A. 现状盘点（已完成阅读）

- [x] 通读实验文档 `TODO. 实验章节.md`。
- [x] 通读核心执行链路：`mas/core/*`, `mas/application/*`, `mas/agents/*`, `mas/infrastructure/*`。
- [x] 通读会话/快照/导出链路：`mas/session/*`, `mas/api/*`。
- [x] 通读现有参数调优体系：`benchmarks/optim/*`。
- [x] 明确当前缺口：
- [x] 缺少“多方法端到端批量评测框架”（当前 `run.py` 仅单案）。
- [x] 缺少实验文档要求的主指标体系实现（E2E Status、Matched Status、E2E F1 FP-sensitive 等）。
- [x] 缺少匹配鲁棒性门槛（Spearman/翻转率）与降级宣告流程。
- [x] 缺少一致性/忠实度离线评测器与异质评测集合。
- [x] 缺少冻结输入 benchmark、反事实预算实验、Pareto 构造与噪声阈值处理。
- [x] 缺少 Holm-Bonferroni、McNemar、Stuart-Maxwell 的全套统计落地。
- [x] 缺少“评测冻结归档”指定路径与 tag 流程。

---

## B. 目录与工程骨架（先搭框架再填逻辑）

- [ ] 新建实验主目录（建议）：`benchmarks/experiments/`。
- [ ] 新建实验配置目录：`configs/`（当前仓库无该目录，仅有 `config/config2.yaml`）。
- [ ] 新建解析规则目录：`parser_rules/`。
- [ ] 新建对齐器配置目录：`aligner/`。
- [ ] 新建预算网格文件：`budget_grid.json`。
- [ ] 新建预注册检验点文件：`prereg_points.json`。
- [ ] 新建实验运行脚本入口（建议）：
- [ ] `benchmarks/experiments/run_claim1_e2e.py`
- [ ] `benchmarks/experiments/run_claim2_consistency_faithfulness.py`
- [ ] `benchmarks/experiments/run_claim3_reasoning_attribution.py`
- [ ] `benchmarks/experiments/run_claim4_counterfactual.py`
- [ ] 新建统一结果汇总与导出脚本：`benchmarks/experiments/export_reports.py`。
- [ ] 新建图表生成脚本：`benchmarks/experiments/plot_figures.py`。
- [ ] 新建附录产物目录（建议）：`benchmarks/experiments/artifacts/appendix/`。
- [ ] 新建主文产物目录（建议）：`benchmarks/experiments/artifacts/main/`。

---

## C. 数据划分与重采样协议（对应文档第 2 节）

### C1. 数据读取与标准化

- [ ] 新建案件清洗与字段标准化模块（从 `data/sampling/cleaned_samples.jsonl` 读取 500 样本）。
- [ ] 明确并固化评测输入字段：`uid`, `cause`, `fact_finding`, `plaintiff_claim`, `defendant_argument`, `court_opinion`, `verdict_result`。
- [ ] 为每个案件生成稳定 `case_id`（优先用 uid，fallback hash）。

### C2. 50 Dev / 其余 Test 划分

- [ ] 实现一次性开发集抽样（50 案）与测试集固定划分。
- [ ] 记录划分索引到文件（防漂移）：`benchmarks/experiments/artifacts/splits/dev_test_split.json`。
- [ ] 保证后续全部实验复用同一划分文件。

### C3. 分层重采样（Dev 内 5 次）

- [ ] 定义分层键：`案由 × 单/多诉求 × 文本长度桶 × 难例标签`。
- [ ] 实现 Dev 内 5 次 stratified resampling（固定种子列表）。
- [ ] 输出每次重采样索引与摘要。
- [ ] 实现参数选择规则：取 5 次结果中位数；稳定区间取 [20%,80%] 分位。
- [ ] 生成“重采样收敛证据图”数据：重采样次数 1..5 下参数区间宽度、主指标均值、CI。
- [ ] 产出附录图文件。

### C4. 难例标签脚本与扰动稳健性

- [ ] 定义难例标签规则（仅使用输入可得特征/纯规则，不泄漏标签）。
- [ ] 输出规则清单文档（附录可公开）。
- [ ] 对难例标签规则做轻量扰动（阈值微调/规则开关）并评估稳定区间抗扰动性。

### C5. 调参优先级闭环

- [ ] 实现“先优化 Step A 抽取 F1，再看主指标”的自动选择逻辑。
- [ ] 在 Step A 近似等价区间内，二级排序使用“事实忠实度覆盖率更高/不确定率更低”。

---

## D. 标签空间协议与匹配协议（对应文档第 2 节）

### D1. 三态与二态映射

- [ ] 新建状态映射器：`HYPOTHETICAL -> DEFEATED`（主评测 2-class）。
- [ ] 同时保留 3-class 评测通道（Balanced Accuracy / Macro-F1）。
- [ ] 保证主表和附表自动分流。

### D2. 统一预测端预去重

- [ ] 在所有方法输出上统一执行层次聚类去重（Cosine）。
- [ ] 实现“簇中心最近 claim 作为代表”逻辑。
- [ ] 阈值来源与版本固定（写入配置）。

### D3. 匈牙利匹配

- [ ] 实现一对一二分图匹配（Hungarian algorithm）。
- [ ] 代价函数实现：语义相似度 + 文本长度差惩罚。
- [ ] 实现匹配审计日志（每对匹配的得分与惩罚分解）。

### D4. 排序稳健性门槛与降级

- [ ] 实现稳健性扰动评测（替代特征 + 阈值扰动）。
- [ ] Dev 集计算方法排序 Spearman rho（要求 >=0.8）。
- [ ] 计算两两排序翻转率（要求 <=10%）。
- [ ] 失败时触发降级流程：
- [ ] 主文仅保留 `E2E Status Acc + CI` 的底线结论。
- [ ] 不输出 Step A/匹配敏感派生指标强结论。
- [ ] 生成附录“扰动通过率 + 主要翻转来源 + 长度分桶误差诊断”。

### D5. Gold Status 与锚点集

- [ ] 实现规则脚本生成基础 gold status。
- [ ] 对规则不确定样本接入辅助判决模型（Qwen3-8B Judge）。
- [ ] 建立双盲人工锚点集（80-150 样本）标注流程。
- [ ] 计算并报告 Cohen's kappa。

---

## E. 对等基线与代码级消融（对应文档第 3 节）

### E1. 统一运行接口

- [ ] 定义 `MethodRunner` 抽象接口（输入 case -> 输出 claim/status/transcript/cost）。
- [ ] 统一日志、随机种子、预算、缓存、输出 schema。

### E2. 主系统 Runner

- [ ] 封装当前 `DebateEngine` 批量运行器（支持逐案运行与批处理）。
- [ ] 保持与在线 API 链路行为一致。

### E3. B1 基线（Vanilla RAG + Structured JSON）

- [ ] 实现单体模型基线 runner。
- [ ] 输出统一 claim-status 结构供后续统一解析。
- [ ] 对接统一检索预算和 token 预算统计。

### E4. B2 基线（Vanilla Multi-Agent Debate）

- [ ] 实现不带图一致性约束的通用多智能体辩论 baseline。
- [ ] 保持同资源上限与同输出 schema。

### E5. B3 基线（Stateful Agent without Axioms）

- [ ] 在现有控制器增加可配置开关，关闭 `PlanTool.validate_json_actions`（即不走 ValidateStep）。
- [ ] 新增配置字段与运行时注入路径。
- [ ] 明确 B3 与主系统唯一差异可审计。

---

## F. 主张 1 指标实现（对应文档 4.1）

### F1. 聚合口径

- [ ] 实现 Case-level macro 聚合器：
- [ ] 先 case 内对 root claim 求均值。
- [ ] 再跨 case 求均值。
- [ ] 配套 case-level paired bootstrap 95% CI。

### F2. 主表指标

- [ ] `E2E Status Acc`：未匹配 gold claim 直接计错。
- [ ] `Status Acc (Matched)`：仅匹配子集的状态准确率。

### F3. 附录指标

- [ ] `Step A Precision/Recall/F1`。
- [ ] `E2E F1 (FP-sensitive)`：
- [ ] 匹配但状态错判计 FP+FN。
- [ ] 未匹配 gold 计 FN。
- [ ] 无法匹配预测计 FP。
- [ ] `Soft E2E`（状态错判仅计 FN）并做方向一致性校验。
- [ ] 过度生成率（over-generation rate）。

### F4. 主文写作钩子数据

- [ ] 自动生成“主张1结论段所需引用片段”：
- [ ] 主结论方向一致性说明。
- [ ] FP-sensitive 与 over-generation 的佐证表。

---

## G. 主张 2：一致性与忠实度（对应文档 4.2）

### G1. 异质评测器框架

- [ ] 新建离线评测器统一接口（支持多模型并行、固定提示词、固定 schema）。
- [ ] 配置主评测三模型集合：
- [ ] 1 个大型指令模型（如 Qwen2.5-72B-Instruct）。
- [ ] 1 个不同家族模型。
- [ ] 1 个 NLI/判别强项模型。
- [ ] 商用模型（如 GPT-4o）仅作为附录交叉验证通道。
- [ ] 记录调用日期、endpoint 版本、模型快照信息。

### G2. 离散标签与一致率

- [ ] 强制输出离散标签 `{E, N, C}`。
- [ ] 计算两两一致率矩阵、多数票一致率、不确定率。
- [ ] 实现一致率降级阈值（<70%）与 ±5% 扰动稳健性检查。
- [ ] 触发降级时自动调整主文结论口径（Faithfulness 优先，冲突率降为探索性）。

### G3. 统一一致性解析管线

- [ ] 为所有方法输出实现统一离线解析器：`Claim -> {status set}`。
- [ ] 基于该映射推断冲突边并计算冲突率。
- [ ] 解析失败计入 Parse-Failure，并输出覆盖率。

### G4. 归并阈值共享与敏感性

- [ ] 先做可解释规则归并（法条号、关键实体），再 embedding 补充。
- [ ] 默认与预测匹配阈值共享。
- [ ] 输出综合敏感性图：阈值 ±δ 下
- [ ] Merge-Error
- [ ] 冲突率上下界
- [ ] 抽取 F1/主指标变化

### G5. 归并/解析审计

- [ ] 输出不可解析原因分布。
- [ ] 输出过度归并率、漏归并率。
- [ ] 基于 parse-failure 样本给出冲突率 upper/lower bound。

### G6. 解耦事实忠实度评估

- [ ] 新建独立证据对齐器（离线模块，不复用在线推理状态）。
- [ ] 固定 embedding 模型与阈值（仅 Dev 调参后冻结）。
- [ ] 输出 top-m 证据窗口（m=3）。
- [ ] 统计“对齐失败率”。

### G7. NLI 多数票与聚合策略

- [ ] 对齐成功子集上运行异质 NLI 判定。
- [ ] 主实验聚合：Any（任一窗口蕴含即蕴含）。
- [ ] 附录聚合：Majority / Max-score。
- [ ] 平局标“不确定”，并输出不确定上下界。

### G8. Overall Faithfulness 标量

- [ ] 主定义：`对齐成功率 × 成功子集蕴含率`。
- [ ] 附录替代：`min(对齐率, 蕴含率)`、均值加权。

---

## H. 主张 3：推理归因（对应文档 4.3）

### H1. 测试集零更新约束

- [ ] 新增运行模式开关：`test_mode_no_learning=true`。
- [ ] 测试集阶段禁用：
- [ ] 记忆写入（`legal_sys.learn`）
- [ ] 提示词自适应
- [ ] 检索缓存学习
- [ ] 保障测试序列单向推进，不回看未来样本。
- [ ] 时间戳不可靠时，回退固定数据集顺序。

### H2. 固定证据包对照

- [ ] 构建“固定证据包”离线缓存格式（fact/law/recall inputs）。
- [ ] 增加检索旁路模式：直接加载缓存，不调用外部检索 API。
- [ ] 对比“仅推演策略变化”下的 Status Accuracy 演化曲线。

### H3. 归因分析输出

- [ ] 产出主图：纯推理 benchmark 的 Status Accuracy 曲线。
- [ ] 输出归因分解：抽取/检索变化贡献 vs 推理变化贡献。

---

## I. 主张 4：反事实边际增益与 Pareto（对应文档 4.4）

### I1. 对称预算控制

- [ ] 定义统一预算网格（token/轮次/工具调用上限）。
- [ ] 所有方法在同预算点运行。
- [ ] 达上限时仅裁剪推演过程，保留输出 schema 合规。

### I2. 重复运行与噪声阈值

- [ ] 每预算点温度固定 `Temperature=0`，默认重复 3 次。
- [ ] 计算预算点内标准差。
- [ ] 判定阈值：`std <= 方法间均值差距的 1/3`。
- [ ] 超阈值时追加到 5 次复跑。
- [ ] 仍超阈值则剔除显著性检验，仅保留均值+CI前沿展示。

### I3. 受控重算（Counterfactual）

- [ ] 建立白名单缓存：仅复用原始检索证据。
- [ ] 在受控条件下重算，避免引入新证据污染。

### I4. 指标与图表

- [ ] 主指标：`Δ_root_flip`。
- [ ] 附图：`Δ_err`, `Δ_conflict`。
- [ ] 基于均值点构造 Pareto frontier。

---

## J. 统计检验与呈现纪律（对应文档第 5 节）

### J1. CI 与显著性标注策略

- [ ] 全主表/主图统一使用 case-level paired bootstrap 95% CI。
- [ ] 仅预注册主点允许显著性星号。
- [ ] 非预注册探索点只报 CI，不做显著性宣告。

### J2. 多重比较校正

- [ ] 为每个主张建立独立检验族。
- [ ] 实现 Holm-Bonferroni 校正。

### J3. 附录检验

- [ ] 实现 McNemar（2-class，配对 case-root 单元）。
- [ ] 实现 Stuart-Maxwell（3-class 辅评）。
- [ ] 输出完整附录检验表。

### J4. 预注册比较点

- [ ] 在 `prereg_points.json` 固化：
- [ ] 比较对：主系统 vs 各基线。
- [ ] 预算点：25%/50%/75%。
- [ ] 回合点：t ∈ {1,3,5}。

---

## K. 全局冻结与可审计归档（对应文档第 5 节）

- [ ] 在 Test 评估启动前冻结所有评测组件。
- [ ] 归档以下路径并记录 hash/tag：
- [ ] `configs/*.yaml`
- [ ] `parser_rules/*.json`
- [ ] `aligner/*.yaml`
- [ ] `budget_grid.json`
- [ ] `prereg_points.json`
- [ ] 生成冻结清单（含 git commit、时间、运行命令、随机种子、硬件）。
- [ ] 创建可回放 tag（如 `exp-freeze-v1`）。

---

## L. 输出产物与附录清单

### L1. 主文最小产物

- [ ] 主张1：主图/主表仅核心指标（最多 2 维或 1 图）。
- [ ] 主张2：冲突率 + Overall Faithfulness（含降级逻辑）。
- [ ] 主张3：纯推理 Status Accuracy 演化图。
- [ ] 主张4：Pareto 前沿 + `Δ_root_flip` 主图。

### L2. 附录必备产物

- [ ] 分层重采样收敛证据图。
- [ ] 匹配稳健性门槛执行摘要（通过率、翻转来源）。
- [ ] 长度分桶误差剖析表。
- [ ] 评测器一致率矩阵与不确定率表。
- [ ] 阈值共享敏感性综合图。
- [ ] 解析失败/归并审计摘要。
- [ ] NLI 聚合策略敏感性对比。
- [ ] Overall Faithfulness 替代聚合检验。
- [ ] 预算点方差估计与超阈值处理记录。
- [ ] McNemar / Stuart-Maxwell 全表。
- [ ] 权重/推理框架/量化/硬件/命令模板复现块。

---

## M. 与现有代码的具体改造点

### M1. 运行与可控开关

- [ ] 在 `mas/config.py` 增加实验模式配置（test 无学习、固定证据包、预算裁剪开关）。
- [ ] 在 `mas/core/engine_adjudication.py` / `mas/core/engine_post_learning.py` 增加可禁用 learning 的条件分支。
- [ ] 在 `mas/application/agents/actions/controller_actions.py` 增加 B3 开关（可选跳过 ValidateStep）。

### M2. 检索与缓存

- [ ] 在 `mas/application/agents/worker.py` 增加“固定证据包读取模式”。
- [ ] 在 `mas/infrastructure/fact_es_tool.py` / `law_es_tool.py` 增加缓存旁路适配接口。

### M3. 指标数据采集

- [ ] 扩展 `mas/core/engine_snapshot_codec.py` turn artifact 字段，补充实验统计必需元信息（预算、token、缓存命中、裁剪信息）。
- [ ] 扩展 `mas/session/exporters.py` replay bundle，确保离线评估可完整复放。

### M4. 离线评测组件

- [ ] 新建实验专用 parser/alignment/nli/statistics 模块，不污染线上推理路径。

---

## N. 测试与验收

### N1. 单元测试

- [ ] 匹配器与聚类阈值逻辑测试。
- [ ] 指标计算测试（E2E Status、Matched Status、E2E F1 FP-sensitive、Soft 版）。
- [ ] 统计检验测试（bootstrap、Holm、McNemar、Stuart-Maxwell）。
- [ ] 一致性解析器与归并审计测试。
- [ ] Faithfulness 对齐器与 NLI 聚合测试。
- [ ] 预算裁剪与噪声阈值判定测试。

### N2. 集成测试

- [ ] 单案端到端评测链路 smoke。
- [ ] 小批量多方法跑通（主系统+B1+B2+B3）。
- [ ] Test 零更新约束回归测试（确保无 memory write）。
- [ ] 固定证据包模式与在线检索模式对照一致性测试。

### N3. 复现验收

- [ ] 同一 seed 重跑结果容差验证。
- [ ] 冻结 tag 下全流程重跑验证。
- [ ] 附录关键图表自动再生验证。

---

## O. 执行顺序（推荐）

- [ ] 第 1 周：B/C/D（骨架 + 数据协议 + 匹配协议）。
- [ ] 第 2 周：E/F（基线 + 主张1指标）。
- [ ] 第 3 周：G（主张2一致性/忠实度）。
- [ ] 第 4 周：H/I（主张3/4实验）。
- [ ] 第 5 周：J/K/L（统计、冻结、产物汇总）。
- [ ] 第 6 周：M/N（代码收敛、测试补全、复现验收）。

---

## P. 完成定义（DoD）

- [ ] 四个主张均有“主文最小结论链条”可自动生成。
- [ ] 所有预注册规则可被脚本检查并自动触发降级逻辑。
- [ ] 主文与附录产物可通过单命令重建。
- [ ] 冻结配置与 tag 可独立审计。
- [ ] 测试覆盖关键计算与关键降级路径。
