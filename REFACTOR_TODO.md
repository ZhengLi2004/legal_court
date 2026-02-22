# 重构 TODO 清单（维护性专项）

## 目标
- 降低“单文件多职责”和“过度 fallback”带来的维护成本。
- 将核心流程拆分为可替换、可测试的模块，减少跨层耦合。
- 统一错误处理与数据协议，避免静默吞错。

## P0（先做，影响最大）
- [ ] 拆分会话管理巨型模块：`mas/api/session_manager.py`（2018 行，38 个方法）
  - 拆成 `session_lifecycle`、`event_stream`、`snapshot_store`、`replay_restore`、`exporters`。
  - 去除重复 JSON 安全转换逻辑，统一复用一个序列化工具。
  - 验收：`SessionManager` 主类仅保留编排职责，不直接处理文件 I/O 与回放细节。

- [ ] 拆分引擎巨型模块：`mas/core/engine.py`（1189 行，31 个方法）
  - 分离 `setup`、`turn_runner`、`convergence`、`snapshot_codec`、`post_adjudication_learning`。
  - 清理重复字段（如快照中的同义字段）和多处默认值回退链。
  - 验收：核心回合推进逻辑可在不触发持久化/序列化代码时单测。

- [ ] 前端状态中心解耦：`frontend/src/app/state/DebateContext.tsx`（1063 行）
  - 拆为 `useSessionActions`、`useGraphActions`、`useTimelineStream`、`useSnapshotActions`。
  - 合并重复错误处理（大量 `err instanceof Error ? ...`）为统一 `toErrorMessage`。
  - 验收：Context 文件 < 400 行，功能行为保持不变。

## P1（第二阶段）
- [ ] 协议归一化去冗余：`frontend/src/compat/protocol.ts`（774 行）
  - 建立 schema 驱动解析（建议 zod/pydantic 对齐），减少层层 `fallback`。
  - 明确“必填失败抛错”与“可选字段回退”边界，避免 silent data fix。
  - 验收：快照/图谱/时间线解析规则可读且单测覆盖关键分支。

- [ ] API 路由统一异常映射：`mas/api/server.py`
  - 抽出统一 `exception -> HTTPException` 映射，删除重复 try/except 模板。
  - 验收：路由函数仅保留参数解析 + 调用服务。

- [ ] 控制器状态机降复杂度：`roles/controller.py`
  - 将 `_act` 的阶段分支拆为独立 handler（assess/plan/push/retry）。
  - 验收：单阶段失败不影响其他阶段可测试性。

## P2（稳定性与规范）
- [ ] 清理宽泛异常与静默分支（全仓当前约 `except Exception` 42 处、`pass` 12 处）
  - 优先处理：`tools/llm.py`、`mas/api/session_manager.py`、`mas/core/engine.py`、`roles/*`。
  - 用结构化日志替换 `print`，禁止无注释 `pass`。

- [ ] 测试与目录治理
  - 将 `tests/optim/` 迁至 `benchmarks/` 或 `tmp/benchmarks/`，避免与回归测试语义混杂。
  - 为 P0/P1 拆分点补契约测试（session 生命周期、snapshot restore、protocol normalize）。

## 执行顺序建议
1. `session_manager` 拆分 + 回归测试
2. `engine` 拆分 + 快照契约测试
3. 前端 `DebateContext`/`protocol` 拆分
4. API 异常映射与日志规范统一
