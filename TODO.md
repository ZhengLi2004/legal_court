# 可执行重构 TODO List（行为不变前提）

## 1) 仓库概览

仓库是 Python 后端 + React 前端的单仓结构。后端入口为 `run.py`（CLI）与 `run_api.py`（FastAPI），HTTP/WS 路由已拆分到 `mas/api/routers/*`；核心辩论流程在 `mas/core/`，但当前 `mas/core` 仍直接依赖 `roles/actions/tools`，存在跨层耦合。最突出的 3 个剩余风险：1）安全风险：`config/config2.yaml:3` 存在明文 API Key；2）架构风险：导入图存在跨包强连通环（`mas.core -> roles -> actions -> tools -> mas.core`）；3）可维护性风险：超大模块集中（`mas/api/session_manager.py` 794 行，`mas/api/serializers.py` 968 行），且测试基线薄（`pytest.ini` 仅匹配一个后端测试文件，前端 vitest 当前无测试文件）。

## 2) TODO List（按 P0/P1/P2）

### P0

#### [RF-002-a]

- [优先级] P0
- [类别] Architecture, Folder Structure
- [证据] `mas/core/engine.py:15` import `roles.controller.ControllerPipelineStep`；`roles/controller.py:21` 依赖 `actions.controller_actions`；`actions/controller_actions.py:13` 依赖 `mas.core.schemas`。
- [症状] `mas/core` 反向依赖上层编排模块，破坏分层与单向依赖。
- [影响] 任意流程改动触发跨层联动（Shotgun Surgery），回归面大。
- [建议改法] 将 `ControllerPipelineStep` 契约下沉到 `mas/core`（或 `mas/application/contracts`），`roles` 与 `engine` 同时依赖该契约；保留旧导出别名一版作为过渡。保持行为不变策略：枚举值与分支判断字符串不变。
- [变更范围] `mas/core/engine.py`, `roles/controller.py`, 新增契约模块。
- [验收标准] `rg -n "from roles.controller import ControllerPipelineStep" mas/core` 结果为 0；核心流程 smoke 通过。
- [工作量] M
- [风险与回滚] 风险：枚举导入路径变更导致运行时错误；回滚：恢复旧 import 并保留新模块不启用。
- [依赖/前置] 无。

#### [RF-002-b]

- [优先级] P0
- [类别] Architecture
- [证据] `mas/core/system.py:14-16` 直接依赖 `tools.*`；`tools/graph_tool.py:11` 依赖 `mas.analysis.executor`。
- [症状] 领域/编排层与基础设施层互相穿透，形成技术细节反向渗透。
- [影响] 难以替换 LLM/embedding/search 实现，单测隔离困难。
- [建议改法] 在核心层定义端口接口（LLM、Embedding、Matcher），由 `tools/` 提供 adapter；`LegalSystem` 只接收端口实例。保持行为不变策略：默认工厂仍装配现有 `tools` 实现。
- [变更范围] `mas/core/system.py`, `tools/`, `mas/session/session_lifecycle.py`（装配点）。
- [验收标准] `mas/core/system.py` 无 `from tools`；`run.py` 主流程与 `/api/v1/health` 行为不变。
- [工作量] M
- [风险与回滚] 风险：依赖注入遗漏；回滚：保留 legacy factory（环境开关回退）。
- [依赖/前置] 无。

### P1

#### [RF-102]

- [优先级] P1
- [类别] Code Smell, Folder Structure
- [证据] `mas/api/session_manager.py` 794 行；同类同时处理会话生命周期、事件流、快照导入导出（如 `save_frontend_snapshot`、`load_frontend_snapshot`）。
- [症状] God Class，职责过载。
- [影响] 任一功能变更都触及大文件，缺陷定位慢。
- [建议改法] 拆分为 `SessionService`、`EventService`、`SnapshotService`，`SessionManager` 退化为 facade。保持行为不变策略：对外方法签名保持兼容。
- [变更范围] `mas/api/session_manager.py`, `mas/session/`。
- [验收标准] 对外 API 无破坏；历史回放/快照导入导出回归用例通过。
- [工作量] M
- [风险与回滚] 风险：服务边界划分不当；回滚：facade 继续代理旧实现。
- [依赖/前置] 无。

#### [RF-103]

- [优先级] P1
- [类别] Code Smell, Folder Structure
- [证据] `mas/api/serializers.py` 968 行；`memory_response` 覆盖 `mas/api/serializers.py:522-968`。
- [症状] 序列化逻辑与聚合业务逻辑混杂。
- [影响] 序列化改动容易引入业务回归，单元测试切分困难。
- [建议改法] 按资源拆分 serializer 模块（snapshot/graph/diff/memory）；将 `memory_response` 拆成 case catalog、insight、task-layer 三个 composer。保持行为不变策略：返回 JSON schema 与字段名严格不变。
- [变更范围] `mas/api/serializers.py`, 新增 `mas/api/serializers/`。
- [验收标准] 关键接口 snapshot 对比一致；回归测试字段 diff 为 0。
- [工作量] M
- [风险与回滚] 风险：字段默认值变化；回滚：保留旧函数包装新实现并逐端点切换。
- [依赖/前置] RF-102。

#### [RF-104]

- [优先级] P1
- [类别] Architecture, Reliability
- [证据] `mas/api/session_manager.py:94` `status: str`；多处字符串状态判断/赋值（如 `mas/api/session_manager.py:265`, `mas/api/session_manager.py:291`）；`mas/session/session_lifecycle.py:79` 返回字符串状态。
- [症状] 状态机用裸字符串驱动，缺少编译期约束。
- [影响] 拼写错误/非法迁移难以及早发现。
- [建议改法] 引入 `SessionStatus(Enum)` + 转移表（allowed transitions），API 序列化层再转字符串。保持行为不变策略：外部响应仍输出原字符串。
- [变更范围] `mas/api/session_manager.py`, `mas/session/session_lifecycle.py`, serializer。
- [验收标准] 非法状态迁移触发显式错误；裸字符串状态引用降为 0（枚举模块除外）。
- [工作量] M
- [风险与回滚] 风险：旧分支未覆盖迁移；回滚：兼容层接受字符串并映射 enum。
- [依赖/前置] RF-102。

#### [RF-105]

- [优先级] P1
- [类别] Architecture, DX
- [证据] `tools/llm.py:19` 模块加载即 `SystemConfig()`；`mas/analysis/executor.py:199` 热路径里创建 `SystemConfig()`；`tools/embedding.py:189` `__post_init__` 回退到全局配置。
- [症状] 隐式依赖与配置生命周期分散。
- [影响] 测试注入困难，运行时配置一致性难保证。
- [建议改法] 建立统一 `SettingsProvider`（启动时构建一次），通过构造函数传入依赖组件。保持行为不变策略：默认 provider 值与当前 `SystemConfig()` 一致。
- [变更范围] `tools/`, `mas/analysis/`, `mas/core/`, 启动装配点。
- [验收标准] 除装配入口外 `SystemConfig()` 调用点显著收敛；集成测试通过。
- [工作量] M
- [风险与回滚] 风险：注入链变长；回滚：保留默认无参构造 fallback。
- [依赖/前置] RF-002-b。

#### [RF-106]

- [优先级] P1
- [类别] Architecture, Testing
- [证据] `frontend/src/app/state/useSnapshotActions.ts` 407 行；`frontend/src/app/state/DebateContext.tsx` 298 行；`frontend/src/app/pages/MemoryPage.tsx:69` 触发 `react-hooks/set-state-in-effect`。
- [症状] 状态管理 Hook 过重，副作用与 UI 状态耦合。
- [影响] 前端 bug 修复成本高，lint 阻塞交付。
- [建议改法] 拆分为 session/memory/snapshot 三个 hooks，移除 effect 内同步 setState，补充 hook 单测。保持行为不变策略：Context 对外 API（方法名/返回值）保持兼容。
- [变更范围] `frontend/src/app/state/`, `frontend/src/app/pages/MemoryPage.tsx`。
- [验收标准] `npm --prefix frontend run lint` 0 error；新增前端单测并通过。
- [工作量] M
- [风险与回滚] 风险：状态来源改变引发 UI 细节差异；回滚：保留旧 hook 并在 provider 层可切换。
- [依赖/前置] RF-103（后端 payload 稳定后进行）。

### P2

#### [RF-202]

- [优先级] P2
- [类别] Observability
- [证据] `tools/llm.py:250`, `tools/llm.py:259` 使用 `print`；`tools/base_es_tool.py:100`、`mas/memory/legal_memory.py:93` 也使用 `print`；前端 `frontend/src/app/state/errorUtils.ts:6` 使用 `console.warn`。
- [症状] 日志出口不统一，结构化字段缺失。
- [影响] 排障链路断裂，难按 `session/turn` 聚合分析。
- [建议改法] 统一后端 logger（带 `session_id/turn_uid/stage` 字段），前端统一封装 warning reporter。保持行为不变策略：仅替换日志输出通道，不改变控制流。
- [变更范围] `tools/`, `mas/memory/`, `frontend/src/app/state/`。
- [验收标准] 生产代码 `print(` 清零；关键日志字段覆盖率达标。
- [工作量] S
- [风险与回滚] 风险：日志量突增；回滚：按模块降级日志级别。
- [依赖/前置] RF-102。

#### [RF-203]

- [优先级] P2
- [类别] Code Smell
- [证据] `frontend/src/compat/protocol.ts:21` 与 `frontend/src/app/utils/payload.ts:1` 重复 `asRecord/asString/unwrapPayload`。
- [症状] 重复概念与重复实现。
- [影响] 修复解析 bug 时易漏改，前后行为分叉。
- [建议改法] 抽到单一 shared 解析工具并统一引用。保持行为不变策略：保留原函数名 re-export 过渡。
- [变更范围] `frontend/src/compat/`, `frontend/src/app/utils/`, 新增 `frontend/src/shared/lib/`。
- [验收标准] 重复 helper 定义仅保留 1 份；协议解析回归通过。
- [工作量] S
- [风险与回滚] 风险：少量边界值解析差异；回滚：逐文件恢复旧 helper。
- [依赖/前置] RF-106。

#### [RF-204]

- [优先级] P2
- [类别] Code Smell, Performance
- [证据] 三个图组件重复 ECharts 生命周期：`frontend/src/app/components/ForceArgumentGraph.tsx:259`、`frontend/src/app/components/TaskLayerGraph.tsx:111`、`frontend/src/app/components/SimpleBafGraph.tsx:77` 均包含 `init/resize/dispose/setOption`。
- [症状] 重复样板代码，维护成本高。
- [影响] 图渲染问题需多处修补，潜在性能回退。
- [建议改法] 抽象 `useEchartsGraph` hook 统一初始化和 resize observer。保持行为不变策略：option 构建逻辑保持在各组件。
- [变更范围] `frontend/src/app/components/`, 新增 `frontend/src/app/hooks/`。
- [验收标准] 3 个组件接入共享 hook；交互与渲染结果一致。
- [工作量] S
- [风险与回滚] 风险：生命周期时序变化；回滚：单组件独立回退。
- [依赖/前置] RF-106。

#### [RF-205]

- [优先级] P2
- [类别] Folder Structure, Performance
- [证据] `frontend/src/App.tsx:12` 手写 `normalizeRoute`，`frontend/src/App.tsx:62` 手动 `history.pushState`；构建产物主 chunk 约 `1,468.86 kB`。
- [症状] 路由逻辑与页面装配耦合，且页面未懒加载。
- [影响] 可维护性一般，首屏包体偏大。
- [建议改法] 引入路由层（React Router 或等价轻量方案）+ 页面级动态导入。保持行为不变策略：URL 与页面映射保持一致。
- [变更范围] `frontend/src/App.tsx`, `frontend/src/app/pages/`。
- [验收标准] 主 chunk 降至 `< 900kB`；路由行为与历史兼容。
- [工作量] M
- [风险与回滚] 风险：前进/后退导航兼容性；回滚：保留 `AppLegacy`。
- [依赖/前置] RF-106。

#### [RF-206]

- [优先级] P2
- [类别] Security
- [证据] `mas/api/server.py:81` 当前 `allow_origins=["*"]`。
- [症状] CORS 策略在非开发环境过宽。
- [影响] 增加跨站调用面，策略治理困难。
- [建议改法] CORS 来源改为环境变量白名单，开发默认 localhost，生产必须显式配置。保持行为不变策略：dev 默认值保持当前联调可用。
- [变更范围] `mas/api/server.py`, 配置文档。
- [验收标准] 生产配置下仅允许白名单域；开发联调不受影响。
- [工作量] S
- [风险与回滚] 风险：白名单误配导致前端不可访问；回滚：临时启用通配并补齐配置。
- [依赖/前置] RF-102。

## 3) 文件夹结构专项建议

### 目标目录结构草案

```text
mas/
  app/
    api/
      main.py
      routers/
        health.py
        sessions.py
        snapshots.py
        memory.py
        events.py
    cli/
      run_experiment.py
  domain/
    debate/
      graph.py
      schemas.py
    memory/
      insights.py
      projection.py
  application/
    debate/
      engine.py
      turn_runner.py
    session/
      service.py
      events.py
      snapshots.py
    controller/
      pipeline.py
  infrastructure/
    llm/
      client.py
    retrieval/
      es/
        fact_tool.py
        law_tool.py
    storage/
      snapshot_store.py
  shared/
    serialization.py

frontend/src/
  app/
    providers/
    routes/
  features/
    launch/
    live/
    memory/
    teamflow/
    judgment/
  infra/
    api/
      client.ts
      protocol.ts
  shared/
    ui/
    lib/
```

### 迁移策略

1. 先加边界再搬迁：先引入 ports/contracts，不立即移动大文件。
2. 引入转发层：旧路径文件保留 re-export，减少一次性改 import。
3. 分目录迁移：先 `mas/api`，再 `mas/core+roles/actions/tools`，最后 `frontend`。
4. 每次迁移只改一条调用链（一个 PR 一条主链）。
5. 删除过渡层延后一里程碑，稳定一轮后清理兼容导出。

### 命名与分层规则

1. 采用“领域 + 应用 + 基础设施”混合：`domain` 不依赖 `infrastructure`。
2. `application` 负责流程编排，不直接做 HTTP/DB/ES SDK 调用。
3. `infrastructure` 实现端口，不承载业务决策。
4. 禁止泛化 `utils` 垃圾桶：共享函数必须归类到 `shared/lib/<topic>`。
5. 反例：一个模块同时承担领域决策与工具调用器职责。

## 4) 分阶段里程碑（最多 4 个）

### 量化质量门槛

1. QG-1 后端测试：`python -m pytest tests -q` 全绿，且测试用例数 >= 20。
2. QG-2 前端 lint：`npm --prefix frontend run lint` 错误数 = 0。
3. QG-3 前端单测：`npm --prefix frontend run test:unit` 通过，测试文件数 >= 8。
4. QG-4 依赖环：核心层强连通分量中不再出现 `mas.core` 与 `roles/actions/tools` 同环。
5. QG-5 复杂度：`roles/controller.py` 中单函数长度 <= 120 行。
6. QG-6 构建性能：前端最大单 chunk < 900kB。
7. QG-7 安全扫描：`config/` 与源码目录无明文 `sk-` 密钥。

### Milestone 1（安全与防回归护栏）

- TODO：已按当前决策移除（RF-001、RF-003-a 不执行）
- 预期收益：先压住高风险泄漏与“无测试重构”风险。
- 验证步骤：执行 QG-1、QG-7。
- 回滚点：保留旧配置模板与原测试入口，失败时仅回滚新增校验脚本与测试改动。

### Milestone 2（核心依赖解耦）

- TODO：RF-002-a, RF-002-b, RF-105（RF-003-b 已完成）
- 预期收益：核心层依赖方向收敛，可测性提升。
- 验证步骤：执行 QG-1、QG-4、QG-5。
- 回滚点：保留 legacy factory 与 `_act_legacy` 开关，按模块回退。

### Milestone 3（API 与会话模块化）

- TODO：RF-102, RF-103, RF-104, RF-202, RF-206
- 预期收益：后端结构清晰，异常与日志可观测性提升。
- 验证步骤：执行 QG-1、QG-4、QG-7。
- 回滚点：`SessionManager` facade 保持旧签名，可逐服务回退。

### Milestone 4（前端结构与性能）

- TODO：RF-106, RF-203, RF-204, RF-205
- 预期收益：前端状态层更可测，包体下降，路由清晰。
- 验证步骤：执行 QG-2、QG-3、QG-6。
- 回滚点：保留 `AppLegacy` 与旧 hooks 适配层。

## 5) 批次进度

### Batch 1

- 已覆盖：`run.py`, `run_api.py`, `mas/core`, `roles`, `actions`, `tools`
- 下一批：`mas/api`, `mas/session`
- 当前覆盖率估计：43%
- 发现：核心层反向依赖上层编排，`_act` 超长。
- 证据：`mas/core/engine.py:15`, `mas/core/system.py:14`, `roles/controller.py:194`, `roles/controller.py:668`。
- 行动：RF-002-a、RF-002-b、RF-003-b、RF-105。
- 验证：导入图检查（SCC）+ 入口调用链核对。
- 剩余风险：缺少回归测试补强（RF-003-a 已按当前决策移除）。

### Batch 2

- 已覆盖：`mas/api`, `mas/session`
- 下一批：`frontend/src`
- 当前覆盖率估计：69%
- 发现：`server.py` 路由内联过多，`SessionManager`/`serializers` 过载，异常处理过粗。
- 证据：`mas/api/server.py:64`, `mas/api/server.py:90`, `mas/api/session_manager.py:159`, `mas/api/session_manager.py:625`, `mas/api/serializers.py:522`。
- 行动：RF-101、RF-201 已完成并从待办移除；后续为 RF-102、RF-103、RF-104、RF-202、RF-206。
- 验证：路由统计 25 条、后端测试基线执行完成。
- 剩余风险：模块拆分前接口稳定性需快照对比保障。

### Batch 3

- 已覆盖：`frontend/src`
- 下一批：`tests/scripts/config/data/docs`
- 当前覆盖率估计：86%
- 发现：状态 Hook 过重、路由手写、图组件重复、单测缺失、包体偏大。
- 证据：`frontend/src/app/state/useSnapshotActions.ts:1`, `frontend/src/App.tsx:12`, `frontend/src/app/components/ForceArgumentGraph.tsx:259`, `frontend/src/app/pages/MemoryPage.tsx:69`。
- 行动：RF-106、RF-203、RF-204、RF-205。
- 验证：已执行 lint/test:unit/build，确认 lint error、0 测试文件、chunk 告警。
- 剩余风险：前端重构依赖后端 payload 稳定（RF-103）。

### Batch 4

- 已覆盖：`tests`, `scripts`, `config`, `data`, `docs` 与运行目录
- 下一批：无
- 当前覆盖率估计：93%（忽略生成物/资产目录）
- 发现：测试入口过窄、明文密钥、运行产物目录膨胀。
- 证据：`pytest.ini:3`, `config/config2.yaml:3`, `scripts/start_dev.sh:10`。
- 行动：RF-001、RF-003-a 已按当前决策从待办移除；已完成 RF-003-b、RF-101、RF-201。
- 验证：后端基线 `1 passed`；前端 lint/test/build 基线已记录。
- 剩余风险：`bge-m3/`、`demo/` 资产目录较大，后续可做仓库瘦身治理。

## 6) 运行清单（供本地执行）

| 命令                                                                      | 目的                                                                                    | 期望输出 / 替代                                |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------- | ----------------- | ------------------------------------- |
| `python -m pytest tests -q`                                               | 后端回归基线                                                                            | 全绿；当前基线为 `1 passed`                    |
| `npm --prefix frontend run lint`                                          | 前端静态检查                                                                            | 0 error；当前有 `MemoryPage.tsx:69`            |
| `npm --prefix frontend run test:unit`                                     | 前端单测基线                                                                            | 通过且有测试；当前为 0 测试文件                |
| `npm --prefix frontend run build`                                         | 类型检查+打包性能                                                                       | 成功且最大 chunk 下降（目标 `<900kB`）         |
| `rg -n "^\\s\*@app\\.(get                                                 | post                                                                                    | websocket)\\(" mas/api/server.py`              | 统计 API 路由数量 | 数量稳定（当前 25）                   |
| `rg -n "except Exception                                                  | except Exception as" mas/api/server.py mas/session/snapshot_store.py mas/core/graph.py` | 异常粗粒度扫描                                 | 数量逐步下降      |
| `rg -n "print\\(                                                          | console.warn\\(" tools mas frontend/src/app/state/errorUtils.ts`                        | 日志出口统一性扫描                             | 生产路径逐步清零  |
| `rg -n "SystemConfig\\(\\)" tools mas/analysis mas/core`                  | 配置注入收敛检查                                                                        | 调用点收敛到装配入口                           |
| `rg -n "sk-[A-Za-z0-9]{20,}" config mas frontend`                         | 明文密钥扫描                                                                            | 应为空；误报可用 `gitleaks detect`             |
| `python <SCC-检查脚本>`                                                   | 验证核心依赖环是否打破                                                                  | `mas.core` 不再与 `roles/actions/tools` 同 SCC |
| `rg --files frontend/src \| rg "\\.test\\.(ts                             | tsx)$" \| wc -l`                                                                        | 前端测试文件计数                               | 目标 >= 8         |
| `bash -lc 'command -v lizard >/dev/null && lizard mas roles actions tools |                                                                                         | echo "lizard not installed"'`                  | 复杂度/长函数检查 | 若无 lizard，可用 `rg` + 手工阈值替代 |

---
