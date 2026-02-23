# 可执行重构 TODO List（仅保留未完成项）

## 1) 仓库概览

仓库为 Python 后端 + React 前端单仓结构。当前后端主线重构已落地，剩余主要风险集中在前端装配层：
1. 路由逻辑仍在 `frontend/src/App.tsx` 手写维护，页面装配与导航状态耦合。
2. 前端构建主 chunk 体积仍偏大（当前约 `1,469.76 kB`），首屏加载与缓存效率受限。
3. 前端测试规模仍偏小（当前 `1` 个测试文件、`3` 个用例），对后续结构调整的回归保障不足。

## 2) TODO List（按 P0/P1/P2）

### P2

#### [RF-205]

- [优先级] P2
- [类别] Folder Structure, Performance
- [证据] `frontend/src/App.tsx:12` 手写 `normalizeRoute`；`frontend/src/App.tsx:62` 手动 `history.pushState`；`npm --prefix frontend run build` 显示主 chunk 约 `1,469.76 kB`。
- [症状] 路由逻辑与页面装配耦合，页面未按路由边界懒加载。
- [影响] 可维护性一般，首屏包体偏大，后续页面演进会继续推高主包体积。
- [建议改法] 引入路由层（React Router 或等价轻量方案）并改为页面级动态导入。保持行为不变策略：URL 与页面映射、现有导航行为、回退/前进语义保持一致。
- [变更范围] `frontend/src/App.tsx`, `frontend/src/app/pages/`, `frontend/src/app/components/MainShell.tsx`。
- [验收标准] 
  1. 主 chunk 降至 `< 900kB`。
  2. 路由行为回归通过（启动页/直播页/协作页/记忆页/裁判页切换正确）。
  3. `npm --prefix frontend run lint`、`npm --prefix frontend run test:unit`、`npm --prefix frontend run build` 全通过。
- [工作量] M
- [风险与回滚] 风险：浏览器前进/后退时序变化。回滚策略：按“路由层接入提交 / 页面懒加载提交”拆分提交，异常时按提交粒度回滚。
- [依赖/前置] 无。

## 3) 文件夹结构专项建议

### 目标目录结构草案（针对前端剩余工作）

```text
frontend/src/
  app/
    routes/
      index.tsx
      routeConfig.ts
    layout/
      MainShell.tsx
  features/
    launch/
      LaunchPage.tsx
    live/
      LivePage.tsx
    teamflow/
      TeamFlowPage.tsx
    memory/
      MemoryPage.tsx
    judgment/
      JudgmentPage.tsx
  shared/
    lib/
      payload.ts
    ui/
```

### 迁移策略

1. 先引入路由配置层，再逐页迁移，不做一次性大搬迁。
2. 每迁移 1 个页面就跑一次 `lint + test + build`，保证可回滚。
3. 页面拆分为动态导入后，再做主包体积复测，避免混合变量过多。

### 命名与分层规则

1. `app/routes` 仅负责路由注册与导航编排，不承载业务渲染逻辑。
2. 页面按 `features/<domain>/<Page>.tsx` 组织，禁止继续向 `App.tsx` 堆叠页面条件分支。
3. `shared/lib` 只放跨 feature 的纯函数工具，避免重新引入 `utils` 大杂烩。

## 4) 分阶段里程碑

### 量化质量门槛

1. QG-1：`npm --prefix frontend run lint` 错误数 = 0。
2. QG-2：`npm --prefix frontend run test:unit` 全通过。
3. QG-3：`npm --prefix frontend run build` 成功。
4. QG-4：主 chunk `< 900kB`。
5. QG-5：路由回归检查覆盖 5 个主页面跳转与浏览器前进/后退。

### Milestone 1（前端路由与包体治理）

- TODO：RF-205
- 预期收益：路由职责清晰、页面装配解耦、首屏包体下降。
- 验证步骤：执行 QG-1~QG-5。
- 回滚点：
  1. 路由层接入提交。
  2. 页面懒加载提交。

## 5) 批次进度

### Batch 1（当前待执行）

- 已覆盖：`frontend/src/App.tsx`, `frontend/src/app/pages/*`, `frontend/src/app/components/MainShell.tsx`
- 下一批：无（完成 RF-205 后关闭当前 TODO）
- 当前覆盖率估计：93%
- 发现：路由装配耦合与主包体积偏大是当前主要未完项。
- 行动：执行 RF-205。
- 验证：以 `lint/test/build + 主 chunk 体积 + 路由回归` 为准。
- 剩余风险：若仅做懒加载不做路由层整理，后续可维护性改善有限。

## 6) 运行清单（供本地执行）

1. `npm --prefix frontend run lint`  
   目的：前端静态检查。  
   期望：0 error。

2. `npm --prefix frontend run test:unit`  
   目的：前端单测回归。  
   期望：全部通过。

3. `npm --prefix frontend run build`  
   目的：类型检查 + 产物构建。  
   期望：构建成功，最大 chunk 下降至目标阈值。

4. `rg -n "normalizeRoute|history.pushState|popstate" frontend/src/App.tsx`  
   目的：检查是否仍保留手写路由核心逻辑。  
   期望：迁移后显著减少或移入路由层文件。

5. `ls -lh frontend/dist/assets/*.js`  
   目的：确认构建后主包体积变化。  
   期望：最大主包满足 `< 900kB`。
