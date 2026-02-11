import { useMemo, useState } from "react";
import { useDebate } from "../state/useDebate";
import type { AppRoute } from "../types";

interface PlaybookStep {
  id: string;
  title: string;
  evidence: string;
  route: AppRoute;
}

const STEPS: PlaybookStep[] = [
  {
    id: "launch",
    title: "1. 启动会话",
    evidence: "看到会话 ID、阶段=初始化/辩论中。",
    route: "/app/launch",
  },
  {
    id: "step3",
    title: "2. 推进至少 3 回合",
    evidence: "回合数递增，庭审对话持续追加。",
    route: "/app/live",
  },
  {
    id: "diff",
    title: "3. 展示图差异",
    evidence: "可见新增/移除/状态变化节点。",
    route: "/app/graph",
  },
  {
    id: "team",
    title: "4. 展示团队协作与重试链",
    evidence: "流程图+turn 工件+重试信息可见。",
    route: "/app/team",
  },
  {
    id: "memory",
    title: "5. 展示记忆与类比",
    evidence: "TaskLayer 案例关系图可见，洞察可跳转回合。",
    route: "/app/memory",
  },
  {
    id: "judgment",
    title: "6. 展示裁决解释",
    evidence: "文书、根主张状态、BAF 图一致呈现。",
    route: "/app/judgment",
  },
  {
    id: "replay",
    title: "7. 回放并导出",
    evidence: "关键帧联动、Replay JSON/GEXF 导出摘要可见。",
    route: "/app/replay",
  },
];

function navigate(to: AppRoute): void {
  if (window.location.pathname !== to) {
    window.history.pushState({}, "", to);
    window.dispatchEvent(new PopStateEvent("popstate"));
  }
}

export function DemoPlaybookPage() {
  const { snapshot, sessionId } = useDebate();
  const [doneIds, setDoneIds] = useState<string[]>([]);
  const doneSet = useMemo(() => new Set(doneIds), [doneIds]);

  return (
    <section className="ux-grid ux-grid-2">
      <article className="ux-card ux-card-full">
        <h2>演示剧本（5-8 分钟）</h2>

        <p className="ux-muted">
          当前会话：{sessionId || "未创建"} · 当前回合：
          {snapshot ? `r${snapshot.round}` : "-"}
        </p>

        <div className="ux-list">
          {STEPS.map((step) => {
            const checked = doneSet.has(step.id);

            return (
              <div className="ux-list-row ux-list-row-static" key={step.id}>
                <span>{step.title}</span>
                <span>{step.evidence}</span>
                <span>
                  <button
                    onClick={() => navigate(step.route)}
                    style={{ marginRight: 8 }}
                    type="button"
                  >
                    跳转
                  </button>

                  <button
                    onClick={() =>
                      setDoneIds((prev) =>
                        checked
                          ? prev.filter((item) => item !== step.id)
                          : [...prev, step.id],
                      )
                    }
                    type="button"
                  >
                    {checked ? "取消完成" : "标记完成"}
                  </button>
                </span>
              </div>
            );
          })}
        </div>
      </article>
    </section>
  );
}
