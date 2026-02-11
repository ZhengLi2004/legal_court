import type { ReactNode } from "react";
import { useDebate } from "../state/useDebate";
import { phaseLabel } from "../utils/payload";
import type { AppRoute } from "../types";

interface MainShellProps {
  route: AppRoute;
  onNavigate: (to: AppRoute) => void;
  children: ReactNode;
}

interface PhaseStep {
  key: string;
  label: string;
}

const PHASE_STEPS: PhaseStep[] = [
  { key: "setup", label: "初始化" },
  { key: "running", label: "辩论中" },
  { key: "ready", label: "待裁决" },
  { key: "finished", label: "已裁决" },
];

function phaseIndex(phase: string): number {
  if (phase === "finished") {
    return 3;
  }

  if (phase === "ready_for_adjudication") {
    return 2;
  }

  if (phase === "running") {
    return 1;
  }

  return 0;
}

function terminationLabel(reason: string): string {
  if (reason === "convergence") {
    return "收敛";
  }

  return "未触发";
}

export function MainShell({ route, onNavigate, children }: MainShellProps) {
  const { snapshot, sessionId, adapterMode, streamStatus, busyAction, error } =
    useDebate();

  const currentPhase = snapshot?.phase ?? "idle";
  const currentStep = phaseIndex(currentPhase);

  return (
    <main className="ux-shell">
      <header className="ux-topbar">
        <div>
          <p className="ux-eyebrow">Legal Debate Assistant</p>
          <h1>可解释法律辩论</h1>

          <p className="ux-muted">
            当前状态：{phaseLabel(currentPhase)} · 会话{" "}
            {sessionId ? sessionId : "未创建"}
          </p>
        </div>

        <div className="ux-meta">
          <span>模式：{adapterMode}</span>
          <span>实时：{streamStatus}</span>

          <span>回合：{snapshot ? `r${snapshot.round}` : "-"}</span>

          <span>
            终止：
            {snapshot
              ? snapshot.termination.ready
                ? terminationLabel(snapshot.termination.reason)
                : "未触发"
              : "-"}
          </span>
        </div>
      </header>

      <section className="ux-stepper">
        {PHASE_STEPS.map((step, index) => (
          <div
            className={`ux-step ${index <= currentStep ? "ux-step-active" : ""}`}
            key={step.key}
          >
            <span className="ux-step-index">{index + 1}</span>
            <span>{step.label}</span>
          </div>
        ))}
      </section>

      <nav className="ux-nav">
        <button
          className={route === "/app/launch" ? "ux-nav-active" : ""}
          onClick={() => onNavigate("/app/launch")}
          type="button"
        >
          案件启动
        </button>

        <button
          className={route === "/app/live" ? "ux-nav-active" : ""}
          onClick={() => onNavigate("/app/live")}
          type="button"
        >
          实时庭审
        </button>

        <button
          className={route === "/app/judgment" ? "ux-nav-active" : ""}
          onClick={() => onNavigate("/app/judgment")}
          type="button"
        >
          裁决解释
        </button>

        <button
          className={route === "/app/graph" ? "ux-nav-active" : ""}
          onClick={() => onNavigate("/app/graph")}
          type="button"
        >
          论证图谱
        </button>

        <button
          className={route === "/app/team" ? "ux-nav-active" : ""}
          onClick={() => onNavigate("/app/team")}
          type="button"
        >
          团队协作
        </button>

        <button
          className={route === "/app/memory" ? "ux-nav-active" : ""}
          onClick={() => onNavigate("/app/memory")}
          type="button"
        >
          记忆类比
        </button>

        <button
          className={route === "/app/replay" ? "ux-nav-active" : ""}
          onClick={() => onNavigate("/app/replay")}
          type="button"
        >
          回放导出
        </button>

        <button onClick={() => onNavigate("/admin/debug")} type="button">
          进入后台调试
        </button>
      </nav>

      {busyAction ? (
        <p className="ux-banner ux-banner-info">执行中：{busyAction}</p>
      ) : null}

      {error ? (
        <p className="ux-banner ux-banner-error">错误：{error}</p>
      ) : null}

      <section className="ux-content">{children}</section>
    </main>
  );
}
