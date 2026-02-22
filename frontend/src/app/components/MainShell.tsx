import type { ReactNode } from "react";
import { useDebate } from "../state/useDebate";
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

export function MainShell({ route, onNavigate, children }: MainShellProps) {
  const { snapshot, sessions, busyAction, error, resetMemory } = useDebate();
  const currentPhase = snapshot?.phase ?? "idle";
  const currentStep = phaseIndex(currentPhase);
  const hasActiveSessions = sessions.length > 0;

  const handleResetMemory = (): void => {
    const confirmed = window.confirm(
      "将删除磁盘上的长期记忆文件（案例、洞察、拓扑）。仅允许在无活动会话时执行，且不可撤销，是否继续？",
    );

    if (!confirmed) {
      return;
    }

    void resetMemory();
  };

  return (
    <main className="ux-shell">
      <header className="ux-topbar">
        <div>
          <p className="ux-eyebrow">Legal Debate Assistant</p>
          <h1>可解释法律辩论</h1>
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
          className="ux-nav-danger"
          disabled={hasActiveSessions || Boolean(busyAction)}
          onClick={handleResetMemory}
          title={
            hasActiveSessions ? "请先关闭所有会话后再清理磁盘长期记忆" : ""
          }
          type="button"
        >
          调试：清空长期记忆
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
