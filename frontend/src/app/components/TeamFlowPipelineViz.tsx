import { useMemo } from "react";
import type { TeamFlowMessage, TeamFlowTurn } from "../../compat";

interface TeamFlowPipelineVizProps {
  turn: TeamFlowTurn | null;
}

const ORDERED_PHASES: TeamFlowMessage["phase"][] = [
  "ASSESS",
  "INSTRUCT",
  "WORKER",
  "DECIDE",
  "NARRATE",
];

const PHASE_LABEL: Record<TeamFlowMessage["phase"], string> = {
  ASSESS: "需求评估",
  INSTRUCT: "任务下发",
  WORKER: "执行反馈",
  DECIDE: "决策执行",
  RETRY: "重试纠偏",
  NARRATE: "叙事整理",
  SYSTEM: "系统消息",
};

const STATUS_LABEL: Record<TeamFlowTurn["status"], string> = {
  done: "已完成",
  retry: "重试后完成",
  partial: "部分完成",
};

export function TeamFlowPipelineViz({ turn }: TeamFlowPipelineVizProps) {
  const activePhases = useMemo(() => {
    const bucket = new Set<TeamFlowMessage["phase"]>();

    if (!turn) {
      return bucket;
    }

    for (const message of turn.messages) {
      bucket.add(message.phase);
    }

    return bucket;
  }, [turn]);

  const actorRows = useMemo(() => {
    if (!turn) {
      return [];
    }

    const bucket = new Map<string, number>();

    for (const message of turn.messages) {
      const actor = message.actor || "unknown";
      bucket.set(actor, (bucket.get(actor) ?? 0) + 1);
    }

    return [...bucket.entries()].sort((a, b) => b[1] - a[1]);
  }, [turn]);

  const maxCount = actorRows.length > 0 ? actorRows[0][1] : 1;

  if (!turn) {
    return <p className="ux-empty">暂无可视化数据。</p>;
  }

  return (
    <div className="ux-teamflow-viz">
      <div className="ux-teamflow-kpis">
        <span>状态：{STATUS_LABEL[turn.status]}</span>
        <span>消息数：{turn.messageCount}</span>
        <span>Worker数：{turn.workerCount}</span>
        <span>重试：{turn.retryCount}</span>
      </div>

      <div className="ux-teamflow-pipeline">
        {ORDERED_PHASES.map((phase, index) => {
          const isActive = activePhases.has(phase);

          return (
            <div
              className={`ux-teamflow-phase-step ${isActive ? "ux-teamflow-phase-step-active" : ""}`}
              key={phase}
            >
              <span className="ux-teamflow-phase-index">{index + 1}</span>
              <span>{PHASE_LABEL[phase]}</span>
            </div>
          );
        })}
      </div>

      {activePhases.has("RETRY") ? (
        <p className="ux-teamflow-retry-note">
          本回合出现重试，控制器进行了纠偏重跑。
        </p>
      ) : null}

      <div className="ux-teamflow-actor-chart">
        {actorRows.map(([actor, count]) => {
          const width = Math.max(8, Math.round((count / maxCount) * 100));

          return (
            <div className="ux-teamflow-actor-row" key={actor}>
              <span className="ux-teamflow-actor-name">{actor}</span>

              <div className="ux-teamflow-actor-track">
                <div
                  className="ux-teamflow-actor-fill"
                  style={{ width: `${width}%` }}
                />
              </div>

              <strong>{count}</strong>
            </div>
          );
        })}
      </div>
    </div>
  );
}
