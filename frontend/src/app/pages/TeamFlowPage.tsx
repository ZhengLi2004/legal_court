import { useEffect, useMemo, useState } from "react";
import { InspectorPanel, TeamFlowPanel } from "../../components/debug";
import { ControllerPipelineGraph } from "../components/ControllerPipelineGraph";
import { useDebate } from "../state/useDebate";
import type { TurnArtifact } from "../../compat";

function hasRetry(artifact: TurnArtifact): boolean {
  return (
    Array.isArray(artifact.retryHistory) && artifact.retryHistory.length > 0
  );
}

export function TeamFlowPage() {
  const { sessionId, snapshot, turnArtifacts, busyAction, loadTurnArtifacts } =
    useDebate();

  const [selectedTurnUid, setSelectedTurnUid] = useState<string>("");

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void loadTurnArtifacts({ limit: 80 });
  }, [loadTurnArtifacts, sessionId]);

  const effectiveTurnUid =
    selectedTurnUid &&
    turnArtifacts.some((item) => item.turnUid === selectedTurnUid)
      ? selectedTurnUid
      : (turnArtifacts[turnArtifacts.length - 1]?.turnUid ?? "");

  const selectedArtifact = useMemo(
    () =>
      turnArtifacts.find((item) => item.turnUid === effectiveTurnUid) ??
      turnArtifacts[turnArtifacts.length - 1] ??
      null,
    [effectiveTurnUid, turnArtifacts],
  );

  const retryCount = useMemo(
    () => turnArtifacts.filter((item) => hasRetry(item)).length,
    [turnArtifacts],
  );

  if (!sessionId || !snapshot) {
    return (
      <article className="ux-card">
        <h2>团队协作</h2>
        <p className="ux-empty">请先在“案件启动”页创建或选择一个会话。</p>
      </article>
    );
  }

  return (
    <section className="ux-grid ux-grid-2">
      <article className="ux-card">
        <h2>协作总览</h2>

        <div className="ux-kv">
          <p>
            <span>当前回合</span>
            <strong>r{snapshot.round}</strong>
          </p>

          <p>
            <span>工件条数</span>
            <strong>{turnArtifacts.length}</strong>
          </p>

          <p>
            <span>含重试回合</span>
            <strong>{retryCount}</strong>
          </p>

          <p>
            <span>当前查看</span>
            <strong>{selectedArtifact?.turnUid ?? "-"}</strong>
          </p>
        </div>

        <div className="ux-row">
          <button
            disabled={Boolean(busyAction)}
            onClick={() => {
              void loadTurnArtifacts({ limit: 80 });
            }}
            type="button"
          >
            刷新协作工件
          </button>
        </div>
      </article>

      <article className="ux-card">
        <h2>流程图（状态机）</h2>

        <p className="ux-muted">
          节点：ASSESS / DELEGATE / WAIT / DECIDE / RETRY /
          DONE。红色链路表示重试或回滚路径。
        </p>

        <ControllerPipelineGraph artifact={selectedArtifact} />
      </article>

      <TeamFlowPanel
        artifacts={turnArtifacts}
        onSelectTurn={setSelectedTurnUid}
        selectedTurnUid={effectiveTurnUid}
      />

      <InspectorPanel artifact={selectedArtifact} snapshot={snapshot} />
    </section>
  );
}
