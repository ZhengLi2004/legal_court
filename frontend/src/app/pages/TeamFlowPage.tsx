import { useEffect, useMemo, useState } from "react";
import { TeamFlowConversation } from "../components/TeamFlowConversation";
import { useDebate } from "../state/useDebate";

function sideLabel(side: string): string {
  const value = side.toLowerCase();

  if (value.includes("plaintiff")) {
    return "原告";
  }

  if (value.includes("defendant")) {
    return "被告";
  }

  return "未知";
}

function statusLabel(status: string): string {
  if (status === "done") {
    return "完成";
  }

  if (status === "retry") {
    return "重试";
  }

  return "部分";
}

export function TeamFlowPage() {
  const {
    sessionId,
    snapshot,
    teamflowStream,
    busyAction,
    loadTeamflowStream,
  } = useDebate();

  const [selectedTurnUid, setSelectedTurnUid] = useState<string>("");

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void loadTeamflowStream(80);
  }, [loadTeamflowStream, sessionId]);

  const orderedTurns = useMemo(
    () => [...teamflowStream].reverse(),
    [teamflowStream],
  );

  const effectiveTurnUid =
    selectedTurnUid &&
    orderedTurns.some((item) => item.turnUid === selectedTurnUid)
      ? selectedTurnUid
      : (orderedTurns[0]?.turnUid ?? "");

  const selectedTurn = useMemo(
    () =>
      orderedTurns.find((item) => item.turnUid === effectiveTurnUid) ??
      orderedTurns[0] ??
      null,
    [effectiveTurnUid, orderedTurns],
  );

  const retryTurns = useMemo(
    () => teamflowStream.filter((item) => item.retryCount > 0).length,
    [teamflowStream],
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
            <span>协作线程</span>
            <strong>{teamflowStream.length}</strong>
          </p>

          <p>
            <span>含重试线程</span>
            <strong>{retryTurns}</strong>
          </p>

          <p>
            <span>当前查看</span>
            <strong>{selectedTurn?.turnUid ?? "-"}</strong>
          </p>
        </div>

        <div className="ux-row">
          <button
            disabled={Boolean(busyAction)}
            onClick={() => {
              void loadTeamflowStream(80);
            }}
            type="button"
          >
            刷新协作流
          </button>
        </div>
      </article>

      <article className="ux-card">
        <h2>回合线程</h2>

        {orderedTurns.length > 0 ? (
          <div className="ux-teamflow-turn-list">
            {orderedTurns.map((item) => (
              <button
                className={`ux-teamflow-turn-row ${item.turnUid === effectiveTurnUid ? "ux-teamflow-turn-row-active" : ""}`}
                key={item.turnUid}
                onClick={() => setSelectedTurnUid(item.turnUid)}
                type="button"
              >
                <span>r{item.round}</span>
                <span>{sideLabel(item.side)}</span>
                <span>{statusLabel(item.status)}</span>
                <span>msg {item.messageCount}</span>
              </button>
            ))}
          </div>
        ) : (
          <p className="ux-empty">暂无协作线程。</p>
        )}
      </article>

      <article className="ux-card ux-card-full">
        <h2>协作对话流</h2>
        <TeamFlowConversation turn={selectedTurn} />
      </article>
    </section>
  );
}
