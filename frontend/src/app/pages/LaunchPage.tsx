import { useState } from "react";

import { useDebate } from "../state/useDebate";
import { phaseLabel } from "../utils/payload";

interface LaunchPageProps {
  onGoLive: () => void;
}

export function LaunchPage({ onGoLive }: LaunchPageProps) {
  const { sessions, snapshot, createSession, selectSession, busyAction } =
    useDebate();

  const [maxRounds, setMaxRounds] = useState<number>(6);

  const handleCreate = async (): Promise<void> => {
    const ok = await createSession(Math.max(1, Math.floor(maxRounds)));

    if (ok) {
      onGoLive();
    }
  };

  const handleContinue = async (sessionId: string): Promise<void> => {
    const ok = await selectSession(sessionId);

    if (ok) {
      onGoLive();
    }
  };

  return (
    <section className="ux-grid ux-grid-2">
      <article className="ux-card">
        <h2>开始一次新庭审</h2>
        <p className="ux-muted">
          默认使用收敛指标自动终止辩论；下方回合数只是兜底安全上限。
        </p>

        <label className="ux-field">
          安全上限回合
          <input
            max={20}
            min={1}
            onChange={(event) => setMaxRounds(Number(event.target.value))}
            type="number"
            value={maxRounds}
          />
        </label>
        <p className="ux-muted">
          正常情况下系统会在“收敛阈值达成”后进入裁决，无需跑满上限回合。
        </p>

        <button
          disabled={Boolean(busyAction)}
          onClick={() => {
            void handleCreate();
          }}
          type="button"
        >
          开始庭审
        </button>
      </article>

      <article className="ux-card">
        <h2>当前会话概览</h2>
        {snapshot ? (
          <div className="ux-kv">
            <p>
              <span>会话</span>
              <strong>{snapshot.sessionId}</strong>
            </p>
            <p>
              <span>阶段</span>
              <strong>{phaseLabel(snapshot.phase)}</strong>
            </p>
            <p>
              <span>回合信息</span>
              <strong>
                {snapshot.round}（安全上限 {snapshot.maxRounds}）
              </strong>
            </p>
            <p>
              <span>图谱</span>
              <strong>
                N{snapshot.metrics.arguments} / E
                {snapshot.metrics.attacks + snapshot.metrics.supports}
              </strong>
            </p>
          </div>
        ) : (
          <p className="ux-empty">尚未创建会话。</p>
        )}
      </article>

      <article className="ux-card ux-card-full">
        <h2>历史会话</h2>
        {sessions.length > 0 ? (
          <div className="ux-list">
            {sessions.map((item) => (
              <button
                className="ux-list-row"
                key={item.sessionId}
                onClick={() => {
                  void handleContinue(item.sessionId);
                }}
                type="button"
              >
                <span>{item.sessionId}</span>
                <span>
                  {item.round} / 上限{item.maxRounds}
                </span>
                <span>{phaseLabel(item.phase)}</span>
              </button>
            ))}
          </div>
        ) : (
          <p className="ux-empty">暂无历史会话。</p>
        )}
      </article>
    </section>
  );
}
