import { useMemo } from "react";
import { useDebate } from "../state/useDebate";
import { asRecord, unwrapPayload } from "../../shared/lib/payload";

export function JudgmentPage() {
  const { sessionId, snapshot } = useDebate();

  const payload = useMemo(
    () => unwrapPayload(snapshot?.raw ?? {}),
    [snapshot?.raw],
  );

  const rootClaimStatusMap = asRecord(payload.root_claims_status);
  const rootClaimEntries = Object.entries(rootClaimStatusMap);

  const validatedCount = rootClaimEntries.filter(([, value]) =>
    String(value).toUpperCase().includes("VALID"),
  ).length;

  const hasSession = Boolean(sessionId && snapshot);
  const isFinished = snapshot?.phase === "finished";
  const currentRound = snapshot?.round ?? 0;

  return (
    <section className="ux-grid ux-grid-2">
      <article className="ux-card">
        <h2>判决总览</h2>

        {!hasSession ? (
          <p className="ux-empty">请先完成会话创建并推进庭审。</p>
        ) : (
          <div className="ux-kv">
            <p>
              <span>裁决状态</span>
              <strong>{isFinished ? "已裁决" : "未裁决"}</strong>
            </p>

            {isFinished ? (
              <p>
                <span>根主张采纳</span>
                <strong>
                  {validatedCount}/{rootClaimEntries.length || 0}
                </strong>
              </p>
            ) : null}

            <p>
              <span>当前回合</span>
              <strong>{currentRound}</strong>
            </p>
          </div>
        )}
      </article>

      <article className="ux-card">
        <h2>裁决详情</h2>

        {!hasSession ? (
          <p className="ux-empty">请先完成会话创建并推进庭审。</p>
        ) : rootClaimEntries.length > 0 ? (
          <details open>
            <summary>展开根主张裁决</summary>

            <div className="ux-log-box">
              {rootClaimEntries.map(([claimId, status]) => (
                <p className="ux-document" key={claimId}>
                  {claimId}: {String(status)}
                </p>
              ))}
            </div>
          </details>
        ) : (
          <p className="ux-empty">尚未生成裁决结果。请先完成裁决。</p>
        )}
      </article>
    </section>
  );
}
