import { useMemo, useState } from "react";

import { ForceArgumentGraph } from "../components/ForceArgumentGraph";
import { useDebate } from "../state/useDebate";
import { asRecord, phaseLabel, unwrapPayload } from "../utils/payload";

import type { TimelineEvent } from "../../compat";

function canAdjudicate(phase: string): boolean {
  return phase === "ready_for_adjudication" || phase === "finished";
}

function terminationReasonLabel(reason: string): string {
  if (reason === "convergence") {
    return "收敛阈值达成";
  }

  if (reason === "max_rounds") {
    return "达到安全上限";
  }

  return "待确认";
}

function timelineReason(row: TimelineEvent): string {
  const data = asRecord(row.data);
  const rawReason = String(data.reason ?? data.message ?? "");

  if (!rawReason) {
    return "";
  }

  const lower = rawReason.toLowerCase();

  if (lower.includes("convergence")) {
    return "收敛阈值达成";
  }

  if (lower.includes("max")) {
    return "达到安全上限";
  }

  return rawReason;
}

export function LivePage() {
  const {
    sessionId,
    snapshot,
    previousSnapshot,
    graphView,
    timeline,
    busyAction,
    step,
    adjudicate,
    refreshSnapshot,
    loadTimeline,
    loadGraphAtRound,
  } = useDebate();

  const [selectedSeq, setSelectedSeq] = useState<number>(0);

  const transcriptDelta = useMemo(() => {
    if (!snapshot) {
      return [];
    }

    if (!previousSnapshot || previousSnapshot.sessionId !== snapshot.sessionId) {
      return snapshot.transcript;
    }

    return snapshot.transcript.slice(previousSnapshot.transcript.length);
  }, [snapshot, previousSnapshot]);

  const rootClaimEntries = useMemo(() => {
    if (!snapshot) {
      return [];
    }

    const payload = unwrapPayload(snapshot.raw ?? {});
    return Object.entries(asRecord(payload.root_claims_status));
  }, [snapshot]);

  const validatedCount = useMemo(
    () =>
      rootClaimEntries.filter(([, status]) =>
        String(status).toUpperCase().includes("VALID"),
      ).length,
    [rootClaimEntries],
  );

  const visibleTimeline = useMemo(
    () => [...timeline].slice(-30).reverse(),
    [timeline],
  );

  const convergenceHistoryText = useMemo(() => {
    const recent = snapshot?.convergence.history.slice(-6) ?? [];

    if (recent.length === 0) {
      return "暂无收敛历史。";
    }

    return recent.map((item) => item.toFixed(2)).join(" -> ");
  }, [snapshot?.convergence.history]);

  const convergenceStateText = useMemo(() => {
    if (!snapshot) {
      return "未知";
    }

    if (snapshot.phase === "finished") {
      return `已进入裁决（${terminationReasonLabel(snapshot.termination.reason)}）`;
    }

    if (snapshot.termination.ready) {
      return `可裁决（${terminationReasonLabel(snapshot.termination.reason)}）`;
    }

    return "收敛推进中";
  }, [snapshot]);

  const onTimelineClick = async (row: TimelineEvent): Promise<void> => {
    setSelectedSeq(row.seq);

    if (typeof row.roundIdx === "number") {
      await loadGraphAtRound(row.roundIdx);
    }
  };

  if (!sessionId || !snapshot) {
    return (
      <article className="ux-card">
        <h2>实时庭审</h2>
        <p className="ux-empty">请先在“案件启动”页创建或选择一个会话。</p>
      </article>
    );
  }

  return (
    <section className="ux-grid ux-grid-2">
      <article className="ux-card">
        <h2>庭审结论总览</h2>
        <div className="ux-kv">
          <p>
            <span>阶段</span>
            <strong>{phaseLabel(snapshot.phase)}</strong>
          </p>
          <p>
            <span>收敛状态</span>
            <strong>{convergenceStateText}</strong>
          </p>
          <p>
            <span>ΔΦ</span>
            <strong>{snapshot.convergence.deltaPhi.toFixed(3)}</strong>
          </p>
          <p>
            <span>SMA</span>
            <strong>{snapshot.convergence.sma.toFixed(3)}</strong>
          </p>
          <p>
            <span>收敛阈值</span>
            <strong>
              SMA &lt; {snapshot.convergence.epsilon.toFixed(2)} 且回合 &gt;=
              {" "}
              {snapshot.convergence.minRounds}
            </strong>
          </p>
          <p>
            <span>回合信息</span>
            <strong>
              {snapshot.round}（安全上限 {snapshot.maxRounds}）
            </strong>
          </p>
          <p>
            <span>论点数</span>
            <strong>{snapshot.metrics.arguments}</strong>
          </p>
          <p>
            <span>冲突边</span>
            <strong>{snapshot.metrics.attacks}</strong>
          </p>
          <p>
            <span>支持边</span>
            <strong>{snapshot.metrics.supports}</strong>
          </p>
          <p>
            <span>根主张采纳</span>
            <strong>
              {validatedCount}/{rootClaimEntries.length || 0}
            </strong>
          </p>
        </div>
        <p className="ux-muted">收敛轨迹：{convergenceHistoryText}</p>

        <div className="ux-row">
          <button
            disabled={Boolean(busyAction)}
            onClick={() => {
              void step();
            }}
            type="button"
          >
            下一步辩论
          </button>
          <button
            disabled={Boolean(busyAction) || !canAdjudicate(snapshot.phase)}
            onClick={() => {
              void adjudicate();
            }}
            type="button"
          >
            发起裁决
          </button>
          <button
            disabled={Boolean(busyAction)}
            onClick={() => {
              void refreshSnapshot();
            }}
            type="button"
          >
            刷新
          </button>
        </div>
      </article>

      <article className="ux-card">
        <h2>事件流</h2>
        <div className="ux-row">
          <button
            disabled={Boolean(busyAction)}
            onClick={() => {
              void loadTimeline();
            }}
            type="button"
          >
            刷新事件
          </button>
        </div>

        {visibleTimeline.length > 0 ? (
          <div className="ux-list">
            {visibleTimeline.map((row) => (
              <button
                className={`ux-list-row ${row.seq === selectedSeq ? "ux-list-row-selected" : ""}`}
                key={`${row.seq}-${row.event}`}
                onClick={() => {
                  void onTimelineClick(row);
                }}
                type="button"
              >
                <span>#{row.seq}</span>
                <span>
                  {row.event}
                  {timelineReason(row) ? ` · ${timelineReason(row)}` : ""}
                </span>
                <span>r{row.roundIdx ?? "-"}</span>
              </button>
            ))}
          </div>
        ) : (
          <p className="ux-empty">暂无事件。</p>
        )}
      </article>

      <article className="ux-card ux-card-full">
        <ForceArgumentGraph graph={graphView} title="论证图（力导布局）" />
      </article>

      <article className="ux-card ux-card-full">
        <details>
          <summary>查看本轮新增庭审陈述</summary>
          {transcriptDelta.length > 0 ? (
            <div className="ux-log-box">
              {transcriptDelta.map((line, idx) => (
                <p className="ux-log-line" key={`${line}-${idx}`}>
                  + {line}
                </p>
              ))}
            </div>
          ) : (
            <p className="ux-empty">本轮暂无新增陈述。</p>
          )}
        </details>
      </article>
    </section>
  );
}

