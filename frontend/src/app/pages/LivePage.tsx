import { useEffect, useMemo, useRef } from "react";
import { ForceArgumentGraph } from "../components/ForceArgumentGraph";
import { ConvergenceSparkline } from "../components/ConvergenceSparkline";
import { useDebate } from "../state/useDebate";
import { asRecord, phaseLabel, unwrapPayload } from "../utils/payload";
type DialogueSide = "plaintiff" | "defendant" | "judge" | "system" | "other";

interface DialogueRow {
  idx: number;
  side: DialogueSide;
  speakerLabel: string;
  text: string;
  isNew: boolean;
}

function canStepDebate(phase: string, isConverged: boolean): boolean {
  if (isConverged) {
    return false;
  }

  return phase === "idle" || phase === "running";
}

function canAdjudicate(phase: string): boolean {
  return phase === "ready_for_adjudication";
}

function normalizeSpeaker(speakerRaw: string): {
  side: DialogueSide;
  label: string;
} {
  const speaker = speakerRaw.toLowerCase();

  if (speaker.includes("plaintiff") || speaker.includes("原告")) {
    return { side: "plaintiff", label: "原告" };
  }

  if (speaker.includes("defendant") || speaker.includes("被告")) {
    return { side: "defendant", label: "被告" };
  }

  if (speaker.includes("judge") || speaker.includes("法官")) {
    return { side: "judge", label: "法官" };
  }

  if (speaker.includes("system") || speaker.includes("系统")) {
    return { side: "system", label: "系统" };
  }

  if (speaker.includes("record") || speaker.includes("记录")) {
    return { side: "other", label: "记录" };
  }

  return { side: "other", label: speakerRaw || "记录" };
}

function inferSpeakerFromText(textRaw: string): {
  side: DialogueSide;
  label: string;
} | null {
  const text = textRaw.trim();
  const lower = text.toLowerCase();

  if (
    text.startsWith("【系统") ||
    text.includes("系统初始化") ||
    lower.startsWith("[system]") ||
    lower.includes("ready for adjudication")
  ) {
    return { side: "system", label: "系统" };
  }

  if (
    /^(原告|原告方|原告诉称|原告认为|原告主张|原告请求)/.test(text) ||
    lower.startsWith("plaintiff")
  ) {
    return { side: "plaintiff", label: "原告" };
  }

  if (
    /^(被告|被告方|被告辩称|被告答辩|答辩意见)/.test(text) ||
    lower.startsWith("defendant")
  ) {
    return { side: "defendant", label: "被告" };
  }

  if (
    /^(法官|审判长|合议庭|法院认为)/.test(text) ||
    lower.startsWith("judge")
  ) {
    return { side: "judge", label: "法官" };
  }

  return null;
}

function toDialogueRows(
  transcript: string[],
  isNewIndexStart: number,
): DialogueRow[] {
  return transcript.map((line, idx) => {
    const match = line.match(/^\[(.+?)\]\s*(.*)$/);
    const speakerRaw = match ? match[1] : "";
    const text = (match ? match[2] : line).trim() || line.trim();
    const normalized = speakerRaw
      ? normalizeSpeaker(speakerRaw)
      : (inferSpeakerFromText(text) ?? { side: "other", label: "记录" });

    return {
      idx,
      side: normalized.side,
      speakerLabel: normalized.label,
      text,
      isNew: idx >= isNewIndexStart,
    };
  });
}

function metricText(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : "-";
}

export function LivePage() {
  const {
    sessionId,
    snapshot,
    previousSnapshot,
    graphView,
    busyAction,
    step,
    adjudicate,
    refreshSnapshot,
    loadGraph,
  } = useDebate();

  const dialogueBoardRef = useRef<HTMLDivElement | null>(null);

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

  const newStart = useMemo(() => {
    if (!snapshot) {
      return 0;
    }

    if (
      !previousSnapshot ||
      previousSnapshot.sessionId !== snapshot.sessionId
    ) {
      return 0;
    }

    return previousSnapshot.transcript.length;
  }, [previousSnapshot, snapshot]);

  const dialogueRows = useMemo(
    () => (snapshot ? toDialogueRows(snapshot.transcript, newStart) : []),
    [newStart, snapshot],
  );

  const latestNewDialogueIdx = useMemo(() => {
    for (let idx = dialogueRows.length - 1; idx >= 0; idx -= 1) {
      if (dialogueRows[idx].isNew) {
        return dialogueRows[idx].idx;
      }
    }

    return null;
  }, [dialogueRows]);

  const stepBlockedByConvergence = snapshot
    ? snapshot.convergence.isConverged ||
      snapshot.phase === "ready_for_adjudication" ||
      snapshot.phase === "finished"
    : false;

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void loadGraph();
  }, [loadGraph, sessionId]);

  const jumpToCurrentDialogue = (): void => {
    if (!dialogueRows.length) {
      return;
    }

    const targetIdx =
      latestNewDialogueIdx ?? dialogueRows[dialogueRows.length - 1].idx;

    const target = dialogueBoardRef.current?.querySelector<HTMLElement>(
      `[data-dialogue-idx="${targetIdx}"]`,
    );

    if (!target) {
      return;
    }

    target.scrollIntoView({ behavior: "smooth", block: "center" });
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
            <span>当前回合</span>
            <strong>{snapshot.round}</strong>
          </p>

          {snapshot.phase === "finished" ? (
            <p>
              <span>根主张采纳</span>

              <strong>
                {validatedCount}/{rootClaimEntries.length || 0}
              </strong>
            </p>
          ) : null}
        </div>

        <div className="ux-row">
          <button
            disabled={
              Boolean(busyAction) ||
              !canStepDebate(snapshot.phase, snapshot.convergence.isConverged)
            }
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

        {stepBlockedByConvergence ? (
          <p className="ux-note">会话已收敛，请直接发起裁决。</p>
        ) : null}
      </article>

      <article className="ux-card">
        <h2>收敛轨迹</h2>

        <ConvergenceSparkline
          deltaPhi={snapshot.convergence.deltaPhi}
          epsilon={snapshot.convergence.epsilon}
          history={snapshot.convergence.history}
          sma={snapshot.convergence.sma}
        />

        <div className="ux-kv" style={{ marginTop: "0.65rem" }}>
          <p>
            <span>最新 ΔΦ</span>
            <strong>{metricText(snapshot.convergence.deltaPhi)}</strong>
          </p>

          <p>
            <span>SMA</span>
            <strong>{metricText(snapshot.convergence.sma)}</strong>
          </p>

          <p>
            <span>阈值 ε</span>
            <strong>{metricText(snapshot.convergence.epsilon)}</strong>
          </p>

          <p>
            <span>收敛状态</span>

            <strong>
              {snapshot.convergence.isConverged ? "已收敛" : "未收敛"}
            </strong>
          </p>
        </div>
      </article>

      <article className="ux-card ux-card-full">
        <h2>庭审对话</h2>

        <div className="ux-row">
          <button
            disabled={!dialogueRows.length}
            onClick={jumpToCurrentDialogue}
            type="button"
          >
            跳转当前对话
          </button>
        </div>

        {dialogueRows.length > 0 ? (
          <div className="ux-dialogue-board" ref={dialogueBoardRef}>
            {dialogueRows.map((row) => (
              <div
                className={`ux-dialogue-row ux-dialogue-${row.side} ${row.isNew ? "ux-dialogue-new" : ""}`}
                data-dialogue-idx={row.idx}
                key={`${row.idx}-${row.speakerLabel}-${row.text}`}
              >
                <div className="ux-dialogue-main">
                  <span className="ux-dialogue-avatar" aria-hidden="true">
                    {row.speakerLabel.slice(0, 1)}
                  </span>

                  <div className="ux-dialogue-bubble">
                    <div className="ux-dialogue-head">
                      <p className="ux-dialogue-speaker">{row.speakerLabel}</p>

                      <span className="ux-dialogue-index">#{row.idx + 1}</span>

                      {row.isNew ? (
                        <span className="ux-dialogue-tag">NEW</span>
                      ) : null}
                    </div>

                    <p className="ux-dialogue-text">{row.text}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="ux-empty">暂无庭审陈述。</p>
        )}
      </article>

      <article className="ux-card ux-card-full">
        <ForceArgumentGraph graph={graphView} title="论证图（力导布局）" />
      </article>
    </section>
  );
}
