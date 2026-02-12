import { useEffect, useMemo, useRef, useState } from "react";
import { ForceArgumentGraph } from "../components/ForceArgumentGraph";
import { ConvergenceSparkline } from "../components/ConvergenceSparkline";
import { useDebate } from "../state/useDebate";
import { asRecord, phaseLabel, unwrapPayload } from "../utils/payload";
import type { TimelineEvent } from "../../compat";
type DialogueSide = "plaintiff" | "defendant" | "judge" | "system" | "other";

interface DialogueRow {
  idx: number;
  side: DialogueSide;
  speakerLabel: string;
  text: string;
  isNew: boolean;
}

function canAdjudicate(phase: string): boolean {
  return phase === "ready_for_adjudication" || phase === "finished";
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

function timelineReason(row: TimelineEvent): string {
  const data = asRecord(row.data);
  const rawReason = String(data.reason ?? data.message ?? "");
  const turnRaw = String(data.turn ?? "").toLowerCase();
  const pipelineStep = String(data.pipeline_step ?? "").toUpperCase();

  if (pipelineStep) {
    const stepLabel =
      pipelineStep === "ASSESS_NEEDS"
        ? "需求评估"
        : pipelineStep === "WAIT_FOR_WORKERS"
          ? "等待工作者"
          : pipelineStep === "DECIDE"
            ? "综合决策"
            : pipelineStep === "DONE"
              ? "回合结束"
              : pipelineStep;

    return `流程阶段：${stepLabel}`;
  }

  if (turnRaw) {
    return `当前方：${turnRaw.includes("plaintiff") ? "原告" : "被告"}`;
  }

  if (!rawReason) {
    return "";
  }

  const lower = rawReason.toLowerCase();

  if (lower.includes("convergence")) {
    return "收敛阈值达成";
  }

  return rawReason;
}

function timelineEventLabel(eventRaw: string): string {
  const event = eventRaw.toLowerCase();
  const TEAM_PREFIX = "team_";

  if (event.startsWith(TEAM_PREFIX)) {
    const teamSide = event.includes("team_plaintiff_")
      ? "原告团队"
      : "被告团队";

    const suffix = event
      .replace("team_plaintiff_", "")
      .replace("team_defendant_", "");

    if (suffix === "turn_start") {
      return `${teamSide}开始执行`;
    }

    if (suffix === "internal_step") {
      return `${teamSide}内部推进`;
    }

    if (suffix === "turn_complete") {
      return `${teamSide}执行完成`;
    }

    if (suffix === "retry") {
      return `${teamSide}触发重试`;
    }

    return `${teamSide}状态更新`;
  }

  if (event === "setup_start") {
    return "系统初始化开始";
  }

  if (event === "setup_complete") {
    return "系统初始化完成";
  }

  if (event === "turn_start") {
    return "庭审回合开始";
  }

  if (event === "transcript_update") {
    return "庭审记录更新";
  }

  if (event === "turn_complete") {
    return "庭审回合完成";
  }

  if (event === "adjudication_ready") {
    return "满足裁决条件";
  }

  if (event === "adjudication_start") {
    return "裁决流程开始";
  }

  if (event === "snapshot_saved") {
    return "回合快照已保存";
  }

  if (event === "session_warning") {
    return "系统运行告警";
  }

  return eventRaw;
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
    loadGraph,
    loadTimeline,
    loadGraphAtRound,
  } = useDebate();

  const [selectedSeq, setSelectedSeq] = useState<number>(0);

  const [currentDialogueIdx, setCurrentDialogueIdx] = useState<number | null>(
    null,
  );

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

  const visibleTimeline = useMemo(
    () => [...timeline].slice(-30).reverse(),
    [timeline],
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

  const onTimelineClick = async (row: TimelineEvent): Promise<void> => {
    setSelectedSeq(row.seq);

    if (typeof row.roundIdx === "number") {
      await loadGraphAtRound(row.roundIdx);
    }
  };

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void loadGraph();
  }, [loadGraph, sessionId]);

  const effectiveCurrentDialogueIdx = useMemo(() => {
    if (!dialogueRows.length || currentDialogueIdx === null) {
      return null;
    }

    const maxIdx = dialogueRows[dialogueRows.length - 1].idx;
    return currentDialogueIdx <= maxIdx ? currentDialogueIdx : null;
  }, [currentDialogueIdx, dialogueRows]);

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

    setCurrentDialogueIdx(targetIdx);
    target.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  useEffect(() => {
    if (currentDialogueIdx === null) {
      return;
    }

    const onDocumentPointerDown = (event: PointerEvent): void => {
      const board = dialogueBoardRef.current;
      const target = event.target as Node | null;

      if (!board || (target && board.contains(target))) {
        return;
      }

      setCurrentDialogueIdx(null);
    };

    document.addEventListener("pointerdown", onDocumentPointerDown);

    return () => {
      document.removeEventListener("pointerdown", onDocumentPointerDown);
    };
  }, [currentDialogueIdx]);

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

        <div className="ux-convergence-wrap">
          <p className="ux-muted">收敛轨迹</p>

          <ConvergenceSparkline
            deltaPhi={snapshot.convergence.deltaPhi}
            epsilon={snapshot.convergence.epsilon}
            history={snapshot.convergence.history}
            sma={snapshot.convergence.sma}
          />
        </div>

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
                  {timelineEventLabel(row.event)}
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
        <h2>庭审对话</h2>

        <p className="ux-muted">
          以下为完整庭审自然语言陈述，按真实发生顺序展示；高亮项为本次刷新后新增内容。
        </p>

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
          <div
            className="ux-dialogue-board"
            onClick={(event) => {
              const target = event.target as HTMLElement;

              if (!target.closest("[data-dialogue-idx]")) {
                setCurrentDialogueIdx(null);
              }
            }}
            ref={dialogueBoardRef}
          >
            {dialogueRows.map((row) => (
              <div
                className={`ux-dialogue-row ux-dialogue-${row.side} ${row.isNew ? "ux-dialogue-new" : ""} ${
                  effectiveCurrentDialogueIdx === row.idx
                    ? "ux-dialogue-current"
                    : ""
                }`}
                data-dialogue-idx={row.idx}
                key={`${row.idx}-${row.speakerLabel}-${row.text}`}
                onClick={() => setCurrentDialogueIdx(row.idx)}
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
