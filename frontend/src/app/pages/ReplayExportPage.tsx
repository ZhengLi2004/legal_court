import { useEffect, useMemo, useState } from "react";
import { GraphDiffPanel, TeamFlowPanel } from "../../components/debug";
import { useDebate } from "../state/useDebate";
import type { DemoKeyframe } from "../../compat";

function uniqueRounds(values: number[]): number[] {
  return [...new Set(values)].filter(Number.isFinite).sort((a, b) => a - b);
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function ReplayExportPage() {
  const {
    sessionId,
    snapshot,
    graphView,
    baselineGraphView,
    graphDiff,
    turnArtifacts,
    snapshotIndex,
    demoKeyframes,
    replayExport,
    busyAction,
    loadReplayBundle,
    loadGraphDiff,
    loadGraphAtRound,
    loadTurnArtifacts,
    exportReplayJson,
    exportGraphGexf,
  } = useDebate();

  const [fromRound, setFromRound] = useState(0);
  const [toRound, setToRound] = useState(0);
  const [selectedTurnUid, setSelectedTurnUid] = useState("");

  const [selectedKeyframe, setSelectedKeyframe] = useState<DemoKeyframe | null>(
    null,
  );

  const rounds = useMemo(
    () => uniqueRounds(snapshotIndex.map((item) => item.round)),
    [snapshotIndex],
  );

  const orderedKeyframes = useMemo(
    () => [...demoKeyframes].sort((a, b) => a.round - b.round || a.ts - b.ts),
    [demoKeyframes],
  );

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void loadReplayBundle();
  }, [loadReplayBundle, sessionId]);

  const effectiveFromRound = useMemo(() => {
    if (!rounds.length) {
      return 0;
    }

    if (rounds.includes(fromRound)) {
      return fromRound;
    }

    return rounds.length > 1 ? rounds[rounds.length - 2] : rounds[0];
  }, [fromRound, rounds]);

  const effectiveToRound = useMemo(() => {
    if (!rounds.length) {
      return 0;
    }

    if (rounds.includes(toRound)) {
      return toRound;
    }

    return rounds[rounds.length - 1];
  }, [rounds, toRound]);

  if (!sessionId || !snapshot) {
    return (
      <article className="ux-card">
        <h2>回放与导出</h2>
        <p className="ux-empty">请先在“案件启动”页创建或选择一个会话。</p>
      </article>
    );
  }

  return (
    <section className="ux-grid ux-grid-2">
      <article className="ux-card">
        <h2>回放控制</h2>

        <div className="ux-row">
          <label className="ux-field">
            from round
            <select
              onChange={(event) => setFromRound(Number(event.target.value))}
              value={effectiveFromRound}
            >
              {rounds.map((item) => (
                <option key={`from-${item}`} value={item}>
                  r{item}
                </option>
              ))}
            </select>
          </label>

          <label className="ux-field">
            to round
            <select
              onChange={(event) => setToRound(Number(event.target.value))}
              value={effectiveToRound}
            >
              {rounds.map((item) => (
                <option key={`to-${item}`} value={item}>
                  r{item}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="ux-row">
          <button
            disabled={Boolean(busyAction)}
            onClick={() => {
              void loadGraphDiff(effectiveFromRound, effectiveToRound);
            }}
            type="button"
          >
            比较图差异
          </button>

          <button
            disabled={Boolean(busyAction)}
            onClick={() => {
              void loadGraphAtRound(effectiveToRound);
            }}
            type="button"
          >
            加载回放回合
          </button>
        </div>
      </article>

      <article className="ux-card">
        <h2>导出</h2>

        <div className="ux-row">
          <button
            disabled={Boolean(busyAction)}
            onClick={() => {
              void exportReplayJson();
            }}
            type="button"
          >
            导出 Replay JSON
          </button>

          <button
            disabled={Boolean(busyAction)}
            onClick={() => {
              void exportGraphGexf(effectiveToRound).then((blob) => {
                if (blob) {
                  downloadBlob(blob, `graph-r${effectiveToRound}.gexf`);
                }
              });
            }}
            type="button"
          >
            导出 Graph GEXF
          </button>
        </div>

        <div className="ux-kv">
          <p>
            <span>事件数</span>
            <strong>{replayExport?.eventCount ?? "-"}</strong>
          </p>

          <p>
            <span>工件数</span>
            <strong>{replayExport?.artifactCount ?? "-"}</strong>
          </p>

          <p>
            <span>快照数</span>
            <strong>{replayExport?.snapshotCount ?? "-"}</strong>
          </p>
        </div>
      </article>

      <article className="ux-card ux-card-full">
        <h2>关键帧轨道</h2>

        {orderedKeyframes.length ? (
          <div className="ux-list">
            {orderedKeyframes.map((item, idx) => (
              <button
                className={`ux-list-row ${
                  selectedKeyframe?.round === item.round &&
                  selectedKeyframe?.reason === item.reason
                    ? "ux-list-row-selected"
                    : ""
                }`}
                key={`${item.reason}-${item.round}-${idx}`}
                onClick={() => {
                  setSelectedKeyframe(item);
                  setSelectedTurnUid(item.turnUid);
                  void loadGraphAtRound(item.round);
                  if (item.turnUid) {
                    void loadTurnArtifacts({
                      turnUid: item.turnUid,
                      limit: 40,
                    });
                  }
                }}
                type="button"
              >
                <span>r{item.round}</span>
                <span>{item.reason || item.event}</span>
                <span>{item.turnUid || "-"}</span>
              </button>
            ))}
          </div>
        ) : (
          <p className="ux-empty">暂无关键帧数据。</p>
        )}
      </article>

      <GraphDiffPanel
        artifacts={turnArtifacts}
        baselineGraph={baselineGraphView}
        currentGraph={graphView}
        diff={graphDiff}
        title="回放图差异画布"
      />

      <TeamFlowPanel
        artifacts={turnArtifacts}
        onSelectTurn={setSelectedTurnUid}
        selectedTurnUid={selectedTurnUid}
      />
    </section>
  );
}
