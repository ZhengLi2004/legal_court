import { useEffect, useMemo, useState } from "react";
import { GraphDiffPanel } from "../../components/debug";
import { useDebate } from "../state/useDebate";

function normalizeRounds(input: number[]): number[] {
  return [...new Set(input)].filter(Number.isFinite).sort((a, b) => a - b);
}

export function GraphPage() {
  const {
    sessionId,
    snapshot,
    graphView,
    baselineGraphView,
    graphDiff,
    snapshotIndex,
    turnArtifacts,
    busyAction,
    loadSnapshots,
    loadGraphDiff,
    loadGraphAtRound,
  } = useDebate();

  const rounds = useMemo(() => {
    const fromIndex = snapshotIndex.map((item) => item.round);
    const current = snapshot ? [snapshot.round] : [];
    return normalizeRounds([...fromIndex, ...current]);
  }, [snapshot, snapshotIndex]);

  const [fromRound, setFromRound] = useState<number>(0);
  const [toRound, setToRound] = useState<number>(0);

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void loadSnapshots();
  }, [loadSnapshots, sessionId]);

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
        <h2>论证图谱</h2>
        <p className="ux-empty">请先在“案件启动”页创建或选择一个会话。</p>
      </article>
    );
  }

  return (
    <section className="ux-grid ux-grid-2">
      <article className="ux-card">
        <h2>图差异控制台</h2>

        <p className="ux-muted">
          选择回合后可直接对比论证图变化，蓝线为支持，红虚线为冲突，紫色用于标注回滚相关节点。
        </p>

        <div className="ux-row">
          <label className="ux-field">
            起始回合
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
            目标回合
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
            计算图差异
          </button>

          <button
            disabled={Boolean(busyAction)}
            onClick={() => {
              void loadGraphAtRound(effectiveToRound);
            }}
            type="button"
          >
            仅加载目标回合图
          </button>
        </div>

        <div className="ux-kv">
          <p>
            <span>当前图回合</span>
            <strong>r{graphView?.round ?? "-"}</strong>
          </p>

          <p>
            <span>基线图回合</span>
            <strong>r{baselineGraphView?.round ?? "-"}</strong>
          </p>

          <p>
            <span>新增节点</span>
            <strong>{graphDiff?.addedNodeIds.length ?? 0}</strong>
          </p>

          <p>
            <span>状态变化节点</span>
            <strong>{graphDiff?.statusChangedNodeIds.length ?? 0}</strong>
          </p>
        </div>
      </article>

      <article className="ux-card">
        <h2>变化摘要</h2>
        <div className="ux-kv">
          <p>
            <span>移除节点</span>
            <strong>{graphDiff?.removedNodeIds.length ?? 0}</strong>
          </p>

          <p>
            <span>新增边</span>
            <strong>{graphDiff?.addedEdgeIds.length ?? 0}</strong>
          </p>

          <p>
            <span>移除边</span>
            <strong>{graphDiff?.removedEdgeIds.length ?? 0}</strong>
          </p>

          <p>
            <span>回滚相关工件</span>
            <strong>{turnArtifacts.length}</strong>
          </p>
        </div>

        <p className="ux-muted">
          重点回答：本回合新增了什么、冲突链在哪、是否发生回滚及修正。
        </p>
      </article>

      <GraphDiffPanel
        artifacts={turnArtifacts}
        baselineGraph={baselineGraphView}
        currentGraph={graphView}
        diff={graphDiff}
        title="论证图差异主画布"
      />
    </section>
  );
}
