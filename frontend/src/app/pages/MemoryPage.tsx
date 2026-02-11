import { useEffect, useMemo, useState } from "react";
import { TaskLayerGraph } from "../components/TaskLayerGraph";
import { useDebate } from "../state/useDebate";

export function MemoryPage() {
  const {
    sessionId,
    snapshot,
    memoryView,
    busyAction,
    loadMemory,
    loadGraphAtRound,
  } = useDebate();

  const [selectedInsight, setSelectedInsight] = useState<string>("");

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void loadMemory();
  }, [loadMemory, sessionId]);

  const insightEntries = useMemo(
    () => memoryView?.insightItems ?? [],
    [memoryView?.insightItems],
  );

  const selectedRow = useMemo(
    () =>
      insightEntries.find((item) => item.content === selectedInsight) ??
      insightEntries[0] ??
      null,
    [insightEntries, selectedInsight],
  );

  const newInsights = useMemo(
    () =>
      insightEntries.filter(
        (item) => item.linkedRound >= Math.max((snapshot?.round ?? 0) - 1, 0),
      ),
    [insightEntries, snapshot?.round],
  );

  const retainedInsights = useMemo(
    () =>
      insightEntries.filter(
        (item) => !newInsights.some((row) => row.content === item.content),
      ),
    [insightEntries, newInsights],
  );

  if (!sessionId || !snapshot) {
    return (
      <article className="ux-card">
        <h2>记忆与类比</h2>
        <p className="ux-empty">请先在“案件启动”页创建或选择一个会话。</p>
      </article>
    );
  }

  return (
    <section className="ux-grid ux-grid-2">
      <article className="ux-card">
        <h2>记忆总览</h2>

        <div className="ux-kv">
          <p>
            <span>静态历史案例</span>
            <strong>{memoryView?.staticHistoryCount ?? 0}</strong>
          </p>

          <p>
            <span>动态法理案例</span>
            <strong>{memoryView?.dynamicLawCaseCount ?? 0}</strong>
          </p>

          <p>
            <span>TaskLayer 节点</span>
            <strong>{memoryView?.taskLayerNodeCount ?? 0}</strong>
          </p>

          <p>
            <span>TaskLayer 边</span>
            <strong>{memoryView?.taskLayerEdgeCount ?? 0}</strong>
          </p>

          <p>
            <span>代表案例数</span>
            <strong>{memoryView?.representativeCaseIds.length ?? 0}</strong>
          </p>

          <p>
            <span>洞察条目</span>
            <strong>{insightEntries.length}</strong>
          </p>
        </div>

        <div className="ux-row">
          <button
            disabled={Boolean(busyAction)}
            onClick={() => {
              void loadMemory();
            }}
            type="button"
          >
            刷新记忆
          </button>
        </div>
      </article>

      <article className="ux-card">
        <h2>洞察分组</h2>
        <p className="ux-muted">新洞察优先展示最近回合新增的可迁移策略。</p>

        <details open>
          <summary>新洞察（{newInsights.length}）</summary>

          <div className="ux-list">
            {newInsights.map((item) => (
              <button
                className="ux-list-row"
                key={`new-${item.content}`}
                onClick={() => setSelectedInsight(item.content)}
                type="button"
              >
                <span>{item.side}</span>
                <span>{item.content}</span>
                <span>r{item.linkedRound}</span>
              </button>
            ))}
          </div>
        </details>

        <details>
          <summary>保留洞察（{retainedInsights.length}）</summary>

          <div className="ux-list">
            {retainedInsights.map((item) => (
              <button
                className="ux-list-row"
                key={`retained-${item.content}`}
                onClick={() => setSelectedInsight(item.content)}
                type="button"
              >
                <span>{item.side}</span>
                <span>{item.content}</span>
                <span>r{item.linkedRound}</span>
              </button>
            ))}
          </div>
        </details>
      </article>

      <TaskLayerGraph memoryView={memoryView} />

      <article className="ux-card">
        <h2>洞察详情与回放</h2>

        {selectedRow ? (
          <div className="ux-kv">
            <p>
              <span>内容</span>
              <strong>{selectedRow.content}</strong>
            </p>

            <p>
              <span>阵营</span>
              <strong>{selectedRow.side}</strong>
            </p>

            <p>
              <span>关联回合</span>
              <strong>r{selectedRow.linkedRound}</strong>
            </p>

            <p>
              <span>代表案例</span>
              <strong>{selectedRow.representatives.join(", ") || "-"}</strong>
            </p>

            <p>
              <span>候选案例</span>
              <strong>{selectedRow.cases.join(", ") || "-"}</strong>
            </p>
          </div>
        ) : (
          <p className="ux-empty">请选择一条洞察查看详情。</p>
        )}

        <div className="ux-row">
          <button
            disabled={!selectedRow}
            onClick={() => {
              if (selectedRow) {
                void loadGraphAtRound(selectedRow.linkedRound);
              }
            }}
            type="button"
          >
            跳转到洞察关联回合图
          </button>
        </div>
      </article>

      <article className="ux-card ux-card-full">
        <h2>案例快照时间轴</h2>

        {memoryView?.caseSnapshots.length ? (
          <div className="ux-list">
            {memoryView.caseSnapshots.map((item) => (
              <button
                className="ux-list-row"
                key={`case-snapshot-${item.round}-${item.ts}`}
                onClick={() => {
                  void loadGraphAtRound(item.round);
                }}
                type="button"
              >
                <span>r{item.round}</span>
                <span>{item.turn || "unknown"}</span>
                <span>
                  N{item.nodeCount} / E{item.edgeCount}
                </span>
              </button>
            ))}
          </div>
        ) : (
          <p className="ux-empty">暂无案例快照。</p>
        )}
      </article>
    </section>
  );
}
