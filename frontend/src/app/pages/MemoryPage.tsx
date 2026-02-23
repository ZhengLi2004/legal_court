import { useEffect, useMemo, useState } from "react";
import { ForceArgumentGraph } from "../components/ForceArgumentGraph";
import { TaskLayerGraph } from "../components/TaskLayerGraph";
import { useDebate } from "../state/useDebate";

export function MemoryPage() {
  const {
    sessionId,
    snapshot,
    memoryView,
    memoryCaseGraphView,
    loadMemory,
    loadMemoryCaseGraph,
  } = useDebate();

  const [pendingCaseId, setPendingCaseId] = useState<string>("");

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void loadMemory();
  }, [loadMemory, sessionId]);

  const recalledCaseIds = useMemo(() => {
    return memoryView?.recalledCaseIds ?? [];
  }, [memoryView?.recalledCaseIds]);

  const recalledRows = useMemo(
    () =>
      recalledCaseIds.map((caseId, index) => ({
        caseId,
        summary:
          memoryView?.caseCatalog?.[caseId]?.summary || `案例 ${index + 1}`,
      })),
    [memoryView?.caseCatalog, recalledCaseIds],
  );

  const selectedCaseId = useMemo(() => {
    if (!recalledRows.length) {
      return "";
    }

    if (recalledRows.some((item) => item.caseId === pendingCaseId)) {
      return pendingCaseId;
    }

    return recalledRows[0].caseId;
  }, [pendingCaseId, recalledRows]);

  const selectedCaseSummary = useMemo(() => {
    if (!selectedCaseId) {
      return "";
    }

    return memoryView?.caseCatalog?.[selectedCaseId]?.summary || "（无摘要）";
  }, [memoryView?.caseCatalog, selectedCaseId]);

  const insightRows = useMemo(
    () => memoryView?.insightItems ?? [],
    [memoryView?.insightItems],
  );

  const formatInsightSide = (sideRaw: string): string => {
    const side = String(sideRaw || "").toUpperCase();

    if (side === "PLAINTIFF") {
      return "原告";
    }

    if (side === "DEFENDANT") {
      return "被告";
    }

    return "通用";
  };

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    if (!selectedCaseId) {
      void loadMemoryCaseGraph("");
      return;
    }

    void loadMemoryCaseGraph(selectedCaseId);
  }, [loadMemoryCaseGraph, selectedCaseId, sessionId]);

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
            <span>召回案例</span>

            <strong>
              {memoryView?.recalledCaseCount ?? recalledRows.length}
            </strong>
          </p>

          <p>
            <span>洞察条目</span>
            <strong>{insightRows.length}</strong>
          </p>
        </div>
      </article>

      <article className="ux-card">
        <h2>洞察列表（{insightRows.length}）</h2>

        {insightRows.length ? (
          <div className="ux-list">
            {insightRows.map((item) => (
              <article
                className="ux-inspector-text"
                key={`${item.content}-${item.linkedRound}`}
              >
                <strong>{item.content}</strong>

                <div className="ux-kv" style={{ marginTop: "0.45rem" }}>
                  <p>
                    <span>侧别</span>
                    <strong>{formatInsightSide(item.side)}</strong>
                  </p>

                  <p>
                    <span>关联案例</span>
                    <strong>{item.relatedCases.length}</strong>
                  </p>

                  <p>
                    <span>关联回合</span>

                    <strong>
                      {item.linkedRound > 0 ? `r${item.linkedRound}` : "-"}
                    </strong>
                  </p>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <p className="ux-empty">暂无洞察。</p>
        )}
      </article>

      <TaskLayerGraph memoryView={memoryView} />

      <article className="ux-card ux-card-full">
        <h2>案例论证图切换窗口</h2>

        <label className="ux-field">
          <span>选择案例</span>

          <select
            onChange={(event) => setPendingCaseId(event.target.value)}
            value={selectedCaseId}
          >
            {!recalledRows.length ? <option value="">暂无案例</option> : null}

            {recalledRows.map((item) => (
              <option key={item.caseId} value={item.caseId}>
                {item.summary}
              </option>
            ))}
          </select>
        </label>

        {selectedCaseId ? (
          <p className="ux-note">
            当前案例：{selectedCaseSummary || "（无摘要）"}
          </p>
        ) : (
          <p className="ux-empty">暂无可查看的案例。</p>
        )}
      </article>

      <ForceArgumentGraph
        cardClassName="ux-card ux-card-full"
        graph={memoryCaseGraphView}
        title="案例论证图"
      />
    </section>
  );
}
