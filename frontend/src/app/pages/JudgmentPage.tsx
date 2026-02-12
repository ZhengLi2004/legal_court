import { useEffect, useMemo } from "react";
import { SimpleBafGraph } from "../components/SimpleBafGraph";
import { useDebate } from "../state/useDebate";

import {
  asNumber,
  asRecord,
  asString,
  nodeStatusLabel,
  unwrapPayload,
} from "../utils/payload";

export function JudgmentPage() {
  const { sessionId, snapshot, graphView, loadGraph } = useDebate();

  useEffect(() => {
    if (sessionId && !graphView) {
      void loadGraph();
    }
  }, [graphView, loadGraph, sessionId]);

  const payload = useMemo(
    () => unwrapPayload(snapshot?.raw ?? {}),
    [snapshot?.raw],
  );

  const judgmentDocument = asString(payload.judgment_document);
  const rootClaimStatusMap = asRecord(payload.root_claims_status);
  const rootClaimEntries = Object.entries(rootClaimStatusMap);

  const preferredExtension = Array.isArray(payload.preferred_extension)
    ? payload.preferred_extension
        .filter((item): item is string => typeof item === "string")
        .map((item) => item.trim())
        .filter(Boolean)
    : [];

  const bafDetails = asRecord(payload.baf_details);

  const preferredExtensionsCount = asNumber(
    bafDetails.preferred_extensions_count ??
      bafDetails.preferredExtensionsCount,
  );

  const chosenExtensionSize = asNumber(
    bafDetails.chosen_extension_size ?? bafDetails.chosenExtensionSize,
  );

  const alignmentRate = asNumber(
    bafDetails.alignment_rate ?? bafDetails.alignmentRate,
    0,
  );

  const validatedCount = rootClaimEntries.filter(([, value]) =>
    String(value).toUpperCase().includes("VALID"),
  ).length;

  if (!sessionId || !snapshot) {
    return (
      <article className="ux-card">
        <h2>裁决解释</h2>
        <p className="ux-empty">请先完成会话创建并推进庭审。</p>
      </article>
    );
  }

  return (
    <section className="ux-grid ux-grid-2">
      <article className="ux-card">
        <h2>判决结论</h2>

        <div className="ux-kv">
          <p>
            <span>裁决状态</span>

            <strong>
              {snapshot.phase === "finished" ? "已裁决" : "未裁决"}
            </strong>
          </p>

          {snapshot.phase === "finished" ? (
            <p>
              <span>根主张采纳</span>

              <strong>
                {validatedCount}/{rootClaimEntries.length || 0}
              </strong>
            </p>
          ) : null}

          <p>
            <span>首选扩展数</span>
            <strong>{preferredExtensionsCount}</strong>
          </p>

          <p>
            <span>选中扩展大小</span>
            <strong>{chosenExtensionSize}</strong>
          </p>

          <p>
            <span>一致率</span>
            <strong>{(alignmentRate * 100).toFixed(1)}%</strong>
          </p>
        </div>
      </article>

      <article className="ux-card">
        <h2>根主张状态表</h2>

        {snapshot.phase !== "finished" ? (
          <p className="ux-empty">尚未裁决，暂无根主张状态。</p>
        ) : rootClaimEntries.length > 0 ? (
          <div className="ux-list">
            {rootClaimEntries.map(([claimId, status]) => (
              <div className="ux-list-row ux-list-row-static" key={claimId}>
                <span>{claimId}</span>
                <span>{nodeStatusLabel(String(status))}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="ux-empty">尚未生成根主张状态。</p>
        )}
      </article>

      <article className="ux-card ux-card-full">
        <h2>法官文书</h2>

        {judgmentDocument ? (
          <details open>
            <summary>展开判决文书</summary>

            <div className="ux-log-box">
              <p className="ux-document">{judgmentDocument}</p>
            </div>
          </details>
        ) : (
          <p className="ux-empty">尚未生成判决文书。请先完成裁决。</p>
        )}
      </article>

      <article className="ux-card ux-card-full">
        <SimpleBafGraph
          graph={graphView}
          preferredExtension={preferredExtension}
          rootClaimStatusMap={Object.fromEntries(
            rootClaimEntries.map(([key, value]) => [key, String(value)]),
          )}
        />
      </article>
    </section>
  );
}
