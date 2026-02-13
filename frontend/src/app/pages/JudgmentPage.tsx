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

function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => {
      if (typeof item === "string") {
        return item.trim();
      }

      if (typeof item === "number" || typeof item === "boolean") {
        return String(item);
      }

      return "";
    })
    .filter(Boolean);
}

function toPrettyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function JudgmentPage() {
  const {
    sessionId,
    snapshot,
    graphView,
    timeline,
    turnArtifacts,
    loadGraph,
    loadTimeline,
    loadTurnArtifacts,
  } = useDebate();

  useEffect(() => {
    if (sessionId && (!graphView || graphView.sessionId !== sessionId)) {
      void loadGraph();
    }
  }, [graphView, loadGraph, sessionId]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void loadTimeline(500);
    void loadTurnArtifacts({ limit: 200 });
  }, [loadTimeline, loadTurnArtifacts, sessionId]);

  const payload = useMemo(
    () => unwrapPayload(snapshot?.raw ?? {}),
    [snapshot?.raw],
  );

  const judgmentDocument = asString(payload.judgment_document);
  const rootClaimStatusMap = asRecord(payload.root_claims_status);
  const rootClaimEntries = Object.entries(rootClaimStatusMap);
  const preferredExtension = toStringArray(payload.preferred_extension);
  const bafDetails = asRecord(payload.baf_details);
  const searchStats = asRecord(bafDetails.search_stats);
  const consistencyReport = asRecord(bafDetails.consistency_report);
  const matchDetails = asRecord(bafDetails.match_details);
  const contextSelection = asRecord(bafDetails.context_selection);
  const fusionCorrections = asRecord(bafDetails.fusion_corrections);
  const llmRootClaimStatus = asRecord(bafDetails.llm_root_claims_status);
  const fusedRootClaimStatus = asRecord(bafDetails.fused_root_claims_status);

  const preferredExtensionsCount = asNumber(
    bafDetails.preferred_extensions_count ??
      bafDetails.preferredExtensionsCount,
  );

  const chosenExtension = (() => {
    const fromBafDetails = toStringArray(bafDetails.chosen_extension);

    if (fromBafDetails.length > 0) {
      return fromBafDetails;
    }

    return preferredExtension;
  })();

  const chosenExtensionSize =
    chosenExtension.length > 0
      ? chosenExtension.length
      : asNumber(
          bafDetails.chosen_extension_size ?? bafDetails.chosenExtensionSize,
          0,
        );

  const alignmentRate = asNumber(
    bafDetails.alignment_rate ?? bafDetails.alignmentRate,
    0,
  );

  const matchScore = asNumber(bafDetails.match_score ?? bafDetails.matchScore);

  const algorithmVersion = asString(
    bafDetails.algorithm_version ?? searchStats.algorithm_version,
    "-",
  );

  const searchTimeMs = asNumber(
    bafDetails.search_time_ms ?? searchStats.search_time_ms,
  );

  const searchedStates = asNumber(
    bafDetails.searched_states ?? searchStats.searched_states,
  );

  const prunedStates = asNumber(
    bafDetails.pruned_states ?? searchStats.pruned_states,
  );

  const terminationReason = asString(
    bafDetails.termination_reason ?? searchStats.termination_reason,
    "-",
  );

  const consistencyIssues = Array.isArray(consistencyReport.issues)
    ? consistencyReport.issues
    : [];

  const validatedCount = rootClaimEntries.filter(([, value]) =>
    String(value).toUpperCase().includes("VALID"),
  ).length;

  const adjudicationTimeline = [...timeline]
    .filter(
      (event) =>
        event.event.includes("adjudication") || event.event === "session_error",
    )
    .sort((lhs, rhs) => lhs.seq - rhs.seq)
    .slice(-80);

  const latestArtifact = [...turnArtifacts]
    .sort((lhs, rhs) => lhs.round - rhs.round)
    .at(-1);

  const payloadJson = useMemo(() => toPrettyJson(payload), [payload]);
  const bafDetailsJson = useMemo(() => toPrettyJson(bafDetails), [bafDetails]);

  const latestArtifactJson = useMemo(
    () => toPrettyJson(latestArtifact ?? {}),
    [latestArtifact],
  );

  const timelineJson = useMemo(
    () => toPrettyJson(adjudicationTimeline),
    [adjudicationTimeline],
  );

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
        <h2>判决总览</h2>

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
            <span>当前回合</span>
            <strong>{snapshot.round}</strong>
          </p>

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

          <p>
            <span>匹配得分</span>
            <strong>{matchScore}</strong>
          </p>
        </div>
      </article>

      <article className="ux-card">
        <h2>搜索与匹配</h2>

        <div className="ux-kv">
          <p>
            <span>算法版本</span>
            <strong>{algorithmVersion}</strong>
          </p>

          <p>
            <span>搜索耗时</span>
            <strong>{searchTimeMs} ms</strong>
          </p>

          <p>
            <span>搜索状态数</span>
            <strong>{searchedStates}</strong>
          </p>

          <p>
            <span>剪枝状态数</span>
            <strong>{prunedStates}</strong>
          </p>

          <p>
            <span>终止原因</span>
            <strong>{terminationReason || "-"}</strong>
          </p>
        </div>
      </article>

      <article className="ux-card ux-card-full">
        <h2>一致性检查（LLM vs BAF）</h2>

        <div className="ux-kv" style={{ marginBottom: "0.6rem" }}>
          <p>
            <span>一致性结论</span>

            <strong>
              {consistencyReport.is_consistent === true ? "一致" : "存在冲突"}
            </strong>
          </p>

          <p>
            <span>LLM 采纳数</span>
            <strong>{asNumber(consistencyReport.validated_count)}</strong>
          </p>

          <p>
            <span>LLM 驳回数</span>
            <strong>{asNumber(consistencyReport.defeated_count)}</strong>
          </p>
        </div>

        {consistencyIssues.length > 0 ? (
          <div className="ux-log-box">
            {consistencyIssues.map((issue, index) => {
              const row = asRecord(issue);
              const issueType = asString(row.type, "unknown");
              const issueMessage = asString(row.message, "");

              return (
                <p className="ux-log-line" key={`${issueType}-${index}`}>
                  [{issueType}] {issueMessage || "无详细信息"}
                </p>
              );
            })}
          </div>
        ) : (
          <p className="ux-empty">未检测到一致性问题。</p>
        )}
      </article>

      <article className="ux-card">
        <h2>扩展匹配详情</h2>

        <div className="ux-kv">
          <p>
            <span>扩展索引</span>
            <strong>{asNumber(matchDetails.extension_index, -1)}</strong>
          </p>

          <p>
            <span>扩展规模</span>
            <strong>{asNumber(matchDetails.size, chosenExtensionSize)}</strong>
          </p>

          <p>
            <span>采纳且在扩展内</span>
            <strong>{asNumber(matchDetails.validated_in_ext)}</strong>
          </p>

          <p>
            <span>采纳但不在扩展内</span>
            <strong>{asNumber(matchDetails.validated_out_ext)}</strong>
          </p>

          <p>
            <span>驳回却在扩展内</span>
            <strong>{asNumber(matchDetails.defeated_in_ext)}</strong>
          </p>

          <p>
            <span>驳回且不在扩展内</span>
            <strong>{asNumber(matchDetails.defeated_out_ext)}</strong>
          </p>
        </div>

        <details style={{ marginTop: "0.6rem" }}>
          <summary>选中扩展节点（{chosenExtension.length}）</summary>
          <div className="ux-log-box">
            {chosenExtension.length > 0 ? (
              chosenExtension.map((nodeId) => (
                <p className="ux-log-line" key={nodeId}>
                  {nodeId}
                </p>
              ))
            ) : (
              <p className="ux-log-line">暂无选中扩展节点。</p>
            )}
          </div>
        </details>
      </article>

      <article className="ux-card">
        <h2>融合修正与上下文</h2>

        <div className="ux-kv">
          <p>
            <span>修正总数</span>
            <strong>{asNumber(fusionCorrections.total_corrections)}</strong>
          </p>

          <p>
            <span>VALIDATED → DEFEATED</span>
            <strong>{asNumber(fusionCorrections.validated_to_defeated)}</strong>
          </p>

          <p>
            <span>DEFEATED → VALIDATED</span>
            <strong>{asNumber(fusionCorrections.defeated_to_validated)}</strong>
          </p>

          <p>
            <span>HYPOTHETICAL → VALIDATED</span>
            <strong>
              {asNumber(fusionCorrections.hypothetical_to_validated)}
            </strong>
          </p>
        </div>

        <details style={{ marginTop: "0.6rem" }}>
          <summary>上下文选择统计</summary>

          <div className="ux-kv" style={{ marginTop: "0.5rem" }}>
            <p>
              <span>模式</span>
              <strong>{asString(contextSelection.mode, "-")}</strong>
            </p>

            <p>
              <span>k-hop</span>
              <strong>{asNumber(contextSelection.k_hop)}</strong>
            </p>

            <p>
              <span>根主张数</span>
              <strong>{asNumber(contextSelection.root_count)}</strong>
            </p>

            <p>
              <span>支持锥节点</span>
              <strong>{asNumber(contextSelection.support_cone_count)}</strong>
            </p>

            <p>
              <span>攻击者节点</span>
              <strong>{asNumber(contextSelection.attacker_count)}</strong>
            </p>

            <p>
              <span>防御者节点</span>
              <strong>{asNumber(contextSelection.defender_count)}</strong>
            </p>

            <p>
              <span>最终上下文节点</span>
              <strong>{asNumber(contextSelection.selected_count)}</strong>
            </p>
          </div>
        </details>
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

      <article className="ux-card">
        <h2>LLM/Fused 根主张对照</h2>

        <details open>
          <summary>LLM 提取状态</summary>

          <div className="ux-log-box">
            <pre className="ux-document">
              {toPrettyJson(llmRootClaimStatus)}
            </pre>
          </div>
        </details>

        <details style={{ marginTop: "0.6rem" }} open>
          <summary>融合后状态</summary>

          <div className="ux-log-box">
            <pre className="ux-document">
              {toPrettyJson(fusedRootClaimStatus)}
            </pre>
          </div>
        </details>
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
        <h2>裁决事件流</h2>

        {adjudicationTimeline.length > 0 ? (
          <div className="ux-log-box">
            {adjudicationTimeline.map((event) => {
              const row = asRecord(event.data);
              const msg = asString(row.message, "");

              return (
                <p className="ux-log-line" key={`event-${event.seq}`}>
                  #{event.seq} [{event.source}] {event.event}
                  {msg ? `: ${msg}` : ""}
                </p>
              );
            })}
          </div>
        ) : (
          <p className="ux-empty">暂无裁决相关事件。</p>
        )}
      </article>

      <article className="ux-card ux-card-full">
        <h2>最新回合产物（Debug）</h2>

        {latestArtifact ? (
          <>
            <div className="ux-kv">
              <p>
                <span>turn_uid</span>
                <strong>{latestArtifact.turnUid}</strong>
              </p>

              <p>
                <span>round</span>
                <strong>{latestArtifact.round}</strong>
              </p>

              <p>
                <span>side</span>
                <strong>{latestArtifact.side}</strong>
              </p>
            </div>

            <details style={{ marginTop: "0.6rem" }} open>
              <summary>展开最新回合完整 JSON</summary>

              <div className="ux-log-box">
                <pre className="ux-document">{latestArtifactJson}</pre>
              </div>
            </details>
          </>
        ) : (
          <p className="ux-empty">暂无回合产物。</p>
        )}
      </article>

      <article className="ux-card ux-card-full">
        <SimpleBafGraph
          graph={graphView}
          preferredExtension={chosenExtension}
          rootClaimStatusMap={Object.fromEntries(
            rootClaimEntries.map(([key, value]) => [key, String(value)]),
          )}
        />
      </article>

      <article className="ux-card ux-card-full">
        <h2>原始 JSON 调试区</h2>

        <details open>
          <summary>baf_details（原始）</summary>

          <div className="ux-log-box">
            <pre className="ux-document">{bafDetailsJson}</pre>
          </div>
        </details>

        <details style={{ marginTop: "0.6rem" }}>
          <summary>payload（unwrap 后）</summary>

          <div className="ux-log-box">
            <pre className="ux-document">{payloadJson}</pre>
          </div>
        </details>

        <details style={{ marginTop: "0.6rem" }}>
          <summary>adjudication timeline（原始）</summary>

          <div className="ux-log-box">
            <pre className="ux-document">{timelineJson}</pre>
          </div>
        </details>
      </article>
    </section>
  );
}
