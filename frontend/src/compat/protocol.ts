import type {
  DebateMetrics,
  DebatePhase,
  DebateSnapshot,
  FrontendSnapshotListItem,
  FrontendSnapshotLoadResult,
  GraphDiffView,
  GraphEdge,
  GraphNode,
  GraphView,
  MemoryInsightItem,
  MemoryView,
  SnapshotIndexItem,
  TaskLayerGraph,
  TeamFlowMessage,
  TeamFlowTurn,
  TimelineEvent,
  TurnArtifact,
} from "./types";

const EMPTY_METRICS: DebateMetrics = {
  arguments: 0,
  attacks: 0,
  supports: 0,
};

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function asMaybeRecord(value: unknown): Record<string, unknown> | undefined {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }

  return value as Record<string, unknown>;
}

function asBoolean(value: unknown): boolean {
  return value === true;
}

function asNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Number(value);

    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return fallback;
}

function asString(value: unknown, fallback = ""): string {
  if (typeof value === "string") {
    return value;
  }

  if (
    typeof value === "number" ||
    typeof value === "boolean" ||
    typeof value === "bigint"
  ) {
    return String(value);
  }

  return fallback;
}

function asStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => {
      if (typeof item === "string") {
        return item;
      }

      const row = asRecord(item);
      const role = asString(row.role, "agent");
      const content = asString(row.content);
      return content ? `[${role}] ${content}` : "";
    })
    .filter(Boolean);
}

function asIdList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map((item) => asString(item).trim()).filter(Boolean);
}

function asNumberList(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => asNumber(item, Number.NaN))
    .filter((item) => Number.isFinite(item));
}

function asTeamflowPhase(value: unknown): TeamFlowMessage["phase"] {
  const candidate = asString(value, "SYSTEM").toUpperCase();

  if (
    candidate === "ASSESS" ||
    candidate === "INSTRUCT" ||
    candidate === "WORKER" ||
    candidate === "DECIDE" ||
    candidate === "RETRY" ||
    candidate === "NARRATE" ||
    candidate === "SYSTEM"
  ) {
    return candidate;
  }

  return "SYSTEM";
}

function asTeamflowRole(value: unknown): TeamFlowMessage["role"] {
  const candidate = asString(value, "system").toLowerCase();

  if (
    candidate === "controller" ||
    candidate === "worker" ||
    candidate === "system" ||
    candidate === "narrator"
  ) {
    return candidate;
  }

  return "system";
}

function normalizeEdgeType(type: string): string {
  const upper = type.toUpperCase();

  if (upper === "ATTACK" || upper === "EDGETYPE.CONFLICT") {
    return "CONFLICT";
  }

  if (upper === "EDGETYPE.SUPPORT") {
    return "SUPPORT";
  }

  return upper || "RELATION";
}

function parseNodes(value: unknown): GraphNode[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map((item, index) => {
    const row = asRecord(item);
    const id = asString(row.id, `node-${index}`);
    const type = asString(row.type, "UNKNOWN");

    return {
      id,
      type,
      label: asString(row.label, id),
      status: asString(row.status) || undefined,
      content: asString(row.content) || undefined,
      agentId: asString(row.agent_id) || undefined,
      metadata: asMaybeRecord(row.metadata),
      raw: item,
    };
  });
}

function parseEdges(value: unknown): GraphEdge[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const edges: GraphEdge[] = [];

  for (const [index, item] of value.entries()) {
    const row = asRecord(item);
    const source = asString(row.source).trim();
    const target = asString(row.target).trim();
    const rawType = asString(row.type, "SUPPORT");

    if (!source || !target) {
      continue;
    }

    const type = normalizeEdgeType(rawType);

    edges.push({
      id: asString(row.id, `${source}->${target}:${type}:${index}`),
      source,
      target,
      type,
      weight: asNumber(row.weight, 1),
      metadata: asMaybeRecord(row.metadata),
      raw: item,
    });
  }

  return edges;
}

function parseMemoryInsightItems(value: unknown): MemoryInsightItem[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const parseRelatedCases = (source: unknown) => {
    if (!Array.isArray(source)) {
      return [];
    }

    return source
      .map((item) => {
        const row = asRecord(item);
        const caseId = asString(row.case_id).trim();

        if (!caseId) {
          return null;
        }

        return {
          caseId,
          summary: asString(row.summary, "（无摘要）"),
          sources: asStringList(row.sources),
        };
      })
      .filter((item): item is NonNullable<typeof item> => item !== null);
  };

  return value.map((item) => {
    const row = asRecord(item);
    const cases = asStringList(row.cases);
    const representatives = asStringList(row.representatives);

    return {
      content: asString(row.content),
      side: asString(row.side, "COMMON"),
      cases,
      representatives,
      relatedCases: parseRelatedCases(row.related_cases),
      caseCount: asNumber(row.case_count, cases.length),
      representativeCount: asNumber(
        row.representative_count,
        representatives.length,
      ),
      linkedRound: asNumber(row.linked_round, 0),
    };
  });
}

function parseCaseCatalog(value: unknown): Record<string, { summary: string }> {
  const source = asRecord(value);
  const output: Record<string, { summary: string }> = {};

  for (const [caseIdRaw, itemRaw] of Object.entries(source)) {
    const caseId = asString(caseIdRaw).trim();

    if (!caseId) {
      continue;
    }

    const row = asRecord(itemRaw);
    output[caseId] = { summary: asString(row.summary, "（无摘要）") };
  }

  return output;
}

function parseTaskLayerGraph(payload: Record<string, unknown>): TaskLayerGraph {
  const source = asRecord(payload.task_layer_graph);
  const nodeRows = Array.isArray(source.nodes) ? source.nodes : [];
  const edgeRows = Array.isArray(source.edges) ? source.edges : [];

  const nodes = nodeRows.map((item, index) => {
    const row = asRecord(item);
    const id = asString(row.id, `case-${index}`);

    return {
      id,
      label: asString(row.label, id),
      kind: asString(row.kind) || undefined,
    };
  });

  const edges = edgeRows.map((item, index) => {
    const row = asRecord(item);
    const sourceId = asString(row.source);
    const targetId = asString(row.target);
    const type = asString(row.type, "reference");

    return {
      id: asString(row.id, `${sourceId}->${targetId}:${type}:${index}`),
      source: sourceId,
      target: targetId,
      type,
    };
  });

  return { nodes, edges };
}

function requireNonEmptyString(value: string, fieldName: string): string {
  const text = value.trim();

  if (!text) {
    throw new Error(`ProtocolError: missing required field \`${fieldName}\`.`);
  }

  return text;
}

export function unwrapPayload(raw: unknown): Record<string, unknown> {
  return asRecord(raw);
}

function derivePhase(payload: Record<string, unknown>): DebatePhase {
  if (asBoolean(payload.is_finished)) {
    return "finished";
  }

  if (asBoolean(payload.is_ready_for_adjudication)) {
    return "ready_for_adjudication";
  }

  if (payload.error !== undefined) {
    return "error";
  }

  return asNumber(payload.current_round, 0) > 0 ? "running" : "idle";
}

function deriveMetrics(payload: Record<string, unknown>): DebateMetrics {
  const metrics = asRecord(payload.metrics);
  const graphStats = asRecord(payload.graph_stats);

  return {
    arguments: asNumber(metrics.arguments, asNumber(graphStats.node_count, 0)),
    attacks: asNumber(
      metrics.attacks,
      asNumber(graphStats.edge_attack_count, 0),
    ),
    supports: asNumber(
      metrics.supports,
      asNumber(graphStats.edge_support_count, 0),
    ),
  };
}

function deriveConvergence(payload: Record<string, unknown>) {
  const lastLog = asRecord(payload.last_log);
  const convergence = asRecord(lastLog.convergence ?? payload.convergence);
  const historyFromPayload = asNumberList(payload.convergence_history);
  const historyFromConvergence = asNumberList(convergence.history);

  const history =
    historyFromPayload.length > 0 ? historyFromPayload : historyFromConvergence;

  const deltaPhi = asNumber(convergence.delta_phi, history.at(-1) ?? 0);
  const sma = asNumber(convergence.sma, deltaPhi);
  const epsilon = asNumber(convergence.epsilon, 3);
  const minRounds = asNumber(convergence.min_rounds, 2);
  const windowSize = asNumber(convergence.window_size, 4);

  return {
    deltaPhi,
    sma,
    history,
    epsilon,
    minRounds,
    windowSize,
    isConverged: convergence.is_converged === true,
  };
}

function deriveTermination(
  payload: Record<string, unknown>,
  round: number,
  convergence: {
    sma: number;
    epsilon: number;
    minRounds: number;
  },
) {
  const ready = asBoolean(
    payload.is_ready_for_adjudication ?? payload.is_finished,
  );

  const hasConvergenceSignal =
    payload.convergence_history !== undefined ||
    asRecord(asRecord(payload.last_log).convergence).sma !== undefined ||
    asRecord(payload.convergence).sma !== undefined;

  const convergenceReached =
    hasConvergenceSignal &&
    round >= convergence.minRounds &&
    convergence.sma < convergence.epsilon;

  return {
    ready,
    reason: convergenceReached
      ? ("convergence" as const)
      : ("unknown" as const),
  };
}

export function normalizeSnapshot(
  raw: unknown,
  fallbackSessionId = "",
): DebateSnapshot {
  const payload = unwrapPayload(raw);

  const sessionId = requireNonEmptyString(
    asString(payload.session_id, fallbackSessionId),
    "session_id",
  );

  const round = asNumber(payload.current_round, 0);
  const convergence = deriveConvergence(payload);
  const termination = deriveTermination(payload, round, convergence);

  return {
    sessionId,
    phase: derivePhase(payload),
    round,
    convergence,
    termination,
    winner: asString(payload.winner, "") || null,
    transcript: asStringList(payload.transcript),
    metrics: deriveMetrics(payload) ?? EMPTY_METRICS,
    updatedAt: new Date().toISOString(),
    raw,
  };
}

export function normalizeSnapshotList(raw: unknown): DebateSnapshot[] {
  const payload = unwrapPayload(raw);
  const rows = payload.sessions ?? payload.items;

  if (!Array.isArray(rows)) {
    return [];
  }

  return rows.map((item) => normalizeSnapshot(item));
}

export function normalizeSnapshotIndex(raw: unknown): SnapshotIndexItem[] {
  const payload = unwrapPayload(raw);
  const rows = payload.items;

  if (!Array.isArray(rows)) {
    return [];
  }

  return rows.map((item, index) => {
    const row = asRecord(item);

    return {
      round: asNumber(row.round_idx, index),
      turn: asString(row.turn, ""),
      ts: asNumber(row.ts_ms, 0),
      nodeCount: asNumber(row.node_count, 0),
      edgeCount: asNumber(row.edge_count, 0),
      raw: item,
    };
  });
}

export function normalizeGraph(
  raw: unknown,
  fallbackSessionId = "",
): GraphView {
  const payload = unwrapPayload(raw);
  const graphData = asRecord(payload.graph_data);
  const nodesRaw = graphData.nodes;
  const edgesRaw = graphData.edges;

  if (!Array.isArray(nodesRaw) || !Array.isArray(edgesRaw)) {
    throw new Error(
      "ProtocolError: graph_data.nodes/graph_data.edges must be arrays.",
    );
  }

  return {
    sessionId: requireNonEmptyString(
      asString(payload.session_id, fallbackSessionId),
      "session_id",
    ),
    round: asNumber(payload.round_idx, asNumber(payload.current_round, 0)),
    nodes: parseNodes(nodesRaw),
    edges: parseEdges(edgesRaw),
    focusNodeIds: asIdList(payload.focus_node_ids),
    raw,
  };
}

export function normalizeGraphDiff(
  raw: unknown,
  sessionId: string,
  fromRound: number,
  toRound: number,
): GraphDiffView {
  const payload = unwrapPayload(raw);
  const addedNodeIds = asStringList(payload.added_node_ids);
  const removedNodeIds = asStringList(payload.removed_node_ids);
  const addedEdgeIds = asStringList(payload.added_edge_ids);
  const removedEdgeIds = asStringList(payload.removed_edge_ids);
  const statusChangedNodeIds = asStringList(payload.status_changed_node_ids);
  const changedNodeIds = asStringList(payload.changed_node_ids);
  const changedEdgeIds = asStringList(payload.changed_edge_ids);

  return {
    sessionId: asString(payload.session_id, sessionId),
    fromRound: asNumber(payload.from_round, fromRound),
    toRound: asNumber(payload.to_round, toRound),
    addedNodeIds,
    removedNodeIds,
    addedEdgeIds,
    removedEdgeIds,
    statusChangedNodeIds,
    changedNodeIds:
      changedNodeIds.length > 0
        ? changedNodeIds
        : [
            ...new Set([
              ...addedNodeIds,
              ...removedNodeIds,
              ...statusChangedNodeIds,
            ]),
          ],
    changedEdgeIds:
      changedEdgeIds.length > 0
        ? changedEdgeIds
        : [...new Set([...addedEdgeIds, ...removedEdgeIds])],
    raw,
  };
}

export function normalizeMemory(
  raw: unknown,
  fallbackSessionId = "",
): MemoryView {
  const payload = unwrapPayload(raw);
  const insightItemsRaw = payload.insight_items;
  const recalledCaseIdsRaw = payload.recalled_case_ids;
  const taskLayerGraphRaw = payload.task_layer_graph;

  if (!Array.isArray(insightItemsRaw)) {
    throw new Error(
      "ProtocolError: memory field `insight_items` must be an array.",
    );
  }

  if (!Array.isArray(recalledCaseIdsRaw)) {
    throw new Error(
      "ProtocolError: memory field `recalled_case_ids` must be an array.",
    );
  }

  if (
    taskLayerGraphRaw === null ||
    typeof taskLayerGraphRaw !== "object" ||
    Array.isArray(taskLayerGraphRaw)
  ) {
    throw new Error(
      "ProtocolError: memory field `task_layer_graph` must be an object.",
    );
  }

  return {
    sessionId: requireNonEmptyString(
      asString(payload.session_id, fallbackSessionId),
      "session_id",
    ),
    insightSummaries: asStringList(payload.insight_summaries),
    insightItems: parseMemoryInsightItems(insightItemsRaw),
    representativeCaseIds: asIdList(payload.representative_case_ids),
    caseCatalog: parseCaseCatalog(payload.case_catalog),
    recalledCaseIds: asIdList(recalledCaseIdsRaw),
    recalledCaseCount: asNumber(payload.recalled_case_count, 0),
    taskLayerGraph: parseTaskLayerGraph(payload),
    raw,
  };
}

export function normalizeTimeline(raw: unknown): TimelineEvent[] {
  const payload = unwrapPayload(raw);
  const events = payload.events ?? payload.items;

  if (!Array.isArray(events)) {
    return [];
  }

  return events.map((item, index) => {
    const row = asRecord(item);
    const hasRound = row.round_idx !== undefined;

    return {
      seq: asNumber(row.seq, index + 1),
      ts: asNumber(row.ts_ms, Date.now()),
      eventId: asString(row.event_id) || undefined,
      event: asString(row.event, "event"),
      source: asString(row.source, "engine"),
      roundIdx: hasRound ? asNumber(row.round_idx, 0) : undefined,
      sessionId: asString(row.session_id) || undefined,
      turnUid: asString(row.turn_uid) || undefined,
      data: row.data,
    };
  });
}

export function normalizeTurnArtifacts(raw: unknown): TurnArtifact[] {
  const payload = unwrapPayload(raw);
  const rows = payload.items;

  if (!Array.isArray(rows)) {
    return [];
  }

  return rows.map((item) => {
    const row = asRecord(item);

    return {
      turnUid: asString(row.turn_uid),
      side: asString(row.side, "unknown"),
      round: asNumber(row.round_idx, 0),
      controllerAssessment: row.controller_assessment,
      batchInstructions: row.batch_instructions,
      decisionRaw: asString(row.decision_raw),
      parsedActions: Array.isArray(row.parsed_actions)
        ? row.parsed_actions
        : [],
      executionLogs: asString(row.execution_logs),
      retryHistory: Array.isArray(row.retry_history) ? row.retry_history : [],
      workerReports: Array.isArray(row.worker_reports)
        ? row.worker_reports
        : [],
      narrativeRawSentences: Array.isArray(row.narrative_raw_sentences)
        ? (row.narrative_raw_sentences as unknown[])
        : [],
      narrativePolished: asString(row.narrative_polished),
      raw: item,
    };
  });
}

export function normalizeTeamflowStream(raw: unknown): TeamFlowTurn[] {
  const payload = unwrapPayload(raw);
  const rows = payload.items;

  if (!Array.isArray(rows)) {
    return [];
  }

  return rows.map((item, index) => {
    const row = asRecord(item);
    const messagesRaw = row.messages;

    const messages = Array.isArray(messagesRaw)
      ? messagesRaw.map((messageItem, messageIndex) => {
          const messageRow = asRecord(messageItem);
          const tsValue = asNumber(messageRow.ts_ms, Number.NaN);

          return {
            id: asString(messageRow.id, `msg-${index + 1}-${messageIndex + 1}`),
            phase: asTeamflowPhase(messageRow.phase),
            actor: asString(messageRow.actor, "System"),
            role: asTeamflowRole(messageRow.role),
            title: asString(messageRow.title, "消息"),
            content: asString(messageRow.content),
            ts: Number.isFinite(tsValue) ? tsValue : undefined,
            meta: asMaybeRecord(messageRow.meta),
            raw: messageItem,
          };
        })
      : [];

    const statusRaw = asString(row.status, "partial").toLowerCase();

    const status =
      statusRaw === "done" || statusRaw === "retry" ? statusRaw : "partial";

    return {
      turnUid: asString(row.turn_uid, `turn-${index + 1}`),
      round: asNumber(row.round_idx, index),
      side: asString(row.side, "unknown"),
      status,
      retryCount: asNumber(row.retry_count, 0),
      workerCount: asNumber(row.worker_count, 0),
      messageCount: asNumber(row.message_count, messages.length),
      messages,
      raw: item,
    };
  });
}

export function normalizeFrontendSnapshotItem(
  raw: unknown,
): FrontendSnapshotListItem {
  const payload = asRecord(raw);
  const metadata = asRecord(payload.metadata);

  return {
    snapshotId: asString(payload.snapshot_id),
    label: asString(payload.label, "snapshot"),
    sourceSessionId: asString(payload.source_session_id, "unknown"),
    createdAt: asString(payload.created_at, new Date(0).toISOString()),
    eventCount: asNumber(
      payload.event_count,
      asNumber(metadata.event_count, 0),
    ),
    artifactCount: asNumber(
      payload.artifact_count,
      asNumber(metadata.artifact_count, 0),
    ),
    snapshotCount: asNumber(
      payload.snapshot_count,
      asNumber(metadata.snapshot_count, 0),
    ),
    raw,
  };
}

export function normalizeFrontendSnapshotList(
  raw: unknown,
): FrontendSnapshotListItem[] {
  const payload = unwrapPayload(raw);
  const rows = payload.items;

  if (!Array.isArray(rows)) {
    return [];
  }

  return rows.map((item) => normalizeFrontendSnapshotItem(item));
}

export function normalizeFrontendSnapshotLoadResult(
  raw: unknown,
): FrontendSnapshotLoadResult {
  const payload = asRecord(raw);
  const snapshot = normalizeFrontendSnapshotItem(payload.snapshot ?? {});
  const frontendStateRaw = payload.frontend_state;

  const frontendState =
    frontendStateRaw !== null &&
    typeof frontendStateRaw === "object" &&
    !Array.isArray(frontendStateRaw)
      ? (frontendStateRaw as Record<string, unknown>)
      : {};

  const session = asRecord(payload.session);
  const parsedSessionId = asString(session.session_id, "");
  const snapshotPayloadRaw = payload.snapshot_payload;
  const snapshotPayload =
    snapshotPayloadRaw !== undefined
      ? normalizeSnapshot(snapshotPayloadRaw, parsedSessionId)
      : null;

  return {
    snapshot,
    frontendState,
    session: {
      sessionId: requireNonEmptyString(
        parsedSessionId || snapshotPayload?.sessionId || "",
        "session.session_id",
      ),
      status: asString(session.status, "UNKNOWN"),
      currentRound: asNumber(session.current_round, 0),
      updatedAt: asString(session.updated_at, new Date(0).toISOString()),
    },
    snapshotPayload,
    raw,
  };
}
