import type {
  MemoryCaseSnapshot,
  MemoryInsightItem,
  DemoKeyframe,
  DemoRunResult,
  DebateMetrics,
  DebatePhase,
  DebateSnapshot,
  DebugBundleView,
  GraphDiffView,
  GraphEdge,
  GraphNode,
  GraphView,
  MemoryView,
  TaskLayerGraph,
  ReplayExportView,
  FrontendSnapshotListItem,
  FrontendSnapshotLoadResult,
  SnapshotIndexItem,
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

  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }

  if (typeof value === "boolean" || typeof value === "bigint") {
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
      const role = asString(row.role, asString(row.speaker, "agent"));
      const content = asString(row.content, asString(row.text));
      return content ? `[${role}] ${content}` : "";
    })
    .filter(Boolean);
}

function asIdList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => {
      if (typeof item === "string") {
        return item.trim();
      }

      if (
        typeof item === "number" ||
        typeof item === "boolean" ||
        typeof item === "bigint"
      ) {
        return String(item);
      }

      return "";
    })
    .filter(Boolean);
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

  if (upper === "ATTACK") {
    return "CONFLICT";
  }

  if (upper === "EDGETYPE.CONFLICT") {
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

    const id = asString(
      row.id ?? row.node_id ?? row.nodeId ?? row.uid,
      `node-${index}`,
    );

    const type = asString(
      row.type ?? row.kind ?? row.node_type ?? row.nodeType,
      "UNKNOWN",
    );

    return {
      id,
      type,
      label: asString(row.label ?? row.name ?? row.content, id),
      status: asString(row.status) || undefined,
      content: asString(row.content) || undefined,
      agentId: asString(row.agent_id ?? row.agentId) || undefined,
      metadata: asMaybeRecord(row.metadata),
      raw: item,
    };
  });
}

function parseMemoryInsightItems(value: unknown): MemoryInsightItem[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map((item) => {
    const row = asRecord(item);
    const cases = asStringList(row.cases);
    const representatives = asStringList(row.representatives);

    return {
      content: asString(row.content),
      side: asString(row.side, "COMMON"),
      cases,
      representatives,
      caseCount: asNumber(row.case_count ?? row.caseCount, cases.length),
      representativeCount: asNumber(
        row.representative_count ?? row.representativeCount,
        representatives.length,
      ),
      linkedRound: asNumber(row.linked_round ?? row.linkedRound, 0),
    };
  });
}

function parseMemoryCaseSnapshots(value: unknown): MemoryCaseSnapshot[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map((item, index) => {
    const row = asRecord(item);

    return {
      round: asNumber(row.round_idx ?? row.round, index),
      turn: asString(row.turn),
      ts: asNumber(row.ts_ms ?? row.ts, 0),
      nodeCount: asNumber(row.node_count ?? row.nodeCount, 0),
      edgeCount: asNumber(row.edge_count ?? row.edgeCount, 0),
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

    const source = asString(
      row.source ??
        row.from ??
        row.u ??
        row.src ??
        row.from_id ??
        row.fromId ??
        row.source_id ??
        row.sourceId,
      "",
    );

    const target = asString(
      row.target ??
        row.to ??
        row.v ??
        row.dst ??
        row.to_id ??
        row.toId ??
        row.target_id ??
        row.targetId,
      "",
    );

    const rawType = asString(
      row.type ?? row.relation ?? row.edge_type ?? row.edgeType ?? row.kind,
      "SUPPORT",
    );

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

function parseTaskLayerGraph(
  payload: Record<string, unknown>,
  taskLayer: Record<string, unknown>,
): TaskLayerGraph {
  const source = asRecord(
    payload.task_layer_graph ??
      payload.taskLayerGraph ??
      taskLayer.graph ??
      taskLayer.task_layer_graph,
  );

  const nodeRows = Array.isArray(source.nodes) ? source.nodes : [];
  const edgeRows = Array.isArray(source.edges) ? source.edges : [];

  const nodes = nodeRows.map((item, index) => {
    const row = asRecord(item);
    const id = asString(row.id ?? row.case_id ?? row.caseId, `case-${index}`);

    return {
      id,
      label: asString(row.label ?? row.title ?? row.name, id),
      kind: asString(row.kind ?? row.type) || undefined,
    };
  });

  const edges = edgeRows.map((item, index) => {
    const row = asRecord(item);
    const sourceId = asString(row.source ?? row.from ?? row.u, "");
    const targetId = asString(row.target ?? row.to ?? row.v, "");
    const type = asString(row.type ?? row.kind ?? row.relation, "reference");

    return {
      id: asString(row.id, `${sourceId}->${targetId}:${type}:${index}`),
      source: sourceId,
      target: targetId,
      type,
    };
  });

  return { nodes, edges };
}

export function unwrapPayload(raw: unknown): Record<string, unknown> {
  const outer = asRecord(raw);

  const candidate =
    outer.data ?? outer.snapshot ?? outer.state ?? outer.payload;

  if (candidate !== undefined) {
    return asRecord(candidate);
  }

  return outer;
}

function derivePhase(payload: Record<string, unknown>): DebatePhase {
  if (asBoolean(payload.is_finished ?? payload.finished)) {
    return "finished";
  }

  if (
    asBoolean(
      payload.is_ready_for_adjudication ?? payload.ready_for_adjudication,
    )
  ) {
    return "ready_for_adjudication";
  }

  if (payload.error !== undefined) {
    return "error";
  }

  const round = asNumber(payload.current_round ?? payload.round);
  return round > 0 ? "running" : "idle";
}

function deriveMetrics(payload: Record<string, unknown>): DebateMetrics {
  const metrics = asRecord(payload.metrics);
  const graphStats = asRecord(payload.graph_stats);
  const graph = asRecord(payload.graph);

  const nodeCount =
    asNumber(metrics.arguments, -1) >= 0
      ? asNumber(metrics.arguments)
      : asNumber(graphStats.node_count, asNumber(graph.nodes_count, 0));

  return {
    arguments: nodeCount,
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
  const lastLog = asRecord(payload.last_log ?? payload.lastLog);
  const convergence = asRecord(lastLog.convergence ?? payload.convergence);

  const historyFromPayload = asNumberList(
    payload.convergence_history ?? payload.convergenceHistory,
  );

  const historyFromConvergence = asNumberList(convergence.history);

  const history =
    historyFromPayload.length > 0 ? historyFromPayload : historyFromConvergence;

  const deltaPhi = asNumber(
    convergence.delta_phi ?? convergence.deltaPhi ?? history.at(-1),
    0,
  );

  const sma = asNumber(convergence.sma, deltaPhi);

  const epsilon = asNumber(
    convergence.epsilon ?? payload.convergence_epsilon ?? payload.epsilon,
    3,
  );

  const minRounds = asNumber(
    convergence.min_rounds ?? convergence.minRounds ?? payload.min_rounds,
    2,
  );

  const windowSize = asNumber(
    convergence.window_size ?? convergence.windowSize ?? payload.window_size,
    4,
  );

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
    payload.is_ready_for_adjudication ??
      payload.ready_for_adjudication ??
      payload.is_finished ??
      payload.finished,
  );

  const convergenceInPayload =
    payload.convergence_history !== undefined ||
    payload.convergenceHistory !== undefined ||
    asRecord(asRecord(payload.last_log ?? payload.lastLog).convergence).sma !==
      undefined ||
    asRecord(payload.convergence).sma !== undefined;

  const convergenceReached =
    convergenceInPayload &&
    round >= convergence.minRounds &&
    convergence.sma < convergence.epsilon;

  if (convergenceReached) {
    return {
      ready,
      reason: "convergence" as const,
    };
  }

  return {
    ready,
    reason: "unknown" as const,
  };
}

export function normalizeSnapshot(
  raw: unknown,
  fallbackSessionId = "",
): DebateSnapshot {
  const payload = unwrapPayload(raw);

  const sessionId = asString(
    payload.session_id ?? payload.sessionId ?? payload.id,
    fallbackSessionId,
  );

  const round = asNumber(payload.current_round ?? payload.round);
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
  const listCandidate = payload.sessions ?? payload.items ?? payload.list;

  if (Array.isArray(listCandidate)) {
    return listCandidate.map((item) => normalizeSnapshot(item));
  }

  if (Array.isArray(raw)) {
    return raw.map((item) => normalizeSnapshot(item));
  }

  return [];
}

export function normalizeSnapshotIndex(raw: unknown): SnapshotIndexItem[] {
  const payload = unwrapPayload(raw);
  const listCandidate = payload.items ?? payload.snapshots ?? raw;

  if (!Array.isArray(listCandidate)) {
    return [];
  }

  return listCandidate.map((item, index) => {
    const row = asRecord(item);

    return {
      round: asNumber(row.round_idx ?? row.round, index),
      turn: asString(row.turn, ""),
      ts: asNumber(row.ts_ms ?? row.ts, 0),
      nodeCount: asNumber(row.node_count ?? row.nodeCount, 0),
      edgeCount: asNumber(row.edge_count ?? row.edgeCount, 0),
      raw: item,
    };
  });
}

export function normalizeGraph(
  raw: unknown,
  fallbackSessionId = "",
): GraphView {
  const payload = unwrapPayload(raw);
  const graphData = asRecord(payload.graph_data ?? payload.graph);

  const sessionId = asString(
    payload.session_id ?? payload.sessionId,
    fallbackSessionId,
  );

  const round = asNumber(
    payload.round_idx ?? payload.current_round ?? payload.round,
  );

  return {
    sessionId,
    round,
    nodes: parseNodes(graphData.nodes ?? payload.nodes),
    edges: parseEdges(
      graphData.edges ?? graphData.links ?? payload.edges ?? payload.links,
    ),
    focusNodeIds: asIdList(
      payload.focus_node_ids ??
        payload.focusNodeIds ??
        graphData.focus_node_ids,
    ),
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

  const addedNodeIds = asStringList(
    payload.added_node_ids ?? payload.addedNodes,
  );

  const removedNodeIds = asStringList(
    payload.removed_node_ids ?? payload.removedNodes,
  );

  const addedEdgeIds = asStringList(
    payload.added_edge_ids ?? payload.addedEdges,
  );

  const removedEdgeIds = asStringList(
    payload.removed_edge_ids ?? payload.removedEdges,
  );

  const statusChangedNodeIds = asStringList(
    payload.status_changed_node_ids ?? payload.statusChangedNodeIds,
  );

  const changedNodeIds = asStringList(
    payload.changed_node_ids ?? payload.changedNodes,
  );

  const changedEdgeIds = asStringList(
    payload.changed_edge_ids ?? payload.changedEdges,
  );

  return {
    sessionId: asString(payload.session_id ?? payload.sessionId, sessionId),
    fromRound: asNumber(payload.from_round ?? payload.fromRound, fromRound),
    toRound: asNumber(payload.to_round ?? payload.toRound, toRound),
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
  const insights = payload.insights ?? payload.insight_summaries;
  const taskLayer = asRecord(payload.task_layer);
  const insightItems = parseMemoryInsightItems(payload.insight_items);
  const caseSnapshots = parseMemoryCaseSnapshots(payload.case_snapshots);
  const taskLayerGraph = parseTaskLayerGraph(payload, taskLayer);

  return {
    sessionId: asString(
      payload.session_id ?? payload.sessionId,
      fallbackSessionId,
    ),
    insightSummaries: asStringList(insights),
    insightItems,
    representativeCaseIds: asStringList(payload.representative_case_ids),
    staticHistoryCount: asNumber(
      payload.static_history_count ?? payload.staticHistoryCount,
    ),
    dynamicLawCaseCount: asNumber(
      payload.dynamic_law_case_count ?? payload.dynamicLawCaseCount,
    ),
    taskLayerNodeCount: asNumber(
      taskLayer.node_count ?? payload.task_layer_node_count,
    ),
    taskLayerEdgeCount: asNumber(
      taskLayer.edge_count ?? payload.task_layer_edge_count,
    ),
    taskLayerGraph,
    caseSnapshots,
    raw,
  };
}

export function normalizeTimeline(raw: unknown): TimelineEvent[] {
  const payload = unwrapPayload(raw);
  const events = payload.events ?? payload.items ?? raw;

  if (!Array.isArray(events)) {
    return [];
  }

  return events.map((item, index) => {
    const row = asRecord(item);

    return {
      seq: asNumber(row.seq, index + 1),
      ts: asNumber(row.ts_ms ?? row.ts, Date.now()),
      eventId: asString(row.event_id ?? row.eventId) || undefined,
      event: asString(row.event, "event"),
      source: asString(row.source, "engine"),
      roundIdx:
        row.round_idx !== undefined || row.roundIdx !== undefined
          ? asNumber(row.round_idx ?? row.roundIdx)
          : undefined,
      sessionId: asString(row.session_id ?? row.sessionId) || undefined,
      turnUid: asString(row.turn_uid ?? row.turnUid) || undefined,
      data: row.data,
    };
  });
}

export function normalizeTurnArtifacts(raw: unknown): TurnArtifact[] {
  const payload = unwrapPayload(raw);
  const list = payload.items ?? payload.artifacts ?? raw;

  if (!Array.isArray(list)) {
    return [];
  }

  return list.map((item) => {
    const row = asRecord(item);

    return {
      turnUid: asString(row.turn_uid ?? row.turnUid),
      side: asString(row.side, "unknown"),
      round: asNumber(row.round_idx ?? row.round),
      controllerAssessment:
        row.controller_assessment ?? row.controllerAssessment,
      batchInstructions: row.batch_instructions ?? row.batchInstructions,
      decisionRaw: asString(row.decision_raw ?? row.decisionRaw),
      parsedActions: Array.isArray(row.parsed_actions)
        ? row.parsed_actions
        : Array.isArray(row.parsedActions)
          ? row.parsedActions
          : [],
      executionLogs: asString(row.execution_logs ?? row.executionLogs),
      retryHistory: Array.isArray(row.retry_history)
        ? row.retry_history
        : Array.isArray(row.retryHistory)
          ? row.retryHistory
          : [],
      workerReports: Array.isArray(row.worker_reports)
        ? row.worker_reports
        : Array.isArray(row.workerReports)
          ? row.workerReports
          : [],
      narrativeRawSentences: Array.isArray(
        row.narrative_raw_sentences ?? row.narrativeRawSentences,
      )
        ? ((row.narrative_raw_sentences ??
            row.narrativeRawSentences) as unknown[])
        : [],
      narrativePolished: asString(
        row.narrative_polished ?? row.narrativePolished,
      ),
      raw: item,
    };
  });
}

export function normalizeTeamflowStream(raw: unknown): TeamFlowTurn[] {
  const payload = unwrapPayload(raw);
  const list = payload.items ?? payload.turns ?? raw;

  if (!Array.isArray(list)) {
    return [];
  }

  return list.map((item, index) => {
    const row = asRecord(item);
    const messagesRaw = row.messages;

    const messages = Array.isArray(messagesRaw)
      ? messagesRaw.map((messageItem, messageIndex) => {
          const messageRow = asRecord(messageItem);
          const meta = asMaybeRecord(messageRow.meta);
          const tsValue = asNumber(
            messageRow.ts_ms ?? messageRow.ts,
            Number.NaN,
          );

          return {
            id: asString(messageRow.id, `msg-${index + 1}-${messageIndex + 1}`),
            phase: asTeamflowPhase(messageRow.phase),
            actor: asString(messageRow.actor, "System"),
            role: asTeamflowRole(messageRow.role),
            title: asString(messageRow.title, "消息"),
            content: asString(messageRow.content),
            ts: Number.isFinite(tsValue) ? tsValue : undefined,
            meta,
            raw: messageItem,
          };
        })
      : [];

    const statusRaw = asString(row.status, "partial").toLowerCase();

    const status =
      statusRaw === "done" || statusRaw === "retry" ? statusRaw : "partial";

    return {
      turnUid: asString(row.turn_uid ?? row.turnUid, `turn-${index + 1}`),
      round: asNumber(row.round_idx ?? row.round, index),
      side: asString(row.side, "unknown"),
      status,
      retryCount: asNumber(row.retry_count ?? row.retryCount, 0),
      workerCount: asNumber(row.worker_count ?? row.workerCount, 0),
      messageCount: asNumber(
        row.message_count ?? row.messageCount,
        messages.length,
      ),
      messages,
      raw: item,
    };
  });
}

export function normalizeDebugBundle(
  raw: unknown,
  fallbackSessionId = "",
): DebugBundleView {
  const payload = unwrapPayload(raw);
  const summary = asRecord(payload.snapshot_summary ?? payload.snapshotSummary);

  const artifactRaw =
    payload.latest_turn_artifact ?? payload.latestTurnArtifact;

  const artifactList = artifactRaw !== undefined ? [artifactRaw] : [];
  const normalizedArtifact = normalizeTurnArtifacts(artifactList)[0];

  return {
    sessionId: asString(
      payload.session_id ?? payload.sessionId,
      fallbackSessionId,
    ),
    round: asNumber(payload.round_idx ?? payload.round),
    turnUid: asString(payload.turn_uid ?? payload.turnUid),
    status: asString(payload.status, "UNKNOWN"),
    lastError: asString(payload.last_error ?? payload.lastError),
    snapshotSummary: {
      phase: asString(summary.phase, "UNKNOWN"),
      nodeCount: asNumber(summary.node_count ?? summary.nodeCount),
      edgeCount: asNumber(summary.edge_count ?? summary.edgeCount),
      claimCount: asNumber(summary.claim_count ?? summary.claimCount),
      conflictCount: asNumber(summary.conflict_count ?? summary.conflictCount),
    },
    recentEvents: normalizeTimeline(
      payload.recent_events ?? payload.events ?? [],
    ),
    latestTurnArtifact: normalizedArtifact,
    generatedAt: asString(payload.generated_at ?? payload.generatedAt),
    raw,
  };
}

export function normalizeDemoKeyframes(raw: unknown): DemoKeyframe[] {
  const payload = unwrapPayload(raw);
  const list = payload.items ?? payload.keyframes ?? raw;

  if (!Array.isArray(list)) {
    return [];
  }

  return list.map((item) => {
    const row = asRecord(item);

    return {
      sessionId: asString(row.session_id ?? row.sessionId),
      event: asString(row.event, "event"),
      reason: asString(row.reason, ""),
      round: asNumber(row.round_idx ?? row.round),
      turnUid: asString(row.turn_uid ?? row.turnUid),
      ts: asNumber(row.ts_ms ?? row.ts, Date.now()),
      data: row.data,
      raw: item,
    };
  });
}

export function normalizeDemoRunResult(
  raw: unknown,
  fallbackSessionId = "",
): DemoRunResult {
  const payload = unwrapPayload(raw);
  const session = asRecord(payload.session);
  const summary = asRecord(payload.demo_summary ?? payload.summary);

  const keyframes = normalizeDemoKeyframes(
    payload.keyframes ?? payload.items ?? [],
  );

  return {
    sessionId: asString(
      session.session_id ?? payload.session_id ?? payload.sessionId,
      fallbackSessionId,
    ),
    status: asString(session.status ?? payload.status, "UNKNOWN"),
    stepsExecuted: asNumber(
      summary.steps_executed ?? summary.stepsExecuted ?? payload.steps_executed,
    ),
    endedBy: asString(
      summary.ended_by ?? summary.endedBy ?? payload.ended_by,
      "unknown",
    ),
    keyframes,
    raw,
  };
}

export function normalizeReplayExport(
  raw: unknown,
  fallbackSessionId = "",
): ReplayExportView {
  const payload = unwrapPayload(raw);
  const session = asRecord(payload.session);
  const metadata = asRecord(payload.metadata);
  const events = payload.events;
  const artifacts = payload.turn_artifacts ?? payload.turnArtifacts;
  const snapshots = payload.snapshots ?? payload.snapshot_list;

  const eventCount = Array.isArray(events)
    ? events.length
    : asNumber(metadata.event_count ?? metadata.eventCount, 0);

  const artifactCount = Array.isArray(artifacts)
    ? artifacts.length
    : asNumber(metadata.artifact_count ?? metadata.artifactCount, 0);

  const snapshotCount = Array.isArray(snapshots)
    ? snapshots.length
    : asNumber(metadata.snapshot_count ?? metadata.snapshotCount, 0);

  return {
    sessionId: asString(
      session.session_id ?? payload.session_id ?? payload.sessionId,
      fallbackSessionId,
    ),
    eventCount,
    artifactCount,
    snapshotCount,
    raw,
  };
}

export function normalizeFrontendSnapshotItem(
  raw: unknown,
): FrontendSnapshotListItem {
  const payload = asRecord(raw);
  const metadata = asRecord(payload.metadata);

  return {
    snapshotId: asString(payload.snapshot_id ?? payload.snapshotId),
    label: asString(payload.label, "snapshot"),
    sourceSessionId: asString(
      payload.source_session_id ?? payload.sourceSessionId,
      "unknown",
    ),
    createdAt: asString(
      payload.created_at ?? payload.createdAt,
      new Date(0).toISOString(),
    ),
    eventCount: asNumber(
      payload.event_count ?? payload.eventCount ?? metadata.event_count,
      0,
    ),
    artifactCount: asNumber(
      payload.artifact_count ??
        payload.artifactCount ??
        metadata.artifact_count,
      0,
    ),
    snapshotCount: asNumber(
      payload.snapshot_count ??
        payload.snapshotCount ??
        metadata.snapshot_count,
      0,
    ),
    raw,
  };
}

export function normalizeFrontendSnapshotList(
  raw: unknown,
): FrontendSnapshotListItem[] {
  const payload = unwrapPayload(raw);
  const rows = payload.items ?? payload.snapshots ?? [];

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
  const frontendStateRaw = payload.frontend_state ?? payload.frontendState;

  const frontendState =
    frontendStateRaw !== null &&
    typeof frontendStateRaw === "object" &&
    !Array.isArray(frontendStateRaw)
      ? (frontendStateRaw as Record<string, unknown>)
      : {};

  const session = asRecord(payload.session);
  const snapshotPayloadRaw =
    payload.snapshot_payload ?? payload.snapshotPayload;

  const parsedSessionId = asString(session.session_id ?? session.sessionId, "");

  const snapshotPayload =
    snapshotPayloadRaw !== undefined
      ? normalizeSnapshot(snapshotPayloadRaw, parsedSessionId)
      : null;

  const resolvedSessionId = parsedSessionId || snapshotPayload?.sessionId || "";

  return {
    snapshot,
    frontendState,
    session: {
      sessionId: resolvedSessionId,
      status: asString(session.status, "UNKNOWN"),
      currentRound: asNumber(session.current_round ?? session.currentRound, 0),
      updatedAt: asString(
        session.updated_at ?? session.updatedAt,
        new Date(0).toISOString(),
      ),
    },
    snapshotPayload,
    raw,
  };
}

export function buildLocalGraphDiff(
  previous: GraphView,
  current: GraphView,
  sessionId: string,
): GraphDiffView {
  const prevNodeIds = new Set(previous.nodes.map((node) => node.id));
  const currNodeIds = new Set(current.nodes.map((node) => node.id));
  const prevEdgeIds = new Set(previous.edges.map((edge) => edge.id));
  const currEdgeIds = new Set(current.edges.map((edge) => edge.id));

  const previousNodeById = new Map(
    previous.nodes.map((node) => [node.id, node]),
  );

  const statusChangedNodeIds: string[] = [];

  for (const node of current.nodes) {
    const prev = previousNodeById.get(node.id);

    if (prev && (prev.status ?? "") !== (node.status ?? "")) {
      statusChangedNodeIds.push(node.id);
    }
  }

  const addedNodeIds = [...currNodeIds].filter((id) => !prevNodeIds.has(id));
  const removedNodeIds = [...prevNodeIds].filter((id) => !currNodeIds.has(id));
  const addedEdgeIds = [...currEdgeIds].filter((id) => !prevEdgeIds.has(id));
  const removedEdgeIds = [...prevEdgeIds].filter((id) => !currEdgeIds.has(id));

  return {
    sessionId,
    fromRound: previous.round,
    toRound: current.round,
    addedNodeIds,
    removedNodeIds,
    addedEdgeIds,
    removedEdgeIds,
    statusChangedNodeIds,
    changedNodeIds: [
      ...new Set([...addedNodeIds, ...removedNodeIds, ...statusChangedNodeIds]),
    ],
    changedEdgeIds: [...new Set([...addedEdgeIds, ...removedEdgeIds])],
  };
}
