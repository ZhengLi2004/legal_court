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
  ReplayExportView,
  SnapshotIndexItem,
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
  return typeof value === "string" ? value : fallback;
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

function parseNodes(value: unknown): GraphNode[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map((item, index) => {
    const row = asRecord(item);
    const id = asString(row.id, `node-${index}`);

    return {
      id,
      type: asString(row.type, "UNKNOWN"),
      label: asString(row.label ?? row.content, id),
      status: asString(row.status) || undefined,
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

  return value.map((item, index) => {
    const row = asRecord(item);
    const source = asString(row.source, "");
    const target = asString(row.target, "");
    const type = asString(row.type, "RELATION");

    return {
      id: asString(row.id, `${source}->${target}:${type}:${index}`),
      source,
      target,
      type,
    };
  });
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

export function normalizeSnapshot(raw: unknown): DebateSnapshot {
  const payload = unwrapPayload(raw);

  const sessionId = asString(
    payload.session_id ?? payload.sessionId ?? payload.id,
    "local-session",
  );

  const round = asNumber(payload.current_round ?? payload.round);
  const maxRounds = asNumber(payload.max_rounds ?? payload.maxRounds, 6);

  return {
    sessionId,
    phase: derivePhase(payload),
    round,
    maxRounds,
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
    return listCandidate.map(normalizeSnapshot);
  }

  if (Array.isArray(raw)) {
    return raw.map(normalizeSnapshot);
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
  fallbackSessionId = "local-session",
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
    edges: parseEdges(graphData.edges ?? payload.edges),
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

  return {
    sessionId: asString(payload.session_id ?? payload.sessionId, sessionId),
    fromRound: asNumber(payload.from_round ?? payload.fromRound, fromRound),
    toRound: asNumber(payload.to_round ?? payload.toRound, toRound),
    addedNodeIds: asStringList(payload.added_node_ids ?? payload.addedNodes),
    removedNodeIds: asStringList(
      payload.removed_node_ids ?? payload.removedNodes,
    ),
    addedEdgeIds: asStringList(payload.added_edge_ids ?? payload.addedEdges),
    removedEdgeIds: asStringList(
      payload.removed_edge_ids ?? payload.removedEdges,
    ),
    raw,
  };
}

export function normalizeMemory(
  raw: unknown,
  fallbackSessionId = "local-session",
): MemoryView {
  const payload = unwrapPayload(raw);
  const insights = payload.insights ?? payload.insight_summaries;
  const taskLayer = asRecord(payload.task_layer);
  const insightItems = parseMemoryInsightItems(payload.insight_items);
  const caseSnapshots = parseMemoryCaseSnapshots(payload.case_snapshots);

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

export function normalizeDebugBundle(
  raw: unknown,
  fallbackSessionId = "local-session",
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
  fallbackSessionId = "local-session",
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
  fallbackSessionId = "local-session",
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

export function buildLocalGraphDiff(
  previous: GraphView,
  current: GraphView,
  sessionId: string,
): GraphDiffView {
  const prevNodeIds = new Set(previous.nodes.map((node) => node.id));
  const currNodeIds = new Set(current.nodes.map((node) => node.id));
  const prevEdgeIds = new Set(previous.edges.map((edge) => edge.id));
  const currEdgeIds = new Set(current.edges.map((edge) => edge.id));

  return {
    sessionId,
    fromRound: previous.round,
    toRound: current.round,
    addedNodeIds: [...currNodeIds].filter((id) => !prevNodeIds.has(id)),
    removedNodeIds: [...prevNodeIds].filter((id) => !currNodeIds.has(id)),
    addedEdgeIds: [...currEdgeIds].filter((id) => !prevEdgeIds.has(id)),
    removedEdgeIds: [...prevEdgeIds].filter((id) => !currEdgeIds.has(id)),
  };
}
