import type {
  DebateMetrics,
  DebatePhase,
  DebateSnapshot,
  GraphDiffView,
  GraphEdge,
  GraphNode,
  GraphView,
  MemoryView,
  TimelineEvent,
} from './types'

const EMPTY_METRICS: DebateMetrics = {
  arguments: 0,
  attacks: 0,
  supports: 0,
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object'
    ? (value as Record<string, unknown>)
    : {}
}

function asBoolean(value: unknown): boolean {
  return value === true
}

function asNumber(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return fallback
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

function asStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value
    .map((item) => {
      if (typeof item === 'string') {
        return item
      }
      const row = asRecord(item)
      const role = asString(row.role, asString(row.speaker, 'agent'))
      const content = asString(row.content, asString(row.text))
      return content ? `[${role}] ${content}` : ''
    })
    .filter(Boolean)
}

function parseNodes(value: unknown): GraphNode[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.map((item, index) => {
    const row = asRecord(item)
    const id = asString(row.id, `node-${index}`)
    return {
      id,
      type: asString(row.type, 'UNKNOWN'),
      label: asString(row.label ?? row.content, id),
      status: asString(row.status) || undefined,
    }
  })
}

function parseEdges(value: unknown): GraphEdge[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.map((item, index) => {
    const row = asRecord(item)
    const source = asString(row.source, '')
    const target = asString(row.target, '')
    const type = asString(row.type, 'RELATION')
    return {
      id: asString(row.id, `${source}->${target}:${type}:${index}`),
      source,
      target,
      type,
    }
  })
}

export function unwrapPayload(raw: unknown): Record<string, unknown> {
  const outer = asRecord(raw)
  const candidate = outer.data ?? outer.snapshot ?? outer.state ?? outer.payload
  if (candidate !== undefined) {
    return asRecord(candidate)
  }
  return outer
}

function derivePhase(payload: Record<string, unknown>): DebatePhase {
  if (asBoolean(payload.is_finished ?? payload.finished)) {
    return 'finished'
  }
  if (asBoolean(payload.is_ready_for_adjudication ?? payload.ready_for_adjudication)) {
    return 'ready_for_adjudication'
  }
  if (payload.error !== undefined) {
    return 'error'
  }
  const round = asNumber(payload.current_round ?? payload.round)
  return round > 0 ? 'running' : 'idle'
}

function deriveMetrics(payload: Record<string, unknown>): DebateMetrics {
  const metrics = asRecord(payload.metrics)
  const graphStats = asRecord(payload.graph_stats)
  const graph = asRecord(payload.graph)

  const nodeCount =
    asNumber(metrics.arguments, -1) >= 0
      ? asNumber(metrics.arguments)
      : asNumber(graphStats.node_count, asNumber(graph.nodes_count, 0))

  return {
    arguments: nodeCount,
    attacks: asNumber(metrics.attacks, asNumber(graphStats.edge_attack_count, 0)),
    supports: asNumber(metrics.supports, asNumber(graphStats.edge_support_count, 0)),
  }
}

export function normalizeSnapshot(raw: unknown): DebateSnapshot {
  const payload = unwrapPayload(raw)
  const sessionId = asString(
    payload.session_id ?? payload.sessionId ?? payload.id,
    'local-session',
  )
  const round = asNumber(payload.current_round ?? payload.round)
  const maxRounds = asNumber(payload.max_rounds ?? payload.maxRounds, 6)

  return {
    sessionId,
    phase: derivePhase(payload),
    round,
    maxRounds,
    winner: asString(payload.winner, '') || null,
    transcript: asStringList(payload.transcript),
    metrics: deriveMetrics(payload) ?? EMPTY_METRICS,
    updatedAt: new Date().toISOString(),
    raw,
  }
}

export function normalizeSnapshotList(raw: unknown): DebateSnapshot[] {
  const payload = unwrapPayload(raw)
  const listCandidate = payload.sessions ?? payload.items ?? payload.list
  if (Array.isArray(listCandidate)) {
    return listCandidate.map(normalizeSnapshot)
  }
  if (Array.isArray(raw)) {
    return raw.map(normalizeSnapshot)
  }
  return []
}

export function normalizeGraph(raw: unknown, fallbackSessionId = 'local-session'): GraphView {
  const payload = unwrapPayload(raw)
  const graphData = asRecord(payload.graph_data ?? payload.graph)
  const sessionId = asString(payload.session_id ?? payload.sessionId, fallbackSessionId)
  const round = asNumber(payload.round_idx ?? payload.current_round ?? payload.round)

  return {
    sessionId,
    round,
    nodes: parseNodes(graphData.nodes ?? payload.nodes),
    edges: parseEdges(graphData.edges ?? payload.edges),
    raw,
  }
}

export function normalizeGraphDiff(
  raw: unknown,
  sessionId: string,
  fromRound: number,
  toRound: number,
): GraphDiffView {
  const payload = unwrapPayload(raw)

  return {
    sessionId: asString(payload.session_id ?? payload.sessionId, sessionId),
    fromRound: asNumber(payload.from_round ?? payload.fromRound, fromRound),
    toRound: asNumber(payload.to_round ?? payload.toRound, toRound),
    addedNodeIds: asStringList(payload.added_node_ids ?? payload.addedNodes),
    removedNodeIds: asStringList(payload.removed_node_ids ?? payload.removedNodes),
    addedEdgeIds: asStringList(payload.added_edge_ids ?? payload.addedEdges),
    removedEdgeIds: asStringList(payload.removed_edge_ids ?? payload.removedEdges),
    raw,
  }
}

export function normalizeMemory(raw: unknown, fallbackSessionId = 'local-session'): MemoryView {
  const payload = unwrapPayload(raw)
  const insights = payload.insights ?? payload.insight_summaries
  const taskLayer = asRecord(payload.task_layer)

  return {
    sessionId: asString(payload.session_id ?? payload.sessionId, fallbackSessionId),
    insightSummaries: asStringList(insights),
    staticHistoryCount: asNumber(payload.static_history_count ?? payload.staticHistoryCount),
    dynamicLawCaseCount: asNumber(payload.dynamic_law_case_count ?? payload.dynamicLawCaseCount),
    taskLayerNodeCount: asNumber(taskLayer.node_count ?? payload.task_layer_node_count),
    raw,
  }
}

export function normalizeTimeline(raw: unknown): TimelineEvent[] {
  const payload = unwrapPayload(raw)
  const events = payload.events ?? payload.items ?? raw
  if (!Array.isArray(events)) {
    return []
  }

  return events.map((item, index) => {
    const row = asRecord(item)
    return {
      seq: asNumber(row.seq, index + 1),
      ts: asNumber(row.ts_ms ?? row.ts, Date.now()),
      event: asString(row.event, 'event'),
      source: asString(row.source, 'engine'),
      data: row.data,
    }
  })
}

export function buildLocalGraphDiff(
  previous: GraphView,
  current: GraphView,
  sessionId: string,
): GraphDiffView {
  const prevNodeIds = new Set(previous.nodes.map((node) => node.id))
  const currNodeIds = new Set(current.nodes.map((node) => node.id))
  const prevEdgeIds = new Set(previous.edges.map((edge) => edge.id))
  const currEdgeIds = new Set(current.edges.map((edge) => edge.id))

  return {
    sessionId,
    fromRound: previous.round,
    toRound: current.round,
    addedNodeIds: [...currNodeIds].filter((id) => !prevNodeIds.has(id)),
    removedNodeIds: [...prevNodeIds].filter((id) => !currNodeIds.has(id)),
    addedEdgeIds: [...currEdgeIds].filter((id) => !prevEdgeIds.has(id)),
    removedEdgeIds: [...prevEdgeIds].filter((id) => !currEdgeIds.has(id)),
  }
}
