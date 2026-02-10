export type DebatePhase =
  | "idle"
  | "running"
  | "ready_for_adjudication"
  | "finished"
  | "error";

export interface DebateMetrics {
  arguments: number;
  attacks: number;
  supports: number;
}

export interface DebateSnapshot {
  sessionId: string;
  phase: DebatePhase;
  round: number;
  maxRounds: number;
  winner: string | null;
  transcript: string[];
  metrics: DebateMetrics;
  updatedAt: string;
  raw?: unknown;
}

export interface CreateSessionInput {
  caseId?: string;
  plaintiffClaim?: string;
  defendantAnswer?: string;
  maxRounds?: number;
}

export interface GraphNode {
  id: string;
  type: string;
  label: string;
  status?: string;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
}

export interface GraphView {
  sessionId: string;
  round: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  raw?: unknown;
}

export interface GraphDiffView {
  sessionId: string;
  fromRound: number;
  toRound: number;
  addedNodeIds: string[];
  removedNodeIds: string[];
  addedEdgeIds: string[];
  removedEdgeIds: string[];
  raw?: unknown;
}

export interface MemoryView {
  sessionId: string;
  insightSummaries: string[];
  staticHistoryCount: number;
  dynamicLawCaseCount: number;
  taskLayerNodeCount: number;
  raw?: unknown;
}

export interface TimelineEvent {
  seq: number;
  ts: number;
  event: string;
  source: string;
  data?: unknown;
}

export interface AdapterCapabilities {
  supportsStreaming: boolean;
  supportsDiff: boolean;
  transport: "mock" | "http";
}

export interface SessionAdapter {
  createSession(input?: CreateSessionInput): Promise<DebateSnapshot>;
  step(sessionId: string): Promise<DebateSnapshot>;
  adjudicate(sessionId: string): Promise<DebateSnapshot>;
  getSnapshot(sessionId: string): Promise<DebateSnapshot>;
  listSessions(): Promise<DebateSnapshot[]>;
}

export interface GraphAdapter {
  getGraph(sessionId: string): Promise<GraphView>;

  getGraphDiff(
    sessionId: string,
    fromRound: number,
    toRound: number,
  ): Promise<GraphDiffView>;
}

export interface InsightAdapter {
  getMemory(sessionId: string): Promise<MemoryView>;
  getTimeline(sessionId: string, limit?: number): Promise<TimelineEvent[]>;
}

export interface EngineAdapter extends SessionAdapter {
  readonly capabilities: AdapterCapabilities;
  readonly session: SessionAdapter;
  readonly graph: GraphAdapter;
  readonly insight: InsightAdapter;
}
