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

export interface SnapshotIndexItem {
  round: number;
  turn: string;
  ts: number;
  nodeCount: number;
  edgeCount: number;
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
  sessionId?: string;
  turnUid?: string;
  data?: unknown;
}

export interface TurnArtifact {
  turnUid: string;
  side: string;
  round: number;
  decisionRaw: string;
  parsedActions: unknown[];
  executionLogs: string;
  retryHistory: unknown[];
  workerReports: unknown[];
  raw?: unknown;
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
  getSnapshots(sessionId: string): Promise<SnapshotIndexItem[]>;
}

export interface GraphAdapter {
  getGraph(sessionId: string): Promise<GraphView>;
  getGraphAtRound(sessionId: string, round: number): Promise<GraphView>;

  getGraphDiff(
    sessionId: string,
    fromRound: number,
    toRound: number,
  ): Promise<GraphDiffView>;
}

export interface InsightAdapter {
  getMemory(sessionId: string): Promise<MemoryView>;
  getTimeline(sessionId: string, limit?: number): Promise<TimelineEvent[]>;

  subscribeTimeline(
    sessionId: string,
    onEvent: (event: TimelineEvent) => void,
    options?: { fromSeq?: number; onError?: (error: Error) => void },
  ): () => void;

  getTurnArtifacts(
    sessionId: string,
    options?: { turnUid?: string; limit?: number },
  ): Promise<TurnArtifact[]>;
}

export interface EngineAdapter extends SessionAdapter {
  readonly capabilities: AdapterCapabilities;
  readonly session: SessionAdapter;
  readonly graph: GraphAdapter;
  readonly insight: InsightAdapter;
}
