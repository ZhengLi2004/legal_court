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
  convergence: DebateConvergence;
  termination: DebateTermination;
  winner: string | null;
  transcript: string[];
  metrics: DebateMetrics;
  updatedAt: string;
  raw?: unknown;
}

export interface DebateConvergence {
  deltaPhi: number;
  sma: number;
  history: number[];
  epsilon: number;
  minRounds: number;
  windowSize: number;
  isConverged: boolean;
}

export interface DebateTermination {
  ready: boolean;
  reason: "convergence" | "max_rounds" | "unknown";
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
  content?: string;
  agentId?: string;
  metadata?: Record<string, unknown>;
  raw?: unknown;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  weight?: number;
  metadata?: Record<string, unknown>;
  raw?: unknown;
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
  insightItems: MemoryInsightItem[];
  representativeCaseIds: string[];
  staticHistoryCount: number;
  dynamicLawCaseCount: number;
  taskLayerNodeCount: number;
  taskLayerEdgeCount: number;
  caseSnapshots: MemoryCaseSnapshot[];
  raw?: unknown;
}

export interface MemoryInsightItem {
  content: string;
  side: string;
  cases: string[];
  representatives: string[];
  caseCount: number;
  representativeCount: number;
  linkedRound: number;
}

export interface MemoryCaseSnapshot {
  round: number;
  turn: string;
  ts: number;
  nodeCount: number;
  edgeCount: number;
}

export interface TimelineEvent {
  seq: number;
  ts: number;
  event: string;
  source: string;
  eventId?: string;
  roundIdx?: number;
  sessionId?: string;
  turnUid?: string;
  data?: unknown;
}

export interface TurnArtifact {
  turnUid: string;
  side: string;
  round: number;
  controllerAssessment?: unknown;
  batchInstructions?: unknown;
  decisionRaw: string;
  parsedActions: unknown[];
  executionLogs: string;
  retryHistory: unknown[];
  workerReports: unknown[];
  narrativeRawSentences?: unknown[];
  narrativePolished?: string;
  raw?: unknown;
}

export interface DebugBundleView {
  sessionId: string;
  round: number;
  turnUid: string;
  status: string;
  lastError: string;

  snapshotSummary: {
    phase: string;
    nodeCount: number;
    edgeCount: number;
    claimCount: number;
    conflictCount: number;
  };

  recentEvents: TimelineEvent[];
  latestTurnArtifact?: TurnArtifact;
  generatedAt: string;
  raw?: unknown;
}

export interface DemoKeyframe {
  sessionId: string;
  event: string;
  reason: string;
  round: number;
  turnUid: string;
  ts: number;
  data?: unknown;
  raw?: unknown;
}

export interface DemoRunResult {
  sessionId: string;
  status: string;
  stepsExecuted: number;
  endedBy: string;
  keyframes: DemoKeyframe[];
  raw?: unknown;
}

export interface ReplayExportView {
  sessionId: string;
  eventCount: number;
  artifactCount: number;
  snapshotCount: number;
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

  getDebugBundle(
    sessionId: string,
    options?: {
      eventLimit?: number;
      includeSnapshot?: boolean;
      includeArtifact?: boolean;
    },
  ): Promise<DebugBundleView>;

  runDemo(
    sessionId: string,
    options?: {
      maxSteps?: number;
      autoAdjudicate?: boolean;
      captureKeyframes?: boolean;
    },
  ): Promise<DemoRunResult>;

  getDemoKeyframes(sessionId: string): Promise<DemoKeyframe[]>;

  setFailureSimulation(
    sessionId: string,
    kind: "es_unavailable" | "llm_timeout",
    enabled: boolean,
  ): Promise<{ sessionId: string; failureSimulation: Record<string, boolean> }>;

  exportReplayJson(sessionId: string): Promise<ReplayExportView>;
  exportGraphGexf(sessionId: string, round?: number): Promise<Blob>;
}

export interface EngineAdapter extends SessionAdapter {
  readonly capabilities: AdapterCapabilities;
  readonly session: SessionAdapter;
  readonly graph: GraphAdapter;
  readonly insight: InsightAdapter;
}
