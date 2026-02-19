import { createContext } from "react";

import type {
  DebateSnapshot,
  GraphDiffView,
  GraphView,
  FrontendSnapshotListItem,
  MemoryView,
  SnapshotIndexItem,
  TeamFlowTurn,
  TimelineEvent,
  TurnArtifact,
} from "../../compat";

import type { AdapterMode, StreamStatus } from "../types";

export interface DebateContextValue {
  adapterMode: AdapterMode;
  sessionId: string;
  streamStatus: StreamStatus;
  snapshot: DebateSnapshot | null;
  previousSnapshot: DebateSnapshot | null;
  sessions: DebateSnapshot[];
  graphView: GraphView | null;
  baselineGraphView: GraphView | null;
  graphDiff: GraphDiffView | null;
  timeline: TimelineEvent[];
  teamflowStream: TeamFlowTurn[];
  snapshotIndex: SnapshotIndexItem[];
  turnArtifacts: TurnArtifact[];
  memoryView: MemoryView | null;
  memoryCaseGraphView: GraphView | null;
  frontendSnapshots: FrontendSnapshotListItem[];
  busyAction: string;
  error: string;
  clearError: () => void;
  listSessions: () => Promise<void>;
  createSession: () => Promise<boolean>;
  selectSession: (nextSessionId: string) => Promise<boolean>;
  step: () => Promise<boolean>;
  adjudicate: () => Promise<boolean>;
  refreshSnapshot: () => Promise<boolean>;
  loadGraph: () => Promise<boolean>;
  loadGraphAtRound: (round: number) => Promise<boolean>;
  loadGraphDiff: (fromRound: number, toRound: number) => Promise<boolean>;
  loadTimeline: (limit?: number) => Promise<boolean>;
  loadTeamflowStream: (limit?: number) => Promise<boolean>;
  loadSnapshots: () => Promise<boolean>;

  loadTurnArtifacts: (options?: {
    turnUid?: string;
    limit?: number;
  }) => Promise<boolean>;

  loadMemory: () => Promise<boolean>;
  loadMemoryCaseGraph: (caseId: string) => Promise<boolean>;
  exportGraphGexf: (round?: number) => Promise<Blob | null>;

  saveFrontendSnapshot: (
    label?: string,
    frontendState?: Record<string, unknown>,
  ) => Promise<boolean>;

  importFrontendSnapshotBundle: (
    bundle: Record<string, unknown>,
    label?: string,
    frontendState?: Record<string, unknown>,
  ) => Promise<boolean>;

  listFrontendSnapshots: (limit?: number) => Promise<boolean>;
  loadFrontendSnapshot: (snapshotId: string) => Promise<boolean>;
}

export const DebateContext = createContext<DebateContextValue | null>(null);
