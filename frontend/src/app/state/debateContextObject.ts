import { createContext } from "react";

import type { DebateSnapshot, GraphView, TimelineEvent } from "../../compat";
import type { AdapterMode, StreamStatus } from "../types";

export interface DebateContextValue {
  adapterMode: AdapterMode;
  sessionId: string;
  streamStatus: StreamStatus;
  snapshot: DebateSnapshot | null;
  previousSnapshot: DebateSnapshot | null;
  sessions: DebateSnapshot[];
  graphView: GraphView | null;
  timeline: TimelineEvent[];
  busyAction: string;
  error: string;
  clearError: () => void;
  listSessions: () => Promise<void>;
  createSession: (maxRounds: number) => Promise<boolean>;
  selectSession: (nextSessionId: string) => Promise<boolean>;
  step: () => Promise<boolean>;
  adjudicate: () => Promise<boolean>;
  refreshSnapshot: () => Promise<boolean>;
  loadGraph: () => Promise<boolean>;
  loadGraphAtRound: (round: number) => Promise<boolean>;
  loadTimeline: (limit?: number) => Promise<boolean>;
}

export const DebateContext = createContext<DebateContextValue | null>(null);

