import type {
  DebateSnapshot,
  EngineAdapter,
  FrontendSnapshotListItem,
  MemoryView,
  SnapshotIndexItem,
  TeamFlowTurn,
  TurnArtifact,
} from "../../compat";

import type { Dispatch, SetStateAction } from "react";
import { useFrontendSnapshotActions } from "./snapshot/useFrontendSnapshotActions";
import { useSnapshotDataActions } from "./snapshot/useSnapshotDataActions";

interface UseSnapshotActionsParams {
  adapter: EngineAdapter;
  sessionId: string;
  sessionsLength: number;
  setBusyAction: (action: string) => void;
  setErrorMessage: (message: string) => void;
  reportError: (err: unknown) => void;
  setActiveSessionId: Dispatch<SetStateAction<string>>;
  applySnapshot: (snapshot: DebateSnapshot) => void;
  listSessions: () => Promise<void>;
  onResetMemoryState: () => void;
}

interface UseSnapshotActionsResult {
  teamflowStream: TeamFlowTurn[];
  snapshotIndex: SnapshotIndexItem[];
  turnArtifacts: TurnArtifact[];
  memoryView: MemoryView | null;
  frontendSnapshots: FrontendSnapshotListItem[];
  loadTeamflowStream: (limit?: number) => Promise<boolean>;
  loadSnapshots: () => Promise<boolean>;

  loadTurnArtifacts: (options?: {
    turnUid?: string;
    limit?: number;
  }) => Promise<boolean>;

  loadMemory: () => Promise<boolean>;
  refreshMemorySilently: () => Promise<boolean>;
  refreshSnapshotsSilently: () => Promise<boolean>;
  resetMemory: () => Promise<boolean>;
  listFrontendSnapshots: (limit?: number) => Promise<boolean>;

  saveFrontendSnapshot: (
    label?: string,
    frontendState?: Record<string, unknown>,
  ) => Promise<boolean>;

  importFrontendSnapshotBundle: (
    bundle: Record<string, unknown>,
    label?: string,
    frontendState?: Record<string, unknown>,
  ) => Promise<boolean>;

  loadFrontendSnapshot: (snapshotId: string) => Promise<boolean>;
  clearSnapshotState: () => void;
}

export function useSnapshotActions({
  adapter,
  sessionId,
  sessionsLength,
  setBusyAction,
  setErrorMessage,
  reportError,
  setActiveSessionId,
  applySnapshot,
  listSessions,
  onResetMemoryState,
}: UseSnapshotActionsParams): UseSnapshotActionsResult {
  const snapshotData = useSnapshotDataActions({
    adapter,
    sessionId,
    sessionsLength,
    setBusyAction,
    setErrorMessage,
    reportError,
    onResetMemoryState,
  });

  const frontendSnapshots = useFrontendSnapshotActions({
    adapter,
    sessionId,
    setBusyAction,
    setErrorMessage,
    reportError,
    setActiveSessionId,
    applySnapshot,
    listSessions,
  });

  return {
    teamflowStream: snapshotData.teamflowStream,
    snapshotIndex: snapshotData.snapshotIndex,
    turnArtifacts: snapshotData.turnArtifacts,
    memoryView: snapshotData.memoryView,
    frontendSnapshots: frontendSnapshots.frontendSnapshots,
    loadTeamflowStream: snapshotData.loadTeamflowStream,
    loadSnapshots: snapshotData.loadSnapshots,
    loadTurnArtifacts: snapshotData.loadTurnArtifacts,
    loadMemory: snapshotData.loadMemory,
    refreshMemorySilently: snapshotData.refreshMemorySilently,
    refreshSnapshotsSilently: snapshotData.refreshSnapshotsSilently,
    resetMemory: snapshotData.resetMemory,
    listFrontendSnapshots: frontendSnapshots.listFrontendSnapshots,
    saveFrontendSnapshot: frontendSnapshots.saveFrontendSnapshot,
    importFrontendSnapshotBundle:
      frontendSnapshots.importFrontendSnapshotBundle,
    loadFrontendSnapshot: frontendSnapshots.loadFrontendSnapshot,
    clearSnapshotState: snapshotData.clearSnapshotState,
  };
}
