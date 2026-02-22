import { useCallback, useRef, useState } from "react";

import type {
  DebateSnapshot,
  EngineAdapter,
  FrontendSnapshotListItem,
  MemoryView,
  SnapshotIndexItem,
  TeamFlowTurn,
  TurnArtifact,
} from "../../compat";

interface UseSnapshotActionsParams {
  adapter: EngineAdapter;
  sessionId: string;
  sessionsLength: number;
  setBusyAction: (action: string) => void;
  setErrorMessage: (message: string) => void;
  reportError: (err: unknown) => void;
  setActiveSessionId: (sessionId: string) => void;
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
  const [teamflowStream, setTeamflowStream] = useState<TeamFlowTurn[]>([]);
  const [snapshotIndex, setSnapshotIndex] = useState<SnapshotIndexItem[]>([]);
  const [turnArtifacts, setTurnArtifacts] = useState<TurnArtifact[]>([]);
  const [memoryView, setMemoryView] = useState<MemoryView | null>(null);

  const [frontendSnapshots, setFrontendSnapshots] = useState<
    FrontendSnapshotListItem[]
  >([]);

  const frontendSnapshotsInFlightRef = useRef<Promise<boolean> | null>(null);
  const restoringSnapshotIdRef = useRef<string>("");

  const clearSnapshotState = useCallback(() => {
    setTeamflowStream([]);
    setSnapshotIndex([]);
    setTurnArtifacts([]);
    setMemoryView(null);
  }, []);

  const loadTeamflowStream = useCallback(
    async (limit = 80): Promise<boolean> => {
      if (!sessionId) {
        return false;
      }

      try {
        const rows = await adapter.insight.getTeamflowStream(sessionId, limit);
        setTeamflowStream(rows);
        return true;
      } catch (err) {
        reportError(err);
        return false;
      }
    },
    [adapter, reportError, sessionId],
  );

  const loadSnapshots = useCallback(async (): Promise<boolean> => {
    if (!sessionId) {
      return false;
    }

    try {
      const rows = await adapter.getSnapshots(sessionId);
      setSnapshotIndex(rows);
      return true;
    } catch (err) {
      reportError(err);
      return false;
    }
  }, [adapter, reportError, sessionId]);

  const loadTurnArtifacts = useCallback(
    async (
      options: { turnUid?: string; limit?: number } = {},
    ): Promise<boolean> => {
      if (!sessionId) {
        return false;
      }

      try {
        const rows = await adapter.insight.getTurnArtifacts(sessionId, {
          limit: options.limit ?? 60,
          turnUid: options.turnUid,
        });

        if (options.turnUid) {
          setTurnArtifacts((prev) => {
            const merged = new Map<string, TurnArtifact>();

            for (const row of prev) {
              merged.set(row.turnUid, row);
            }

            for (const row of rows) {
              merged.set(row.turnUid, row);
            }

            return [...merged.values()].sort((a, b) => a.round - b.round);
          });
        } else {
          setTurnArtifacts(rows);
        }

        return true;
      } catch (err) {
        reportError(err);
        return false;
      }
    },
    [adapter, reportError, sessionId],
  );

  const loadMemory = useCallback(async (): Promise<boolean> => {
    if (!sessionId) {
      return false;
    }

    try {
      const memory = await adapter.insight.getMemory(sessionId);
      setMemoryView(memory);
      return true;
    } catch (err) {
      reportError(err);
      return false;
    }
  }, [adapter, reportError, sessionId]);

  const refreshMemorySilently = useCallback(async (): Promise<boolean> => {
    if (!sessionId) {
      return false;
    }

    try {
      const memory = await adapter.insight.getMemory(sessionId);
      setMemoryView(memory);
      return true;
    } catch {
      return false;
    }
  }, [adapter, sessionId]);

  const refreshSnapshotsSilently = useCallback(async (): Promise<boolean> => {
    if (!sessionId) {
      return false;
    }

    try {
      const rows = await adapter.getSnapshots(sessionId);
      setSnapshotIndex(rows);
      return true;
    } catch {
      return false;
    }
  }, [adapter, sessionId]);

  const resetMemory = useCallback(async (): Promise<boolean> => {
    if (sessionsLength > 0) {
      setErrorMessage("存在活动会话，禁止清理磁盘长期记忆。请先关闭所有会话。");
      return false;
    }

    setBusyAction("resetMemoryStorage");
    setErrorMessage("");

    try {
      await adapter.resetMemory();
      setMemoryView(null);
      onResetMemoryState();
      return true;
    } catch (err) {
      reportError(err);
      return false;
    } finally {
      setBusyAction("");
    }
  }, [
    adapter,
    onResetMemoryState,
    reportError,
    sessionsLength,
    setBusyAction,
    setErrorMessage,
  ]);

  const listFrontendSnapshots = useCallback(
    async (limit = 20): Promise<boolean> => {
      if (frontendSnapshotsInFlightRef.current) {
        return frontendSnapshotsInFlightRef.current;
      }

      const task = (async (): Promise<boolean> => {
        try {
          const rows = await adapter.listFrontendSnapshots(limit, 0);
          setFrontendSnapshots(rows);
          return true;
        } catch (err) {
          reportError(err);
          return false;
        }
      })();

      frontendSnapshotsInFlightRef.current = task.finally(() => {
        frontendSnapshotsInFlightRef.current = null;
      });

      return frontendSnapshotsInFlightRef.current;
    },
    [adapter, reportError],
  );

  const saveFrontendSnapshot = useCallback(
    async (
      label = "",
      frontendState: Record<string, unknown> = {},
    ): Promise<boolean> => {
      if (!sessionId) {
        return false;
      }

      setBusyAction("saveFrontendSnapshot");
      setErrorMessage("");

      try {
        await adapter.saveFrontendSnapshot({
          sessionId,
          label,
          frontendState,
        });

        await listFrontendSnapshots();
        return true;
      } catch (err) {
        reportError(err);
        return false;
      } finally {
        setBusyAction("");
      }
    },
    [
      adapter,
      listFrontendSnapshots,
      reportError,
      sessionId,
      setBusyAction,
      setErrorMessage,
    ],
  );

  const importFrontendSnapshotBundle = useCallback(
    async (
      bundle: Record<string, unknown>,
      label = "",
      frontendState: Record<string, unknown> = {},
    ): Promise<boolean> => {
      setBusyAction("importFrontendSnapshot");
      setErrorMessage("");

      try {
        await adapter.importFrontendSnapshot({
          bundle,
          label,
          frontendState,
        });

        await listFrontendSnapshots();
        return true;
      } catch (err) {
        reportError(err);
        return false;
      } finally {
        setBusyAction("");
      }
    },
    [
      adapter,
      listFrontendSnapshots,
      reportError,
      setBusyAction,
      setErrorMessage,
    ],
  );

  const loadFrontendSnapshot = useCallback(
    async (snapshotId: string): Promise<boolean> => {
      if (!snapshotId || restoringSnapshotIdRef.current === snapshotId) {
        return false;
      }

      restoringSnapshotIdRef.current = snapshotId;
      setBusyAction("loadFrontendSnapshot");
      setErrorMessage("");

      try {
        const loaded = await adapter.loadFrontendSnapshot(snapshotId);
        const restoredSessionId = loaded.session.sessionId;

        if (!restoredSessionId) {
          setErrorMessage("loadFrontendSnapshot: missing restored session_id");
          return false;
        }

        setActiveSessionId(restoredSessionId);

        if (
          loaded.snapshotPayload &&
          loaded.snapshotPayload.sessionId === restoredSessionId
        ) {
          applySnapshot(loaded.snapshotPayload);
        } else {
          const refreshed = await adapter.getSnapshot(restoredSessionId);
          applySnapshot(refreshed);
        }

        await listSessions();
        return true;
      } catch (err) {
        reportError(err);
        return false;
      } finally {
        restoringSnapshotIdRef.current = "";
        setBusyAction("");
      }
    },
    [
      adapter,
      applySnapshot,
      listSessions,
      reportError,
      setActiveSessionId,
      setBusyAction,
      setErrorMessage,
    ],
  );

  return {
    teamflowStream,
    snapshotIndex,
    turnArtifacts,
    memoryView,
    frontendSnapshots,
    loadTeamflowStream,
    loadSnapshots,
    loadTurnArtifacts,
    loadMemory,
    refreshMemorySilently,
    refreshSnapshotsSilently,
    resetMemory,
    listFrontendSnapshots,
    saveFrontendSnapshot,
    importFrontendSnapshotBundle,
    loadFrontendSnapshot,
    clearSnapshotState,
  };
}
