import { useCallback, useState } from "react";

import type {
  EngineAdapter,
  MemoryView,
  SnapshotIndexItem,
  TeamFlowTurn,
  TurnArtifact,
} from "../../../compat";

interface UseSnapshotDataActionsParams {
  adapter: EngineAdapter;
  sessionId: string;
  sessionsLength: number;
  setBusyAction: (action: string) => void;
  setErrorMessage: (message: string) => void;
  reportError: (err: unknown) => void;
  onResetMemoryState: () => void;
}

interface UseSnapshotDataActionsResult {
  teamflowStream: TeamFlowTurn[];
  snapshotIndex: SnapshotIndexItem[];
  turnArtifacts: TurnArtifact[];
  memoryView: MemoryView | null;
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
  clearSnapshotState: () => void;
}

export function useSnapshotDataActions({
  adapter,
  sessionId,
  sessionsLength,
  setBusyAction,
  setErrorMessage,
  reportError,
  onResetMemoryState,
}: UseSnapshotDataActionsParams): UseSnapshotDataActionsResult {
  const [teamflowStream, setTeamflowStream] = useState<TeamFlowTurn[]>([]);
  const [snapshotIndex, setSnapshotIndex] = useState<SnapshotIndexItem[]>([]);
  const [turnArtifacts, setTurnArtifacts] = useState<TurnArtifact[]>([]);
  const [memoryView, setMemoryView] = useState<MemoryView | null>(null);

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

  return {
    teamflowStream,
    snapshotIndex,
    turnArtifacts,
    memoryView,
    loadTeamflowStream,
    loadSnapshots,
    loadTurnArtifacts,
    loadMemory,
    refreshMemorySilently,
    refreshSnapshotsSilently,
    resetMemory,
    clearSnapshotState,
  };
}
