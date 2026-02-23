import {
  useCallback,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";

import type {
  DebateSnapshot,
  EngineAdapter,
  FrontendSnapshotListItem,
} from "../../../compat";

interface UseFrontendSnapshotActionsParams {
  adapter: EngineAdapter;
  sessionId: string;
  setBusyAction: (action: string) => void;
  setErrorMessage: (message: string) => void;
  reportError: (err: unknown) => void;
  setActiveSessionId: Dispatch<SetStateAction<string>>;
  applySnapshot: (snapshot: DebateSnapshot) => void;
  listSessions: () => Promise<void>;
}

interface UseFrontendSnapshotActionsResult {
  frontendSnapshots: FrontendSnapshotListItem[];
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
}

export function useFrontendSnapshotActions({
  adapter,
  sessionId,
  setBusyAction,
  setErrorMessage,
  reportError,
  setActiveSessionId,
  applySnapshot,
  listSessions,
}: UseFrontendSnapshotActionsParams): UseFrontendSnapshotActionsResult {
  const [frontendSnapshots, setFrontendSnapshots] = useState<
    FrontendSnapshotListItem[]
  >([]);

  const frontendSnapshotsInFlightRef = useRef<Promise<boolean> | null>(null);
  const restoringSnapshotIdRef = useRef<string>("");

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
        const restoredSnapshot = loaded.snapshotPayload;

        if (!restoredSessionId) {
          setErrorMessage("loadFrontendSnapshot: missing restored session_id");
          return false;
        }

        if (
          !restoredSnapshot ||
          restoredSnapshot.sessionId !== restoredSessionId
        ) {
          setErrorMessage("loadFrontendSnapshot: invalid snapshot payload");
          return false;
        }

        setActiveSessionId(restoredSessionId);
        applySnapshot(restoredSnapshot);
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
    frontendSnapshots,
    listFrontendSnapshots,
    saveFrontendSnapshot,
    importFrontendSnapshotBundle,
    loadFrontendSnapshot,
  };
}
