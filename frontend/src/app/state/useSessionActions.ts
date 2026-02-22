import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";

import type { DebateSnapshot, EngineAdapter } from "../../compat";
import { toErrorMessage } from "./errorUtils";

interface UseSessionActionsResult {
  sessionId: string;
  snapshot: DebateSnapshot | null;
  previousSnapshot: DebateSnapshot | null;
  sessions: DebateSnapshot[];
  busyAction: string;
  error: string;
  snapshotRef: MutableRefObject<DebateSnapshot | null>;
  sessionIdRef: MutableRefObject<string>;
  setBusyAction: Dispatch<SetStateAction<string>>;
  setErrorMessage: (message: string) => void;
  reportError: (err: unknown) => void;
  clearError: () => void;
  setActiveSessionId: Dispatch<SetStateAction<string>>;
  applySnapshot: (nextSnapshot: DebateSnapshot) => void;
  listSessions: () => Promise<void>;
  createSession: () => Promise<boolean>;
  selectSession: (nextSessionId: string) => Promise<boolean>;
  refreshSnapshot: () => Promise<boolean>;
  stepSession: () => Promise<boolean>;
  adjudicateSession: () => Promise<boolean>;
}

export function useSessionActions(
  adapter: EngineAdapter,
): UseSessionActionsResult {
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [snapshot, setSnapshot] = useState<DebateSnapshot | null>(null);

  const [previousSnapshot, setPreviousSnapshot] =
    useState<DebateSnapshot | null>(null);

  const [sessions, setSessions] = useState<DebateSnapshot[]>([]);
  const [busyAction, setBusyAction] = useState<string>("");
  const [error, setError] = useState<string>("");
  const snapshotRef = useRef<DebateSnapshot | null>(null);
  const sessionIdRef = useRef<string>("");
  const snapshotInFlightRef = useRef<Promise<boolean> | null>(null);
  const snapshotInFlightSessionRef = useRef<string>("");

  useEffect(() => {
    snapshotRef.current = snapshot;
  }, [snapshot]);

  useEffect(() => {
    sessionIdRef.current = activeSessionId;
  }, [activeSessionId]);

  const clearError = useCallback(() => setError(""), []);

  const setErrorMessage = useCallback((message: string): void => {
    setError(message);
  }, []);

  const reportError = useCallback((err: unknown): void => {
    setError(toErrorMessage(err));
  }, []);

  const applySnapshot = useCallback((nextSnapshot: DebateSnapshot): void => {
    const prev = snapshotRef.current;
    setPreviousSnapshot(prev);
    setSnapshot(nextSnapshot);
    snapshotRef.current = nextSnapshot;
  }, []);

  const listSessions = useCallback(async (): Promise<void> => {
    setBusyAction("listSessions");
    setError("");

    try {
      const rows = await adapter.listSessions();
      setSessions(rows);

      setActiveSessionId(
        (prev) => prev || (rows.length > 0 ? rows[0].sessionId : prev),
      );
    } catch (err) {
      reportError(err);
    } finally {
      setBusyAction("");
    }
  }, [adapter, reportError]);

  const createSession = useCallback(async (): Promise<boolean> => {
    setBusyAction("createSession");
    setError("");

    try {
      const created = await adapter.createSession();
      applySnapshot(created);
      setActiveSessionId(created.sessionId);
      const rows = await adapter.listSessions();
      setSessions(rows);
      return true;
    } catch (err) {
      reportError(err);
      return false;
    } finally {
      setBusyAction("");
    }
  }, [adapter, applySnapshot, reportError]);

  const selectSession = useCallback(
    async (nextSessionId: string): Promise<boolean> => {
      setActiveSessionId(nextSessionId);
      setBusyAction("getSnapshot");
      setError("");

      try {
        const result = await adapter.getSnapshot(nextSessionId);
        applySnapshot(result);
        return true;
      } catch (err) {
        reportError(err);
        return false;
      } finally {
        setBusyAction("");
      }
    },
    [adapter, applySnapshot, reportError],
  );

  const refreshSnapshot = useCallback(async (): Promise<boolean> => {
    const targetSessionId = activeSessionId;

    if (!targetSessionId) {
      return false;
    }

    if (
      snapshotInFlightRef.current &&
      snapshotInFlightSessionRef.current === targetSessionId
    ) {
      return snapshotInFlightRef.current;
    }

    const task = (async (): Promise<boolean> => {
      setBusyAction("getSnapshot");
      setError("");

      try {
        const result = await adapter.getSnapshot(targetSessionId);

        if (sessionIdRef.current !== targetSessionId) {
          return false;
        }

        applySnapshot(result);
        return true;
      } catch (err) {
        reportError(err);
        return false;
      } finally {
        setBusyAction("");
      }
    })();

    const inFlight = task.finally(() => {
      if (snapshotInFlightRef.current === inFlight) {
        snapshotInFlightRef.current = null;
        snapshotInFlightSessionRef.current = "";
      }
    });

    snapshotInFlightSessionRef.current = targetSessionId;
    snapshotInFlightRef.current = inFlight;

    return inFlight;
  }, [activeSessionId, adapter, applySnapshot, reportError]);

  const stepSession = useCallback(async (): Promise<boolean> => {
    const sessionId = activeSessionId;

    if (!sessionId) {
      return false;
    }

    const currentSnapshot = snapshotRef.current;

    if (currentSnapshot && currentSnapshot.sessionId === sessionId) {
      const blockedByPhase =
        currentSnapshot.phase === "ready_for_adjudication" ||
        currentSnapshot.phase === "finished";

      if (blockedByPhase || currentSnapshot.convergence.isConverged) {
        setError("当前会话已收敛，请直接发起裁决，无法继续下一步辩论。");
        return false;
      }
    }

    setBusyAction("step");
    setError("");

    try {
      const result = await adapter.step(sessionId);
      applySnapshot(result);
      return true;
    } catch (err) {
      reportError(err);
      return false;
    } finally {
      setBusyAction("");
    }
  }, [activeSessionId, adapter, applySnapshot, reportError]);

  const adjudicateSession = useCallback(async (): Promise<boolean> => {
    const sessionId = activeSessionId;

    if (!sessionId) {
      return false;
    }

    setBusyAction("adjudicate");
    setError("");

    try {
      const result = await adapter.adjudicate(sessionId);
      applySnapshot(result);
      return true;
    } catch (err) {
      reportError(err);
      return false;
    } finally {
      setBusyAction("");
    }
  }, [activeSessionId, adapter, applySnapshot, reportError]);

  return {
    sessionId: activeSessionId,
    snapshot,
    previousSnapshot,
    sessions,
    busyAction,
    error,
    snapshotRef,
    sessionIdRef,
    setBusyAction,
    setErrorMessage,
    reportError,
    clearError,
    setActiveSessionId,
    applySnapshot,
    listSessions,
    createSession,
    selectSession,
    refreshSnapshot,
    stepSession,
    adjudicateSession,
  };
}
