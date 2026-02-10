import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { createCompatAdapter } from "../../compat";
import type { DebateSnapshot, GraphView, TimelineEvent } from "../../compat";
import type { AdapterMode, StreamStatus } from "../types";

import { DebateContext, type DebateContextValue } from "./debateContextObject";

const envMode = import.meta.env.VITE_COMPAT_MODE;
const envBaseUrl = import.meta.env.VITE_API_BASE_URL;

function sortedTimeline(rows: TimelineEvent[]): TimelineEvent[] {
  return [...rows].sort((a, b) => a.seq - b.seq || a.ts - b.ts).slice(-180);
}

export function DebateProvider({ children }: { children: ReactNode }) {
  const adapterMode: AdapterMode =
    envMode === "http" || envMode === "mock" ? envMode : "auto";

  const adapter = useMemo(
    () =>
      createCompatAdapter({ mode: adapterMode, baseUrl: envBaseUrl || "/api" }),
    [adapterMode],
  );

  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [snapshot, setSnapshot] = useState<DebateSnapshot | null>(null);

  const [previousSnapshot, setPreviousSnapshot] =
    useState<DebateSnapshot | null>(null);

  const [sessions, setSessions] = useState<DebateSnapshot[]>([]);
  const [graphView, setGraphView] = useState<GraphView | null>(null);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [streamStatus, setStreamStatus] = useState<StreamStatus>("idle");
  const [busyAction, setBusyAction] = useState<string>("");
  const [error, setError] = useState<string>("");
  const lastSeqRef = useRef<number>(0);
  const wsLiveRef = useRef<boolean>(false);
  const snapshotRef = useRef<DebateSnapshot | null>(null);
  const sessionId = activeSessionId;
  const clearError = useCallback(() => setError(""), []);

  const applySnapshot = useCallback((nextSnapshot: DebateSnapshot): void => {
    const prev = snapshotRef.current;
    setPreviousSnapshot(prev);
    setSnapshot(nextSnapshot);
    snapshotRef.current = nextSnapshot;
  }, []);

  const replaceTimeline = useCallback((rows: TimelineEvent[]): void => {
    const sorted = sortedTimeline(rows);
    const latest = sorted.length > 0 ? sorted[sorted.length - 1] : null;
    lastSeqRef.current = latest?.seq ?? 0;
    setTimeline(sorted);
  }, []);

  const mergeTimeline = useCallback((rows: TimelineEvent[]): void => {
    if (!rows.length) {
      return;
    }

    setTimeline((prev) => {
      const bucket = new Map<number, TimelineEvent>();

      for (const row of prev) {
        bucket.set(row.seq, row);
      }

      for (const row of rows) {
        bucket.set(row.seq, row);
      }

      const merged = sortedTimeline([...bucket.values()]);
      const latest = merged.length > 0 ? merged[merged.length - 1] : null;
      lastSeqRef.current = latest?.seq ?? lastSeqRef.current;
      return merged;
    });
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
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    } finally {
      setBusyAction("");
    }
  }, [adapter]);

  const createSession = useCallback(
    async (maxRounds: number): Promise<boolean> => {
      setBusyAction("createSession");
      setError("");

      try {
        const created = await adapter.createSession({ maxRounds });
        applySnapshot(created);
        setActiveSessionId(created.sessionId);
        const rows = await adapter.listSessions();
        setSessions(rows);
        return true;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        return false;
      } finally {
        setBusyAction("");
      }
    },
    [adapter, applySnapshot],
  );

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
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        return false;
      } finally {
        setBusyAction("");
      }
    },
    [adapter, applySnapshot],
  );

  const refreshSnapshot = useCallback(async (): Promise<boolean> => {
    if (!sessionId) {
      return false;
    }

    setBusyAction("getSnapshot");
    setError("");

    try {
      const result = await adapter.getSnapshot(sessionId);
      applySnapshot(result);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      return false;
    } finally {
      setBusyAction("");
    }
  }, [adapter, sessionId, applySnapshot]);

  const step = useCallback(async (): Promise<boolean> => {
    if (!sessionId) {
      return false;
    }

    setBusyAction("step");
    setError("");

    try {
      const result = await adapter.step(sessionId);
      applySnapshot(result);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      return false;
    } finally {
      setBusyAction("");
    }
  }, [adapter, sessionId, applySnapshot]);

  const adjudicate = useCallback(async (): Promise<boolean> => {
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
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      return false;
    } finally {
      setBusyAction("");
    }
  }, [adapter, sessionId, applySnapshot]);

  const loadGraph = useCallback(async (): Promise<boolean> => {
    if (!sessionId) {
      return false;
    }

    try {
      const result = await adapter.graph.getGraph(sessionId);
      setGraphView(result);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      return false;
    }
  }, [adapter, sessionId]);

  const loadGraphAtRound = useCallback(
    async (round: number): Promise<boolean> => {
      if (!sessionId || !Number.isFinite(round)) {
        return false;
      }

      try {
        const result = await adapter.graph.getGraphAtRound(
          sessionId,
          Math.max(0, Math.floor(round)),
        );

        setGraphView(result);
        return true;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        return false;
      }
    },
    [adapter, sessionId],
  );

  const loadTimeline = useCallback(
    async (limit = 80): Promise<boolean> => {
      if (!sessionId) {
        return false;
      }

      try {
        const rows = await adapter.insight.getTimeline(sessionId, limit);
        replaceTimeline(rows);
        return true;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        return false;
      }
    },
    [adapter, replaceTimeline, sessionId],
  );

  useEffect(() => {
    void listSessions();
  }, [listSessions]);

  useEffect(() => {
    snapshotRef.current = snapshot;
  }, [snapshot]);

  useEffect(() => {
    if (!sessionId) {
      setGraphView(null);
      setTimeline([]);
      setStreamStatus("idle");
      lastSeqRef.current = 0;
      wsLiveRef.current = false;
      snapshotRef.current = null;
      return;
    }

    void Promise.all([refreshSnapshot(), loadGraph()]);
  }, [loadGraph, refreshSnapshot, sessionId]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void loadGraph();
  }, [sessionId, snapshot?.round, loadGraph]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    let alive = true;
    wsLiveRef.current = false;

    const pullTimeline = async (): Promise<void> => {
      try {
        const rows = await adapter.insight.getTimeline(sessionId, 80);

        if (!alive) {
          return;
        }

        replaceTimeline(rows);

        if (!wsLiveRef.current) {
          setStreamStatus("poll");
        }
      } catch {
        if (alive) {
          setStreamStatus("poll");
        }
      }
    };

    void pullTimeline();

    const stopStreaming = adapter.capabilities.supportsStreaming
      ? adapter.insight.subscribeTimeline(
          sessionId,
          (event) => {
            if (!alive) {
              return;
            }

            wsLiveRef.current = true;
            setStreamStatus("ws");
            mergeTimeline([event]);
          },
          {
            fromSeq:
              lastSeqRef.current > 0 ? lastSeqRef.current + 1 : undefined,
            onError: () => {
              if (!alive) {
                return;
              }

              wsLiveRef.current = false;
              setStreamStatus("poll");
            },
          },
        )
      : () => {};

    const timer = window.setInterval(() => {
      if (!wsLiveRef.current) {
        void pullTimeline();
      }
    }, 3500);

    return () => {
      alive = false;
      stopStreaming();
      window.clearInterval(timer);
    };
  }, [adapter, mergeTimeline, replaceTimeline, sessionId]);

  const contextValue = useMemo<DebateContextValue>(
    () => ({
      adapterMode,
      sessionId,
      streamStatus,
      snapshot,
      previousSnapshot,
      sessions,
      graphView,
      timeline,
      busyAction,
      error,
      clearError,
      listSessions,
      createSession,
      selectSession,
      step,
      adjudicate,
      refreshSnapshot,
      loadGraph,
      loadGraphAtRound,
      loadTimeline,
    }),
    [
      adapterMode,
      adjudicate,
      busyAction,
      clearError,
      createSession,
      error,
      graphView,
      listSessions,
      loadGraph,
      loadGraphAtRound,
      loadTimeline,
      previousSnapshot,
      refreshSnapshot,
      selectSession,
      sessionId,
      sessions,
      snapshot,
      step,
      streamStatus,
      timeline,
    ],
  );

  return (
    <DebateContext.Provider value={contextValue}>
      {children}
    </DebateContext.Provider>
  );
}
