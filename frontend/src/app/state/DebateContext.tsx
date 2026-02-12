import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { createCompatAdapter } from "../../compat";

import type {
  DebateSnapshot,
  DemoKeyframe,
  GraphDiffView,
  GraphView,
  MemoryView,
  ReplayExportView,
  SnapshotIndexItem,
  TimelineEvent,
  TurnArtifact,
} from "../../compat";

import type { AdapterMode, StreamStatus } from "../types";
import { DebateContext, type DebateContextValue } from "./debateContextObject";
const envBaseUrl = import.meta.env.VITE_API_BASE_URL;

function sortedTimeline(rows: TimelineEvent[]): TimelineEvent[] {
  return [...rows].sort((a, b) => a.seq - b.seq || a.ts - b.ts).slice(-180);
}

export function DebateProvider({ children }: { children: ReactNode }) {
  const adapterMode: AdapterMode = "http";

  const adapter = useMemo(
    () => createCompatAdapter({ baseUrl: envBaseUrl || "/api" }),
    [],
  );

  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [snapshot, setSnapshot] = useState<DebateSnapshot | null>(null);

  const [previousSnapshot, setPreviousSnapshot] =
    useState<DebateSnapshot | null>(null);

  const [sessions, setSessions] = useState<DebateSnapshot[]>([]);
  const [graphView, setGraphView] = useState<GraphView | null>(null);

  const [baselineGraphView, setBaselineGraphView] = useState<GraphView | null>(
    null,
  );

  const [graphDiff, setGraphDiff] = useState<GraphDiffView | null>(null);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [snapshotIndex, setSnapshotIndex] = useState<SnapshotIndexItem[]>([]);
  const [turnArtifacts, setTurnArtifacts] = useState<TurnArtifact[]>([]);
  const [memoryView, setMemoryView] = useState<MemoryView | null>(null);
  const [demoKeyframes, setDemoKeyframes] = useState<DemoKeyframe[]>([]);

  const [replayExport, setReplayExport] = useState<ReplayExportView | null>(
    null,
  );

  const [streamStatus, setStreamStatus] = useState<StreamStatus>("idle");
  const [busyAction, setBusyAction] = useState<string>("");
  const [error, setError] = useState<string>("");
  const lastSeqRef = useRef<number>(0);
  const wsLiveRef = useRef<boolean>(false);
  const snapshotRef = useRef<DebateSnapshot | null>(null);
  const sessionIdRef = useRef<string>("");
  const snapshotInFlightRef = useRef<Promise<boolean> | null>(null);
  const graphInFlightRef = useRef<Promise<boolean> | null>(null);
  const timelineInFlightRef = useRef<Promise<boolean> | null>(null);
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
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      return false;
    } finally {
      setBusyAction("");
    }
  }, [adapter, applySnapshot]);

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
    const targetSessionId = sessionId;

    if (!targetSessionId) {
      return false;
    }

    if (snapshotInFlightRef.current) {
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
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        return false;
      } finally {
        setBusyAction("");
      }
    })();

    snapshotInFlightRef.current = task.finally(() => {
      snapshotInFlightRef.current = null;
    });

    return snapshotInFlightRef.current;
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
    const targetSessionId = sessionId;

    if (!targetSessionId) {
      return false;
    }

    if (graphInFlightRef.current) {
      return graphInFlightRef.current;
    }

    const task = (async (): Promise<boolean> => {
      try {
        const result = await adapter.graph.getGraph(targetSessionId);

        if (sessionIdRef.current !== targetSessionId) {
          return false;
        }

        setGraphView(result);
        return true;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        return false;
      }
    })();

    graphInFlightRef.current = task.finally(() => {
      graphInFlightRef.current = null;
    });

    return graphInFlightRef.current;
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
        setGraphDiff(null);
        return true;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        return false;
      }
    },
    [adapter, sessionId],
  );

  const loadGraphDiff = useCallback(
    async (fromRound: number, toRound: number): Promise<boolean> => {
      if (
        !sessionId ||
        !Number.isFinite(fromRound) ||
        !Number.isFinite(toRound)
      ) {
        return false;
      }

      const from = Math.max(0, Math.floor(fromRound));
      const to = Math.max(0, Math.floor(toRound));
      const lhs = Math.min(from, to);
      const rhs = Math.max(from, to);

      try {
        const [diff, fromGraph, toGraph] = await Promise.all([
          adapter.graph.getGraphDiff(sessionId, lhs, rhs),
          adapter.graph.getGraphAtRound(sessionId, lhs),
          adapter.graph.getGraphAtRound(sessionId, rhs),
        ]);

        setGraphDiff(diff);
        setBaselineGraphView(fromGraph);
        setGraphView(toGraph);
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
      const targetSessionId = sessionId;

      if (!targetSessionId) {
        return false;
      }

      if (timelineInFlightRef.current) {
        return timelineInFlightRef.current;
      }

      const task = (async (): Promise<boolean> => {
        try {
          const rows = await adapter.insight.getTimeline(
            targetSessionId,
            limit,
          );

          if (sessionIdRef.current !== targetSessionId) {
            return false;
          }

          replaceTimeline(rows);
          return true;
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          setError(message);
          return false;
        }
      })();

      timelineInFlightRef.current = task.finally(() => {
        timelineInFlightRef.current = null;
      });

      return timelineInFlightRef.current;
    },
    [adapter, replaceTimeline, sessionId],
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
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      return false;
    }
  }, [adapter, sessionId]);

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
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        return false;
      }
    },
    [adapter, sessionId],
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
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      return false;
    }
  }, [adapter, sessionId]);

  const loadDemoKeyframes = useCallback(async (): Promise<boolean> => {
    if (!sessionId) {
      return false;
    }

    try {
      const rows = await adapter.insight.getDemoKeyframes(sessionId);
      setDemoKeyframes(rows);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      return false;
    }
  }, [adapter, sessionId]);

  const exportReplayJson = useCallback(async (): Promise<boolean> => {
    if (!sessionId) {
      return false;
    }

    try {
      const row = await adapter.insight.exportReplayJson(sessionId);
      setReplayExport(row);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      return false;
    }
  }, [adapter, sessionId]);

  const exportGraphGexf = useCallback(
    async (round?: number): Promise<Blob | null> => {
      if (!sessionId) {
        return null;
      }

      try {
        return await adapter.insight.exportGraphGexf(sessionId, round);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        return null;
      }
    },
    [adapter, sessionId],
  );

  const loadReplayBundle = useCallback(async (): Promise<boolean> => {
    if (!sessionId) {
      return false;
    }

    const results = await Promise.all([
      loadSnapshots(),
      loadTurnArtifacts({ limit: 80 }),
      loadDemoKeyframes(),
      loadTimeline(120),
      loadMemory(),
    ]);

    return results.every(Boolean);
  }, [
    loadDemoKeyframes,
    loadMemory,
    loadSnapshots,
    loadTimeline,
    loadTurnArtifacts,
    sessionId,
  ]);

  const snapshotRound = snapshot?.round;
  const snapshotSession = snapshot?.sessionId;
  const graphRound = graphView?.round;
  const graphSession = graphView?.sessionId;

  useEffect(() => {
    void listSessions();
  }, [listSessions]);

  useEffect(() => {
    snapshotRef.current = snapshot;
  }, [snapshot]);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) {
      setGraphView(null);
      setBaselineGraphView(null);
      setGraphDiff(null);
      setTimeline([]);
      setSnapshotIndex([]);
      setTurnArtifacts([]);
      setMemoryView(null);
      setDemoKeyframes([]);
      setReplayExport(null);
      setStreamStatus("idle");
      lastSeqRef.current = 0;
      wsLiveRef.current = false;
      snapshotRef.current = null;
      sessionIdRef.current = "";
      snapshotInFlightRef.current = null;
      graphInFlightRef.current = null;
      timelineInFlightRef.current = null;
      return;
    }

    if (!snapshotRef.current || snapshotRef.current.sessionId !== sessionId) {
      void refreshSnapshot();
    }
  }, [refreshSnapshot, sessionId]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void Promise.all([
      loadSnapshots(),
      loadTurnArtifacts({ limit: 80 }),
      loadMemory(),
      loadDemoKeyframes(),
    ]);
  }, [
    loadDemoKeyframes,
    loadMemory,
    loadSnapshots,
    loadTurnArtifacts,
    sessionId,
  ]);

  useEffect(() => {
    if (!sessionId || !snapshotSession || snapshotSession !== sessionId) {
      return;
    }

    if (graphSession === sessionId && graphRound === snapshotRound) {
      return;
    }

    void loadGraph();
  }, [
    graphRound,
    graphSession,
    loadGraph,
    sessionId,
    snapshotRound,
    snapshotSession,
  ]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    let alive = true;
    wsLiveRef.current = false;

    const pullTimeline = async (): Promise<void> => {
      const ok = await loadTimeline(80);

      if (!alive) {
        return;
      }

      if (!wsLiveRef.current) {
        setStreamStatus("poll");
      }

      if (!ok) {
        setStreamStatus("poll");
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

    let timerId: number | null = null;

    const schedulePoll = (): void => {
      timerId = window.setTimeout(() => {
        if (!alive) {
          return;
        }

        const pageVisible =
          typeof document === "undefined" ||
          document.visibilityState === "visible";

        if (!wsLiveRef.current && pageVisible) {
          void pullTimeline();
        }

        schedulePoll();
      }, 3500);
    };

    schedulePoll();

    const handleVisibilityChange = (): void => {
      if (
        alive &&
        !wsLiveRef.current &&
        typeof document !== "undefined" &&
        document.visibilityState === "visible"
      ) {
        void pullTimeline();
        void loadGraph();
      }
    };

    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", handleVisibilityChange);
    }

    return () => {
      alive = false;
      stopStreaming();

      if (timerId !== null) {
        window.clearTimeout(timerId);
      }

      if (typeof document !== "undefined") {
        document.removeEventListener(
          "visibilitychange",
          handleVisibilityChange,
        );
      }
    };
  }, [adapter, loadGraph, loadTimeline, mergeTimeline, sessionId]);

  const contextValue = useMemo<DebateContextValue>(
    () => ({
      adapterMode,
      sessionId,
      streamStatus,
      snapshot,
      previousSnapshot,
      sessions,
      graphView,
      baselineGraphView,
      graphDiff,
      timeline,
      snapshotIndex,
      turnArtifacts,
      memoryView,
      demoKeyframes,
      replayExport,
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
      loadGraphDiff,
      loadTimeline,
      loadSnapshots,
      loadTurnArtifacts,
      loadMemory,
      loadDemoKeyframes,
      exportReplayJson,
      exportGraphGexf,
      loadReplayBundle,
    }),
    [
      adapterMode,
      adjudicate,
      baselineGraphView,
      busyAction,
      clearError,
      createSession,
      demoKeyframes,
      error,
      exportGraphGexf,
      exportReplayJson,
      graphDiff,
      graphView,
      listSessions,
      loadDemoKeyframes,
      loadGraph,
      loadGraphAtRound,
      loadGraphDiff,
      loadMemory,
      loadReplayBundle,
      loadSnapshots,
      loadTimeline,
      loadTurnArtifacts,
      memoryView,
      previousSnapshot,
      refreshSnapshot,
      replayExport,
      selectSession,
      sessionId,
      sessions,
      snapshot,
      snapshotIndex,
      step,
      streamStatus,
      timeline,
      turnArtifacts,
    ],
  );

  return (
    <DebateContext.Provider value={contextValue}>
      {children}
    </DebateContext.Provider>
  );
}
