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
  FrontendSnapshotListItem,
  GraphDiffView,
  GraphView,
  MemoryView,
  SnapshotIndexItem,
  TeamFlowTurn,
  TimelineEvent,
  TurnArtifact,
} from "../../compat";

import type { AdapterMode, StreamStatus } from "../types";
import { DebateContext, type DebateContextValue } from "./debateContextObject";
const envBaseUrl = import.meta.env.VITE_API_BASE_URL;

function sortedTimeline(rows: TimelineEvent[]): TimelineEvent[] {
  return [...rows].sort((a, b) => a.seq - b.seq || a.ts - b.ts).slice(-180);
}

function toErrorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function warnDebateContext(scope: string, err: unknown): void {
  console.warn(`[DebateContext] ${scope} failed: ${toErrorMessage(err)}`);
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
  const [teamflowStream, setTeamflowStream] = useState<TeamFlowTurn[]>([]);
  const [snapshotIndex, setSnapshotIndex] = useState<SnapshotIndexItem[]>([]);
  const [turnArtifacts, setTurnArtifacts] = useState<TurnArtifact[]>([]);
  const [memoryView, setMemoryView] = useState<MemoryView | null>(null);

  const [memoryCaseGraphView, setMemoryCaseGraphView] =
    useState<GraphView | null>(null);

  const [frontendSnapshots, setFrontendSnapshots] = useState<
    FrontendSnapshotListItem[]
  >([]);

  const [streamStatus, setStreamStatus] = useState<StreamStatus>("idle");
  const [busyAction, setBusyAction] = useState<string>("");
  const [error, setError] = useState<string>("");
  const lastSeqRef = useRef<number>(0);
  const wsLiveRef = useRef<boolean>(false);
  const snapshotRef = useRef<DebateSnapshot | null>(null);
  const sessionIdRef = useRef<string>("");
  const snapshotInFlightRef = useRef<Promise<boolean> | null>(null);
  const snapshotInFlightSessionRef = useRef<string>("");
  const graphInFlightRef = useRef<Promise<boolean> | null>(null);
  const graphInFlightSessionRef = useRef<string>("");
  const timelineInFlightRef = useRef<Promise<boolean> | null>(null);
  const timelineInFlightSessionRef = useRef<string>("");
  const frontendSnapshotsInFlightRef = useRef<Promise<boolean> | null>(null);
  const restoringSnapshotIdRef = useRef<string>("");
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
      setError(toErrorMessage(err));
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
      setError(toErrorMessage(err));
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
        setError(toErrorMessage(err));
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
        setError(toErrorMessage(err));
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
  }, [adapter, sessionId, applySnapshot]);

  const step = useCallback(async (): Promise<boolean> => {
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

      void (async () => {
        try {
          const memory = await adapter.insight.getMemory(sessionId);
          setMemoryView(memory);
        } catch (err) {
          warnDebateContext("getMemory after step", err);
        }
      })();

      return true;
    } catch (err) {
      setError(toErrorMessage(err));
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

      void (async () => {
        try {
          const memory = await adapter.insight.getMemory(sessionId);
          setMemoryView(memory);
        } catch (err) {
          warnDebateContext("getMemory after adjudicate", err);
        }
      })();

      void (async () => {
        try {
          const rows = await adapter.getSnapshots(sessionId);
          setSnapshotIndex(rows);
        } catch (err) {
          warnDebateContext("getSnapshots after adjudicate", err);
        }
      })();

      return true;
    } catch (err) {
      setError(toErrorMessage(err));
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

    if (
      graphInFlightRef.current &&
      graphInFlightSessionRef.current === targetSessionId
    ) {
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
        setMemoryCaseGraphView(null);
        setError(toErrorMessage(err));
        return false;
      }
    })();

    const inFlight = task.finally(() => {
      if (graphInFlightRef.current === inFlight) {
        graphInFlightRef.current = null;
        graphInFlightSessionRef.current = "";
      }
    });

    graphInFlightSessionRef.current = targetSessionId;
    graphInFlightRef.current = inFlight;

    return inFlight;
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
        setError(toErrorMessage(err));
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
        setError(toErrorMessage(err));
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

      if (
        timelineInFlightRef.current &&
        timelineInFlightSessionRef.current === targetSessionId
      ) {
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
          setError(toErrorMessage(err));
          return false;
        }
      })();

      const inFlight = task.finally(() => {
        if (timelineInFlightRef.current === inFlight) {
          timelineInFlightRef.current = null;
          timelineInFlightSessionRef.current = "";
        }
      });

      timelineInFlightSessionRef.current = targetSessionId;
      timelineInFlightRef.current = inFlight;

      return inFlight;
    },
    [adapter, replaceTimeline, sessionId],
  );

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
        setError(toErrorMessage(err));
        return false;
      }
    },
    [adapter, sessionId],
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
      setError(toErrorMessage(err));
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
        setError(toErrorMessage(err));
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
      setError(toErrorMessage(err));
      return false;
    }
  }, [adapter, sessionId]);

  const resetMemory = useCallback(async (): Promise<boolean> => {
    if (sessions.length > 0) {
      setError("存在活动会话，禁止清理磁盘长期记忆。请先关闭所有会话。");
      return false;
    }

    setBusyAction("resetMemoryStorage");
    setError("");

    try {
      await adapter.resetMemory();
      setMemoryView(null);
      setMemoryCaseGraphView(null);
      return true;
    } catch (err) {
      setError(toErrorMessage(err));
      return false;
    } finally {
      setBusyAction("");
    }
  }, [adapter, sessions.length]);

  const loadMemoryCaseGraph = useCallback(
    async (caseId: string): Promise<boolean> => {
      if (!sessionId) {
        return false;
      }

      const normalizedCaseId = String(caseId || "").trim();

      if (!normalizedCaseId) {
        setMemoryCaseGraphView(null);
        return false;
      }

      try {
        const graph = await adapter.insight.getMemoryCaseGraph(
          sessionId,
          normalizedCaseId,
        );

        setMemoryCaseGraphView(graph);
        return true;
      } catch (err) {
        setMemoryCaseGraphView(null);
        setError(toErrorMessage(err));
        return false;
      }
    },
    [adapter, sessionId],
  );

  const exportGraphGexf = useCallback(
    async (round?: number): Promise<Blob | null> => {
      if (!sessionId) {
        return null;
      }

      try {
        return await adapter.insight.exportGraphGexf(sessionId, round);
      } catch (err) {
        setError(toErrorMessage(err));
        return null;
      }
    },
    [adapter, sessionId],
  );

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
          setError(toErrorMessage(err));
          return false;
        }
      })();

      frontendSnapshotsInFlightRef.current = task.finally(() => {
        frontendSnapshotsInFlightRef.current = null;
      });

      return frontendSnapshotsInFlightRef.current;
    },
    [adapter],
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
      setError("");

      try {
        await adapter.saveFrontendSnapshot({
          sessionId,
          label,
          frontendState,
        });

        await listFrontendSnapshots();
        return true;
      } catch (err) {
        setError(toErrorMessage(err));
        return false;
      } finally {
        setBusyAction("");
      }
    },
    [adapter, listFrontendSnapshots, sessionId],
  );

  const importFrontendSnapshotBundle = useCallback(
    async (
      bundle: Record<string, unknown>,
      label = "",
      frontendState: Record<string, unknown> = {},
    ): Promise<boolean> => {
      setBusyAction("importFrontendSnapshot");
      setError("");

      try {
        await adapter.importFrontendSnapshot({
          bundle,
          label,
          frontendState,
        });

        await listFrontendSnapshots();
        return true;
      } catch (err) {
        setError(toErrorMessage(err));
        return false;
      } finally {
        setBusyAction("");
      }
    },
    [adapter, listFrontendSnapshots],
  );

  const loadFrontendSnapshot = useCallback(
    async (snapshotId: string): Promise<boolean> => {
      if (!snapshotId || restoringSnapshotIdRef.current === snapshotId) {
        return false;
      }

      restoringSnapshotIdRef.current = snapshotId;
      setBusyAction("loadFrontendSnapshot");
      setError("");

      try {
        const loaded = await adapter.loadFrontendSnapshot(snapshotId);
        const restoredSessionId = loaded.session.sessionId;

        if (!restoredSessionId) {
          setError("loadFrontendSnapshot: missing restored session_id");
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
        setError(toErrorMessage(err));
        return false;
      } finally {
        restoringSnapshotIdRef.current = "";
        setBusyAction("");
      }
    },
    [adapter, applySnapshot, listSessions],
  );

  const snapshotRound = snapshot?.round;
  const snapshotSession = snapshot?.sessionId;
  const graphRound = graphView?.round;
  const graphSession = graphView?.sessionId;

  useEffect(() => {
    void listSessions();
    void listFrontendSnapshots();
  }, [listFrontendSnapshots, listSessions]);

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
      setTeamflowStream([]);
      setSnapshotIndex([]);
      setTurnArtifacts([]);
      setMemoryView(null);
      setMemoryCaseGraphView(null);
      setStreamStatus("idle");
      lastSeqRef.current = 0;
      wsLiveRef.current = false;
      snapshotRef.current = null;
      sessionIdRef.current = "";
      snapshotInFlightRef.current = null;
      snapshotInFlightSessionRef.current = "";
      graphInFlightRef.current = null;
      graphInFlightSessionRef.current = "";
      timelineInFlightRef.current = null;
      timelineInFlightSessionRef.current = "";
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
      loadTeamflowStream(80),
      loadMemory(),
    ]);
  }, [
    loadMemory,
    loadSnapshots,
    loadTeamflowStream,
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
      teamflowStream,
      snapshotIndex,
      turnArtifacts,
      memoryView,
      memoryCaseGraphView,
      frontendSnapshots,
      busyAction,
      error,
      clearError,
      listSessions,
      createSession,
      selectSession,
      step,
      adjudicate,
      resetMemory,
      refreshSnapshot,
      loadGraph,
      loadGraphAtRound,
      loadGraphDiff,
      loadTimeline,
      loadTeamflowStream,
      loadSnapshots,
      loadTurnArtifacts,
      loadMemory,
      loadMemoryCaseGraph,
      exportGraphGexf,
      saveFrontendSnapshot,
      importFrontendSnapshotBundle,
      listFrontendSnapshots,
      loadFrontendSnapshot,
    }),
    [
      adapterMode,
      adjudicate,
      baselineGraphView,
      busyAction,
      clearError,
      createSession,
      error,
      exportGraphGexf,
      frontendSnapshots,
      graphDiff,
      graphView,
      importFrontendSnapshotBundle,
      listFrontendSnapshots,
      listSessions,
      loadFrontendSnapshot,
      loadGraph,
      loadGraphAtRound,
      loadGraphDiff,
      loadMemory,
      loadMemoryCaseGraph,
      loadSnapshots,
      loadTimeline,
      loadTeamflowStream,
      loadTurnArtifacts,
      memoryView,
      memoryCaseGraphView,
      previousSnapshot,
      refreshSnapshot,
      resetMemory,
      saveFrontendSnapshot,
      selectSession,
      sessionId,
      sessions,
      snapshot,
      snapshotIndex,
      step,
      streamStatus,
      teamflowStream,
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
