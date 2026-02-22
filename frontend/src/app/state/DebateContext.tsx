import { useCallback, useEffect, useMemo, type ReactNode } from "react";
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
import { useGraphActions } from "./useGraphActions";
import { useSessionActions } from "./useSessionActions";
import { useSnapshotActions } from "./useSnapshotActions";
import { useTimelineStream } from "./useTimelineStream";

const envBaseUrl = import.meta.env.VITE_API_BASE_URL;

export function DebateProvider({ children }: { children: ReactNode }) {
  const adapterMode: AdapterMode = "http";

  const adapter = useMemo(
    () => createCompatAdapter({ baseUrl: envBaseUrl || "/api" }),
    [],
  );

  const {
    sessionId,
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
  } = useSessionActions(adapter);

  const {
    graphView,
    baselineGraphView,
    graphDiff,
    memoryCaseGraphView,
    loadGraph,
    loadGraphAtRound,
    loadGraphDiff,
    loadMemoryCaseGraph,
    exportGraphGexf,
    clearGraphState,
    clearMemoryCaseGraphView,
  } = useGraphActions({
    adapter,
    sessionId,
    sessionIdRef,
    reportError,
  });

  const {
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
  } = useSnapshotActions({
    adapter,
    sessionId,
    sessionsLength: sessions.length,
    setBusyAction,
    setErrorMessage,
    reportError,
    setActiveSessionId,
    applySnapshot,
    listSessions,
    onResetMemoryState: clearMemoryCaseGraphView,
  });

  const { timeline, streamStatus, loadTimeline, clearTimelineState } =
    useTimelineStream({
      adapter,
      sessionId,
      loadGraph,
      reportError,
    });

  const step = useCallback(async (): Promise<boolean> => {
    const ok = await stepSession();

    if (ok) {
      void refreshMemorySilently();
    }

    return ok;
  }, [refreshMemorySilently, stepSession]);

  const adjudicate = useCallback(async (): Promise<boolean> => {
    const ok = await adjudicateSession();

    if (ok) {
      void refreshMemorySilently();
      void refreshSnapshotsSilently();
    }

    return ok;
  }, [adjudicateSession, refreshMemorySilently, refreshSnapshotsSilently]);

  useEffect(() => {
    void listSessions();
    void listFrontendSnapshots();
  }, [listFrontendSnapshots, listSessions]);

  useEffect(() => {
    if (!sessionId) {
      clearGraphState();
      clearSnapshotState();
      clearTimelineState();
      snapshotRef.current = null;
      sessionIdRef.current = "";
      return;
    }

    if (!snapshotRef.current || snapshotRef.current.sessionId !== sessionId) {
      void refreshSnapshot();
    }
  }, [
    clearGraphState,
    clearSnapshotState,
    clearTimelineState,
    refreshSnapshot,
    sessionId,
    sessionIdRef,
    snapshotRef,
  ]);

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

  const snapshotRound = snapshot?.round;
  const snapshotSession = snapshot?.sessionId;
  const graphRound = graphView?.round;
  const graphSession = graphView?.sessionId;

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

  const contextValue = useMemo<DebateContextValue>(
    () => ({
      adapterMode,
      sessionId,
      streamStatus: streamStatus as StreamStatus,
      snapshot: snapshot as DebateSnapshot | null,
      previousSnapshot: previousSnapshot as DebateSnapshot | null,
      sessions: sessions as DebateSnapshot[],
      graphView: graphView as GraphView | null,
      baselineGraphView: baselineGraphView as GraphView | null,
      graphDiff: graphDiff as GraphDiffView | null,
      timeline: timeline as TimelineEvent[],
      teamflowStream: teamflowStream as TeamFlowTurn[],
      snapshotIndex: snapshotIndex as SnapshotIndexItem[],
      turnArtifacts: turnArtifacts as TurnArtifact[],
      memoryView: memoryView as MemoryView | null,
      memoryCaseGraphView: memoryCaseGraphView as GraphView | null,
      frontendSnapshots: frontendSnapshots as FrontendSnapshotListItem[],
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
      memoryCaseGraphView,
      memoryView,
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
