import { useCallback, useRef, useState, type MutableRefObject } from "react";

import type { EngineAdapter, GraphDiffView, GraphView } from "../../compat";

interface UseGraphActionsParams {
  adapter: EngineAdapter;
  sessionId: string;
  sessionIdRef: MutableRefObject<string>;
  reportError: (err: unknown) => void;
}

interface UseGraphActionsResult {
  graphView: GraphView | null;
  baselineGraphView: GraphView | null;
  graphDiff: GraphDiffView | null;
  memoryCaseGraphView: GraphView | null;
  loadGraph: () => Promise<boolean>;
  loadGraphAtRound: (round: number) => Promise<boolean>;
  loadGraphDiff: (fromRound: number, toRound: number) => Promise<boolean>;
  loadMemoryCaseGraph: (caseId: string) => Promise<boolean>;
  exportGraphGexf: (round?: number) => Promise<Blob | null>;
  clearGraphState: () => void;
  clearMemoryCaseGraphView: () => void;
}

export function useGraphActions({
  adapter,
  sessionId,
  sessionIdRef,
  reportError,
}: UseGraphActionsParams): UseGraphActionsResult {
  const [graphView, setGraphView] = useState<GraphView | null>(null);

  const [baselineGraphView, setBaselineGraphView] = useState<GraphView | null>(
    null,
  );

  const [graphDiff, setGraphDiff] = useState<GraphDiffView | null>(null);

  const [memoryCaseGraphView, setMemoryCaseGraphView] =
    useState<GraphView | null>(null);

  const graphInFlightRef = useRef<Promise<boolean> | null>(null);
  const graphInFlightSessionRef = useRef<string>("");

  const clearMemoryCaseGraphView = useCallback(() => {
    setMemoryCaseGraphView(null);
  }, []);

  const clearGraphState = useCallback(() => {
    setGraphView(null);
    setBaselineGraphView(null);
    setGraphDiff(null);
    setMemoryCaseGraphView(null);
    graphInFlightRef.current = null;
    graphInFlightSessionRef.current = "";
  }, []);

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
        reportError(err);
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
  }, [adapter, reportError, sessionId, sessionIdRef]);

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
        reportError(err);
        return false;
      }
    },
    [adapter, reportError, sessionId],
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
        reportError(err);
        return false;
      }
    },
    [adapter, reportError, sessionId],
  );

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
        reportError(err);
        return false;
      }
    },
    [adapter, reportError, sessionId],
  );

  const exportGraphGexf = useCallback(
    async (round?: number): Promise<Blob | null> => {
      if (!sessionId) {
        return null;
      }

      try {
        return await adapter.insight.exportGraphGexf(sessionId, round);
      } catch (err) {
        reportError(err);
        return null;
      }
    },
    [adapter, reportError, sessionId],
  );

  return {
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
  };
}
