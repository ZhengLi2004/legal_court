import { useEffect, useMemo, useState } from "react";
import { createCompatAdapter } from "./compat";

import type {
  DebateSnapshot,
  GraphDiffView,
  GraphView,
  MemoryView,
  TimelineEvent,
} from "./compat";

const envMode = import.meta.env.VITE_COMPAT_MODE;
const envBaseUrl = import.meta.env.VITE_API_BASE_URL;
type AdapterMode = "auto" | "http" | "mock";

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString();
}

function App() {
  const adapterMode: AdapterMode =
    envMode === "http" || envMode === "mock" ? envMode : "auto";

  const adapter = useMemo(
    () =>
      createCompatAdapter({ mode: adapterMode, baseUrl: envBaseUrl || "/api" }),
    [adapterMode],
  );

  const [snapshot, setSnapshot] = useState<DebateSnapshot | null>(null);

  const [previousSnapshot, setPreviousSnapshot] =
    useState<DebateSnapshot | null>(null);

  const [sessions, setSessions] = useState<DebateSnapshot[]>([]);
  const [graphView, setGraphView] = useState<GraphView | null>(null);
  const [graphDiff, setGraphDiff] = useState<GraphDiffView | null>(null);
  const [memoryView, setMemoryView] = useState<MemoryView | null>(null);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [busyAction, setBusyAction] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [logs, setLogs] = useState<string[]>([]);
  const sessionId = snapshot?.sessionId ?? sessions[0]?.sessionId ?? "";

  const transcriptDelta = useMemo(() => {
    if (!snapshot) {
      return [];
    }

    if (
      !previousSnapshot ||
      previousSnapshot.sessionId !== snapshot.sessionId
    ) {
      return snapshot.transcript;
    }

    return snapshot.transcript.slice(previousSnapshot.transcript.length);
  }, [snapshot, previousSnapshot]);

  const appendLog = (message: string): void => {
    const row = `${new Date().toISOString()} ${message}`;
    setLogs((prev) => [row, ...prev].slice(0, 30));
  };

  const runAction = async (
    actionName: string,
    action: () => Promise<DebateSnapshot | DebateSnapshot[]>,
  ): Promise<void> => {
    setBusyAction(actionName);
    setError("");

    try {
      const result = await action();

      if (Array.isArray(result)) {
        setSessions(result);
        appendLog(`${actionName}: loaded ${result.length} sessions`);
      } else {
        setPreviousSnapshot(snapshot);
        setSnapshot(result);

        appendLog(
          `${actionName}: ${result.sessionId} round=${result.round} phase=${result.phase} transport=${adapter.capabilities.transport}`,
        );
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      appendLog(`${actionName} failed: ${message}`);
    } finally {
      setBusyAction("");
    }
  };

  const loadGraph = async (): Promise<void> => {
    if (!sessionId) {
      return;
    }

    setBusyAction("loadGraph");
    setError("");

    try {
      const result = await adapter.graph.getGraph(sessionId);
      setGraphView(result);

      appendLog(
        `loadGraph: nodes=${result.nodes.length}, edges=${result.edges.length}`,
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      appendLog(`loadGraph failed: ${message}`);
    } finally {
      setBusyAction("");
    }
  };

  const loadDiff = async (): Promise<void> => {
    if (!sessionId || !snapshot) {
      return;
    }

    const fromRound =
      previousSnapshot?.round ?? Math.max(snapshot.round - 1, 0);

    const toRound = snapshot.round;
    setBusyAction("loadDiff");
    setError("");

    try {
      const result = await adapter.graph.getGraphDiff(
        sessionId,
        fromRound,
        toRound,
      );

      setGraphDiff(result);

      appendLog(
        `loadDiff: +N${result.addedNodeIds.length} -N${result.removedNodeIds.length} +E${result.addedEdgeIds.length} -E${result.removedEdgeIds.length}`,
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      appendLog(`loadDiff failed: ${message}`);
    } finally {
      setBusyAction("");
    }
  };

  const loadMemory = async (): Promise<void> => {
    if (!sessionId) {
      return;
    }

    setBusyAction("loadMemory");
    setError("");

    try {
      const result = await adapter.insight.getMemory(sessionId);
      setMemoryView(result);
      appendLog(`loadMemory: insights=${result.insightSummaries.length}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      appendLog(`loadMemory failed: ${message}`);
    } finally {
      setBusyAction("");
    }
  };

  const loadTimeline = async (): Promise<void> => {
    if (!sessionId) {
      return;
    }

    setBusyAction("loadTimeline");
    setError("");

    try {
      const result = await adapter.insight.getTimeline(sessionId, 20);
      setTimeline(result);
      appendLog(`loadTimeline: events=${result.length}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      appendLog(`loadTimeline failed: ${message}`);
    } finally {
      setBusyAction("");
    }
  };

  useEffect(() => {
    let alive = true;
    setBusyAction("listSessions");
    setError("");

    void adapter
      .listSessions()
      .then((result) => {
        if (!alive) {
          return;
        }

        setSessions(result);
        appendLog(`listSessions: loaded ${result.length} sessions`);
      })
      .catch((err) => {
        if (!alive) {
          return;
        }

        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        appendLog(`listSessions failed: ${message}`);
      })
      .finally(() => {
        if (alive) {
          setBusyAction("");
        }
      });

    return () => {
      alive = false;
    };
  }, [adapter]);

  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Bridge Layer Demo</p>
          <h1>Frontend/Backend Compat Adapter</h1>
        </div>

        <div className="transport">
          <span>mode: {adapterMode}</span>
          <span>active: {adapter.capabilities.transport}</span>
          <span>diff: {adapter.capabilities.supportsDiff ? "on" : "off"}</span>
        </div>
      </section>

      <section className="actions">
        <button
          disabled={Boolean(busyAction)}
          onClick={() =>
            void runAction("createSession", () =>
              adapter.createSession({ maxRounds: 6 }),
            )
          }
        >
          Create Session
        </button>

        <button
          disabled={Boolean(busyAction) || !sessionId}
          onClick={() => void runAction("step", () => adapter.step(sessionId))}
        >
          Step
        </button>

        <button
          disabled={Boolean(busyAction) || !sessionId}
          onClick={() =>
            void runAction("adjudicate", () => adapter.adjudicate(sessionId))
          }
        >
          Adjudicate
        </button>

        <button
          disabled={Boolean(busyAction) || !sessionId}
          onClick={() =>
            void runAction("getSnapshot", () => adapter.getSnapshot(sessionId))
          }
        >
          Refresh Snapshot
        </button>

        <button
          disabled={Boolean(busyAction)}
          onClick={() =>
            void runAction("listSessions", () => adapter.listSessions())
          }
        >
          List Sessions
        </button>

        <button
          disabled={Boolean(busyAction) || !sessionId}
          onClick={() => void loadGraph()}
        >
          Load Graph
        </button>

        <button
          disabled={Boolean(busyAction) || !sessionId || !snapshot}
          onClick={() => void loadDiff()}
        >
          Load Diff
        </button>

        <button
          disabled={Boolean(busyAction) || !sessionId}
          onClick={() => void loadMemory()}
        >
          Load Memory
        </button>

        <button
          disabled={Boolean(busyAction) || !sessionId}
          onClick={() => void loadTimeline()}
        >
          Load Timeline
        </button>
      </section>

      {busyAction && <p className="hint">running: {busyAction}</p>}
      {error && <p className="error">error: {error}</p>}

      <section className="layout">
        <article className="card">
          <h2>Session Overview</h2>
          {snapshot ? (
            <>
              <p className="line">session: {snapshot.sessionId}</p>

              <p className="line">
                phase: <strong>{snapshot.phase}</strong>
              </p>

              <p className="line">
                round: {snapshot.round} / {snapshot.maxRounds}
              </p>

              <p className="line">winner: {snapshot.winner ?? "pending"}</p>

              <div className="metrics">
                <div>
                  <span>Arguments</span>
                  <strong>{snapshot.metrics.arguments}</strong>
                </div>

                <div>
                  <span>Attacks</span>
                  <strong>{snapshot.metrics.attacks}</strong>
                </div>

                <div>
                  <span>Supports</span>
                  <strong>{snapshot.metrics.supports}</strong>
                </div>
              </div>

              <p className="line">updated: {formatTime(snapshot.updatedAt)}</p>
            </>
          ) : (
            <p className="hint">No active snapshot.</p>
          )}
        </article>

        <article className="card">
          <h2>Domain Modules</h2>
          <p className="line">graph nodes: {graphView?.nodes.length ?? "-"}</p>
          <p className="line">graph edges: {graphView?.edges.length ?? "-"}</p>

          <p className="line">
            memory insights: {memoryView?.insightSummaries.length ?? "-"}
          </p>

          <p className="line">timeline events: {timeline.length || "-"}</p>

          <p className="line">
            diff: +N{graphDiff?.addedNodeIds.length ?? 0} -N
            {graphDiff?.removedNodeIds.length ?? 0}
          </p>
        </article>

        <article className="card">
          <h2>Transcript Diff</h2>
          <p className="hint">New lines from latest state transition</p>

          <div className="scrollbox">
            {transcriptDelta.length > 0 ? (
              transcriptDelta.map((line, idx) => (
                <p className="diff" key={`${line}-${idx}`}>
                  + {line}
                </p>
              ))
            ) : (
              <p className="hint">No new lines.</p>
            )}
          </div>
        </article>

        <article className="card">
          <h2>Known Sessions</h2>

          <div className="scrollbox">
            {sessions.length === 0 && (
              <p className="hint">No sessions loaded.</p>
            )}

            {sessions.map((item) => (
              <button
                className="session-row"
                key={item.sessionId}
                onClick={() =>
                  void runAction("getSnapshot", () =>
                    adapter.getSnapshot(item.sessionId),
                  )
                }
                type="button"
              >
                <span>{item.sessionId}</span>

                <span>
                  r{item.round}/{item.maxRounds}
                </span>

                <span>{item.phase}</span>
              </button>
            ))}
          </div>
        </article>

        <article className="card">
          <h2>Memory Preview</h2>

          <div className="scrollbox">
            {memoryView?.insightSummaries.length ? (
              memoryView.insightSummaries.map((line, idx) => (
                <p className="log" key={`${line}-${idx}`}>
                  {line}
                </p>
              ))
            ) : (
              <p className="hint">No memory data loaded.</p>
            )}
          </div>
        </article>

        <article className="card wide">
          <h2>Adapter Event Log</h2>

          <div className="scrollbox">
            {logs.length === 0 ? (
              <p className="hint">No events yet.</p>
            ) : (
              logs.map((line, idx) => (
                <p className="log" key={`${line}-${idx}`}>
                  {line}
                </p>
              ))
            )}
          </div>
        </article>
      </section>
    </main>
  );
}

export default App;
