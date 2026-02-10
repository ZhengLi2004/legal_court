import { useEffect, useMemo, useRef, useState } from "react";
import { createCompatAdapter } from "./compat";

import type {
  DebateSnapshot,
  GraphDiffView,
  GraphView,
  MemoryView,
  SnapshotIndexItem,
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
  const [snapshotIndex, setSnapshotIndex] = useState<SnapshotIndexItem[]>([]);
  const [replayFromRound, setReplayFromRound] = useState<number>(0);
  const [replayToRound, setReplayToRound] = useState<number>(0);

  const [streamStatus, setStreamStatus] = useState<"idle" | "ws" | "poll">(
    "idle",
  );

  const [busyAction, setBusyAction] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [logs, setLogs] = useState<string[]>([]);
  const lastSeqRef = useRef<number>(0);
  const wsLiveRef = useRef<boolean>(false);
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
    setLogs((prev) => [row, ...prev].slice(0, 50));
  };

  const syncSnapshotIndex = (
    items: SnapshotIndexItem[],
    resetSelection: boolean,
  ): void => {
    setSnapshotIndex(items);

    if (!items.length) {
      setReplayFromRound(0);
      setReplayToRound(0);
      return;
    }

    const rounds = items.map((item) => item.round);
    const latest = rounds[rounds.length - 1];
    const previous = rounds.length > 1 ? rounds[rounds.length - 2] : rounds[0];

    if (resetSelection) {
      setReplayFromRound(previous);
      setReplayToRound(latest);
      return;
    }

    setReplayFromRound((prev) => {
      if (!rounds.includes(prev)) {
        return previous;
      }

      return prev === 0 && latest > 0 ? previous : prev;
    });

    setReplayToRound((prev) => {
      if (!rounds.includes(prev)) {
        return latest;
      }

      return prev === 0 && latest > 0 ? latest : prev;
    });
  };

  const replaceTimeline = (rows: TimelineEvent[]): void => {
    const sorted = [...rows].sort((a, b) => a.seq - b.seq || a.ts - b.ts);
    const latest = sorted.length > 0 ? sorted[sorted.length - 1] : null;
    lastSeqRef.current = latest?.seq ?? 0;
    setTimeline(sorted.slice(-120));
  };

  const mergeTimeline = (rows: TimelineEvent[]): void => {
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

      const merged = [...bucket.values()].sort(
        (a, b) => a.seq - b.seq || a.ts - b.ts,
      );

      const latest = merged.length > 0 ? merged[merged.length - 1] : null;
      lastSeqRef.current = latest?.seq ?? lastSeqRef.current;

      return merged.slice(-120);
    });
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
      const result = await adapter.insight.getTimeline(sessionId, 40);
      replaceTimeline(result);
      appendLog(`loadTimeline: events=${result.length}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      appendLog(`loadTimeline failed: ${message}`);
    } finally {
      setBusyAction("");
    }
  };

  const loadSnapshots = async (
    options: { silent?: boolean } = {},
  ): Promise<void> => {
    if (!sessionId) {
      return;
    }

    const silent = options.silent === true;

    if (!silent) {
      setBusyAction("loadSnapshots");
    }

    setError("");

    try {
      const items = await adapter.session.getSnapshots(sessionId);
      syncSnapshotIndex(items, !silent);

      if (!silent) {
        appendLog(`loadSnapshots: items=${items.length}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);

      appendLog(
        silent
          ? `loadSnapshots(auto) failed: ${message}`
          : `loadSnapshots failed: ${message}`,
      );
    } finally {
      if (!silent) {
        setBusyAction("");
      }
    }
  };

  const loadReplayRound = async (): Promise<void> => {
    if (!sessionId) {
      return;
    }

    setBusyAction("loadReplayRound");
    setError("");

    try {
      const graph = await adapter.graph.getGraphAtRound(
        sessionId,
        replayToRound,
      );

      setGraphView(graph);

      appendLog(
        `loadReplayRound: round=${replayToRound}, nodes=${graph.nodes.length}, edges=${graph.edges.length}`,
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      appendLog(`loadReplayRound failed: ${message}`);
    } finally {
      setBusyAction("");
    }
  };

  const loadReplayDiff = async (): Promise<void> => {
    if (!sessionId) {
      return;
    }

    setBusyAction("loadReplayDiff");
    setError("");

    try {
      const result = await adapter.graph.getGraphDiff(
        sessionId,
        replayFromRound,
        replayToRound,
      );

      setGraphDiff(result);

      appendLog(
        `loadReplayDiff: ${replayFromRound} -> ${replayToRound}, +N${result.addedNodeIds.length}, +E${result.addedEdgeIds.length}`,
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      appendLog(`loadReplayDiff failed: ${message}`);
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

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    void loadSnapshots({ silent: true });
  }, [adapter, sessionId, snapshot?.round]);

  useEffect(() => {
    if (!sessionId) {
      setStreamStatus("idle");
      setTimeline([]);
      setSnapshotIndex([]);
      lastSeqRef.current = 0;
      wsLiveRef.current = false;
      return;
    }

    let alive = true;
    wsLiveRef.current = false;

    const pullTimeline = async (): Promise<void> => {
      try {
        const rows = await adapter.insight.getTimeline(sessionId, 40);

        if (!alive) {
          return;
        }

        replaceTimeline(rows);

        if (!wsLiveRef.current) {
          setStreamStatus("poll");
        }
      } catch (err) {
        if (!alive) {
          return;
        }

        const message = err instanceof Error ? err.message : String(err);
        appendLog(`timeline poll failed: ${message}`);
      }
    };

    void pullTimeline();
    void loadSnapshots({ silent: true });

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
            onError: (streamErr) => {
              if (!alive) {
                return;
              }

              if (wsLiveRef.current) {
                appendLog(`stream fallback to polling: ${streamErr.message}`);
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
  }, [adapter, sessionId]);

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

          <span>
            stream:{" "}
            {adapter.capabilities.supportsStreaming ? streamStatus : "off"}
          </span>
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

        <button
          disabled={Boolean(busyAction) || !sessionId}
          onClick={() => void loadSnapshots()}
        >
          Load Snapshots
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
          <p className="line">snapshot index: {snapshotIndex.length || "-"}</p>

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

        <article className="card">
          <h2>Timeline Stream</h2>
          <p className="hint">latest seq: {lastSeqRef.current || "-"}</p>

          <div className="scrollbox">
            {timeline.length ? (
              timeline
                .slice(-20)
                .reverse()
                .map((row) => (
                  <p className="log" key={`${row.seq}-${row.event}`}>
                    #{row.seq} [{row.source}] {row.event}
                  </p>
                ))
            ) : (
              <p className="hint">No timeline events.</p>
            )}
          </div>
        </article>

        <article className="card wide">
          <h2>Replay Controls</h2>

          {snapshotIndex.length ? (
            <div className="replay-controls">
              <label>
                from round
                <select
                  value={replayFromRound}
                  onChange={(event) =>
                    setReplayFromRound(Number(event.target.value))
                  }
                >
                  {snapshotIndex.map((item) => (
                    <option key={`from-${item.round}`} value={item.round}>
                      {item.round} ({item.turn || "unknown"})
                    </option>
                  ))}
                </select>
              </label>

              <label>
                to round
                <select
                  value={replayToRound}
                  onChange={(event) =>
                    setReplayToRound(Number(event.target.value))
                  }
                >
                  {snapshotIndex.map((item) => (
                    <option key={`to-${item.round}`} value={item.round}>
                      {item.round} ({item.turn || "unknown"})
                    </option>
                  ))}
                </select>
              </label>

              <button
                disabled={Boolean(busyAction) || !sessionId}
                onClick={() => void loadReplayRound()}
                type="button"
              >
                Load Replay Round
              </button>

              <button
                disabled={Boolean(busyAction) || !sessionId}
                onClick={() => void loadReplayDiff()}
                type="button"
              >
                Compare Replay Diff
              </button>
            </div>
          ) : (
            <p className="hint">Load snapshots to enable replay controls.</p>
          )}
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
