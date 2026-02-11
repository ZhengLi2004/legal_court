import { useEffect, useMemo, useRef, useState } from "react";
import "@xyflow/react/dist/style.css";
import { createCompatAdapter } from "./compat";

import {
  DebugBundlePanel,
  GraphDiffPanel,
  InspectorPanel,
  TeamFlowPanel,
  TimelinePanel,
} from "./components/debug";

import type {
  DebateSnapshot,
  DemoKeyframe,
  DemoRunResult,
  DebugBundleView,
  GraphDiffView,
  GraphView,
  MemoryInsightItem,
  MemoryView,
  ReplayExportView,
  SnapshotIndexItem,
  TimelineEvent,
  TurnArtifact,
} from "./compat";

const envMode = import.meta.env.VITE_COMPAT_MODE;
const envBaseUrl = import.meta.env.VITE_API_BASE_URL;
type AdapterMode = "auto" | "http" | "mock";

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString();
}

interface AuditRow {
  id: string;
  round: number;
  side: string;
  actionType: string;
  status: "accepted" | "rejected" | "unknown";
  axiom: string;
  reason: string;
}

interface MemoryRepresentativeChange {
  content: string;
  before: string[];
  after: string[];
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function asNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Number(value);

    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return fallback;
}

function unwrapPayload(raw: unknown): Record<string, unknown> {
  const outer = asRecord(raw);
  const nested = outer.data ?? outer.snapshot ?? outer.state ?? outer.payload;
  return nested !== undefined ? asRecord(nested) : outer;
}

function inferAuditStatus(text: string): "accepted" | "rejected" | "unknown" {
  const normalized = text.toLowerCase();

  if (
    normalized.includes("reject") ||
    normalized.includes("rollback") ||
    normalized.includes("failed") ||
    normalized.includes("error")
  ) {
    return "rejected";
  }

  if (
    normalized.includes("success") ||
    normalized.includes("applied") ||
    normalized.includes("completed") ||
    normalized.includes("ok")
  ) {
    return "accepted";
  }

  return "unknown";
}

function inferAxiom(text: string): string {
  const normalized = text.toLowerCase();

  if (normalized.includes("cycle")) {
    return "No Directed Cycle";
  }

  if (normalized.includes("support") && normalized.includes("claim")) {
    return "Support -> CLAIM";
  }

  if (normalized.includes("conflict") && normalized.includes("claim")) {
    return "Conflict endpoints -> CLAIM";
  }

  return "N/A";
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

  const [baselineGraphView, setBaselineGraphView] = useState<GraphView | null>(
    null,
  );

  const [graphDiff, setGraphDiff] = useState<GraphDiffView | null>(null);
  const [memoryView, setMemoryView] = useState<MemoryView | null>(null);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [turnArtifacts, setTurnArtifacts] = useState<TurnArtifact[]>([]);
  const [snapshotIndex, setSnapshotIndex] = useState<SnapshotIndexItem[]>([]);
  const [replayFromRound, setReplayFromRound] = useState<number>(0);
  const [replayToRound, setReplayToRound] = useState<number>(0);
  const [selectedTimelineSeq, setSelectedTimelineSeq] = useState<number>(0);
  const [selectedTurnUid, setSelectedTurnUid] = useState<string>("");
  const [debugBundle, setDebugBundle] = useState<DebugBundleView | null>(null);
  const [debugBundleLoading, setDebugBundleLoading] = useState<boolean>(false);
  const [demoRunning, setDemoRunning] = useState<boolean>(false);
  const [demoResult, setDemoResult] = useState<DemoRunResult | null>(null);
  const [demoKeyframes, setDemoKeyframes] = useState<DemoKeyframe[]>([]);

  const [failureMode, setFailureMode] = useState<
    "off" | "es_unavailable" | "llm_timeout"
  >("off");

  const [replayExport, setReplayExport] = useState<ReplayExportView | null>(
    null,
  );

  const [memoryNewInsights, setMemoryNewInsights] = useState<string[]>([]);

  const [memoryRetainedInsights, setMemoryRetainedInsights] = useState<
    string[]
  >([]);

  const [memoryRepresentativeChanges, setMemoryRepresentativeChanges] =
    useState<MemoryRepresentativeChange[]>([]);

  const [streamStatus, setStreamStatus] = useState<"idle" | "ws" | "poll">(
    "idle",
  );

  const [busyAction, setBusyAction] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [logs, setLogs] = useState<string[]>([]);
  const lastSeqRef = useRef<number>(0);
  const wsLiveRef = useRef<boolean>(false);
  const sessionId = snapshot?.sessionId ?? sessions[0]?.sessionId ?? "";

  const selectedArtifact = useMemo(() => {
    if (!turnArtifacts.length) {
      return null;
    }

    if (selectedTurnUid) {
      return (
        turnArtifacts.find((item) => item.turnUid === selectedTurnUid) ?? null
      );
    }

    return turnArtifacts[turnArtifacts.length - 1] ?? null;
  }, [selectedTurnUid, turnArtifacts]);

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

  const judgmentPayload = useMemo(
    () => unwrapPayload(snapshot?.raw ?? {}),
    [snapshot],
  );

  const judgmentDocument = asString(judgmentPayload.judgment_document);

  const rootClaimEntries = Object.entries(
    asRecord(judgmentPayload.root_claims_status),
  );

  const bafDetails = asRecord(judgmentPayload.baf_details);

  const rootClaimValidatedCount = rootClaimEntries.filter(([, status]) =>
    String(status).toUpperCase().includes("VALID"),
  ).length;

  const auditRows = useMemo(() => {
    const rows: AuditRow[] = [];

    for (const artifact of turnArtifacts) {
      const executionLogs = asString(artifact.executionLogs);
      const status = inferAuditStatus(executionLogs);
      const axiom = inferAxiom(executionLogs);
      const reason = executionLogs.split("\n")[0]?.trim() ?? "";

      const actionRows = Array.isArray(artifact.parsedActions)
        ? artifact.parsedActions
        : [];

      if (actionRows.length === 0) {
        rows.push({
          id: `${artifact.turnUid}-none`,
          round: artifact.round,
          side: artifact.side,
          actionType: "(none)",
          status,
          axiom,
          reason,
        });

        continue;
      }

      actionRows.forEach((item, index) => {
        const action = asRecord(item);

        rows.push({
          id: `${artifact.turnUid}-${index}`,
          round: artifact.round,
          side: artifact.side,
          actionType: asString(
            action.action_type ?? action.actionType ?? action.type,
            "unknown",
          ),
          status,
          axiom,
          reason,
        });
      });
    }

    return rows.slice(-60).reverse();
  }, [turnArtifacts]);

  const warningEvents = useMemo(
    () => timeline.filter((row) => row.event === "session_warning"),
    [timeline],
  );

  const latestWarningText = useMemo(() => {
    if (!warningEvents.length) {
      return "";
    }

    const latest = warningEvents[warningEvents.length - 1];
    const payload = asRecord(latest.data);
    const kind = asString(payload.kind);
    const stage = asString(payload.stage);
    const message = asString(payload.message);

    return [stage, kind, message].filter(Boolean).join(" | ");
  }, [warningEvents]);

  const appendLog = (message: string): void => {
    const row = `${new Date().toISOString()} ${message}`;
    setLogs((prev) => [row, ...prev].slice(0, 80));
  };

  const jumpToInsightReplay = async (
    item: MemoryInsightItem,
  ): Promise<void> => {
    const linkedRound = Number.isFinite(item.linkedRound)
      ? item.linkedRound
      : 0;

    await loadReplayRound(linkedRound);
    appendLog(`memory insight jump: round=${linkedRound} | ${item.content}`);
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

    setReplayFromRound((prev) => (rounds.includes(prev) ? prev : previous));
    setReplayToRound((prev) => (rounds.includes(prev) ? prev : latest));
  };

  const replaceTimeline = (rows: TimelineEvent[]): void => {
    const sorted = [...rows].sort((a, b) => a.seq - b.seq || a.ts - b.ts);
    const latest = sorted.length > 0 ? sorted[sorted.length - 1] : null;
    lastSeqRef.current = latest?.seq ?? 0;
    setTimeline(sorted.slice(-180));
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
      return merged.slice(-180);
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
          `${actionName}: ${result.sessionId} round=${result.round} phase=${result.phase}`,
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

  const loadGraph = async (
    options: { silent?: boolean } = {},
  ): Promise<void> => {
    if (!sessionId) {
      return;
    }

    const silent = options.silent === true;

    if (!silent) {
      setBusyAction("loadGraph");
      setError("");
    }

    try {
      const result = await adapter.graph.getGraph(sessionId);
      setGraphView(result);

      if (!silent) {
        appendLog(
          `loadGraph: nodes=${result.nodes.length}, edges=${result.edges.length}`,
        );
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);

      if (!silent) {
        setError(message);
        appendLog(`loadGraph failed: ${message}`);
      }
    } finally {
      if (!silent) {
        setBusyAction("");
      }
    }
  };

  const loadDiffWithRounds = async (
    fromRound: number,
    toRound: number,
    actionName: string,
    options: { silent?: boolean } = {},
  ): Promise<void> => {
    if (!sessionId) {
      return;
    }

    const silent = options.silent === true;

    if (!silent) {
      setBusyAction(actionName);
      setError("");
    }

    try {
      const [diffResult, fromGraph, toGraph] = await Promise.all([
        adapter.graph.getGraphDiff(sessionId, fromRound, toRound),
        adapter.graph.getGraphAtRound(sessionId, fromRound),
        adapter.graph.getGraphAtRound(sessionId, toRound),
      ]);

      setGraphDiff(diffResult);
      setBaselineGraphView(fromGraph);
      setGraphView(toGraph);

      if (!silent) {
        appendLog(
          `${actionName}: ${fromRound}->${toRound} +N${diffResult.addedNodeIds.length} +E${diffResult.addedEdgeIds.length}`,
        );
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);

      if (!silent) {
        setError(message);
        appendLog(`${actionName} failed: ${message}`);
      }
    } finally {
      if (!silent) {
        setBusyAction("");
      }
    }
  };

  const loadDiff = async (): Promise<void> => {
    if (!snapshot) {
      return;
    }

    const fromRound =
      previousSnapshot?.round ?? Math.max(snapshot.round - 1, 0);

    await loadDiffWithRounds(fromRound, snapshot.round, "loadDiff");
  };

  const loadMemory = async (
    options: { silent?: boolean } = {},
  ): Promise<void> => {
    if (!sessionId) {
      return;
    }

    const silent = options.silent === true;

    if (!silent) {
      setBusyAction("loadMemory");
      setError("");
    }

    try {
      const result = await adapter.insight.getMemory(sessionId);
      const prevInsights = memoryView?.insightSummaries ?? [];
      const previousInsightItems = memoryView?.insightItems ?? [];

      const newInsights = result.insightSummaries.filter(
        (item) => !prevInsights.includes(item),
      );

      const retainedInsights = result.insightSummaries.filter((item) =>
        prevInsights.includes(item),
      );

      const previousRepresentatives = new Map<string, Set<string>>();

      previousInsightItems.forEach((item) => {
        previousRepresentatives.set(
          item.content,
          new Set(item.representatives),
        );
      });

      const changedRepresentatives: MemoryRepresentativeChange[] = [];

      result.insightItems.forEach((item) => {
        const previous = previousRepresentatives.get(item.content);

        if (!previous) {
          return;
        }

        const current = new Set(item.representatives);
        const isSameSize = previous.size === current.size;

        const hasOnlyKnown = [...current].every((caseId) =>
          previous.has(caseId),
        );

        if (!isSameSize || !hasOnlyKnown) {
          changedRepresentatives.push({
            content: item.content,
            before: [...previous].sort(),
            after: [...current].sort(),
          });
        }
      });

      setMemoryView(result);
      setMemoryNewInsights(newInsights);
      setMemoryRetainedInsights(retainedInsights);
      setMemoryRepresentativeChanges(changedRepresentatives);

      if (!silent) {
        appendLog(
          `loadMemory: insights=${result.insightSummaries.length}, changed-representatives=${changedRepresentatives.length}`,
        );
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);

      if (!silent) {
        setError(message);
        appendLog(`loadMemory failed: ${message}`);
      }
    } finally {
      if (!silent) {
        setBusyAction("");
      }
    }
  };

  const loadTimeline = async (): Promise<void> => {
    if (!sessionId) {
      return;
    }

    setBusyAction("loadTimeline");
    setError("");

    try {
      const result = await adapter.insight.getTimeline(sessionId, 80);
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

  const loadTurnArtifacts = async (
    options: { silent?: boolean; turnUid?: string } = {},
  ): Promise<void> => {
    if (!sessionId) {
      return;
    }

    const silent = options.silent === true;

    if (!silent) {
      setBusyAction("loadTurnArtifacts");
    }

    setError("");

    try {
      const rows = await adapter.insight.getTurnArtifacts(sessionId, {
        limit: 60,
        turnUid: options.turnUid,
      });

      if (options.turnUid) {
        setTurnArtifacts((prev) => {
          const byTurn = new Map(prev.map((item) => [item.turnUid, item]));

          for (const row of rows) {
            byTurn.set(row.turnUid, row);
          }

          return [...byTurn.values()].sort((a, b) => a.round - b.round);
        });
      } else {
        setTurnArtifacts(rows);
      }

      if (rows.length > 0 && !selectedTurnUid) {
        setSelectedTurnUid(rows[rows.length - 1].turnUid);
      }

      if (!silent) {
        appendLog(`loadTurnArtifacts: turns=${rows.length}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);

      appendLog(
        silent
          ? `loadTurnArtifacts(auto) failed: ${message}`
          : `loadTurnArtifacts failed: ${message}`,
      );
    } finally {
      if (!silent) {
        setBusyAction("");
      }
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

  const loadReplayRound = async (targetRound?: number): Promise<void> => {
    if (!sessionId) {
      return;
    }

    const resolvedRound =
      typeof targetRound === "number" && Number.isFinite(targetRound)
        ? targetRound
        : replayToRound;

    setBusyAction("loadReplayRound");
    setError("");

    try {
      const graph = await adapter.graph.getGraphAtRound(
        sessionId,
        resolvedRound,
      );

      setGraphView(graph);
      setReplayToRound(resolvedRound);

      appendLog(
        `loadReplayRound: round=${resolvedRound}, nodes=${graph.nodes.length}, edges=${graph.edges.length}`,
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
    await loadDiffWithRounds(replayFromRound, replayToRound, "loadReplayDiff");
  };

  const loadDebugBundle = async (
    options: { silent?: boolean } = {},
  ): Promise<void> => {
    if (!sessionId) {
      return;
    }

    const silent = options.silent === true;

    setDebugBundleLoading(true);

    try {
      const bundle = await adapter.insight.getDebugBundle(sessionId, {
        eventLimit: 20,
        includeArtifact: true,
        includeSnapshot: true,
      });

      setDebugBundle(bundle);

      if (bundle.turnUid) {
        setSelectedTurnUid(bundle.turnUid);
      }

      if (bundle.latestTurnArtifact) {
        setTurnArtifacts((prev) => {
          const byTurn = new Map(prev.map((item) => [item.turnUid, item]));

          byTurn.set(
            bundle.latestTurnArtifact!.turnUid,
            bundle.latestTurnArtifact!,
          );

          return [...byTurn.values()].sort((a, b) => a.round - b.round);
        });
      }

      if (!silent) {
        appendLog(`loadDebugBundle: events=${bundle.recentEvents.length}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);

      if (!silent) {
        setError(message);
        appendLog(`loadDebugBundle failed: ${message}`);
      }
    } finally {
      setDebugBundleLoading(false);
    }
  };

  const loadDemoKeyframes = async (
    options: { silent?: boolean } = {},
  ): Promise<void> => {
    if (!sessionId) {
      return;
    }

    const silent = options.silent === true;

    if (!silent) {
      setBusyAction("loadDemoKeyframes");
    }

    try {
      const rows = await adapter.insight.getDemoKeyframes(sessionId);
      setDemoKeyframes(rows);

      if (!silent) {
        appendLog(`loadDemoKeyframes: items=${rows.length}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);

      if (!silent) {
        setError(message);
        appendLog(`loadDemoKeyframes failed: ${message}`);
      }
    } finally {
      if (!silent) {
        setBusyAction("");
      }
    }
  };

  const runDemo = async (): Promise<void> => {
    if (!sessionId) {
      return;
    }

    setBusyAction("runDemo");
    setError("");
    setDemoRunning(true);

    try {
      const result = await adapter.insight.runDemo(sessionId, {
        maxSteps: 20,
        autoAdjudicate: true,
        captureKeyframes: true,
      });

      setDemoResult(result);
      setDemoKeyframes(result.keyframes);

      appendLog(
        `runDemo: steps=${result.stepsExecuted}, endedBy=${result.endedBy}, keyframes=${result.keyframes.length}`,
      );

      const refreshed = await adapter.getSnapshot(sessionId);
      setPreviousSnapshot(snapshot);
      setSnapshot(refreshed);

      await Promise.all([
        loadSnapshots({ silent: true }),
        loadGraph({ silent: true }),
        loadMemory({ silent: true }),
        loadTurnArtifacts({ silent: true }),
        loadDebugBundle({ silent: true }),
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      appendLog(`runDemo failed: ${message}`);
    } finally {
      setDemoRunning(false);
      setBusyAction("");
    }
  };

  const jumpToKeyframe = async (frame: DemoKeyframe): Promise<void> => {
    if (!sessionId) {
      return;
    }

    setReplayToRound(frame.round);

    try {
      const graph = await adapter.graph.getGraphAtRound(sessionId, frame.round);
      setGraphView(graph);

      if (frame.turnUid) {
        setSelectedTurnUid(frame.turnUid);
        await loadTurnArtifacts({ silent: true, turnUid: frame.turnUid });
      }

      appendLog(
        `jumpToKeyframe: ${frame.reason} round=${frame.round} turn=${frame.turnUid || "-"}`,
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      appendLog(`jumpToKeyframe failed: ${message}`);
    }
  };

  const setFailureSimulationMode = async (
    mode: "off" | "es_unavailable" | "llm_timeout",
  ): Promise<void> => {
    if (!sessionId) {
      return;
    }

    setBusyAction("setFailureSimulation");

    try {
      await adapter.insight.setFailureSimulation(
        sessionId,
        "es_unavailable",
        mode === "es_unavailable",
      );

      await adapter.insight.setFailureSimulation(
        sessionId,
        "llm_timeout",
        mode === "llm_timeout",
      );

      setFailureMode(mode);
      appendLog(`setFailureSimulation: mode=${mode}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      appendLog(`setFailureSimulation failed: ${message}`);
    } finally {
      setBusyAction("");
    }
  };

  const exportReplayJson = async (): Promise<void> => {
    if (!sessionId) {
      return;
    }

    setBusyAction("exportReplayJson");

    try {
      const summary = await adapter.insight.exportReplayJson(sessionId);
      setReplayExport(summary);
      const content = JSON.stringify(summary.raw ?? summary, null, 2);
      const blob = new Blob([content], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${sessionId}-replay.json`;
      anchor.click();
      URL.revokeObjectURL(url);

      appendLog(
        `exportReplayJson: events=${summary.eventCount}, artifacts=${summary.artifactCount}, snapshots=${summary.snapshotCount}`,
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      appendLog(`exportReplayJson failed: ${message}`);
    } finally {
      setBusyAction("");
    }
  };

  const exportGraphGexf = async (): Promise<void> => {
    if (!sessionId) {
      return;
    }

    setBusyAction("exportGraphGexf");

    try {
      const blob = await adapter.insight.exportGraphGexf(
        sessionId,
        replayToRound,
      );

      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${sessionId}-round-${replayToRound}.gexf`;
      anchor.click();
      URL.revokeObjectURL(url);
      appendLog(`exportGraphGexf: round=${replayToRound}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      appendLog(`exportGraphGexf failed: ${message}`);
    } finally {
      setBusyAction("");
    }
  };

  const handleTimelineSelect = async (event: TimelineEvent): Promise<void> => {
    setSelectedTimelineSeq(event.seq);

    if (event.turnUid) {
      setSelectedTurnUid(event.turnUid);
      await loadTurnArtifacts({ silent: true, turnUid: event.turnUid });
    }

    if (sessionId && typeof event.roundIdx === "number") {
      try {
        const graph = await adapter.graph.getGraphAtRound(
          sessionId,
          event.roundIdx,
        );

        setGraphView(graph);
        setReplayToRound(event.roundIdx);
        appendLog(`timeline jump: seq=${event.seq} -> round=${event.roundIdx}`);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        appendLog(`timeline jump failed: ${message}`);
      }
    }
  };

  useEffect(() => {
    let alive = true;
    setBusyAction("listSessions");

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
    void loadGraph({ silent: true });
    void loadMemory({ silent: true });
    void loadTurnArtifacts({ silent: true });
    void loadDebugBundle({ silent: true });
    void loadDemoKeyframes({ silent: true });
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId || !snapshot) {
      return;
    }

    const round = snapshot.round;
    const fromRound = Math.max(round - 1, 0);
    void loadSnapshots({ silent: true });
    void loadGraph({ silent: true });
    void loadMemory({ silent: true });
    void loadTurnArtifacts({ silent: true });
    void loadDebugBundle({ silent: true });
    void loadDemoKeyframes({ silent: true });

    if (round > 0) {
      void loadDiffWithRounds(fromRound, round, "autoLoadDiff", {
        silent: true,
      });
    }
  }, [sessionId, snapshot?.round]);

  useEffect(() => {
    if (!sessionId) {
      setStreamStatus("idle");
      setTimeline([]);
      setSnapshotIndex([]);
      lastSeqRef.current = 0;
      wsLiveRef.current = false;
      setSelectedTimelineSeq(0);
      setSelectedTurnUid("");
      setDebugBundle(null);
      return;
    }

    let alive = true;
    wsLiveRef.current = false;

    const pullTimeline = async (): Promise<void> => {
      try {
        const rows = await adapter.insight.getTimeline(sessionId, 60);

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
          <h1>Phase 2.5 Debug Console</h1>
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
            void runAction("createSession", () => adapter.createSession())
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
          onClick={() => void loadTurnArtifacts()}
        >
          Load Artifacts
        </button>

        <button
          disabled={Boolean(busyAction) || !sessionId}
          onClick={() => void loadSnapshots()}
        >
          Load Snapshots
        </button>

        <button
          disabled={Boolean(busyAction) || !sessionId}
          onClick={() => void loadDebugBundle()}
        >
          Load Debug Bundle
        </button>

        <button
          disabled={Boolean(busyAction) || !sessionId || demoRunning}
          onClick={() => void runDemo()}
        >
          Run Demo
        </button>

        <button
          disabled={Boolean(busyAction) || !sessionId}
          onClick={() => void loadDemoKeyframes()}
        >
          Load Keyframes
        </button>

        <button
          disabled={Boolean(busyAction) || !sessionId}
          onClick={() => void exportReplayJson()}
        >
          Export Replay JSON
        </button>

        <button
          disabled={Boolean(busyAction) || !sessionId}
          onClick={() => void exportGraphGexf()}
        >
          Export Graph GEXF
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

              <p className="line">round: {snapshot.round}</p>

              <p className="line">
                root-claim status:{" "}
                {rootClaimEntries.length > 0
                  ? `${rootClaimValidatedCount}/${rootClaimEntries.length} validated`
                  : "pending adjudication"}
              </p>

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

          <p className="line">
            graph nodes:{" "}
            {graphView?.nodes.length ?? snapshot?.metrics.arguments ?? "-"}
          </p>

          <p className="line">
            graph edges:{" "}
            {graphView?.edges.length ??
              (snapshot
                ? snapshot.metrics.attacks + snapshot.metrics.supports
                : "-")}
          </p>

          <p className="line">
            memory insights:{" "}
            {memoryView?.insightItems.length ??
              memoryView?.insightSummaries.length ??
              "-"}
          </p>

          <p className="line">timeline events: {timeline.length || "-"}</p>
          <p className="line">snapshot index: {snapshotIndex.length || "-"}</p>
          <p className="line">turn artifacts: {turnArtifacts.length || "-"}</p>

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

                <span>r{item.round}</span>

                <span>{item.phase}</span>
              </button>
            ))}
          </div>
        </article>

        <article className="card">
          <h2>Memory Preview</h2>

          <p className="line">
            static={memoryView?.staticHistoryCount ?? "-"}, dynamic-law=
            {memoryView?.dynamicLawCaseCount ?? "-"}, task-layer-nodes=
            {memoryView?.taskLayerNodeCount ?? "-"}, task-layer-edges=
            {memoryView?.taskLayerEdgeCount ?? "-"}
          </p>

          <p className="line">
            representative cases:{" "}
            {memoryView?.representativeCaseIds.length ?? 0}
          </p>

          <div className="scrollbox">
            {memoryView?.insightItems.length ? (
              memoryView.insightItems.map((item, idx) => (
                <div className="log" key={`memory-preview-${idx}`}>
                  <p className="line">
                    [{item.side}] {item.content}
                  </p>

                  <p className="line">
                    reps({item.representativeCount}):{" "}
                    {item.representatives.join(", ") || "-"}
                  </p>
                </div>
              ))
            ) : (
              <p className="hint">
                No insight summaries yet (this can be normal before enough
                rounds accumulate).
              </p>
            )}
          </div>
        </article>

        <article className="card wide">
          <h2>Memory Deep View</h2>

          <div className="memory-grid">
            <section className="memory-pane">
              <h3>Insights Layer</h3>
              <p className="line">new: {memoryNewInsights.length}</p>
              <p className="line">retained: {memoryRetainedInsights.length}</p>

              <p className="line">
                representative changed: {memoryRepresentativeChanges.length}
              </p>

              <div className="scrollbox">
                {memoryView?.insightItems.length ? (
                  memoryView.insightItems.map((item, idx) => {
                    const isNew = memoryNewInsights.includes(item.content);

                    return (
                      <button
                        className={`timeline-row ${isNew ? "timeline-row-selected" : ""}`}
                        key={`memory-insight-${idx}`}
                        onClick={() => {
                          void jumpToInsightReplay(item);
                        }}
                        type="button"
                      >
                        <span>{item.side}</span>
                        <span>{item.content}</span>
                        <span>r{item.linkedRound}</span>
                      </button>
                    );
                  })
                ) : (
                  <p className="hint">No insights this round.</p>
                )}
              </div>
            </section>

            <section className="memory-pane">
              <h3>TaskLayer Layer</h3>

              <p className="line">
                node_count: {memoryView?.taskLayerNodeCount ?? "-"}
              </p>

              <p className="line">
                edge_count: {memoryView?.taskLayerEdgeCount ?? "-"}
              </p>

              <p className="line">
                static_history_count: {memoryView?.staticHistoryCount ?? "-"}
              </p>

              <p className="line">
                dynamic_law_case_count: {memoryView?.dynamicLawCaseCount ?? "-"}
              </p>

              <div className="scrollbox">
                {memoryRepresentativeChanges.length ? (
                  memoryRepresentativeChanges.map((item, idx) => (
                    <div className="audit-row" key={`rep-change-${idx}`}>
                      <p className="audit-head">{item.content}</p>

                      <p className="audit-reason">
                        before: {item.before.join(", ") || "-"}
                      </p>

                      <p className="audit-reason">
                        after: {item.after.join(", ") || "-"}
                      </p>
                    </div>
                  ))
                ) : (
                  <p className="hint">No representative-case changes.</p>
                )}
              </div>
            </section>

            <section className="memory-pane">
              <h3>Case Snapshots Layer</h3>
              <p className="line">click one round to jump replay context</p>

              <div className="scrollbox">
                {(memoryView?.caseSnapshots.length
                  ? memoryView.caseSnapshots
                  : snapshotIndex
                ).length ? (
                  (memoryView?.caseSnapshots.length
                    ? memoryView.caseSnapshots.map((item) => ({
                        round: item.round,
                        turn: item.turn,
                        nodeCount: item.nodeCount,
                        edgeCount: item.edgeCount,
                      }))
                    : snapshotIndex
                  ).map((item) => (
                    <button
                      className="timeline-row"
                      key={`memory-round-${item.round}`}
                      onClick={() => {
                        void loadReplayRound(item.round);
                      }}
                      type="button"
                    >
                      <span>r{item.round}</span>
                      <span>{item.turn || "unknown"}</span>

                      <span>
                        N{item.nodeCount}/E{item.edgeCount}
                      </span>
                    </button>
                  ))
                ) : (
                  <p className="hint">No snapshot index yet.</p>
                )}
              </div>
            </section>
          </div>
        </article>

        <article className="card wide">
          <h2>Demo Mode & Failure Simulation</h2>

          <div className="sub-actions">
            <button
              disabled={Boolean(busyAction) || !sessionId || demoRunning}
              onClick={() => void runDemo()}
              type="button"
            >
              {demoRunning ? "Running Demo..." : "Run Demo"}
            </button>

            <button
              disabled={Boolean(busyAction) || !sessionId}
              onClick={() => void loadDemoKeyframes()}
              type="button"
            >
              Refresh Keyframes
            </button>

            <label className="graph-control">
              Failure Mode
              <select
                value={failureMode}
                onChange={(event) =>
                  void setFailureSimulationMode(
                    event.target.value as
                      | "off"
                      | "es_unavailable"
                      | "llm_timeout",
                  )
                }
              >
                <option value="off">off</option>
                <option value="es_unavailable">es_unavailable</option>
                <option value="llm_timeout">llm_timeout</option>
              </select>
            </label>
          </div>

          <p className="line">
            demo status: {demoResult?.status ?? "-"} | endedBy:{" "}
            {demoResult?.endedBy ?? "-"} | steps:{" "}
            {demoResult?.stepsExecuted ?? "-"}
          </p>

          <p className="line">
            failure mode: {failureMode} | warnings: {warningEvents.length}
          </p>

          {latestWarningText ? (
            <p className="error">latest warning: {latestWarningText}</p>
          ) : null}

          {replayExport ? (
            <p className="line">
              latest export: events={replayExport.eventCount}, artifacts=
              {replayExport.artifactCount}, snapshots=
              {replayExport.snapshotCount}
            </p>
          ) : null}

          <div className="scrollbox">
            {demoKeyframes.length ? (
              [...demoKeyframes]
                .sort((a, b) => a.round - b.round || a.ts - b.ts)
                .map((frame, idx) => (
                  <button
                    className="timeline-row"
                    key={`${frame.reason}-${idx}-${frame.round}`}
                    onClick={() => {
                      void jumpToKeyframe(frame);
                    }}
                    type="button"
                  >
                    <span>r{frame.round}</span>
                    <span>{frame.reason || frame.event}</span>
                    <span>{frame.turnUid || "-"}</span>
                  </button>
                ))
            ) : (
              <p className="hint">No demo keyframes yet.</p>
            )}
          </div>
        </article>

        <TimelinePanel
          timeline={timeline}
          selectedSeq={selectedTimelineSeq}
          onSelectEvent={(event) => {
            void handleTimelineSelect(event);
          }}
        />

        <GraphDiffPanel
          artifacts={turnArtifacts}
          baselineGraph={baselineGraphView}
          currentGraph={graphView}
          diff={graphDiff}
        />

        <TeamFlowPanel
          artifacts={turnArtifacts}
          selectedTurnUid={selectedTurnUid}
          onSelectTurn={(turnUid) => setSelectedTurnUid(turnUid)}
        />

        <InspectorPanel artifact={selectedArtifact} snapshot={snapshot} />

        <DebugBundlePanel
          bundle={debugBundle}
          loading={debugBundleLoading}
          onLoad={loadDebugBundle}
        />

        <article className="card wide">
          <h2>Graph Action Audit</h2>

          <p className="hint">
            action-level audit with heuristic axiom mapping from executor logs
          </p>

          <div className="scrollbox">
            {auditRows.length ? (
              auditRows.map((row) => (
                <div className="audit-row" key={row.id}>
                  <p className="audit-head">
                    r{row.round} [{row.side}] {row.actionType}
                  </p>

                  <p className="audit-meta">
                    <span className={`tag tag-${row.status}`}>
                      {row.status}
                    </span>
                    <span className="tag tag-axiom">{row.axiom}</span>
                  </p>

                  {row.reason ? (
                    <p className="audit-reason">{row.reason}</p>
                  ) : null}
                </div>
              ))
            ) : (
              <p className="hint">No action audit data loaded.</p>
            )}
          </div>
        </article>

        <article className="card wide">
          <h2>Judgment & BAF</h2>

          {judgmentDocument ? (
            <>
              <p className="line">
                preferred extensions:{" "}
                {asNumber(
                  bafDetails.preferred_extensions_count ??
                    bafDetails.preferredExtensionsCount,
                ) || 0}
              </p>

              <p className="line">
                chosen extension size:{" "}
                {asNumber(
                  bafDetails.chosen_extension_size ??
                    bafDetails.chosenExtensionSize,
                ) || 0}
              </p>

              <p className="line">
                alignment rate:{" "}
                {(
                  asNumber(
                    bafDetails.alignment_rate ?? bafDetails.alignmentRate,
                    0,
                  ) * 100
                ).toFixed(1)}
                %
              </p>

              <div className="scrollbox">
                <p className="log">{judgmentDocument}</p>

                {rootClaimEntries.length ? (
                  rootClaimEntries.map(([claimId, status]) => (
                    <p className="log" key={claimId}>
                      {claimId}: {String(status)}
                    </p>
                  ))
                ) : (
                  <p className="hint">No root-claim status available.</p>
                )}
              </div>
            </>
          ) : (
            <p className="hint">
              No judgment yet. Run adjudication to render judgment and BAF
              panel.
            </p>
          )}
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
