import {
  normalizeDebugBundle,
  normalizeDemoKeyframes,
  normalizeDemoRunResult,
  normalizeMemory,
  normalizeReplayExport,
  normalizeTeamflowStream,
  normalizeTimeline,
  normalizeTurnArtifacts,
} from "../protocol";

import { withQuery } from "../client";
import type { CompatClient } from "../client";

import type {
  DemoKeyframe,
  DemoRunResult,
  DebugBundleView,
  InsightAdapter,
  MemoryView,
  ReplayExportView,
  SessionAdapter,
  TeamFlowTurn,
  TimelineEvent,
  TurnArtifact,
} from "../types";

export class InsightDomainAdapter implements InsightAdapter {
  private readonly client: CompatClient;
  private readonly sessionAdapter: SessionAdapter;

  constructor(client: CompatClient, sessionAdapter: SessionAdapter) {
    this.client = client;
    this.sessionAdapter = sessionAdapter;
  }

  private static buildWebSocketUrl(
    baseUrl: string,
    path: string,
    fromSeq?: number,
  ): string | null {
    if (typeof window === "undefined" || typeof WebSocket === "undefined") {
      return null;
    }

    let absoluteBase = baseUrl.trim();

    if (!/^https?:\/\//i.test(absoluteBase)) {
      if (absoluteBase.startsWith("/")) {
        absoluteBase = `${window.location.origin}${absoluteBase}`;
      } else {
        absoluteBase = `${window.location.origin}/${absoluteBase}`;
      }
    }

    const wsBase = absoluteBase.replace(/^http/i, "ws").replace(/\/$/, "");
    const query = new URLSearchParams();

    if (
      typeof fromSeq === "number" &&
      Number.isFinite(fromSeq) &&
      fromSeq > 0
    ) {
      query.set("from_seq", String(Math.floor(fromSeq)));
    }

    const suffix = query.toString();
    return `${wsBase}${path}${suffix ? `?${suffix}` : ""}`;
  }

  async getMemory(sessionId: string): Promise<MemoryView> {
    try {
      const raw = await this.client.callWithCandidates([
        { method: "GET", path: `/api/v1/sessions/${sessionId}/memory` },
        { method: "GET", path: `/sessions/${sessionId}/memory` },
      ]);

      return normalizeMemory(raw, sessionId);
    } catch {
      const snapshot = await this.sessionAdapter.getSnapshot(sessionId);
      return normalizeMemory(snapshot.raw ?? {}, sessionId);
    }
  }

  async getTimeline(sessionId: string, limit = 100): Promise<TimelineEvent[]> {
    const v1Path = withQuery(`/api/v1/sessions/${sessionId}/events/history`, {
      limit,
    });

    const basePath = withQuery(`/sessions/${sessionId}/events`, {
      limit,
    });

    const raw = await this.client.callWithCandidates([
      { method: "GET", path: v1Path },
      { method: "GET", path: basePath },
    ]);

    return normalizeTimeline(raw);
  }

  async getTeamflowStream(
    sessionId: string,
    limit = 80,
  ): Promise<TeamFlowTurn[]> {
    const path = withQuery(`/api/v1/sessions/${sessionId}/teamflow/stream`, {
      limit,
    });

    const raw = await this.client.callWithCandidates([{ method: "GET", path }]);
    return normalizeTeamflowStream(raw);
  }

  subscribeTimeline(
    sessionId: string,
    onEvent: (event: TimelineEvent) => void,
    options: { fromSeq?: number; onError?: (error: Error) => void } = {},
  ): () => void {
    if (this.client.transportKind !== "http") {
      return () => {};
    }

    const wsUrl = InsightDomainAdapter.buildWebSocketUrl(
      this.client.baseUrl,
      `/api/v1/sessions/${sessionId}/events`,
      options.fromSeq,
    );

    if (!wsUrl) {
      return () => {};
    }

    let socket: WebSocket;

    try {
      socket = new WebSocket(wsUrl);
    } catch (err) {
      if (options.onError) {
        const message = err instanceof Error ? err.message : String(err);
        options.onError(new Error(`WebSocket init failed: ${message}`));
      }

      return () => {};
    }

    let isClosed = false;
    socket.onmessage = (event) => {
      try {
        const raw = JSON.parse(String(event.data));
        const rows = normalizeTimeline({ events: [raw] });

        if (rows.length > 0) {
          onEvent(rows[0]);
        }
      } catch (err) {
        if (options.onError) {
          const message = err instanceof Error ? err.message : String(err);
          options.onError(new Error(`WebSocket parse failed: ${message}`));
        }
      }
    };

    socket.onerror = () => {
      if (options.onError) {
        options.onError(new Error("WebSocket stream error"));
      }
    };

    socket.onclose = () => {
      if (!isClosed && options.onError) {
        options.onError(new Error("WebSocket stream closed"));
      }
    };

    return () => {
      isClosed = true;
      socket.close();
    };
  }

  async getTurnArtifacts(
    sessionId: string,
    options: { turnUid?: string; limit?: number } = {},
  ): Promise<TurnArtifact[]> {
    const limit = options.limit ?? 50;

    if (options.turnUid) {
      const byTurnPath = withQuery(
        `/api/v1/sessions/${sessionId}/turns/${options.turnUid}/artifacts`,
        { limit },
      );

      const raw = await this.client.callWithCandidates([
        { method: "GET", path: byTurnPath },
      ]);

      return normalizeTurnArtifacts(raw);
    }

    const listPath = withQuery(
      `/api/v1/sessions/${sessionId}/turns/artifacts`,
      {
        limit,
      },
    );

    const raw = await this.client.callWithCandidates([
      { method: "GET", path: listPath },
    ]);

    return normalizeTurnArtifacts(raw);
  }

  async getDebugBundle(
    sessionId: string,
    options: {
      eventLimit?: number;
      includeSnapshot?: boolean;
      includeArtifact?: boolean;
    } = {},
  ): Promise<DebugBundleView> {
    const eventLimit = options.eventLimit ?? 20;
    const includeSnapshot = options.includeSnapshot ?? true;
    const includeArtifact = options.includeArtifact ?? true;

    const path = withQuery(`/api/v1/sessions/${sessionId}/debug-bundle`, {
      event_limit: eventLimit,
      include_snapshot: includeSnapshot ? 1 : 0,
      include_artifact: includeArtifact ? 1 : 0,
    });

    try {
      const raw = await this.client.callWithCandidates([
        { method: "GET", path },
      ]);

      return normalizeDebugBundle(raw, sessionId);
    } catch {
      const [snapshot, timeline, artifacts] = await Promise.all([
        this.sessionAdapter.getSnapshot(sessionId),
        this.getTimeline(sessionId, eventLimit),
        includeArtifact
          ? this.getTurnArtifacts(sessionId, { limit: 1 })
          : Promise.resolve([]),
      ]);

      return normalizeDebugBundle(
        {
          session_id: sessionId,
          round_idx: snapshot.round,
          turn_uid: timeline[timeline.length - 1]?.turnUid ?? "",
          status: snapshot.phase,
          last_error: "",
          snapshot_summary: {
            phase: snapshot.phase,
            node_count: snapshot.metrics.arguments,
            edge_count: snapshot.metrics.attacks + snapshot.metrics.supports,
            claim_count: 0,
            conflict_count: snapshot.metrics.attacks,
          },
          recent_events: timeline,
          latest_turn_artifact: artifacts[0]?.raw ?? artifacts[0],
          generated_at: new Date().toISOString(),
        },
        sessionId,
      );
    }
  }

  async runDemo(
    sessionId: string,
    options: {
      maxSteps?: number;
      autoAdjudicate?: boolean;
      captureKeyframes?: boolean;
    } = {},
  ): Promise<DemoRunResult> {
    const raw = await this.client.callWithCandidates([
      {
        method: "POST",
        path: `/api/v1/sessions/${sessionId}/demo/run`,
        body: {
          max_steps: options.maxSteps ?? 20,
          auto_adjudicate: options.autoAdjudicate ?? true,
          capture_keyframes: options.captureKeyframes ?? true,
        },
      },
    ]);

    return normalizeDemoRunResult(raw, sessionId);
  }

  async getDemoKeyframes(sessionId: string): Promise<DemoKeyframe[]> {
    const raw = await this.client.callWithCandidates([
      {
        method: "GET",
        path: `/api/v1/sessions/${sessionId}/demo/keyframes`,
      },
    ]);

    return normalizeDemoKeyframes(raw);
  }

  async setFailureSimulation(
    sessionId: string,
    kind: "es_unavailable" | "llm_timeout",
    enabled: boolean,
  ): Promise<{
    sessionId: string;
    failureSimulation: Record<string, boolean>;
  }> {
    const raw = await this.client.callWithCandidates([
      {
        method: "POST",
        path: `/api/v1/sessions/${sessionId}/simulate/failure`,
        body: { kind, enabled },
      },
    ]);

    const row =
      raw !== null && typeof raw === "object"
        ? (raw as Record<string, unknown>)
        : {};

    const failureRaw = row.failure_simulation;

    const failureSimulation =
      failureRaw !== null && typeof failureRaw === "object"
        ? (failureRaw as Record<string, boolean>)
        : {};

    return {
      sessionId:
        typeof row.session_id === "string"
          ? row.session_id
          : typeof row.sessionId === "string"
            ? row.sessionId
            : sessionId,
      failureSimulation,
    };
  }

  async exportReplayJson(sessionId: string): Promise<ReplayExportView> {
    const raw = await this.client.callWithCandidates([
      {
        method: "GET",
        path: `/api/v1/sessions/${sessionId}/export/replay.json`,
      },
    ]);

    return normalizeReplayExport(raw, sessionId);
  }

  async exportGraphGexf(sessionId: string, round?: number): Promise<Blob> {
    const query =
      typeof round === "number" && Number.isFinite(round)
        ? `?round_idx=${Math.max(0, Math.floor(round))}`
        : "";

    const url = `${this.client.baseUrl}/api/v1/sessions/${sessionId}/export/graph.gexf${query}`;
    const response = await fetch(url, { method: "GET" });

    if (!response.ok) {
      throw new Error(`exportGraphGexf failed: HTTP ${response.status}`);
    }

    return await response.blob();
  }
}
