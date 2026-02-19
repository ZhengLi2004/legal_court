import {
  normalizeGraph,
  normalizeMemory,
  normalizeTeamflowStream,
  normalizeTimeline,
  normalizeTurnArtifacts,
} from "../protocol";

import { withQuery } from "../client";
import type { CompatClient } from "../client";

import type {
  InsightAdapter,
  MemoryView,
  GraphView,
  TeamFlowTurn,
  TimelineEvent,
  TurnArtifact,
} from "../types";

export class InsightDomainAdapter implements InsightAdapter {
  private readonly client: CompatClient;

  constructor(client: CompatClient) {
    this.client = client;
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
    const raw = await this.client.request({
      method: "GET",
      path: `/api/v1/sessions/${sessionId}/memory`,
    });

    return normalizeMemory(raw, sessionId);
  }

  async getMemoryCaseGraph(
    sessionId: string,
    caseId: string,
  ): Promise<GraphView> {
    const encodedCaseId = encodeURIComponent(caseId);

    const raw = await this.client.request({
      method: "GET",
      path: `/api/v1/sessions/${sessionId}/memory/cases/${encodedCaseId}/graph`,
    });

    return normalizeGraph(raw, sessionId);
  }

  async getTimeline(sessionId: string, limit = 100): Promise<TimelineEvent[]> {
    const path = withQuery(`/api/v1/sessions/${sessionId}/events`, {
      limit,
    });

    const raw = await this.client.request({ method: "GET", path });

    return normalizeTimeline(raw);
  }

  async getTeamflowStream(
    sessionId: string,
    limit = 80,
  ): Promise<TeamFlowTurn[]> {
    const path = withQuery(`/api/v1/sessions/${sessionId}/teamflow/stream`, {
      limit,
    });

    const raw = await this.client.request({ method: "GET", path });
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

      const raw = await this.client.request({ method: "GET", path: byTurnPath });

      return normalizeTurnArtifacts(raw);
    }

    const listPath = withQuery(
      `/api/v1/sessions/${sessionId}/turns/artifacts`,
      {
        limit,
      },
    );

    const raw = await this.client.request({ method: "GET", path: listPath });

    return normalizeTurnArtifacts(raw);
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
