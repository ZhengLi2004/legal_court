import {
  normalizeMemory,
  normalizeTimeline,
  normalizeTurnArtifacts,
} from "../protocol";

import { withQuery } from "../client";
import type { CompatClient } from "../client";

import type {
  InsightAdapter,
  MemoryView,
  SessionAdapter,
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
}
