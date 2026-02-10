import { normalizeMemory, normalizeTimeline } from "../protocol";
import { withQuery } from "../client";
import type { CompatClient } from "../client";

import type {
  InsightAdapter,
  MemoryView,
  SessionAdapter,
  TimelineEvent,
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
        { method: "GET", path: `/sessions/${sessionId}/memory` },
        { method: "GET", path: `/api/v1/sessions/${sessionId}/memory` },
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
      { method: "GET", path: basePath },
      { method: "GET", path: v1Path },
    ]);

    return normalizeTimeline(raw);
  }
}
