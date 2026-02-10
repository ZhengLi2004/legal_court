import {
  buildLocalGraphDiff,
  normalizeGraph,
  normalizeGraphDiff,
} from "../protocol";

import { withQuery } from "../client";
import type { CompatClient } from "../client";

import type {
  GraphAdapter,
  GraphDiffView,
  GraphView,
  SessionAdapter,
} from "../types";

export class GraphDomainAdapter implements GraphAdapter {
  private readonly client: CompatClient;
  private readonly sessionAdapter: SessionAdapter;

  constructor(client: CompatClient, sessionAdapter: SessionAdapter) {
    this.client = client;
    this.sessionAdapter = sessionAdapter;
  }

  async getGraph(sessionId: string): Promise<GraphView> {
    try {
      const raw = await this.client.callWithCandidates([
        { method: "GET", path: `/sessions/${sessionId}/graph` },
        { method: "GET", path: `/api/v1/sessions/${sessionId}/graph` },
        { method: "GET", path: `/api/v1/sessions/${sessionId}/snapshot` },
      ]);

      return normalizeGraph(raw, sessionId);
    } catch {
      const snapshot = await this.sessionAdapter.getSnapshot(sessionId);
      return normalizeGraph(snapshot.raw ?? snapshot, sessionId);
    }
  }

  async getGraphDiff(
    sessionId: string,
    fromRound: number,
    toRound: number,
  ): Promise<GraphDiffView> {
    const v1Path = withQuery(`/api/v1/sessions/${sessionId}/diff`, {
      from_round: fromRound,
      to_round: toRound,
    });

    const basePath = withQuery(`/sessions/${sessionId}/diff`, {
      from_round: fromRound,
      to_round: toRound,
    });

    try {
      const raw = await this.client.callWithCandidates([
        { method: "GET", path: basePath },
        { method: "GET", path: v1Path },
      ]);

      return normalizeGraphDiff(raw, sessionId, fromRound, toRound);
    } catch {
      const [fromGraph, toGraph] = await Promise.all([
        this.getGraphAtRound(sessionId, fromRound),
        this.getGraphAtRound(sessionId, toRound),
      ]);

      return buildLocalGraphDiff(fromGraph, toGraph, sessionId);
    }
  }

  private async getGraphAtRound(
    sessionId: string,
    round: number,
  ): Promise<GraphView> {
    const basePath = `/sessions/${sessionId}/snapshots/${round}`;
    const v1Path = `/api/v1/sessions/${sessionId}/snapshots/${round}`;

    try {
      const raw = await this.client.callWithCandidates([
        { method: "GET", path: basePath },
        { method: "GET", path: v1Path },
      ]);

      return normalizeGraph(raw, sessionId);
    } catch {
      return this.getGraph(sessionId);
    }
  }
}
