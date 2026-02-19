import { normalizeGraph, normalizeGraphDiff } from "../protocol";

import { withQuery } from "../client";
import type { CompatClient } from "../client";

import type { GraphAdapter, GraphDiffView, GraphView } from "../types";

export class GraphDomainAdapter implements GraphAdapter {
  private readonly client: CompatClient;

  constructor(client: CompatClient) {
    this.client = client;
  }

  async getGraph(sessionId: string): Promise<GraphView> {
    const raw = await this.client.request({
      method: "GET",
      path: `/api/v1/sessions/${sessionId}/graph`,
    });

    return normalizeGraph(raw, sessionId);
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

    const raw = await this.client.request({
      method: "GET",
      path: v1Path,
    });

    return normalizeGraphDiff(raw, sessionId, fromRound, toRound);
  }

  async getGraphAtRound(sessionId: string, round: number): Promise<GraphView> {
    const v1Path = `/api/v1/sessions/${sessionId}/snapshots/${round}`;

    const raw = await this.client.request({
      method: "GET",
      path: v1Path,
    });

    return normalizeGraph(raw, sessionId);
  }
}
