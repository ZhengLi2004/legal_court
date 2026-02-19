import {
  normalizeFrontendSnapshotItem,
  normalizeFrontendSnapshotList,
  normalizeFrontendSnapshotLoadResult,
  normalizeSnapshot,
  normalizeSnapshotIndex,
  normalizeSnapshotList,
} from "../protocol";

import { withQuery } from "../client";
import type { CompatClient } from "../client";

import type {
  CreateSessionInput,
  DebateSnapshot,
  FrontendSnapshotListItem,
  FrontendSnapshotLoadResult,
  SnapshotIndexItem,
  SessionAdapter,
} from "../types";

export class SessionDomainAdapter implements SessionAdapter {
  private readonly client: CompatClient;

  constructor(client: CompatClient) {
    this.client = client;
  }

  async createSession(input: CreateSessionInput = {}): Promise<DebateSnapshot> {
    const body = {
      case_id: input.caseId,
      case_uid: input.caseUid,
      case_data: input.caseData,
    };

    const raw = await this.client.request({
      method: "POST",
      path: "/api/v1/sessions",
      body,
    });

    return normalizeSnapshot(raw);
  }

  async step(sessionId: string): Promise<DebateSnapshot> {
    const raw = await this.client.request({
      method: "POST",
      path: `/api/v1/sessions/${sessionId}/step`,
    });

    return normalizeSnapshot(raw);
  }

  async adjudicate(sessionId: string): Promise<DebateSnapshot> {
    const raw = await this.client.request({
      method: "POST",
      path: `/api/v1/sessions/${sessionId}/adjudicate`,
    });

    return normalizeSnapshot(raw);
  }

  async getSnapshot(sessionId: string): Promise<DebateSnapshot> {
    const raw = await this.client.request({
      method: "GET",
      path: `/api/v1/sessions/${sessionId}/snapshot`,
    });

    return normalizeSnapshot(raw);
  }

  async listSessions(): Promise<DebateSnapshot[]> {
    const raw = await this.client.request({
      method: "GET",
      path: "/api/v1/sessions",
    });

    return normalizeSnapshotList(raw);
  }

  async getSnapshots(sessionId: string): Promise<SnapshotIndexItem[]> {
    const raw = await this.client.request({
      method: "GET",
      path: `/api/v1/sessions/${sessionId}/snapshots`,
    });

    return normalizeSnapshotIndex(raw);
  }

  async saveFrontendSnapshot(input: {
    sessionId: string;
    label?: string;
    frontendState?: Record<string, unknown>;
  }): Promise<FrontendSnapshotListItem> {
    const raw = await this.client.request({
      method: "POST",
      path: "/api/v1/frontend-snapshots",
      body: {
        session_id: input.sessionId,
        label: input.label ?? "",
        frontend_state: input.frontendState ?? {},
      },
    });

    return normalizeFrontendSnapshotItem(raw);
  }

  async importFrontendSnapshot(input: {
    bundle: Record<string, unknown>;
    label?: string;
    frontendState?: Record<string, unknown>;
  }): Promise<FrontendSnapshotListItem> {
    const raw = await this.client.request({
      method: "POST",
      path: "/api/v1/frontend-snapshots/import",
      body: {
        bundle: input.bundle,
        label: input.label ?? "",
        frontend_state: input.frontendState ?? {},
      },
    });

    return normalizeFrontendSnapshotItem(raw);
  }

  async listFrontendSnapshots(
    limit = 20,
    offset = 0,
  ): Promise<FrontendSnapshotListItem[]> {
    const path = withQuery("/api/v1/frontend-snapshots", { limit, offset });
    const raw = await this.client.request({ method: "GET", path });
    return normalizeFrontendSnapshotList(raw);
  }

  async loadFrontendSnapshot(
    snapshotId: string,
  ): Promise<FrontendSnapshotLoadResult> {
    const raw = await this.client.request({
      method: "POST",
      path: `/api/v1/frontend-snapshots/${snapshotId}/load`,
    });

    return normalizeFrontendSnapshotLoadResult(raw);
  }
}
