import {
  normalizeFrontendSnapshotItem,
  normalizeFrontendSnapshotList,
  normalizeFrontendSnapshotLoadResult,
  normalizeSnapshot,
  normalizeSnapshotIndex,
  normalizeSnapshotList,
} from "../protocol";

import type { CompatClient } from "../client";

import {
  LegalCourtApiClient,
  type CreateSessionRequest,
} from "../../shared/api/generated";

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
  private readonly generatedClient: LegalCourtApiClient;

  constructor(client: CompatClient) {
    this.client = client;
    this.generatedClient = new LegalCourtApiClient({ baseUrl: client.baseUrl });
  }

  async createSession(input: CreateSessionInput = {}): Promise<DebateSnapshot> {
    const body: CreateSessionRequest = {
      case_data: input.caseData,
    };

    const raw = await this.generatedClient.createSession(body);
    return normalizeSnapshot(raw);
  }

  async step(sessionId: string): Promise<DebateSnapshot> {
    const raw = await this.generatedClient.stepSession(sessionId);
    return normalizeSnapshot(raw);
  }

  async adjudicate(sessionId: string): Promise<DebateSnapshot> {
    const raw = await this.generatedClient.adjudicateSession(sessionId);
    return normalizeSnapshot(raw);
  }

  async resetMemory(): Promise<void> {
    await this.generatedClient.resetMemoryStorage();
  }

  async getSnapshot(sessionId: string): Promise<DebateSnapshot> {
    const raw = await this.generatedClient.getSessionSnapshot(sessionId);
    return normalizeSnapshot(raw);
  }

  async listSessions(): Promise<DebateSnapshot[]> {
    const raw = await this.generatedClient.listSessions();
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
    const raw = await this.generatedClient.saveFrontendSnapshot({
      session_id: input.sessionId,
      label: input.label ?? "",
      frontend_state: input.frontendState ?? {},
    });

    return normalizeFrontendSnapshotItem(raw);
  }

  async importFrontendSnapshot(input: {
    bundle: Record<string, unknown>;
    label?: string;
    frontendState?: Record<string, unknown>;
  }): Promise<FrontendSnapshotListItem> {
    const raw = await this.generatedClient.importFrontendSnapshot({
      bundle: input.bundle,
      label: input.label ?? "",
      frontend_state: input.frontendState ?? {},
    });

    return normalizeFrontendSnapshotItem(raw);
  }

  async listFrontendSnapshots(
    limit = 20,
    offset = 0,
  ): Promise<FrontendSnapshotListItem[]> {
    const raw = await this.generatedClient.listFrontendSnapshots(limit, offset);
    return normalizeFrontendSnapshotList(raw);
  }

  async loadFrontendSnapshot(
    snapshotId: string,
  ): Promise<FrontendSnapshotLoadResult> {
    const raw = await this.generatedClient.loadFrontendSnapshot(snapshotId);
    return normalizeFrontendSnapshotLoadResult(raw);
  }
}
