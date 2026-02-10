import { normalizeSnapshot, normalizeSnapshotList } from "../protocol";
import type { CompatClient } from "../client";

import type {
  CreateSessionInput,
  DebateSnapshot,
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
      plaintiff_claim: input.plaintiffClaim,
      defendant_answer: input.defendantAnswer,
      max_rounds: input.maxRounds,
    };

    const raw = await this.client.callWithCandidates([
      { method: "POST", path: "/sessions", body },
      { method: "POST", path: "/engine/init", body },
      { method: "POST", path: "/engine/create", body },
      { method: "POST", path: "/api/v1/sessions", body },
    ]);

    return normalizeSnapshot(raw);
  }

  async step(sessionId: string): Promise<DebateSnapshot> {
    const raw = await this.client.callWithCandidates([
      { method: "POST", path: `/sessions/${sessionId}/step` },
      { method: "POST", path: "/engine/step", body: { session_id: sessionId } },
      { method: "POST", path: `/api/v1/sessions/${sessionId}/step` },
    ]);

    return normalizeSnapshot(raw);
  }

  async adjudicate(sessionId: string): Promise<DebateSnapshot> {
    const raw = await this.client.callWithCandidates([
      { method: "POST", path: `/sessions/${sessionId}/adjudicate` },
      {
        method: "POST",
        path: "/engine/adjudicate",
        body: { session_id: sessionId },
      },
      { method: "POST", path: `/api/v1/sessions/${sessionId}/adjudicate` },
    ]);

    return normalizeSnapshot(raw);
  }

  async getSnapshot(sessionId: string): Promise<DebateSnapshot> {
    const raw = await this.client.callWithCandidates([
      { method: "GET", path: `/sessions/${sessionId}/snapshot` },
      { method: "GET", path: `/sessions/${sessionId}` },
      {
        method: "POST",
        path: "/engine/snapshot",
        body: { session_id: sessionId },
      },
      { method: "GET", path: `/api/v1/sessions/${sessionId}/snapshot` },
      { method: "GET", path: `/api/v1/sessions/${sessionId}` },
    ]);

    return normalizeSnapshot(raw);
  }

  async listSessions(): Promise<DebateSnapshot[]> {
    const raw = await this.client.callWithCandidates([
      { method: "GET", path: "/sessions" },
      { method: "GET", path: "/engine/sessions" },
      { method: "GET", path: "/api/v1/sessions" },
    ]);

    return normalizeSnapshotList(raw);
  }
}
