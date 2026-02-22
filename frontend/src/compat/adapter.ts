import { CompatClient } from "./client";
import { GraphDomainAdapter } from "./domains/graphAdapter";
import { InsightDomainAdapter } from "./domains/insightAdapter";
import { SessionDomainAdapter } from "./domains/sessionAdapter";
import type { AdapterOptions } from "./client";

import type {
  AdapterCapabilities,
  CreateSessionInput,
  DebateSnapshot,
  EngineAdapter,
  FrontendSnapshotListItem,
  FrontendSnapshotLoadResult,
  GraphAdapter,
  InsightAdapter,
  SessionAdapter,
  SnapshotIndexItem,
} from "./types";

export class CompatAdapterFacade implements EngineAdapter {
  readonly session: SessionAdapter;
  readonly graph: GraphAdapter;
  readonly insight: InsightAdapter;
  private readonly client: CompatClient;

  constructor(options: AdapterOptions = {}) {
    this.client = new CompatClient(options);
    this.session = new SessionDomainAdapter(this.client);
    this.graph = new GraphDomainAdapter(this.client);
    this.insight = new InsightDomainAdapter(this.client);
  }

  get capabilities(): AdapterCapabilities {
    const hasWebSocket =
      typeof window !== "undefined" && typeof WebSocket !== "undefined";

    return {
      supportsStreaming: this.client.transportKind === "http" && hasWebSocket,
      supportsDiff: true,
      transport: this.client.transportKind,
    };
  }

  createSession(input?: CreateSessionInput): Promise<DebateSnapshot> {
    return this.session.createSession(input);
  }

  step(sessionId: string): Promise<DebateSnapshot> {
    return this.session.step(sessionId);
  }

  adjudicate(sessionId: string): Promise<DebateSnapshot> {
    return this.session.adjudicate(sessionId);
  }

  resetMemory(): Promise<void> {
    return this.session.resetMemory();
  }

  getSnapshot(sessionId: string): Promise<DebateSnapshot> {
    return this.session.getSnapshot(sessionId);
  }

  listSessions(): Promise<DebateSnapshot[]> {
    return this.session.listSessions();
  }

  getSnapshots(sessionId: string): Promise<SnapshotIndexItem[]> {
    return this.session.getSnapshots(sessionId);
  }

  saveFrontendSnapshot(input: {
    sessionId: string;
    label?: string;
    frontendState?: Record<string, unknown>;
  }): Promise<FrontendSnapshotListItem> {
    return this.session.saveFrontendSnapshot(input);
  }

  importFrontendSnapshot(input: {
    bundle: Record<string, unknown>;
    label?: string;
    frontendState?: Record<string, unknown>;
  }): Promise<FrontendSnapshotListItem> {
    return this.session.importFrontendSnapshot(input);
  }

  listFrontendSnapshots(
    limit?: number,
    offset?: number,
  ): Promise<FrontendSnapshotListItem[]> {
    return this.session.listFrontendSnapshots(limit, offset);
  }

  loadFrontendSnapshot(
    snapshotId: string,
  ): Promise<FrontendSnapshotLoadResult> {
    return this.session.loadFrontendSnapshot(snapshotId);
  }
}

export function createCompatAdapter(
  options: AdapterOptions = {},
): EngineAdapter {
  return new CompatAdapterFacade(options);
}
