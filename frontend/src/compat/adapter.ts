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
  GraphAdapter,
  InsightAdapter,
  SessionAdapter,
} from "./types";

export class CompatAdapterFacade implements EngineAdapter {
  readonly session: SessionAdapter;
  readonly graph: GraphAdapter;
  readonly insight: InsightAdapter;
  private readonly client: CompatClient;

  constructor(options: AdapterOptions = {}) {
    this.client = new CompatClient(options);
    this.session = new SessionDomainAdapter(this.client);
    this.graph = new GraphDomainAdapter(this.client, this.session);
    this.insight = new InsightDomainAdapter(this.client, this.session);
  }

  get capabilities(): AdapterCapabilities {
    return {
      supportsStreaming: false,
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

  getSnapshot(sessionId: string): Promise<DebateSnapshot> {
    return this.session.getSnapshot(sessionId);
  }

  listSessions(): Promise<DebateSnapshot[]> {
    return this.session.listSessions();
  }
}

export function createCompatAdapter(
  options: AdapterOptions = {},
): EngineAdapter {
  return new CompatAdapterFacade(options);
}
