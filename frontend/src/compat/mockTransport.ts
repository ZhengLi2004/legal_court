import type { CompatTransport, TransportRequest } from "./transport";

interface MockGraphNode {
  id: string;
  type: string;
  label: string;
  status: string;
}

interface MockGraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
}

interface MockGraphSnapshot {
  round: number;
  nodes: MockGraphNode[];
  edges: MockGraphEdge[];
}

interface MockEvent {
  seq: number;
  ts_ms: number;
  event: string;
  source: string;
  data?: unknown;
}

interface MockSession {
  session_id: string;
  current_round: number;
  max_rounds: number;
  is_ready_for_adjudication: boolean;
  is_finished: boolean;
  winner: string | null;
  transcript: string[];

  metrics: {
    arguments: number;
    attacks: number;
    supports: number;
  };

  snapshots: MockGraphSnapshot[];
  events: MockEvent[];
  nextSeq: number;
}

function makeId(): string {
  return `mock-${Math.random().toString(36).slice(2, 10)}`;
}

function parseUrl(path: string): URL {
  return new URL(path, "http://mock.local");
}

function buildGraph(round: number): MockGraphSnapshot {
  const nodes: MockGraphNode[] = [
    {
      id: "CLAIM_ROOT",
      type: "CLAIM",
      label: "Root Claim",
      status: "HYPOTHETICAL",
    },
  ];

  const edges: MockGraphEdge[] = [];

  for (let idx = 1; idx <= round; idx += 1) {
    const factId = `FACT_${idx}`;
    const claimId = `CLAIM_${idx}`;

    nodes.push({
      id: factId,
      type: "FACT",
      label: `Fact ${idx}`,
      status: "ACCEPTED",
    });

    nodes.push({
      id: claimId,
      type: "CLAIM",
      label: `Claim ${idx}`,
      status: idx % 2 === 0 ? "SUPPORTED" : "CONTESTED",
    });

    edges.push({
      id: `E_SUP_${idx}`,
      source: factId,
      target: claimId,
      type: "SUPPORT",
    });

    edges.push({
      id: `E_ATK_${idx}`,
      source: claimId,
      target: "CLAIM_ROOT",
      type: "ATTACK",
    });
  }

  return {
    round,
    nodes,
    edges,
  };
}

function cloneSession(session: MockSession): MockSession {
  return {
    ...session,
    transcript: [...session.transcript],
    metrics: { ...session.metrics },
    snapshots: session.snapshots.map((item) => ({
      round: item.round,
      nodes: item.nodes.map((node) => ({ ...node })),
      edges: item.edges.map((edge) => ({ ...edge })),
    })),
    events: session.events.map((event) => ({ ...event })),
  };
}

function getGraphAtRound(
  session: MockSession,
  round: number,
): MockGraphSnapshot {
  const exact = session.snapshots.find((item) => item.round === round);

  if (exact) {
    return exact;
  }

  if (round <= 0) {
    return session.snapshots[0];
  }

  return session.snapshots[session.snapshots.length - 1];
}

function computeDiff(
  session: MockSession,
  fromRound: number,
  toRound: number,
): Record<string, unknown> {
  const from = getGraphAtRound(session, fromRound);
  const to = getGraphAtRound(session, toRound);
  const fromNodeIds = new Set(from.nodes.map((node) => node.id));
  const toNodeIds = new Set(to.nodes.map((node) => node.id));
  const fromEdgeIds = new Set(from.edges.map((edge) => edge.id));
  const toEdgeIds = new Set(to.edges.map((edge) => edge.id));

  return {
    session_id: session.session_id,
    from_round: from.round,
    to_round: to.round,
    added_node_ids: [...toNodeIds].filter((id) => !fromNodeIds.has(id)),
    removed_node_ids: [...fromNodeIds].filter((id) => !toNodeIds.has(id)),
    added_edge_ids: [...toEdgeIds].filter((id) => !fromEdgeIds.has(id)),
    removed_edge_ids: [...fromEdgeIds].filter((id) => !toEdgeIds.has(id)),
  };
}

function toSnapshotResponse(session: MockSession): Record<string, unknown> {
  const graph = getGraphAtRound(session, session.current_round);

  return {
    session_id: session.session_id,
    current_round: session.current_round,
    max_rounds: session.max_rounds,
    is_ready_for_adjudication: session.is_ready_for_adjudication,
    is_finished: session.is_finished,
    winner: session.winner,
    transcript: session.transcript,
    metrics: session.metrics,
    graph_data: {
      nodes: graph.nodes,
      edges: graph.edges,
    },
  };
}

export class MockTransport implements CompatTransport {
  readonly kind = "mock" as const;
  private readonly sessions = new Map<string, MockSession>();

  async request<T = unknown>(req: TransportRequest): Promise<T> {
    const { method, body } = req;
    const url = parseUrl(req.path);
    const pathname = url.pathname;

    if (
      method === "POST" &&
      (pathname === "/sessions" ||
        pathname === "/engine/init" ||
        pathname === "/api/v1/sessions")
    ) {
      const payload = body as Record<string, unknown> | undefined;
      const maxRoundsRaw = payload?.max_rounds ?? payload?.maxRounds;
      const parsed =
        typeof maxRoundsRaw === "number" && Number.isFinite(maxRoundsRaw)
          ? Math.max(1, Math.floor(maxRoundsRaw))
          : 6;

      const graph = buildGraph(0);

      const session: MockSession = {
        session_id: makeId(),
        current_round: 0,
        max_rounds: parsed,
        is_ready_for_adjudication: false,
        is_finished: false,
        winner: null,
        transcript: ["[system] mock debate session created"],
        metrics: {
          arguments: graph.nodes.length,
          attacks: 0,
          supports: 0,
        },
        snapshots: [graph],
        events: [],
        nextSeq: 1,
      };

      this.recordEvent(session, "setup_complete", "engine", {
        round: 0,
      });

      this.sessions.set(session.session_id, session);
      return toSnapshotResponse(cloneSession(session)) as T;
    }

    if (
      method === "GET" &&
      (pathname === "/sessions" || pathname === "/api/v1/sessions")
    ) {
      return {
        sessions: [...this.sessions.values()].map((session) =>
          toSnapshotResponse(cloneSession(session)),
        ),
      } as T;
    }

    if (method === "POST" && pathname === "/engine/step") {
      const payload = body as Record<string, unknown>;
      const sessionId = String(payload.session_id ?? payload.sessionId ?? "");
      const session = this.mustGetSession(sessionId);
      this.advance(session);
      return toSnapshotResponse(cloneSession(session)) as T;
    }

    if (method === "POST" && pathname === "/engine/adjudicate") {
      const payload = body as Record<string, unknown>;
      const sessionId = String(payload.session_id ?? payload.sessionId ?? "");
      const session = this.mustGetSession(sessionId);
      this.adjudicate(session);
      return toSnapshotResponse(cloneSession(session)) as T;
    }

    const match = pathname.match(
      /^\/(?:api\/v1\/)?sessions\/([^/]+)(?:\/(.+))?$/,
    );
    if (match) {
      const [, sessionId, action] = match;
      const session = this.mustGetSession(sessionId);
      const roundQuery = Number(url.searchParams.get("round"));

      if (method === "GET" && (!action || action === "snapshot")) {
        return toSnapshotResponse(cloneSession(session)) as T;
      }

      if (method === "POST" && action === "step") {
        this.advance(session);
        return toSnapshotResponse(cloneSession(session)) as T;
      }

      if (method === "POST" && action === "adjudicate") {
        this.adjudicate(session);
        return toSnapshotResponse(cloneSession(session)) as T;
      }

      if (method === "GET" && action === "graph") {
        const graph = getGraphAtRound(
          session,
          Number.isFinite(roundQuery) ? roundQuery : session.current_round,
        );

        return {
          session_id: session.session_id,
          round_idx: graph.round,
          graph_data: {
            nodes: graph.nodes,
            edges: graph.edges,
          },
        } as T;
      }

      if (method === "GET" && action.startsWith("snapshots/")) {
        const roundRaw = action.split("/")[1];
        const round = Number(roundRaw);

        const graph = getGraphAtRound(
          session,
          Number.isFinite(round) ? round : session.current_round,
        );

        return {
          session_id: session.session_id,
          round_idx: graph.round,
          graph_data: {
            nodes: graph.nodes,
            edges: graph.edges,
          },
        } as T;
      }

      if (method === "GET" && action === "memory") {
        return {
          session_id: session.session_id,
          insight_summaries: [
            "When opponent introduces ungrounded claims, prioritize source-backed attack chains.",
            "Escalate from fact retrieval to precedent retrieval after repeated rebuttal cycles.",
          ],
          static_history_count: 12,
          dynamic_law_case_count: Math.max(
            2,
            Math.floor(session.current_round / 2) + 1,
          ),
          task_layer: {
            node_count: 5 + session.current_round,
          },
        } as T;
      }

      if (
        method === "GET" &&
        (action === "events" || action.startsWith("events/history"))
      ) {
        const limitRaw = Number(url.searchParams.get("limit"));
        const limit = Number.isFinite(limitRaw) ? Math.max(1, limitRaw) : 100;

        return {
          events: session.events.slice(-limit),
        } as T;
      }

      if (method === "GET" && action === "diff") {
        const fromRoundRaw = Number(url.searchParams.get("from_round"));
        const toRoundRaw = Number(url.searchParams.get("to_round"));

        const fromRound = Number.isFinite(fromRoundRaw)
          ? fromRoundRaw
          : Math.max(session.current_round - 1, 0);

        const toRound = Number.isFinite(toRoundRaw)
          ? toRoundRaw
          : session.current_round;

        return computeDiff(session, fromRound, toRound) as T;
      }
    }

    throw new Error(`Mock transport does not support ${method} ${req.path}`);
  }

  private mustGetSession(sessionId: string): MockSession {
    const session = this.sessions.get(sessionId);

    if (!session) {
      throw new Error(`Mock session not found: ${sessionId}`);
    }

    return session;
  }

  private recordEvent(
    session: MockSession,
    event: string,
    source: string,
    data?: Record<string, unknown>,
  ): void {
    session.events.push({
      seq: session.nextSeq,
      ts_ms: Date.now(),
      event,
      source,
      data,
    });

    session.nextSeq += 1;
  }

  private pushSnapshot(session: MockSession): void {
    const existing = session.snapshots.find(
      (item) => item.round === session.current_round,
    );

    if (!existing) {
      session.snapshots.push(buildGraph(session.current_round));
    }

    const graph = getGraphAtRound(session, session.current_round);
    session.metrics.arguments = graph.nodes.length;

    session.metrics.attacks = graph.edges.filter(
      (edge) => edge.type === "ATTACK",
    ).length;

    session.metrics.supports = graph.edges.filter(
      (edge) => edge.type === "SUPPORT",
    ).length;
  }

  private advance(session: MockSession): void {
    if (session.is_finished) {
      return;
    }

    this.recordEvent(session, "turn_start", "engine", {
      round: session.current_round + 1,
    });

    session.current_round += 1;

    session.transcript.push(
      `[plaintiff] mock argument at round ${session.current_round}`,
    );

    session.transcript.push(
      `[defendant] mock rebuttal at round ${session.current_round}`,
    );

    if (session.current_round >= session.max_rounds) {
      session.is_ready_for_adjudication = true;
      session.transcript.push("[system] ready for adjudication");

      this.recordEvent(session, "adjudication_ready", "engine", {
        round: session.current_round,
      });
    }

    this.pushSnapshot(session);

    this.recordEvent(session, "turn_complete", "engine", {
      round: session.current_round,
      phase: session.is_ready_for_adjudication
        ? "ready_for_adjudication"
        : "running",
    });
  }

  private adjudicate(session: MockSession): void {
    if (session.is_finished) {
      return;
    }

    this.recordEvent(session, "adjudication_start", "judge", {
      round: session.current_round,
    });

    if (!session.is_ready_for_adjudication) {
      session.transcript.push(
        "[judge] adjudication requested early; accepted in mock mode",
      );
    }

    session.is_ready_for_adjudication = false;
    session.is_finished = true;

    session.winner =
      session.current_round % 2 === 0 ? "plaintiff" : "defendant";

    session.transcript.push(`[judge] verdict: ${session.winner}`);

    this.recordEvent(session, "adjudication_complete", "judge", {
      winner: session.winner,
    });
  }
}
