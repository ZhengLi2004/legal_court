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
  event_id: string;
  seq: number;
  ts_ms: number;
  session_id: string;
  turn_uid: string;
  round_idx?: number;
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
  turn_artifacts: Record<string, unknown>[];
  demo_keyframes: Record<string, unknown>[];

  failure_simulation: {
    es_unavailable: boolean;
    llm_timeout: boolean;
  };

  latest_turn_uid: string;
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
    turn_artifacts: session.turn_artifacts.map((item) => ({ ...item })),
    demo_keyframes: session.demo_keyframes.map((item) => ({ ...item })),
    failure_simulation: { ...session.failure_simulation },
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
        turn_artifacts: [],
        demo_keyframes: [],
        failure_simulation: {
          es_unavailable: false,
          llm_timeout: false,
        },
        latest_turn_uid: "",
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
        if (session.failure_simulation.es_unavailable) {
          this.recordEvent(session, "session_warning", "api", {
            kind: "es_unavailable",
            message: "Simulated ES unavailable; degrade continue",
            round_idx: session.current_round,
          });
        }

        if (session.failure_simulation.llm_timeout) {
          this.recordEvent(session, "session_warning", "api", {
            kind: "llm_timeout",
            message: "Simulated LLM timeout; degrade continue",
            round_idx: session.current_round,
          });
        }

        this.advance(session);
        return toSnapshotResponse(cloneSession(session)) as T;
      }

      if (method === "POST" && action === "adjudicate") {
        if (session.failure_simulation.llm_timeout) {
          this.recordEvent(session, "session_warning", "api", {
            kind: "llm_timeout",
            message: "Simulated adjudication timeout flag enabled",
            round_idx: session.current_round,
          });
        }

        this.adjudicate(session);
        return toSnapshotResponse(cloneSession(session)) as T;
      }

      if (method === "POST" && action === "simulate/failure") {
        const payload = body as Record<string, unknown>;
        const kind = String(payload.kind ?? "");
        const enabled = payload.enabled === true;

        if (kind === "es_unavailable" || kind === "llm_timeout") {
          session.failure_simulation[kind] = enabled;

          this.recordEvent(session, "failure_simulation_set", "api", {
            kind,
            enabled,
            round_idx: session.current_round,
          });
        }

        return {
          session_id: session.session_id,
          failure_simulation: { ...session.failure_simulation },
          updated_at: new Date().toISOString(),
        } as T;
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

      if (
        method === "GET" &&
        typeof action === "string" &&
        action.startsWith("snapshots/")
      ) {
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

      if (method === "GET" && action === "snapshots") {
        return {
          items: session.snapshots.map((item) => ({
            round_idx: item.round,
            turn: item.round % 2 === 0 ? "defendant" : "plaintiff",
            ts_ms: Date.now(),
            node_count: item.nodes.length,
            edge_count: item.edges.length,
          })),
        } as T;
      }

      if (method === "GET" && action === "memory") {
        const baseInsights = [
          {
            content:
              "When opponent introduces ungrounded claims, prioritize source-backed attack chains.",
            side: "PLAINTIFF",
            cases: ["case-A", "case-C"],
            representatives: ["case-C"],
            linked_round: Math.max(0, session.current_round - 1),
          },
          {
            content:
              "Escalate from fact retrieval to precedent retrieval after repeated rebuttal cycles.",
            side: "COMMON",
            cases: ["case-B", "case-C", "case-D"],
            representatives: ["case-B", "case-D"],
            linked_round: session.current_round,
          },
        ];

        return {
          session_id: session.session_id,
          insight_summaries: baseInsights.map((item) => item.content),
          insight_items: baseInsights.map((item) => ({
            ...item,
            case_count: item.cases.length,
            representative_count: item.representatives.length,
          })),
          representative_case_ids: ["case-B", "case-C", "case-D"],
          static_history_count: 12,
          dynamic_law_case_count: Math.max(
            2,
            Math.floor(session.current_round / 2) + 1,
          ),
          task_layer: {
            node_count: 5 + session.current_round,
            edge_count: 4 + session.current_round,
          },
          case_snapshots: session.snapshots.map((item) => ({
            round_idx: item.round,
            turn: item.round % 2 === 0 ? "defendant" : "plaintiff",
            ts_ms: Date.now(),
            node_count: item.nodes.length,
            edge_count: item.edges.length,
          })),
        } as T;
      }

      if (
        method === "GET" &&
        (action === "events" ||
          (typeof action === "string" && action.startsWith("events/history")))
      ) {
        const limitRaw = Number(url.searchParams.get("limit"));
        const limit = Number.isFinite(limitRaw) ? Math.max(1, limitRaw) : 100;

        return {
          events: session.events.slice(-limit),
        } as T;
      }

      if (method === "GET" && action === "turns/artifacts") {
        const limitRaw = Number(url.searchParams.get("limit"));
        const limit = Number.isFinite(limitRaw) ? Math.max(1, limitRaw) : 50;

        return {
          items: session.turn_artifacts.slice(-limit),
        } as T;
      }

      if (
        method === "GET" &&
        typeof action === "string" &&
        action.startsWith("turns/")
      ) {
        const chunks = action.split("/");

        if (chunks.length >= 3 && chunks[2] === "artifacts") {
          const turnUid = chunks[1];
          const limitRaw = Number(url.searchParams.get("limit"));
          const limit = Number.isFinite(limitRaw) ? Math.max(1, limitRaw) : 50;

          return {
            items: session.turn_artifacts
              .filter((item) => item.turn_uid === turnUid)
              .slice(-limit),
          } as T;
        }
      }

      if (method === "GET" && action === "debug-bundle") {
        const limitRaw = Number(url.searchParams.get("event_limit"));

        const eventLimit = Number.isFinite(limitRaw)
          ? Math.max(1, limitRaw)
          : 20;

        const events = session.events.slice(-eventLimit);
        const latestGraph = getGraphAtRound(session, session.current_round);

        return {
          session_id: session.session_id,
          round_idx: session.current_round,
          turn_uid: session.latest_turn_uid,
          status: session.is_finished
            ? "FINISHED"
            : session.is_ready_for_adjudication
              ? "READY_FOR_ADJUDICATION"
              : "DEBATING",
          last_error: "",
          snapshot_summary: {
            phase: session.is_finished ? "finished" : "running",
            node_count: latestGraph.nodes.length,
            edge_count: latestGraph.edges.length,
            claim_count: latestGraph.nodes.filter(
              (item) => item.type === "CLAIM",
            ).length,
            conflict_count: latestGraph.edges.filter(
              (item) => item.type === "ATTACK",
            ).length,
          },
          recent_events: events,
          latest_turn_artifact: {
            turn_uid: session.latest_turn_uid,
            side: session.current_round % 2 === 0 ? "defendant" : "plaintiff",
            round_idx: session.current_round,
            controller_assessment: { needs_fact: true, needs_law: true },
            batch_instructions: {
              fact: "mock fact task",
              law: "mock law task",
            },
            decision_raw: `mock decision round ${session.current_round}`,
            parsed_actions: [{ action_type: "add_support" }],
            execution_logs:
              session.current_round % 2 === 0 ? "success" : "rejected",
            retry_history:
              session.current_round % 2 === 0 ? [] : [{ kind: "format" }],
            worker_reports: [
              { worker: "FactWorker", status: "FOUND", duration_ms: 88 },
            ],
            narrative_raw_sentences: [`raw sentence ${session.current_round}`],
            narrative_polished: `polished sentence ${session.current_round}`,
          },
          generated_at: new Date().toISOString(),
        } as T;
      }

      if (method === "POST" && action === "demo/run") {
        const payload = body as Record<string, unknown> | undefined;
        const maxStepsRaw = Number(payload?.max_steps ?? 20);

        const maxSteps = Number.isFinite(maxStepsRaw)
          ? Math.max(1, Math.floor(maxStepsRaw))
          : 20;

        const autoAdjudicate = payload?.auto_adjudicate !== false;
        const keyframes: Record<string, unknown>[] = [];

        if (session.current_round === 0) {
          keyframes.push({
            session_id: session.session_id,
            event: "demo_start",
            reason: "demo_start",
            round_idx: 0,
            turn_uid: session.latest_turn_uid,
            ts_ms: Date.now(),
          });
        }

        let stepsExecuted = 0;
        let endedBy = "max_steps";

        for (let i = 0; i < maxSteps; i += 1) {
          if (session.is_finished) {
            endedBy = "finished";
            break;
          }

          if (session.is_ready_for_adjudication && autoAdjudicate) {
            keyframes.push({
              session_id: session.session_id,
              event: "adjudication_ready",
              reason: "ready_for_adjudication",
              round_idx: session.current_round,
              turn_uid: session.latest_turn_uid,
              ts_ms: Date.now(),
            });

            this.adjudicate(session);
            stepsExecuted += 1;
            endedBy = "adjudicated";

            keyframes.push({
              session_id: session.session_id,
              event: "adjudication_complete",
              reason: "adjudication_complete",
              round_idx: session.current_round,
              turn_uid: session.latest_turn_uid,
              ts_ms: Date.now(),
            });

            break;
          }

          this.advance(session);
          stepsExecuted += 1;

          if (session.current_round === 1) {
            keyframes.push({
              session_id: session.session_id,
              event: "turn_complete",
              reason: "first_round",
              round_idx: session.current_round,
              turn_uid: session.latest_turn_uid,
              ts_ms: Date.now(),
            });
          }
        }

        session.demo_keyframes = keyframes;

        return {
          session: {
            session_id: session.session_id,
            status: session.is_finished ? "FINISHED" : "DEBATING",
            updated_at: new Date().toISOString(),
          },
          keyframes,
          demo_summary: {
            steps_executed: stepsExecuted,
            ended_by: endedBy,
          },
          snapshot: toSnapshotResponse(session),
        } as T;
      }

      if (method === "GET" && action === "demo/keyframes") {
        return {
          items: session.demo_keyframes,
          total: session.demo_keyframes.length,
        } as T;
      }

      if (method === "GET" && action === "export/replay.json") {
        return {
          session: {
            session_id: session.session_id,
            status: session.is_finished ? "FINISHED" : "DEBATING",
            failure_simulation: session.failure_simulation,
          },
          snapshot: toSnapshotResponse(session),
          snapshot_index: session.snapshots.map((item) => ({
            round_idx: item.round,
            turn: item.round % 2 === 0 ? "defendant" : "plaintiff",
            ts_ms: Date.now(),
            node_count: item.nodes.length,
            edge_count: item.edges.length,
          })),
          snapshots: session.snapshots,
          events: session.events,
          turn_artifacts: session.turn_artifacts,
          metadata: {
            generated_at: new Date().toISOString(),
            event_count: session.events.length,
            artifact_count: session.turn_artifacts.length,
            snapshot_count: session.snapshots.length,
          },
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
    const roundRaw = data?.round ?? data?.round_idx;

    const round =
      typeof roundRaw === "number" && Number.isFinite(roundRaw)
        ? Math.floor(roundRaw)
        : session.current_round;

    const turnUidRaw = data?.turn_uid;

    const turnUid =
      typeof turnUidRaw === "string" && turnUidRaw.length > 0
        ? turnUidRaw
        : `turn_${round}_${session.nextSeq}_mock`;

    session.latest_turn_uid = turnUid;

    session.events.push({
      event_id: `${session.session_id}-${String(session.nextSeq).padStart(6, "0")}`,
      seq: session.nextSeq,
      ts_ms: Date.now(),
      session_id: session.session_id,
      turn_uid: turnUid,
      round_idx: round,
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
      turn_uid: `turn_${session.current_round + 1}_start_mock`,
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
    session.turn_artifacts.push({
      turn_uid: session.latest_turn_uid,
      side: session.current_round % 2 === 0 ? "defendant" : "plaintiff",
      round_idx: session.current_round,
      controller_assessment: {
        needs_fact: true,
        needs_law: session.current_round % 2 === 0,
      },
      batch_instructions: {
        fact_worker: "collect supporting fact",
        law_worker: "collect legal basis",
        recall_worker: "retrieve prior similar case",
      },
      decision_raw: `mock decision round ${session.current_round}`,
      parsed_actions: [
        {
          action_type:
            session.current_round % 2 === 0 ? "add_support" : "add_conflict",
          source_id: `FACT_${session.current_round}`,
          target_id: "CLAIM_ROOT",
        },
      ],
      execution_logs:
        session.current_round % 3 === 0
          ? "validation reject: conflict endpoint must be claim"
          : "success: action applied",
      retry_history:
        session.current_round % 3 === 0
          ? [
              {
                attempt: 1,
                error_type: "validation",
                message: "endpoint not claim",
              },
            ]
          : [],
      worker_reports: [
        {
          worker_name: "FactWorker",
          status: "FOUND",
          duration_ms: 80 + session.current_round,
          max_score: 0.84,
        },
        {
          worker_name: "LawWorker",
          status: "FOUND",
          duration_ms: 90 + session.current_round,
          max_score: 0.78,
        },
      ],
      narrative_raw_sentences: [`raw sentence round ${session.current_round}`],
      narrative_polished: `polished sentence round ${session.current_round}`,
    });

    this.recordEvent(session, "turn_complete", "engine", {
      round: session.current_round,
      turn_uid: `turn_${session.current_round}_complete_mock`,
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
