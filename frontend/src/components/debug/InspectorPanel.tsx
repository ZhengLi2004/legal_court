import { useMemo, useState } from "react";
import type { DebateSnapshot, TurnArtifact } from "../../compat";

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function toPretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function unwrapPayload(raw: unknown): Record<string, unknown> {
  const outer = asRecord(raw);

  const candidate =
    outer.data ?? outer.snapshot ?? outer.state ?? outer.payload;

  return candidate !== undefined ? asRecord(candidate) : outer;
}

interface InspectorPanelProps {
  snapshot: DebateSnapshot | null;
  artifact: TurnArtifact | null;
}

export function InspectorPanel({ snapshot, artifact }: InspectorPanelProps) {
  const [tab, setTab] = useState<"context" | "decision" | "narrative">(
    "context",
  );

  const payload = useMemo(() => unwrapPayload(snapshot?.raw ?? {}), [snapshot]);

  const contextPayload =
    payload.latest_context ?? payload.context ?? payload.latestContext ?? {};

  const executionText = (artifact?.executionLogs ?? "").toLowerCase();

  const hasParserIssue =
    executionText.includes("parse") ||
    executionText.includes("invalid") ||
    executionText.includes("error") ||
    executionText.includes("failed");

  return (
    <article className="card wide">
      <h2>Context / Decision / Narrative Inspector</h2>

      <div className="sub-actions">
        <button type="button" onClick={() => setTab("context")}>
          Context
        </button>

        <button type="button" onClick={() => setTab("decision")}>
          Decision
        </button>

        <button type="button" onClick={() => setTab("narrative")}>
          Narrative
        </button>
      </div>

      {tab === "context" ? (
        <>
          <p className="line">latest_context + id_inventory</p>
          <pre className="json-block">{toPretty(contextPayload)}</pre>

          <pre className="json-block">
            {toPretty(payload.id_inventory ?? {})}
          </pre>
        </>
      ) : null}

      {tab === "decision" ? (
        <>
          <p className="line">raw decision vs parsed actions</p>

          {hasParserIssue ? (
            <p className="error">
              parser issue detected from execution logs. inspect raw response
              and logs.
            </p>
          ) : null}

          <pre className="json-block">{artifact?.decisionRaw || "(empty)"}</pre>

          <pre className="json-block">
            {toPretty(artifact?.parsedActions ?? [])}
          </pre>

          <pre className="json-block">
            {artifact?.executionLogs || "(empty)"}
          </pre>
        </>
      ) : null}

      {tab === "narrative" ? (
        <>
          <p className="line">raw narrative vs polished narrative</p>

          <pre className="json-block">
            {toPretty(artifact?.narrativeRawSentences ?? [])}
          </pre>

          <pre className="json-block">
            {artifact?.narrativePolished || "(empty)"}
          </pre>
        </>
      ) : null}
    </article>
  );
}
