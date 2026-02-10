import { useMemo, useState } from "react";
import type { TurnArtifact } from "../../compat";

function toPretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

interface TeamFlowPanelProps {
  artifacts: TurnArtifact[];
  selectedTurnUid?: string;
  onSelectTurn?: (turnUid: string) => void;
}

export function TeamFlowPanel({
  artifacts,
  selectedTurnUid,
  onSelectTurn,
}: TeamFlowPanelProps) {
  const [expanded, setExpanded] = useState<
    "assessment" | "instructions" | "decision" | "workers" | "retry" | "none"
  >("none");

  const ordered = useMemo(
    () => [...artifacts].slice(-20).reverse(),
    [artifacts],
  );

  const selected = useMemo(() => {
    if (!ordered.length) {
      return null;
    }

    if (selectedTurnUid) {
      return (
        ordered.find((item) => item.turnUid === selectedTurnUid) ?? ordered[0]
      );
    }

    return ordered[0];
  }, [ordered, selectedTurnUid]);

  return (
    <article className="card wide">
      <h2>Team Flow & Retry Timeline</h2>
      <div className="team-flow-layout">
        <div className="scrollbox">
          {ordered.length ? (
            ordered.map((item) => {
              const isSelected = selected?.turnUid === item.turnUid;

              return (
                <button
                  className={`turn-row ${isSelected ? "turn-row-selected" : ""}`}
                  key={item.turnUid}
                  onClick={() => onSelectTurn?.(item.turnUid)}
                  type="button"
                >
                  <span>
                    r{item.round} [{item.side}]
                  </span>

                  <span>{item.turnUid}</span>

                  <span>
                    retries:{" "}
                    {Array.isArray(item.retryHistory)
                      ? item.retryHistory.length
                      : 0}
                  </span>
                </button>
              );
            })
          ) : (
            <p className="hint">No turn artifacts.</p>
          )}
        </div>

        <div className="team-flow-detail">
          {selected ? (
            <>
              <p className="line">
                selected turn: <strong>{selected.turnUid}</strong>
              </p>

              <p className="line">
                pipeline: ASSESS_NEEDS -&gt; WAIT_FOR_WORKERS -&gt; DECIDE -&gt;
                DONE
              </p>

              <div className="sub-actions">
                <button type="button" onClick={() => setExpanded("assessment")}>
                  Assessment
                </button>

                <button
                  type="button"
                  onClick={() => setExpanded("instructions")}
                >
                  Instructions
                </button>

                <button type="button" onClick={() => setExpanded("decision")}>
                  Decision
                </button>

                <button type="button" onClick={() => setExpanded("workers")}>
                  Workers
                </button>

                <button type="button" onClick={() => setExpanded("retry")}>
                  Retry
                </button>
              </div>

              {expanded === "assessment" ? (
                <pre className="json-block">
                  {toPretty(selected.controllerAssessment ?? "(empty)")}
                </pre>
              ) : null}

              {expanded === "instructions" ? (
                <pre className="json-block">
                  {toPretty(selected.batchInstructions ?? "(empty)")}
                </pre>
              ) : null}

              {expanded === "decision" ? (
                <>
                  <pre className="json-block">
                    {selected.decisionRaw || "(empty)"}
                  </pre>

                  <pre className="json-block">
                    {toPretty(selected.parsedActions)}
                  </pre>

                  <pre className="json-block">
                    {selected.executionLogs || "(empty)"}
                  </pre>
                </>
              ) : null}

              {expanded === "workers" ? (
                <pre className="json-block">
                  {toPretty(selected.workerReports)}
                </pre>
              ) : null}

              {expanded === "retry" ? (
                <pre className="json-block">
                  {toPretty(selected.retryHistory)}
                </pre>
              ) : null}
            </>
          ) : (
            <p className="hint">Select one turn to inspect.</p>
          )}
        </div>
      </div>
    </article>
  );
}
