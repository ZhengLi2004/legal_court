import { useMemo } from "react";
import CytoscapeComponent from "react-cytoscapejs";
import type { TurnArtifact } from "../../compat";
type StepStatus = "done" | "active" | "pending" | "warning";

function hasValue(value: unknown): boolean {
  if (value === null || value === undefined) {
    return false;
  }

  if (typeof value === "string") {
    return value.trim().length > 0;
  }

  if (Array.isArray(value)) {
    return value.length > 0;
  }

  if (typeof value === "object") {
    return Object.keys(value as Record<string, unknown>).length > 0;
  }

  return true;
}

function deriveStatuses(
  artifact: TurnArtifact | null,
): Record<string, StepStatus> {
  if (!artifact) {
    return {
      ASSESS: "pending",
      DELEGATE: "pending",
      WAIT: "pending",
      DECIDE: "pending",
      RETRY: "pending",
      DONE: "pending",
    };
  }

  const execution = artifact.executionLogs.toLowerCase();

  const hasRetry =
    Array.isArray(artifact.retryHistory) && artifact.retryHistory.length > 0;

  const failed =
    execution.includes("reject") ||
    execution.includes("failed") ||
    execution.includes("error") ||
    execution.includes("rollback");

  const assessDone = hasValue(artifact.controllerAssessment);
  const delegateDone = hasValue(artifact.batchInstructions);
  const waitDone = hasValue(artifact.workerReports);

  const decideDone =
    hasValue(artifact.decisionRaw) || hasValue(artifact.parsedActions);

  return {
    ASSESS: assessDone ? "done" : "active",
    DELEGATE: delegateDone ? "done" : assessDone ? "active" : "pending",
    WAIT: waitDone ? "done" : delegateDone ? "active" : "pending",
    DECIDE: decideDone ? "done" : waitDone ? "active" : "pending",
    RETRY: hasRetry ? "warning" : failed ? "warning" : "pending",
    DONE: failed ? "warning" : decideDone ? "done" : "pending",
  };
}

const PIPELINE_STYLESHEET = [
  {
    selector: "node",
    style: {
      shape: "round-rectangle",
      label: "data(label)",
      width: 126,
      height: 52,
      "font-size": "11px",
      "font-weight": 700,
      "text-valign": "center",
      "text-halign": "center",
      "background-color": "#f1f5f9",
      "border-width": 2,
      "border-color": "#64748b",
      color: "#334155",
      "overlay-opacity": 0,
    },
  },
  {
    selector: 'node[status = "done"]',
    style: {
      "background-color": "#dcfce7",
      "border-color": "#15803d",
      color: "#14532d",
    },
  },
  {
    selector: 'node[status = "active"]',
    style: {
      "background-color": "#dbeafe",
      "border-color": "#1d4ed8",
      color: "#1e3a8a",
    },
  },
  {
    selector: 'node[status = "warning"]',
    style: {
      "background-color": "#fee2e2",
      "border-color": "#dc2626",
      color: "#7f1d1d",
    },
  },
  {
    selector: "edge",
    style: {
      width: 2,
      "curve-style": "bezier",
      "line-color": "#0284c7",
      "target-arrow-color": "#0284c7",
      "target-arrow-shape": "triangle",
      "arrow-scale": 0.9,
    },
  },
  {
    selector: ".retry-edge",
    style: {
      "line-color": "#dc2626",
      "target-arrow-color": "#dc2626",
      "line-style": "dashed",
    },
  },
];

interface ControllerPipelineGraphProps {
  artifact: TurnArtifact | null;
}

export function ControllerPipelineGraph({
  artifact,
}: ControllerPipelineGraphProps) {
  const status = deriveStatuses(artifact);

  const elements = useMemo(
    () => [
      {
        data: { id: "ASSESS", label: "ASSESS", status: status.ASSESS },
        position: { x: 80, y: 120 },
      },
      {
        data: { id: "DELEGATE", label: "DELEGATE", status: status.DELEGATE },
        position: { x: 250, y: 120 },
      },
      {
        data: { id: "WAIT", label: "WAIT", status: status.WAIT },
        position: { x: 430, y: 120 },
      },
      {
        data: { id: "DECIDE", label: "DECIDE", status: status.DECIDE },
        position: { x: 600, y: 120 },
      },
      {
        data: { id: "DONE", label: "DONE", status: status.DONE },
        position: { x: 770, y: 120 },
      },
      {
        data: { id: "RETRY", label: "RETRY", status: status.RETRY },
        position: { x: 600, y: 260 },
      },
      {
        data: { id: "A-D", source: "ASSESS", target: "DELEGATE" },
      },
      {
        data: { id: "D-W", source: "DELEGATE", target: "WAIT" },
      },
      {
        data: { id: "W-C", source: "WAIT", target: "DECIDE" },
      },
      {
        data: { id: "C-F", source: "DECIDE", target: "DONE" },
      },
      {
        data: { id: "C-R", source: "DECIDE", target: "RETRY" },
        classes: "retry-edge",
      },
      {
        data: { id: "R-D", source: "RETRY", target: "DELEGATE" },
        classes: "retry-edge",
      },
    ],
    [status],
  );

  return (
    <div className="ux-graph-canvas ux-graph-canvas-compact">
      <CytoscapeComponent
        boxSelectionEnabled={false}
        elements={elements}
        layout={{
          name: "preset",
          fit: true,
          padding: 24,
        }}
        maxZoom={1.8}
        minZoom={0.4}
        stylesheet={PIPELINE_STYLESHEET}
        style={{ width: "100%", height: "100%" }}
        wheelSensitivity={0.18}
      />
    </div>
  );
}
