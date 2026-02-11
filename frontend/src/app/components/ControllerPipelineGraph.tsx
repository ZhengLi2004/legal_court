import {
  Background,
  MarkerType,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";

import type { TurnArtifact } from "../../compat";
type StepStatus = "done" | "active" | "pending" | "warning";

function statusColor(status: StepStatus): {
  bg: string;
  border: string;
  text: string;
} {
  if (status === "done") {
    return { bg: "#dcfce7", border: "#15803d", text: "#14532d" };
  }

  if (status === "active") {
    return { bg: "#dbeafe", border: "#1d4ed8", text: "#1e3a8a" };
  }

  if (status === "warning") {
    return { bg: "#fee2e2", border: "#dc2626", text: "#7f1d1d" };
  }

  return { bg: "#f1f5f9", border: "#64748b", text: "#334155" };
}

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

interface ControllerPipelineGraphProps {
  artifact: TurnArtifact | null;
}

export function ControllerPipelineGraph({
  artifact,
}: ControllerPipelineGraphProps) {
  const status = deriveStatuses(artifact);

  const nodes: Node[] = [
    { id: "ASSESS", position: { x: 20, y: 120 }, data: { label: "ASSESS" } },
    {
      id: "DELEGATE",
      position: { x: 190, y: 120 },
      data: { label: "DELEGATE" },
    },
    { id: "WAIT", position: { x: 370, y: 120 }, data: { label: "WAIT" } },
    { id: "DECIDE", position: { x: 540, y: 120 }, data: { label: "DECIDE" } },
    { id: "DONE", position: { x: 710, y: 120 }, data: { label: "DONE" } },
    { id: "RETRY", position: { x: 540, y: 260 }, data: { label: "RETRY" } },
  ].map((node) => {
    const row = statusColor(status[node.id] ?? "pending");

    return {
      ...node,
      draggable: false,
      selectable: false,
      style: {
        width: 132,
        borderRadius: 10,
        border: `1.5px solid ${row.border}`,
        background: row.bg,
        color: row.text,
        textAlign: "center",
        fontWeight: 700,
      },
    };
  });

  const edges: Edge[] = [
    { id: "A-D", source: "ASSESS", target: "DELEGATE" },
    { id: "D-W", source: "DELEGATE", target: "WAIT" },
    { id: "W-C", source: "WAIT", target: "DECIDE" },
    { id: "C-F", source: "DECIDE", target: "DONE" },
    { id: "C-R", source: "DECIDE", target: "RETRY" },
    { id: "R-D", source: "RETRY", target: "DELEGATE" },
  ].map((edge) => ({
    ...edge,
    type: "smoothstep",
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: edge.id === "C-R" || edge.id === "R-D" ? "#dc2626" : "#0284c7",
    },
    style: {
      stroke: edge.id === "C-R" || edge.id === "R-D" ? "#dc2626" : "#0284c7",
      strokeDasharray:
        edge.id === "C-R" || edge.id === "R-D" ? "6 4" : undefined,
      strokeWidth: 1.8,
    },
    selectable: false,
  }));

  return (
    <div className="ux-graph-canvas ux-graph-canvas-compact">
      <ReactFlow
        edges={edges}
        fitView
        fitViewOptions={{ padding: 0.18 }}
        maxZoom={1.5}
        minZoom={0.4}
        nodes={nodes}
        nodesConnectable={false}
        nodesDraggable={false}
        panOnDrag={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={18} size={1} />
      </ReactFlow>
    </div>
  );
}
