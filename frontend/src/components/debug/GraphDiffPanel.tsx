import { useMemo, useState } from "react";

import {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";

import type { GraphDiffView, GraphView, TurnArtifact } from "../../compat";
type FocusMode = "all" | "changed" | "changedNeighbors" | "rejected";

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function normalizeEdgeType(raw: string): string {
  const upper = raw.toUpperCase();

  if (upper === "ATTACK") {
    return "CONFLICT";
  }

  return upper;
}

function gatherRejectedNodeIds(artifacts: TurnArtifact[]): Set<string> {
  const ids = new Set<string>();

  for (const artifact of artifacts) {
    const logs = artifact.executionLogs.toLowerCase();

    if (
      !(
        logs.includes("reject") ||
        logs.includes("failed") ||
        logs.includes("error") ||
        logs.includes("rollback")
      )
    ) {
      continue;
    }

    const actions = Array.isArray(artifact.parsedActions)
      ? artifact.parsedActions
      : [];

    for (const action of actions) {
      const row = asRecord(action);

      const candidates = [
        row.node_id,
        row.nodeId,
        row.source_id,
        row.sourceId,
        row.target_id,
        row.targetId,
        row.source,
        row.target,
      ];

      for (const item of candidates) {
        const id = asString(item);

        if (id) {
          ids.add(id);
        }
      }
    }
  }

  return ids;
}

interface GraphDiffPanelProps {
  currentGraph: GraphView | null;
  baselineGraph: GraphView | null;
  diff: GraphDiffView | null;
  artifacts: TurnArtifact[];
  title?: string;
}

export function GraphDiffPanel({
  currentGraph,
  baselineGraph,
  diff,
  artifacts,
  title = "Graph Diff Highlighter",
}: GraphDiffPanelProps) {
  const [focusMode, setFocusMode] = useState<FocusMode>("all");

  const model = useMemo(() => {
    if (!currentGraph) {
      return null;
    }

    const previousNodes = new Map(
      (baselineGraph?.nodes ?? []).map((node) => [node.id, node]),
    );

    const currentNodeIds = new Set(currentGraph.nodes.map((node) => node.id));

    const previousNodeIds = new Set(
      (baselineGraph?.nodes ?? []).map((node) => node.id),
    );

    const addedNodeIds = new Set(diff?.addedNodeIds ?? []);
    const addedEdgeIds = new Set(diff?.addedEdgeIds ?? []);
    const statusChangedNodeIds = new Set<string>();

    for (const node of currentGraph.nodes) {
      const prev = previousNodes.get(node.id);

      if (prev && (prev.status ?? "") !== (node.status ?? "")) {
        statusChangedNodeIds.add(node.id);
      }
    }

    const reusedNodeIds = new Set<string>();

    for (const nodeId of currentNodeIds) {
      if (previousNodeIds.has(nodeId) && !statusChangedNodeIds.has(nodeId)) {
        reusedNodeIds.add(nodeId);
      }
    }

    const rejectedNodeIds = gatherRejectedNodeIds(artifacts);

    const changedNodeIds = new Set<string>([
      ...addedNodeIds,
      ...statusChangedNodeIds,
      ...rejectedNodeIds,
    ]);

    const adjacency = new Map<string, Set<string>>();

    for (const edge of currentGraph.edges) {
      if (!adjacency.has(edge.source)) {
        adjacency.set(edge.source, new Set<string>());
      }

      if (!adjacency.has(edge.target)) {
        adjacency.set(edge.target, new Set<string>());
      }

      adjacency.get(edge.source)?.add(edge.target);
      adjacency.get(edge.target)?.add(edge.source);
    }

    const changedWithNeighbors = new Set<string>(changedNodeIds);

    for (const nodeId of changedNodeIds) {
      for (const neighbor of adjacency.get(nodeId) ?? []) {
        changedWithNeighbors.add(neighbor);
      }
    }

    let visibleIds = new Set(currentNodeIds);

    if (focusMode === "changed") {
      visibleIds = changedNodeIds;
    } else if (focusMode === "changedNeighbors") {
      visibleIds = changedWithNeighbors;
    } else if (focusMode === "rejected") {
      visibleIds = rejectedNodeIds;
    }

    const nodes: Node[] = currentGraph.nodes
      .filter((node) => visibleIds.has(node.id))
      .map((node, index) => {
        const isAdded = addedNodeIds.has(node.id);
        const isRejected = rejectedNodeIds.has(node.id);
        const isChanged = statusChangedNodeIds.has(node.id);
        const isReused = reusedNodeIds.has(node.id);
        let borderColor = "#64748b";
        let background = "#f8fafc";
        let borderStyle: "solid" | "dashed" = "solid";

        if (isAdded) {
          borderColor = "#16a34a";
          background = "#dcfce7";
        } else if (isRejected) {
          borderColor = "#e11d48";
          background = "#ffe4e6";
        } else if (isChanged) {
          borderColor = "#2563eb";
          background = "#dbeafe";
        } else if (isReused) {
          borderColor = "#0ea5e9";
          borderStyle = "dashed";
          background = "#ecfeff";
        }

        const tags = [
          isAdded ? "Added" : "",
          isReused ? "Reused" : "",
          isChanged ? "Status Changed" : "",
          isRejected ? "Rejected" : "",
        ]
          .filter(Boolean)
          .join(" | ");

        return {
          id: node.id,
          position: {
            x: 40 + (index % 5) * 240,
            y: 30 + Math.floor(index / 5) * 140,
          },
          data: {
            label: (
              <div>
                <strong>{node.id}</strong>
                <div style={{ fontSize: "0.75rem", marginTop: 2 }}>
                  {node.type} {node.status ? `• ${node.status}` : ""}
                </div>
                {tags ? (
                  <div style={{ fontSize: "0.72rem", marginTop: 4 }}>
                    {tags}
                  </div>
                ) : null}
              </div>
            ),
          },
          style: {
            border: `2px ${borderStyle} ${borderColor}`,
            borderRadius: 12,
            padding: 8,
            width: 210,
            background,
            color: "#0f172a",
          },
        };
      });

    const edges: Edge[] = currentGraph.edges
      .filter(
        (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
      )
      .map((edge) => {
        const isAdded = addedEdgeIds.has(edge.id);
        const type = normalizeEdgeType(edge.type);
        let stroke = "#64748b";

        if (type === "CONFLICT") {
          stroke = isAdded ? "#dc2626" : "#f97316";
        } else if (type === "SUPPORT") {
          stroke = isAdded ? "#0891b2" : "#0ea5e9";
        }

        return {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          markerEnd: {
            type: MarkerType.ArrowClosed,
            width: 20,
            height: 20,
            color: stroke,
          },
          style: {
            stroke,
            strokeWidth: isAdded ? 3 : 2,
          },
          label: type,
          labelStyle: {
            fill: "#334155",
            fontSize: 11,
          },
          animated: isAdded,
        } as Edge;
      });

    return {
      nodes,
      edges,
      summary: {
        addedNodes: addedNodeIds.size,
        addedEdges: addedEdgeIds.size,
        statusChanged: statusChangedNodeIds.size,
        reused: reusedNodeIds.size,
        rejected: rejectedNodeIds.size,
      },
    };
  }, [artifacts, baselineGraph, currentGraph, diff, focusMode]);

  return (
    <article className="card wide">
      <h2>{title}</h2>

      <div className="sub-actions">
        <button type="button" onClick={() => setFocusMode("all")}>
          All
        </button>

        <button type="button" onClick={() => setFocusMode("changed")}>
          Changed
        </button>

        <button type="button" onClick={() => setFocusMode("changedNeighbors")}>
          Changed + 1-hop
        </button>

        <button type="button" onClick={() => setFocusMode("rejected")}>
          Rejected
        </button>
      </div>

      {model ? (
        <>
          <p className="line">
            +N{model.summary.addedNodes} +E{model.summary.addedEdges} | status
            changes {model.summary.statusChanged} | reused{" "}
            {model.summary.reused}| rejected {model.summary.rejected}
          </p>

          <div className="graph-canvas">
            <ReactFlow nodes={model.nodes} edges={model.edges} fitView>
              <MiniMap pannable zoomable />
              <Controls />
              <Background gap={18} size={1} />
            </ReactFlow>
          </div>
        </>
      ) : (
        <p className="hint">Load graph and diff first.</p>
      )}
    </article>
  );
}
