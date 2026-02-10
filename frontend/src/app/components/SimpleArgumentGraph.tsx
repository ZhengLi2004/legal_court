import { useMemo } from "react";

import {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";

import { nodeStatusLabel } from "../utils/payload";
import type { GraphView } from "../../compat";

interface SimpleArgumentGraphProps {
  graph: GraphView | null;
  title?: string;
}

type Lane = "FACT" | "LAW" | "CLAIM" | "OTHER";

const LANE_Y: Record<Lane, number> = {
  FACT: 40,
  LAW: 180,
  CLAIM: 320,
  OTHER: 460,
};

function laneForType(type: string): Lane {
  const upper = type.toUpperCase();

  if (upper === "FACT") {
    return "FACT";
  }

  if (upper === "LAW") {
    return "LAW";
  }

  if (upper === "CLAIM") {
    return "CLAIM";
  }

  return "OTHER";
}

function statusColor(status: string): { border: string; bg: string } {
  const upper = status.toUpperCase();

  if (upper === "VALIDATED") {
    return { border: "#15803d", bg: "#dcfce7" };
  }

  if (upper === "DEFEATED") {
    return { border: "#be123c", bg: "#ffe4e6" };
  }

  return { border: "#1d4ed8", bg: "#dbeafe" };
}

function edgeTypeText(type: string): string {
  const upper = type.toUpperCase();
  return upper === "ATTACK" ? "CONFLICT" : upper;
}

export function SimpleArgumentGraph({
  graph,
  title = "论证图谱",
}: SimpleArgumentGraphProps) {
  const model = useMemo(() => {
    if (!graph) {
      return null;
    }

    const laneSlots = new Map<string, number>();

    const nodes: Node[] = graph.nodes.map((node) => {
      const lane = laneForType(node.type);
      const slot = laneSlots.get(lane) ?? 0;
      laneSlots.set(lane, slot + 1);
      const x = 40 + slot * 210;
      const y = LANE_Y[lane];
      const palette = statusColor(node.status ?? "HYPOTHETICAL");

      return {
        id: node.id,
        position: { x, y },
        data: {
          label: (
            <div>
              <strong>{node.label || node.id}</strong>

              <div style={{ fontSize: "0.74rem", marginTop: 4 }}>
                {lane} · {nodeStatusLabel(node.status ?? "HYPOTHETICAL")}
              </div>
            </div>
          ),
        },
        style: {
          width: 190,
          border: `2px solid ${palette.border}`,
          borderRadius: 12,
          background: palette.bg,
          color: "#0f172a",
          padding: 8,
        },
      } as Node;
    });

    const edges: Edge[] = graph.edges.map((edge) => {
      const kind = edgeTypeText(edge.type);
      const isConflict = kind === "CONFLICT";
      const stroke = isConflict ? "#dc2626" : "#0284c7";

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: "smoothstep",
        label: kind,
        style: {
          stroke,
          strokeWidth: 2,
          strokeDasharray: isConflict ? "6 3" : undefined,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 20,
          height: 20,
          color: stroke,
        },
        labelStyle: {
          fill: "#334155",
          fontSize: 10,
        },
      } as Edge;
    });

    return { nodes, edges };
  }, [graph]);

  return (
    <article className="ux-card">
      <h2>{title}</h2>

      <p className="ux-muted">
        蓝线代表支持，红色虚线代表冲突；节点颜色代表当前结论状态。
      </p>

      {model ? (
        <div className="ux-graph-canvas">
          <ReactFlow
            nodes={model.nodes}
            edges={model.edges}
            fitView
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable
            proOptions={{ hideAttribution: true }}
          >
            <MiniMap pannable zoomable />
            <Controls />
            <Background gap={18} size={1} />
          </ReactFlow>
        </div>
      ) : (
        <p className="ux-empty">当前暂无图谱数据。</p>
      )}
    </article>
  );
}
