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

interface SimpleBafGraphProps {
  graph: GraphView | null;
  preferredExtension: string[];
  rootClaimStatusMap: Record<string, string>;
}

function edgeKind(type: string): "SUPPORT" | "CONFLICT" | "OTHER" {
  const upper = type.toUpperCase();

  if (upper === "SUPPORT") {
    return "SUPPORT";
  }

  if (upper === "CONFLICT" || upper === "ATTACK") {
    return "CONFLICT";
  }

  return "OTHER";
}

function fillByStatus(status: string): string {
  const upper = status.toUpperCase();

  if (upper === "VALIDATED") {
    return "#dcfce7";
  }

  if (upper === "DEFEATED") {
    return "#ffe4e6";
  }

  return "#dbeafe";
}

export function SimpleBafGraph({
  graph,
  preferredExtension,
  rootClaimStatusMap,
}: SimpleBafGraphProps) {
  const model = useMemo(() => {
    if (!graph) {
      return null;
    }

    const claimNodes = graph.nodes.filter(
      (node) => node.type.toUpperCase() === "CLAIM",
    );

    if (!claimNodes.length) {
      return null;
    }

    const claimIdSet = new Set(claimNodes.map((node) => node.id));
    const preferredSet = new Set(preferredExtension);
    const total = claimNodes.length;
    const centerX = 360;
    const centerY = 240;
    const radius = Math.max(140, Math.min(250, 40 * total));

    const nodes: Node[] = claimNodes.map((node, index) => {
      const angle = (index / Math.max(total, 1)) * Math.PI * 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      const rawStatus =
        rootClaimStatusMap[node.id] ?? node.status ?? "HYPOTHETICAL";

      const preferred = preferredSet.has(node.id);

      return {
        id: node.id,
        position: { x, y },
        data: {
          label: (
            <div>
              <strong>{node.id}</strong>

              <div style={{ fontSize: "0.74rem", marginTop: 4 }}>
                {nodeStatusLabel(rawStatus)}
                {preferred ? " · 选中扩展" : ""}
              </div>
            </div>
          ),
        },
        style: {
          width: 170,
          border: preferred ? "3px solid #15803d" : "2px solid #334155",
          borderRadius: 12,
          background: fillByStatus(rawStatus),
          color: "#0f172a",
          padding: 8,
        },
      } as Node;
    });

    const edges: Edge[] = graph.edges
      .filter(
        (edge) => claimIdSet.has(edge.source) && claimIdSet.has(edge.target),
      )
      .filter((edge) => edgeKind(edge.type) !== "OTHER")
      .map((edge) => {
        const kind = edgeKind(edge.type);
        const conflict = kind === "CONFLICT";
        const stroke = conflict ? "#dc2626" : "#0284c7";

        return {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          type: "smoothstep",
          label: kind,
          style: {
            stroke,
            strokeWidth: 2,
            strokeDasharray: conflict ? "6 3" : undefined,
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
  }, [graph, preferredExtension, rootClaimStatusMap]);

  return (
    <article className="ux-card">
      <h2>BAF 关系图</h2>

      <p className="ux-muted">
        绿色粗边节点为选中扩展；蓝线为支持，红色虚线为冲突。
      </p>

      {model ? (
        <div className="ux-graph-canvas ux-graph-canvas-compact">
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
        <p className="ux-empty">暂无可渲染的 BAF 图数据。</p>
      )}
    </article>
  );
}
