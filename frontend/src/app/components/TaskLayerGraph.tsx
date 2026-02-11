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

import type { MemoryView } from "../../compat";

function nodeColor(kind?: string): {
  bg: string;
  border: string;
  text: string;
} {
  const value = (kind ?? "").toLowerCase();

  if (value.includes("current")) {
    return { bg: "#dbeafe", border: "#1d4ed8", text: "#1e3a8a" };
  }

  if (value.includes("representative")) {
    return { bg: "#dcfce7", border: "#15803d", text: "#14532d" };
  }

  if (value.includes("related")) {
    return { bg: "#fef3c7", border: "#a16207", text: "#713f12" };
  }

  return { bg: "#f1f5f9", border: "#64748b", text: "#334155" };
}

interface TaskLayerGraphProps {
  memoryView: MemoryView | null;
}

export function TaskLayerGraph({ memoryView }: TaskLayerGraphProps) {
  const [selectedId, setSelectedId] = useState<string>("");

  const model = useMemo(() => {
    if (!memoryView) {
      return { nodes: [], edges: [] };
    }

    const taskLayer = memoryView.taskLayerGraph;
    const count = Math.max(taskLayer.nodes.length, 1);

    const nodes: Node[] = taskLayer.nodes.map((node, idx) => {
      const angle = (Math.PI * 2 * idx) / count;
      const x = 360 + Math.cos(angle) * 210;
      const y = 240 + Math.sin(angle) * 160;
      const palette = nodeColor(node.kind);

      return {
        id: node.id,
        position: { x, y },
        data: {
          label: (
            <div className="ux-force-node">
              <strong className="ux-force-node-title">
                {node.label || node.id}
              </strong>
              <div className="ux-force-node-meta">{node.kind ?? "case"}</div>
            </div>
          ),
        },
        style: {
          width: 138,
          borderRadius: 10,
          border: `1.5px solid ${palette.border}`,
          background: palette.bg,
          color: palette.text,
          padding: 6,
        },
      };
    });

    const edges: Edge[] = taskLayer.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: "smoothstep",
      label: edge.type ?? "reference",
      style: {
        stroke: "#0284c7",
        strokeWidth: 1.7,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: "#0284c7",
      },
      labelStyle: {
        fontSize: 10,
        fill: "#334155",
      },
    }));

    return { nodes, edges };
  }, [memoryView]);

  const selectedNode = memoryView?.taskLayerGraph.nodes.find(
    (item) => item.id === selectedId,
  );

  return (
    <article className="ux-card">
      <h2>TaskLayer 案例关系图</h2>

      <p className="ux-muted">
        节点表示案例，边表示引用或相似关系。点击节点查看案例详情。
      </p>

      {memoryView ? (
        <div className="ux-graph-layout">
          <div className="ux-graph-canvas">
            <ReactFlow
              edges={model.edges}
              fitView
              fitViewOptions={{ padding: 0.2 }}
              minZoom={0.25}
              nodes={model.nodes}
              nodesConnectable={false}
              onNodeClick={(_, node) => setSelectedId(node.id)}
              proOptions={{ hideAttribution: true }}
            >
              <MiniMap pannable zoomable />
              <Controls />
              <Background gap={18} size={1} />
            </ReactFlow>
          </div>

          <aside className="ux-node-inspector">
            <h3>案例详情</h3>

            {selectedNode ? (
              <div className="ux-kv">
                <p>
                  <span>案例 ID</span>
                  <strong>{selectedNode.id}</strong>
                </p>

                <p>
                  <span>标题</span>
                  <strong>{selectedNode.label}</strong>
                </p>

                <p>
                  <span>节点类型</span>
                  <strong>{selectedNode.kind ?? "case"}</strong>
                </p>
              </div>
            ) : (
              <p className="ux-empty">点击图中的案例节点查看详情。</p>
            )}
          </aside>
        </div>
      ) : (
        <p className="ux-empty">暂无记忆图数据。</p>
      )}
    </article>
  );
}
