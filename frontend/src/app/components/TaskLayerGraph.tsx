import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Core, ElementDefinition } from "cytoscape";
import CytoscapeComponent from "react-cytoscapejs";

import {
  clearNeighborhoodFocus,
  ensureCytoscapeFcoseRegistered,
  runFcoseLayout,
  setNeighborhoodFocus,
} from "../graph/cytoscape/debateGraphCytoscape";

import type { MemoryView } from "../../compat";

interface TaskLayerGraphProps {
  memoryView: MemoryView | null;
}

const TASK_LAYER_STYLESHEET = [
  {
    selector: "node",
    style: {
      shape: "round-rectangle",
      label: "data(label)",
      width: "label",
      height: "label",
      padding: "12px",
      "font-size": "10px",
      color: "#0f172a",
      "text-wrap": "wrap",
      "text-max-width": "180px",
      "background-color": "#f1f5f9",
      "border-width": 2,
      "border-color": "#64748b",
      "overlay-opacity": 0,
    },
  },
  {
    selector: 'node[kindGroup = "current"]',
    style: {
      "background-color": "#dbeafe",
      "border-color": "#1d4ed8",
      color: "#1e3a8a",
    },
  },
  {
    selector: 'node[kindGroup = "representative"]',
    style: {
      "background-color": "#dcfce7",
      "border-color": "#15803d",
      color: "#14532d",
    },
  },
  {
    selector: 'node[kindGroup = "related"]',
    style: {
      "background-color": "#fef3c7",
      "border-color": "#a16207",
      color: "#713f12",
    },
  },
  {
    selector: "edge",
    style: {
      width: 1.8,
      "curve-style": "bezier",
      "line-color": "#0284c7",
      "target-arrow-color": "#0284c7",
      "target-arrow-shape": "triangle",
      opacity: 0.88,
    },
  },
  {
    selector: ".is-dimmed",
    style: {
      opacity: 0.14,
      "text-opacity": 0.25,
    },
  },
  {
    selector: ".is-focus-node",
    style: {
      "underlay-color": "#0f172a",
      "underlay-opacity": 0.14,
      "underlay-padding": 8,
      "z-index": 999,
    },
  },
  {
    selector: ".is-focus-neighbor",
    style: {
      "underlay-color": "#0284c7",
      "underlay-opacity": 0.1,
      "underlay-padding": 5,
    },
  },
  {
    selector: ".is-focus-edge",
    style: {
      width: 2.8,
      opacity: 1,
    },
  },
];

function normalizeKind(
  kindRaw?: string,
): "current" | "representative" | "related" | "other" {
  const value = (kindRaw ?? "").toLowerCase();

  if (value.includes("current")) {
    return "current";
  }

  if (value.includes("representative")) {
    return "representative";
  }

  if (value.includes("related")) {
    return "related";
  }

  return "other";
}

ensureCytoscapeFcoseRegistered();

export function TaskLayerGraph({ memoryView }: TaskLayerGraphProps) {
  const [selectedId, setSelectedId] = useState<string>("");
  const cyRef = useRef<Core | null>(null);

  const model = useMemo(() => {
    if (!memoryView) {
      return { elements: [] as ElementDefinition[] };
    }

    const taskLayer = memoryView.taskLayerGraph;

    return {
      elements: [
        ...taskLayer.nodes.map((node) => ({
          data: {
            id: node.id,
            label: node.label || node.id,
            kind: node.kind ?? "case",
            kindGroup: normalizeKind(node.kind),
          },
        })),
        ...taskLayer.edges.map((edge, index) => ({
          data: {
            id: edge.id || `${edge.source}->${edge.target}#${index}`,
            source: edge.source,
            target: edge.target,
            type: edge.type ?? "reference",
          },
        })),
      ] as ElementDefinition[],
    };
  }, [memoryView]);

  const selectedNode = memoryView?.taskLayerGraph.nodes.find(
    (item) => item.id === selectedId,
  );

  const onCyMount = useCallback((cy: Core) => {
    cyRef.current = cy;
    cy.off("tap.taskLayer");
    cy.off("tap.taskLayer", "node");

    cy.on("tap.taskLayer", "node", (event) => {
      const nodeId = event.target.id();
      setSelectedId(nodeId);
      setNeighborhoodFocus(cy, nodeId);
    });

    cy.on("tap.taskLayer", (event) => {
      if (event.target !== cy) {
        return;
      }

      setSelectedId("");
      clearNeighborhoodFocus(cy);
    });
  }, []);

  useEffect(() => {
    const cy = cyRef.current;

    if (!cy) {
      return;
    }

    const frame = requestAnimationFrame(() => {
      runFcoseLayout(cy, {
        padding: 36,
        idealEdgeLength: 140,
        nodeRepulsion: 7600,
      });
    });

    return () => cancelAnimationFrame(frame);
  }, [model]);

  return (
    <article className="ux-card">
      <h2>TaskLayer 案例关系图</h2>

      <p className="ux-muted">
        节点表示案例，边表示引用或相似关系。点击节点查看案例详情。
      </p>

      {memoryView ? (
        <div className="ux-graph-layout">
          <div className="ux-graph-canvas">
            <CytoscapeComponent
              boxSelectionEnabled={false}
              cy={onCyMount}
              elements={model.elements}
              maxZoom={2.2}
              minZoom={0.12}
              stylesheet={TASK_LAYER_STYLESHEET}
              style={{ width: "100%", height: "100%" }}
              wheelSensitivity={0.18}
            />
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
