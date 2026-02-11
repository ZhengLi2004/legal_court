import { useMemo } from "react";
import CytoscapeComponent from "react-cytoscapejs";

import {
  DEBATE_GRAPH_STYLESHEET,
  mapDebateGraphToElements,
} from "../graph/cytoscape/debateGraphCytoscape";

import { nodeStatusLabel } from "../utils/payload";
import type { GraphView } from "../../compat";

interface SimpleBafGraphProps {
  graph: GraphView | null;
  preferredExtension: string[];
  rootClaimStatusMap: Record<string, string>;
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

    const claimNodes = graph.nodes
      .filter((node) => node.type.toUpperCase() === "CLAIM")
      .map((node) => ({
        ...node,
        status: rootClaimStatusMap[node.id] ?? node.status,
      }));

    if (!claimNodes.length) {
      return null;
    }

    const claimIdSet = new Set(claimNodes.map((node) => node.id));

    const claimEdges = graph.edges.filter(
      (edge) =>
        claimIdSet.has(edge.source) &&
        claimIdSet.has(edge.target) &&
        ["SUPPORT", "CONFLICT", "ATTACK"].includes(edge.type.toUpperCase()),
    );

    const preferredSet = new Set(preferredExtension);
    const nodeClassById: Record<string, string[]> = {};

    for (const node of claimNodes) {
      nodeClassById[node.id] = preferredSet.has(node.id)
        ? ["node-preferred"]
        : [];
    }

    const mapped = mapDebateGraphToElements(claimNodes, claimEdges, {
      nodeClassById,
    });

    return {
      ...mapped,
      statusMap: rootClaimStatusMap,
      preferredSet,
    };
  }, [graph, preferredExtension, rootClaimStatusMap]);

  return (
    <article className="ux-card">
      <h2>BAF 关系图</h2>

      <p className="ux-muted">
        绿色粗边节点为选中扩展；蓝线为支持，红色虚线为冲突。
      </p>

      {model ? (
        <>
          <div className="ux-graph-canvas ux-graph-canvas-compact">
            <CytoscapeComponent
              boxSelectionEnabled={false}
              elements={model.elements}
              layout={{
                name: "circle",
                fit: true,
                padding: 40,
              }}
              maxZoom={2.2}
              minZoom={0.15}
              stylesheet={DEBATE_GRAPH_STYLESHEET}
              style={{ width: "100%", height: "100%" }}
              wheelSensitivity={0.18}
            />
          </div>

          <div className="ux-kv" style={{ marginTop: "0.65rem" }}>
            {model.nodes.map((node) => {
              const status = model.statusMap[node.id] ?? node.statusFamily;

              return (
                <p key={node.id}>
                  <span>{node.id}</span>

                  <strong>
                    {nodeStatusLabel(status)}
                    {model.preferredSet.has(node.id) ? " · 选中扩展" : ""}
                  </strong>
                </p>
              );
            })}
          </div>
        </>
      ) : (
        <p className="ux-empty">暂无可渲染的 BAF 图数据。</p>
      )}
    </article>
  );
}
