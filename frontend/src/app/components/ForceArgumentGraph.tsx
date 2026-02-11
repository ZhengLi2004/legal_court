import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Core } from "cytoscape";
import CytoscapeComponent from "react-cytoscapejs";
import type { GraphView } from "../../compat";

import {
  clearNeighborhoodFocus,
  DEBATE_GRAPH_STYLESHEET,
  ensureCytoscapeFcoseRegistered,
  mapDebateGraphToElements,
  runFcoseLayout,
  setNeighborhoodFocus,
  type DebateNodeFamily,
} from "../graph/cytoscape/debateGraphCytoscape";

import { nodeStatusLabel } from "../utils/payload";

interface ForceArgumentGraphProps {
  graph: GraphView | null;
  title?: string;
}

const NODE_FAMILY_LABEL: Record<DebateNodeFamily, string> = {
  FACT: "事实",
  LAW: "法条",
  CLAIM: "主张",
  OTHER: "其他",
};

ensureCytoscapeFcoseRegistered();

export function ForceArgumentGraph({
  graph,
  title = "论证图谱（Cytoscape + fcose）",
}: ForceArgumentGraphProps) {
  const [selectedNodeId, setSelectedNodeId] = useState<string>("");
  const cyRef = useRef<Core | null>(null);

  const model = useMemo(() => {
    if (!graph) {
      return null;
    }

    return mapDebateGraphToElements(graph.nodes, graph.edges);
  }, [graph]);

  const effectiveSelectedNodeId = useMemo(() => {
    if (!selectedNodeId || !model) {
      return "";
    }

    return model.nodes.some((node) => node.id === selectedNodeId)
      ? selectedNodeId
      : "";
  }, [model, selectedNodeId]);

  useEffect(() => {
    const cy = cyRef.current;

    if (!cy) {
      return;
    }

    if (!effectiveSelectedNodeId) {
      clearNeighborhoodFocus(cy);
      return;
    }

    setNeighborhoodFocus(cy, effectiveSelectedNodeId);
  }, [effectiveSelectedNodeId, model?.elements.length]);

  useEffect(() => {
    const cy = cyRef.current;

    if (!cy || !model) {
      return;
    }

    const frame = requestAnimationFrame(() => {
      runFcoseLayout(cy);
    });

    return () => cancelAnimationFrame(frame);
  }, [model]);

  const onCyMount = useCallback((cy: Core) => {
    cyRef.current = cy;
    cy.off("tap.forceGraph");
    cy.off("tap.forceGraph", "node");

    cy.on("tap.forceGraph", "node", (event) => {
      const nodeId = event.target.id();
      setSelectedNodeId(nodeId);
      setNeighborhoodFocus(cy, nodeId);
    });

    cy.on("tap.forceGraph", (event) => {
      if (event.target !== cy) {
        return;
      }

      setSelectedNodeId("");
      clearNeighborhoodFocus(cy);
    });
  }, []);

  const selectedNode = useMemo(
    () => model?.nodes.find((node) => node.id === effectiveSelectedNodeId),
    [effectiveSelectedNodeId, model],
  );

  const selectedEdges = useMemo(() => {
    if (!selectedNode || !model) {
      return [];
    }

    return model.edges.filter(
      (edge) =>
        edge.source === selectedNode.id || edge.target === selectedNode.id,
    );
  }, [model, selectedNode]);

  return (
    <article className="ux-card">
      <h2>{title}</h2>

      <p className="ux-muted">
        默认采用 fcose 网状团簇布局。支持拖拽、缩放/平移与节点邻域高亮。
      </p>

      <div className="ux-graph-legend">
        <span>节点类型着色：FACT / LAW / CLAIM / OTHER</span>
        <span>边关系：support 实线、attack 虚线、cite 点线（bezier）</span>

        <span>
          当前可见：{model?.nodes.length ?? 0} 节点 / {model?.edges.length ?? 0}{" "}
          条边
        </span>
      </div>

      {model ? (
        <div className="ux-graph-layout">
          <div className="ux-graph-canvas">
            <CytoscapeComponent
              boxSelectionEnabled={false}
              cy={onCyMount}
              elements={model.elements}
              maxZoom={2.2}
              minZoom={0.08}
              stylesheet={DEBATE_GRAPH_STYLESHEET}
              style={{ width: "100%", height: "100%" }}
              wheelSensitivity={0.18}
            />
          </div>

          <aside className="ux-node-inspector">
            <h3>节点详情</h3>

            {selectedNode ? (
              <div className="ux-kv">
                <p>
                  <span>ID</span>
                  <strong>{selectedNode.id}</strong>
                </p>

                <p>
                  <span>类型</span>
                  <strong>{NODE_FAMILY_LABEL[selectedNode.family]}</strong>
                </p>

                <p>
                  <span>状态</span>
                  <strong>{nodeStatusLabel(selectedNode.statusFamily)}</strong>
                </p>

                <p>
                  <span>来源</span>
                  <strong>{selectedNode.agentId}</strong>
                </p>

                <div className="ux-inspector-text">
                  <strong>内容</strong>
                  <p>{selectedNode.content || "(empty)"}</p>
                </div>

                <div className="ux-inspector-text">
                  <strong>相邻边</strong>

                  {selectedEdges.length > 0 ? (
                    <pre>
                      {selectedEdges
                        .map(
                          (edge) =>
                            `${edge.id}: ${edge.source} -> ${edge.target} [${edge.relation}]`,
                        )
                        .join("\n")}
                    </pre>
                  ) : (
                    <p>无直接相邻边。</p>
                  )}
                </div>
              </div>
            ) : (
              <p className="ux-empty">点击节点查看详情；点击空白恢复全图。</p>
            )}
          </aside>
        </div>
      ) : (
        <p className="ux-empty">当前暂无图谱数据。</p>
      )}
    </article>
  );
}
