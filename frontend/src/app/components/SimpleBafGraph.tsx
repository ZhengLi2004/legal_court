import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { GraphView } from "../../compat";
import { useEChart } from "../hooks/useEChart";

import {
  shortText,
  toEdgeRelation,
  toNodeFamily,
} from "../graph/echarts/debateGraphEcharts";

import { renderScrollableNodeTooltip } from "../graph/echarts/tooltip";
import { nodeStatusLabel } from "../../shared/lib/payload";

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
      .filter((node) => toNodeFamily(node.type) === "CLAIM")
      .map((node) => ({
        id: node.id,
        label: node.label || node.id,
        status: rootClaimStatusMap[node.id] ?? node.status ?? "HYPOTHETICAL",
      }));

    if (!claimNodes.length) {
      return null;
    }

    const claimIdSet = new Set(claimNodes.map((node) => node.id));

    const claimEdges = graph.edges
      .map((edge, index) => ({
        id: edge.id || `${edge.source}->${edge.target}#${index}`,
        source: edge.source,
        target: edge.target,
        relation: toEdgeRelation(edge.type),
      }))
      .filter(
        (edge) =>
          edge.relation !== "cite" &&
          claimIdSet.has(edge.source) &&
          claimIdSet.has(edge.target),
      );

    return {
      nodes: claimNodes,
      edges: claimEdges,
      preferredSet: new Set(preferredExtension),
      statusMap: rootClaimStatusMap,
    };
  }, [graph, preferredExtension, rootClaimStatusMap]);
  const option = useMemo<EChartsOption | null>(() => {
    if (!model) {
      return null;
    }

    const nodes = model.nodes.map((node) => {
      const preferred = model.preferredSet.has(node.id);

      return {
        id: node.id,
        name: shortText(node.label, 16),
        tooltipTitle: node.label,
        value: node.label,
        symbolSize: preferred ? 54 : 42,
        itemStyle: {
          color: preferred ? "#86efac" : "#93c5fd",
          borderColor: preferred ? "#15803d" : "#1d4ed8",
          borderWidth: preferred ? 4 : 2,
        },
        label: {
          show: true,
          color: "#0f172a",
          fontSize: 10,
        },
      };
    });

    const links = model.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      lineStyle: {
        color: edge.relation === "support" ? "#0284c7" : "#dc2626",
        width: 2.4,
        type: edge.relation === "support" ? "solid" : "dashed",
      },
    }));

    return {
      backgroundColor: "#f8fafc",
      tooltip: {
        trigger: "item",
        confine: true,
        enterable: true,
        formatter: (params: unknown) => {
          const row = params as {
            dataType?: string;
            data?: { id?: string; tooltipTitle?: string; value?: string };
          };

          if (row.dataType !== "node") {
            return "";
          }

          const titleText = row.data?.tooltipTitle ?? row.data?.id ?? "";

          return renderScrollableNodeTooltip(
            String(titleText),
            String(row.data?.value ?? ""),
          );
        },
      },
      series: [
        {
          type: "graph",
          layout: "circular",
          circular: {
            rotateLabel: false,
          },
          roam: true,
          draggable: true,
          data: nodes,
          links,
          edgeSymbol: ["none", "arrow"],
          edgeSymbolSize: 8,
          lineStyle: {
            opacity: 0.88,
          },
          label: {
            position: "inside",
            overflow: "truncate",
          },
          emphasis: {
            focus: "adjacency",
          },
        },
      ],
    } as EChartsOption;
  }, [model]);

  const { containerRef } = useEChart({
    option,
    enabled: Boolean(model),
  });

  return (
    <article className="ux-card">
      <h2>BAF 关系图</h2>

      {model ? (
        <>
          <div className="ux-graph-canvas ux-graph-canvas-compact">
            <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
          </div>

          <div className="ux-kv" style={{ marginTop: "0.65rem" }}>
            {model.nodes.map((node) => {
              const status = model.statusMap[node.id] ?? node.status;

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
