import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { MemoryView } from "../../compat";
import { useEChart } from "../hooks/useEChart";
import { shortText } from "../graph/echarts/debateGraphEcharts";
import { renderScrollableNodeTooltip } from "../graph/echarts/tooltip";

interface TaskLayerGraphProps {
  memoryView: MemoryView | null;
}

function normalizeKind(
  kindRaw?: string,
): "current" | "retrieved" | "topology" | "other" {
  const value = (kindRaw ?? "").toLowerCase();

  if (value.includes("current")) {
    return "current";
  }

  if (value.includes("static") || value.includes("dynamic")) {
    return "retrieved";
  }

  if (value.includes("topology") || value.includes("reference")) {
    return "topology";
  }

  if (value.includes("case")) {
    return "topology";
  }

  return "other";
}

const NODE_COLOR_BY_KIND: Record<string, string> = {
  current: "#3b82f6",
  retrieved: "#16a34a",
  topology: "#0ea5e9",
  other: "#64748b",
};

function formatNodeKind(kindRaw?: string): string {
  const normalized = normalizeKind(kindRaw);

  if (normalized === "current") {
    return "当前案件";
  }

  if (normalized === "retrieved") {
    return "自动召回";
  }

  if (normalized === "topology") {
    return "拓扑关联";
  }

  return "案例节点";
}

export function TaskLayerGraph({ memoryView }: TaskLayerGraphProps) {
  const model = useMemo(() => {
    if (!memoryView) {
      return null;
    }

    const nodes = memoryView.taskLayerGraph.nodes.map((node, index) => {
      const rawId = String(node.id);

      const summary =
        node.label ||
        memoryView.caseCatalog[rawId]?.summary ||
        `案例 ${index + 1}`;

      return {
        id: rawId,
        label: summary,
        kind: node.kind ?? "case",
        kindGroup: normalizeKind(node.kind),
      };
    });

    const nodeIdSet = new Set(nodes.map((node) => node.id));

    const edges = memoryView.taskLayerGraph.edges
      .map((edge, index) => ({
        id: edge.id || `${edge.source}->${edge.target}#${index}`,
        source: String(edge.source),
        target: String(edge.target),
        type: edge.type ?? "reference",
      }))
      .filter(
        (edge) => nodeIdSet.has(edge.source) && nodeIdSet.has(edge.target),
      );

    return { nodes, edges };
  }, [memoryView]);

  const hasModel = Boolean(model);

  const option = useMemo<EChartsOption | null>(() => {
    if (!model) {
      return null;
    }

    const nodes = model.nodes.map((node) => {
      return {
        id: node.id,
        name: shortText(node.label, 14),
        tooltipTitle: node.label,
        value: `${node.label}\n${formatNodeKind(node.kind)}`,
        symbol: "circle",
        symbolSize: 34,
        itemStyle: {
          color: NODE_COLOR_BY_KIND[node.kindGroup],
          borderColor: "#ffffff",
          borderWidth: 1.5,
          opacity: 1,
        },
        label: {
          show: true,
          color: "#0f172a",
          fontSize: 10,
        },
      };
    });

    const links = model.edges.map((edge) => {
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        lineStyle: {
          color: "#60a5fa",
          width: 2,
          opacity: 0.86,
          curveness: 0.08,
        },
      };
    });

    return {
      backgroundColor: "#f1f5f9",
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
          layout: "force",
          roam: true,
          draggable: true,
          data: nodes,
          links,
          force: {
            repulsion: 480,
            gravity: 0.08,
            edgeLength: [80, 150],
            friction: 0.12,
          },
          edgeSymbol: ["none", "arrow"],
          edgeSymbolSize: 8,
          emphasis: {
            focus: "adjacency",
          },
        },
      ],
    } as EChartsOption;
  }, [model]);

  const { containerRef } = useEChart({
    option,
    enabled: hasModel,
  });

  return (
    <article className="ux-card ux-card-full">
      <h2>TaskLayer 案例关系图</h2>

      {memoryView ? (
        <div className="ux-graph-canvas">
          <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
        </div>
      ) : (
        <p className="ux-empty">暂无记忆图数据。</p>
      )}
    </article>
  );
}
