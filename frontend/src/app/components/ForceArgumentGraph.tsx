import { useEffect, useMemo, useRef } from "react";
import * as echarts from "echarts";
import type { EChartsOption, EChartsType } from "echarts";
import type { GraphEdge, GraphNode, GraphView } from "../../compat";

import {
  buildLouvainCommunities,
  shortText,
  toEdgeRelation,
  toNodeFamily,
  type DebateNodeFamily,
} from "../graph/echarts/debateGraphEcharts";

interface ForceArgumentGraphProps {
  graph: GraphView | null;
  title?: string;
}

type NodeLegendCategory =
  | "事实"
  | "原告观点"
  | "核心诉求"
  | "法条"
  | "被告观点";

interface GraphNodeView {
  id: string;
  label: string;
  content: string;
  family: DebateNodeFamily;
  agentId: string;
  community: number;
  category: NodeLegendCategory;
}

interface GraphEdgeView {
  id: string;
  source: string;
  target: string;
  relation: "support" | "attack";
}

interface GraphModel {
  nodes: GraphNodeView[];
  edges: GraphEdgeView[];
  clusterCount: number;
}

const LEGEND_ORDER: NodeLegendCategory[] = [
  "事实",
  "原告观点",
  "核心诉求",
  "法条",
  "被告观点",
];

const CATEGORY_COLOR: Record<NodeLegendCategory, string> = {
  事实: "#14b8a6",
  原告观点: "#2b90e8",
  核心诉求: "#2f855a",
  法条: "#facc15",
  被告观点: "#e11d48",
};

const CATEGORY_SIZE: Record<NodeLegendCategory, number> = {
  事实: 20,
  原告观点: 34,
  核心诉求: 40,
  法条: 16,
  被告观点: 34,
};

function resolveLegendCategory(node: {
  id: string;
  family: DebateNodeFamily;
  label: string;
  content: string;
  agentId: string;
}): NodeLegendCategory {
  if (node.family === "FACT") {
    return "事实";
  }

  if (node.family === "LAW") {
    return "法条";
  }

  const context =
    `${node.id} ${node.label} ${node.content} ${node.agentId}`.toLowerCase();

  if (
    context.includes("system_init") ||
    context.includes("claim_root") ||
    context.includes("root_claim") ||
    context.includes("核心诉求") ||
    context.includes("核心") ||
    context.includes("root")
  ) {
    return "核心诉求";
  }

  if (
    context.includes("defendant") ||
    context.includes("defense") ||
    context.includes("被告")
  ) {
    return "被告观点";
  }

  if (context.includes("plaintiff") || context.includes("原告")) {
    return "原告观点";
  }

  return "原告观点";
}

function mapGraphToModel(graph: GraphView): GraphModel {
  const baseNodes = graph.nodes
    .map((node: GraphNode) => {
      const family = toNodeFamily(node.type);
      const label = (node.label ?? node.id).trim() || String(node.id);
      const content = (node.content ?? label).trim();
      const id = String(node.id);
      const agentId = String(node.agentId ?? "unknown");

      return {
        id,
        label,
        content,
        family,
        agentId,
        category: resolveLegendCategory({
          id,
          family,
          label,
          content,
          agentId,
        }),
      };
    })
    .filter((node) => node.family !== "OTHER");

  const nodeIdSet = new Set(baseNodes.map((node) => node.id));

  const baseEdges = graph.edges
    .map((edge: GraphEdge, index) => {
      const source = String(edge.source);
      const target = String(edge.target);
      const relation = toEdgeRelation(String(edge.type ?? ""));

      return {
        id: String(edge.id || `${source}=>${target}#${index}`),
        source,
        target,
        relation,
      };
    })
    .filter(
      (edge): edge is GraphEdgeView =>
        edge.relation !== "cite" &&
        nodeIdSet.has(edge.source) &&
        nodeIdSet.has(edge.target),
    );

  const communities = buildLouvainCommunities(
    baseNodes.map((node) => node.id),
    baseEdges,
  );

  const nodes: GraphNodeView[] = baseNodes.map((node) => ({
    ...node,
    community: communities.get(node.id) ?? 0,
  }));

  return {
    nodes,
    edges: baseEdges,
    clusterCount: new Set(nodes.map((node) => node.community)).size,
  };
}

export function ForceArgumentGraph({
  graph,
  title = "论证图（ECharts）",
}: ForceArgumentGraphProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<EChartsType | null>(null);
  const model = useMemo(() => (graph ? mapGraphToModel(graph) : null), [graph]);

  useEffect(() => {
    const container = containerRef.current;

    if (!container) {
      return;
    }

    const chart = echarts.init(container);
    chartRef.current = chart;
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(container);

    return () => {
      observer.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;

    if (!chart || !model) {
      return;
    }

    const categoryIndex = new Map(
      LEGEND_ORDER.map((category, idx) => [category, idx]),
    );

    const nodes = model.nodes.map((node) => ({
      id: node.id,
      name: shortText(node.label, 16),
      value: node.content,
      category: categoryIndex.get(node.category) ?? 0,
      symbol: "circle",
      symbolSize: CATEGORY_SIZE[node.category],
      itemStyle: {
        color: CATEGORY_COLOR[node.category],
        borderColor: "#ffffff",
        borderWidth: 1.5,
      },
      label: {
        show: false,
      },
    }));

    const links = model.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      lineStyle: {
        color: edge.relation === "support" ? "#86d694" : "#f39b76",
        type: edge.relation === "support" ? "solid" : "dashed",
        width: 2.6,
        opacity: 0.92,
        curveness: edge.relation === "attack" ? 0.15 : 0.06,
      },
    }));

    const option = {
      backgroundColor: "#ffffff",
      animationDurationUpdate: 420,
      animationEasingUpdate: "quarticInOut",
      tooltip: {
        trigger: "item",
        confine: true,
        backgroundColor: "rgba(30, 41, 59, 0.96)",
        borderWidth: 0,
        textStyle: {
          color: "#f8fafc",
          fontSize: 12,
          lineHeight: 18,
        },
        formatter: (params: unknown) => {
          const row = params as {
            dataType?: string;
            data?: { id?: string; name?: string; value?: string };
          };

          if (row.dataType === "node") {
            const titleText = row.data?.name ?? row.data?.id ?? "";
            return `${titleText}<br/>${String(row.data?.value ?? "").replace(/\n/g, "<br/>")}`;
          }

          return "";
        },
      },
      legend: {
        top: 10,
        left: "center",
        icon: "roundRect",
        itemWidth: 36,
        itemHeight: 18,
        data: LEGEND_ORDER,
        textStyle: {
          color: "#334155",
          fontSize: 24,
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
          categories: LEGEND_ORDER.map((category) => ({
            name: category,
            itemStyle: {
              color: CATEGORY_COLOR[category],
            },
          })),
          force: {
            repulsion: 680,
            gravity: 0.05,
            edgeLength: [120, 260],
            friction: 0.08,
            layoutAnimation: true,
          },
          edgeSymbol: ["none", "none"],
          emphasis: {
            focus: "adjacency",
            lineStyle: {
              width: 3.6,
            },
          },
        },
      ],
    } as EChartsOption;

    chart.setOption(option, true);
  }, [model]);

  return (
    <article className="ux-card">
      <h2>{title}</h2>

      <p className="ux-muted">
        悬停即可查看节点全文；图中按角色语义着色并通过力导布局自然形成聚类。
      </p>

      <div className="ux-graph-legend">
        <span>节点图例：事实 / 原告观点 / 核心诉求 / 法条 / 被告观点</span>
        <span>边关系：support 绿色实线，attack 橙色虚线</span>

        <span>
          当前可见：{model?.nodes.length ?? 0} 节点 / {model?.edges.length ?? 0}{" "}
          条边 /{` ${model?.clusterCount ?? 0} 个聚类`}
        </span>
      </div>

      {model ? (
        <div className="ux-graph-canvas">
          <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
        </div>
      ) : (
        <p className="ux-empty">当前暂无图谱数据。</p>
      )}
    </article>
  );
}
