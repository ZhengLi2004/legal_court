import { useEffect, useMemo, useRef, useState } from "react";
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

import { renderScrollableNodeTooltip } from "../graph/echarts/tooltip";

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

type EdgeLegendCategory = "support" | "attack";

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
  relation: EdgeLegendCategory;
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

const EDGE_LEGEND_ORDER: EdgeLegendCategory[] = ["support", "attack"];

const EDGE_LEGEND_LABEL: Record<EdgeLegendCategory, string> = {
  support: "支持边",
  attack: "冲突边",
};

const CATEGORY_COLOR: Record<NodeLegendCategory, string> = {
  事实: "#14b8a6",
  原告观点: "#2b90e8",
  核心诉求: "#2f855a",
  法条: "#facc15",
  被告观点: "#e11d48",
};

const CATEGORY_SIZE: Record<NodeLegendCategory, number> = {
  事实: 28,
  原告观点: 44,
  核心诉求: 52,
  法条: 24,
  被告观点: 44,
};

function resolveLegendCategory(node: {
  id: string;
  family: DebateNodeFamily;
  label: string;
  content: string;
  agentId: string;
  metadata?: Record<string, unknown>;
}): NodeLegendCategory {
  if (node.family === "FACT") {
    return "事实";
  }

  if (node.family === "LAW") {
    return "法条";
  }

  const agent = node.agentId.toLowerCase();
  const idAndLabel = `${node.id} ${node.label}`.toLowerCase();
  const contentHead = node.content.trim().slice(0, 24).toLowerCase();
  const isRootClaim = node.metadata?.is_root_claim === true;

  if (isRootClaim) {
    return "核心诉求";
  }

  if (agent.includes("plaintiff") || agent.includes("原告")) {
    return "原告观点";
  }

  if (agent.includes("defendant") || agent.includes("被告")) {
    return "被告观点";
  }

  if (
    idAndLabel.includes("system_init") ||
    idAndLabel.includes("claim_root") ||
    idAndLabel.includes("root_claim") ||
    idAndLabel.includes("核心诉求") ||
    idAndLabel.includes("核心") ||
    idAndLabel.includes("root") ||
    agent.includes("system")
  ) {
    return "核心诉求";
  }

  if (idAndLabel.includes("plaintiff") || idAndLabel.includes("原告")) {
    return "原告观点";
  }

  if (
    idAndLabel.includes("defendant") ||
    idAndLabel.includes("defense") ||
    idAndLabel.includes("被告")
  ) {
    return "被告观点";
  }

  if (contentHead.startsWith("原告") || contentHead.startsWith("plaintiff")) {
    return "原告观点";
  }

  if (contentHead.startsWith("被告") || contentHead.startsWith("defendant")) {
    return "被告观点";
  }

  return "核心诉求";
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
          metadata: node.metadata,
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

  const [activeCategories, setActiveCategories] = useState<
    Set<NodeLegendCategory>
  >(() => new Set<NodeLegendCategory>(LEGEND_ORDER));

  const [activeRelations, setActiveRelations] = useState<
    Set<EdgeLegendCategory>
  >(() => new Set<EdgeLegendCategory>(EDGE_LEGEND_ORDER));

  const [focusOnlyEnabled, setFocusOnlyEnabled] = useState<boolean>(false);

  const focusNodeIds = useMemo(
    () => new Set((graph?.focusNodeIds ?? []).map((item) => String(item))),
    [graph],
  );

  const focusNodeCount = useMemo(() => {
    if (!model) {
      return 0;
    }

    return model.nodes.filter((node) => focusNodeIds.has(node.id)).length;
  }, [focusNodeIds, model]);

  const effectiveFocusOnly = focusOnlyEnabled && focusNodeCount > 0;

  const visibleStats = useMemo(() => {
    if (!model) {
      return { nodes: 0, edges: 0 };
    }

    if (!effectiveFocusOnly) {
      return { nodes: model.nodes.length, edges: model.edges.length };
    }

    const nodeIds = new Set(
      model.nodes
        .filter((node) => focusNodeIds.has(node.id))
        .map((node) => node.id),
    );

    const edges = model.edges.filter(
      (edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target),
    );

    return {
      nodes: nodeIds.size,
      edges: edges.length,
    };
  }, [effectiveFocusOnly, focusNodeIds, model]);

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

    const focusVisibleNodeIds = effectiveFocusOnly
      ? new Set(
          model.nodes
            .filter((node) => focusNodeIds.has(node.id))
            .map((node) => node.id),
        )
      : null;

    const renderedNodes = focusVisibleNodeIds
      ? model.nodes.filter((node) => focusVisibleNodeIds.has(node.id))
      : model.nodes;

    const renderedNodeIds = new Set(renderedNodes.map((node) => node.id));
    const nodeOpacityMap = new Map<string, number>();

    const nodes = renderedNodes.map((node) => {
      let fadeFactor = 1;

      if (!activeCategories.has(node.category)) {
        fadeFactor *= 0.22;
      }

      const opacity = Math.max(fadeFactor, 0.08);
      nodeOpacityMap.set(node.id, opacity);
      const isFocusNode = focusNodeIds.has(node.id);

      return {
        id: node.id,
        name: shortText(node.label, 16),
        tooltipTitle: node.label,
        value: node.content,
        category: categoryIndex.get(node.category) ?? 0,
        symbol: "circle",
        symbolSize: CATEGORY_SIZE[node.category],
        itemStyle: {
          color: CATEGORY_COLOR[node.category],
          borderColor: isFocusNode ? "#0f172a" : "#ffffff",
          borderWidth: isFocusNode ? 2.2 : 1.5,
          opacity,
        },
        label: {
          show: false,
        },
      };
    });

    const links = model.edges
      .filter(
        (edge) =>
          renderedNodeIds.has(edge.source) && renderedNodeIds.has(edge.target),
      )
      .map((edge) => {
        let fadeFactor = 1;

        if (!activeRelations.has(edge.relation)) {
          fadeFactor *= 0.2;
        }

        fadeFactor *= Math.min(
          nodeOpacityMap.get(edge.source) ?? 1,
          nodeOpacityMap.get(edge.target) ?? 1,
        );

        return {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          lineStyle: {
            color: edge.relation === "support" ? "#86d694" : "#f39b76",
            type: edge.relation === "support" ? "solid" : "dashed",
            width: 2.6,
            opacity: Math.max(0.06, 0.92 * fadeFactor),
            curveness: edge.relation === "attack" ? 0.15 : 0.06,
          },
        };
      });

    const option = {
      backgroundColor: "#ffffff",
      animationDurationUpdate: 420,
      animationEasingUpdate: "quarticInOut",
      tooltip: {
        trigger: "item",
        confine: true,
        enterable: true,
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

            data?: {
              id?: string;
              name?: string;
              tooltipTitle?: string;
              value?: string;
            };
          };

          if (row.dataType === "node") {
            const titleText =
              row.data?.tooltipTitle ?? row.data?.name ?? row.data?.id ?? "";

            return renderScrollableNodeTooltip(
              String(titleText),
              String(row.data?.value ?? ""),
            );
          }

          return "";
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
            repulsion: 760,
            gravity: 0.05,
            edgeLength: [130, 280],
            friction: 0.08,
            layoutAnimation: true,
          },
          edgeSymbol: ["none", "arrow"],
          edgeSymbolSize: 8,
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
  }, [
    activeCategories,
    activeRelations,
    effectiveFocusOnly,
    focusNodeIds,
    model,
  ]);

  return (
    <article className="ux-card">
      <h2>{title}</h2>

      <div className="ux-row">
        <button
          disabled={focusNodeCount === 0}
          onClick={() => setFocusOnlyEnabled((prev) => !prev)}
          type="button"
        >
          {effectiveFocusOnly ? "显示全部节点" : "仅显示 Focus 节点"}
        </button>

        <span className="ux-chip">
          Focus：{focusNodeCount}/{model?.nodes.length ?? 0}
        </span>
      </div>

      <div className="ux-graph-legend">
        <span>节点图例（点击筛选）</span>
        <span>边图例（点击筛选）</span>

        <span>
          当前可见：{visibleStats.nodes} 节点 / {visibleStats.edges} 条边 /
          {` ${model?.clusterCount ?? 0} 个聚类`}
        </span>
      </div>

      <div className="ux-chip-row">
        {LEGEND_ORDER.map((category) => (
          <button
            className={`ux-chip ${activeCategories.has(category) ? "ux-chip-active" : ""}`}
            key={category}
            onClick={() => {
              setActiveCategories((prev) => {
                const next = new Set(prev);

                if (next.has(category)) {
                  next.delete(category);
                } else {
                  next.add(category);
                }

                return next;
              });
            }}
            type="button"
          >
            {category}
          </button>
        ))}
      </div>

      <div className="ux-chip-row">
        {EDGE_LEGEND_ORDER.map((relation) => (
          <button
            className={`ux-chip ${activeRelations.has(relation) ? "ux-chip-active" : ""}`}
            key={relation}
            onClick={() => {
              setActiveRelations((prev) => {
                const next = new Set(prev);

                if (next.has(relation)) {
                  next.delete(relation);
                } else {
                  next.add(relation);
                }

                return next;
              });
            }}
            type="button"
          >
            {EDGE_LEGEND_LABEL[relation]}
          </button>
        ))}
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
