import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts";
import type { EChartsOption, EChartsType } from "echarts";
import type { MemoryView } from "../../compat";
import { shortText } from "../graph/echarts/debateGraphEcharts";

interface TaskLayerGraphProps {
  memoryView: MemoryView | null;
}

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

const NODE_COLOR_BY_KIND: Record<string, string> = {
  current: "#3b82f6",
  representative: "#16a34a",
  related: "#f59e0b",
  other: "#64748b",
};

export function TaskLayerGraph({ memoryView }: TaskLayerGraphProps) {
  const [selectedId, setSelectedId] = useState<string>("");
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<EChartsType | null>(null);

  const model = useMemo(() => {
    if (!memoryView) {
      return null;
    }

    const nodes = memoryView.taskLayerGraph.nodes.map((node) => ({
      id: String(node.id),
      label: node.label || node.id,
      kind: node.kind ?? "case",
      kindGroup: normalizeKind(node.kind),
    }));

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

  const effectiveSelectedId = useMemo(() => {
    if (!selectedId || !model) {
      return "";
    }

    return model.nodes.some((node) => node.id === selectedId) ? selectedId : "";
  }, [model, selectedId]);

  useEffect(() => {
    const chart = chartRef.current;

    if (!chart || !model) {
      return;
    }

    const selectedNeighborIds = new Set<string>();

    if (effectiveSelectedId) {
      selectedNeighborIds.add(effectiveSelectedId);

      for (const edge of model.edges) {
        if (edge.source === effectiveSelectedId) {
          selectedNeighborIds.add(edge.target);
        } else if (edge.target === effectiveSelectedId) {
          selectedNeighborIds.add(edge.source);
        }
      }
    }

    const focused = selectedNeighborIds.size > 0;

    const nodes = model.nodes.map((node) => {
      const isFocused = selectedNeighborIds.has(node.id);
      const isSelected = node.id === effectiveSelectedId;

      return {
        id: node.id,
        name: shortText(node.label, 14),
        value: `${node.label}\n${node.kind}`,
        symbol: "circle",
        symbolSize: isSelected ? 44 : 34,
        itemStyle: {
          color: NODE_COLOR_BY_KIND[node.kindGroup],
          borderColor: "#ffffff",
          borderWidth: isSelected ? 3 : 1.5,
          opacity: focused ? (isFocused ? 1 : 0.2) : 1,
        },
        label: {
          show: true,
          color: "#0f172a",
          fontSize: 10,
        },
      };
    });

    const links = model.edges.map((edge) => {
      const isFocused =
        effectiveSelectedId &&
        (edge.source === effectiveSelectedId ||
          edge.target === effectiveSelectedId);

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        lineStyle: {
          color: "#60a5fa",
          width: isFocused ? 3 : 2,
          opacity: effectiveSelectedId ? (isFocused ? 1 : 0.14) : 0.86,
          curveness: 0.08,
        },
      };
    });

    const option = {
      backgroundColor: "#f1f5f9",
      tooltip: {
        trigger: "item",
        confine: true,
        formatter: (params: unknown) => {
          const row = params as {
            dataType?: string;
            data?: { value?: string };
          };

          return row.dataType === "node"
            ? String(row.data?.value ?? "").replace(/\n/g, "<br/>")
            : "";
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

    chart.setOption(option, true);
    chart.off("click");

    chart.on("click", (params) => {
      if (params.dataType !== "node") {
        return;
      }

      const payload = params.data as { id?: string } | undefined;
      setSelectedId(String(payload?.id ?? ""));
    });

    const zr = chart.getZr();
    zr.off("click");

    zr.on("click", (event) => {
      if (event.target) {
        return;
      }

      setSelectedId("");
    });
  }, [effectiveSelectedId, model]);

  const selectedNode = model?.nodes.find(
    (item) => item.id === effectiveSelectedId,
  );

  return (
    <article className="ux-card">
      <h2>TaskLayer 案例关系图</h2>

      <p className="ux-muted">
        节点表示案例，边表示引用或相似关系。悬停预览，点击节点固定详情。
      </p>

      {memoryView ? (
        <div className="ux-graph-layout">
          <div className="ux-graph-canvas">
            <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
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
                  <strong>{selectedNode.kind}</strong>
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
