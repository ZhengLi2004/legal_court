import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts";
import type { EChartsOption, EChartsType } from "echarts";

import {
  buildLouvainCommunities,
  shortText,
  toEdgeRelation,
  toNodeFamily,
  type DebateNodeFamily,
  type DebateEdgeRelation,
} from "../../app/graph/echarts/debateGraphEcharts";

import { renderScrollableNodeTooltip } from "../../app/graph/echarts/tooltip";
import type { GraphDiffView, GraphView, TurnArtifact } from "../../compat";
type FocusMode = "all" | "changed" | "changedNeighbors" | "rejected";
type ChainMode = "all" | "support" | "conflict";

interface GraphDiffPanelProps {
  currentGraph: GraphView | null;
  baselineGraph: GraphView | null;
  diff: GraphDiffView | null;
  artifacts: TurnArtifact[];
  title?: string;
}

interface DiffNodeView {
  id: string;
  label: string;
  content: string;
  family: DebateNodeFamily;
  status: string;
  agentId: string;
  isAdded: boolean;
  isRejected: boolean;
  isStatusChanged: boolean;
  isReused: boolean;
  isChain: boolean;
  isAnchor: boolean;
}

interface DiffEdgeView {
  id: string;
  source: string;
  target: string;
  relation: DebateEdgeRelation;
  isAdded: boolean;
  isChain: boolean;
  isMuted: boolean;
}

interface DiffModel {
  nodes: DiffNodeView[];
  edges: DiffEdgeView[];
  claimIds: string[];
  resolvedClaimId: string;

  summary: {
    addedNodes: number;
    addedEdges: number;
    statusChanged: number;
    reused: number;
    rejected: number;
    visibleNodes: number;
    visibleEdges: number;
  };
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function asString(value: unknown, fallback = ""): string {
  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }

  return fallback;
}

function shortId(id: string): string {
  if (!id) {
    return "N/A";
  }

  if (id.length <= 20) {
    return id;
  }

  return `${id.slice(0, 14)}...${id.slice(-3)}`;
}

function gatherRejectedNodeIds(artifacts: TurnArtifact[]): Set<string> {
  const rejected = new Set<string>();

  for (const artifact of artifacts) {
    const logs = artifact.executionLogs.toLowerCase();

    if (
      !(
        logs.includes("reject") ||
        logs.includes("failed") ||
        logs.includes("error") ||
        logs.includes("rollback")
      )
    ) {
      continue;
    }

    const actions = Array.isArray(artifact.parsedActions)
      ? artifact.parsedActions
      : [];

    for (const action of actions) {
      const row = asRecord(action);
      const candidates = [
        row.node_id,
        row.nodeId,
        row.source_id,
        row.sourceId,
        row.target_id,
        row.targetId,
        row.source,
        row.target,
      ];

      for (const item of candidates) {
        const nodeId = asString(item);

        if (nodeId) {
          rejected.add(nodeId);
        }
      }
    }
  }

  return rejected;
}

function buildUndirectedAdjacency(
  edges: Array<{ source: string; target: string }>,
): Map<string, Set<string>> {
  const adjacency = new Map<string, Set<string>>();

  for (const edge of edges) {
    if (!adjacency.has(edge.source)) {
      adjacency.set(edge.source, new Set<string>());
    }

    if (!adjacency.has(edge.target)) {
      adjacency.set(edge.target, new Set<string>());
    }

    adjacency.get(edge.source)?.add(edge.target);
    adjacency.get(edge.target)?.add(edge.source);
  }

  return adjacency;
}

function collectWithinHops(
  roots: Set<string>,
  adjacency: Map<string, Set<string>>,
  hops: number,
): Set<string> {
  const visited = new Set<string>(roots);
  let frontier = new Set<string>(roots);

  for (let depth = 0; depth < hops; depth += 1) {
    const next = new Set<string>();

    for (const nodeId of frontier) {
      for (const neighbor of adjacency.get(nodeId) ?? []) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          next.add(neighbor);
        }
      }
    }

    frontier = next;

    if (!frontier.size) {
      break;
    }
  }

  return visited;
}

function collectClaimChain(
  startClaimId: string,
  edges: Array<{
    id: string;
    source: string;
    target: string;
    relation: DebateEdgeRelation;
  }>,
  hops: number,
  chainMode: ChainMode,
): { nodeIds: Set<string>; edgeIds: Set<string> } {
  const nodeIds = new Set<string>([startClaimId]);
  const edgeIds = new Set<string>();
  let frontier = new Set<string>([startClaimId]);

  for (let depth = 0; depth < hops; depth += 1) {
    const next = new Set<string>();

    for (const edge of edges) {
      if (chainMode === "support" && edge.relation !== "support") {
        continue;
      }

      if (chainMode === "conflict" && edge.relation !== "attack") {
        continue;
      }

      const touchSource = frontier.has(edge.source);
      const touchTarget = frontier.has(edge.target);

      if (!touchSource && !touchTarget) {
        continue;
      }

      edgeIds.add(edge.id);

      if (!nodeIds.has(edge.source)) {
        nodeIds.add(edge.source);
        next.add(edge.source);
      }

      if (!nodeIds.has(edge.target)) {
        nodeIds.add(edge.target);
        next.add(edge.target);
      }
    }

    frontier = next;

    if (!frontier.size) {
      break;
    }
  }

  return { nodeIds, edgeIds };
}

function nodeColorByFamily(family: DebateNodeFamily): string {
  if (family === "FACT") {
    return "#14b8a6";
  }

  if (family === "LAW") {
    return "#facc15";
  }

  if (family === "CLAIM") {
    return "#2563eb";
  }

  return "#94a3b8";
}

export function GraphDiffPanel({
  currentGraph,
  baselineGraph,
  diff,
  artifacts,
  title = "Graph Diff Highlighter",
}: GraphDiffPanelProps) {
  const [focusMode, setFocusMode] = useState<FocusMode>("changedNeighbors");
  const [chainMode, setChainMode] = useState<ChainMode>("all");
  const [selectedClaimId, setSelectedClaimId] = useState<string>("auto");
  const [selectedNodeId, setSelectedNodeId] = useState<string>("");
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<EChartsType | null>(null);

  const model = useMemo<DiffModel | null>(() => {
    if (!currentGraph) {
      return null;
    }

    const mappedNodes = currentGraph.nodes
      .map((node) => ({
        id: asString(node.id),
        label: asString(node.label, asString(node.id)),
        content: asString(
          node.content,
          asString(node.label, asString(node.id)),
        ),
        family: toNodeFamily(asString(node.type)),
        status: asString(node.status, "HYPOTHETICAL"),
        agentId: asString(node.agentId, "unknown"),
      }))
      .filter((node) => node.family !== "OTHER");

    const nodeIdSet = new Set(mappedNodes.map((node) => node.id));

    const mappedEdges = currentGraph.edges
      .map((edge, index) => {
        const source = asString(edge.source);
        const target = asString(edge.target);
        const relation = toEdgeRelation(asString(edge.type));

        return {
          id: asString(edge.id, `${source}->${target}#${index}`),
          source,
          target,
          relation,
        };
      })
      .filter(
        (edge) =>
          edge.relation !== "cite" &&
          nodeIdSet.has(edge.source) &&
          nodeIdSet.has(edge.target),
      );

    const previousNodes = new Map(
      (baselineGraph?.nodes ?? []).map((node) => [asString(node.id), node]),
    );

    const currentNodeIds = new Set(mappedNodes.map((node) => node.id));

    const previousNodeIds = new Set(
      (baselineGraph?.nodes ?? []).map((n) => asString(n.id)),
    );

    const addedNodeIds = new Set(
      (diff?.addedNodeIds ?? []).map((id) => asString(id)),
    );

    const addedEdgeIds = new Set(
      (diff?.addedEdgeIds ?? []).map((id) => asString(id)),
    );

    const statusChangedNodeIds = new Set<string>();

    for (const node of mappedNodes) {
      const prev = previousNodes.get(node.id);

      if (prev && asString(prev.status) !== node.status) {
        statusChangedNodeIds.add(node.id);
      }
    }

    const reusedNodeIds = new Set<string>();

    for (const nodeId of currentNodeIds) {
      if (previousNodeIds.has(nodeId) && !statusChangedNodeIds.has(nodeId)) {
        reusedNodeIds.add(nodeId);
      }
    }

    const rejectedNodeIds = gatherRejectedNodeIds(artifacts);

    const changedNodeIds = new Set<string>([
      ...addedNodeIds,
      ...statusChangedNodeIds,
      ...rejectedNodeIds,
    ]);

    const claimIds = mappedNodes
      .filter((node) => node.family === "CLAIM")
      .map((node) => node.id)
      .sort();

    const adjacency = buildUndirectedAdjacency(mappedEdges);
    let baseVisibleIds = new Set<string>(currentNodeIds);

    if (focusMode === "changed") {
      baseVisibleIds = new Set(changedNodeIds);
    } else if (focusMode === "changedNeighbors") {
      baseVisibleIds = collectWithinHops(changedNodeIds, adjacency, 1);
    } else if (focusMode === "rejected") {
      baseVisibleIds = new Set(rejectedNodeIds);
    }

    if (!baseVisibleIds.size) {
      baseVisibleIds =
        claimIds.length > 0
          ? new Set(claimIds.slice(0, 10))
          : new Set(currentNodeIds);
    }

    const autoClaimFromChanged = claimIds.find((id) => changedNodeIds.has(id));

    const resolvedClaimId =
      selectedClaimId === "none"
        ? ""
        : selectedClaimId === "auto"
          ? (autoClaimFromChanged ?? claimIds[0] ?? "")
          : selectedClaimId;

    const chain = resolvedClaimId
      ? collectClaimChain(resolvedClaimId, mappedEdges, 3, chainMode)
      : { nodeIds: new Set<string>(), edgeIds: new Set<string>() };

    const visibleNodeIds = new Set<string>(baseVisibleIds);

    for (const nodeId of chain.nodeIds) {
      visibleNodeIds.add(nodeId);
    }

    let visibleEdges = mappedEdges.filter(
      (edge) =>
        visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target),
    );

    if (resolvedClaimId && chainMode !== "all") {
      visibleEdges = visibleEdges.filter((edge) => chain.edgeIds.has(edge.id));
    }

    const nodes: DiffNodeView[] = mappedNodes
      .filter((node) => visibleNodeIds.has(node.id))
      .map((node) => ({
        ...node,
        isAdded: addedNodeIds.has(node.id),
        isRejected: rejectedNodeIds.has(node.id),
        isStatusChanged: statusChangedNodeIds.has(node.id),
        isReused: reusedNodeIds.has(node.id),
        isChain: chain.nodeIds.has(node.id),
        isAnchor: resolvedClaimId !== "" && node.id === resolvedClaimId,
      }));

    const edges: DiffEdgeView[] = visibleEdges.map((edge) => ({
      ...edge,
      isAdded: addedEdgeIds.has(edge.id),
      isChain: chain.edgeIds.has(edge.id),
      isMuted:
        resolvedClaimId !== "" &&
        focusMode !== "all" &&
        !chain.edgeIds.has(edge.id),
    }));

    return {
      nodes,
      edges,
      claimIds,
      resolvedClaimId,
      summary: {
        addedNodes: addedNodeIds.size,
        addedEdges: addedEdgeIds.size,
        statusChanged: statusChangedNodeIds.size,
        reused: reusedNodeIds.size,
        rejected: rejectedNodeIds.size,
        visibleNodes: nodes.length,
        visibleEdges: edges.length,
      },
    };
  }, [
    artifacts,
    baselineGraph,
    chainMode,
    currentGraph,
    diff,
    focusMode,
    selectedClaimId,
  ]);

  const effectiveSelectedNodeId = useMemo(() => {
    if (!selectedNodeId || !model) {
      return "";
    }

    return model.nodes.some((node) => node.id === selectedNodeId)
      ? selectedNodeId
      : "";
  }, [model, selectedNodeId]);

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

    const selectedNeighborIds = new Set<string>();

    if (effectiveSelectedNodeId) {
      selectedNeighborIds.add(effectiveSelectedNodeId);

      for (const edge of model.edges) {
        if (edge.source === effectiveSelectedNodeId) {
          selectedNeighborIds.add(edge.target);
        } else if (edge.target === effectiveSelectedNodeId) {
          selectedNeighborIds.add(edge.source);
        }
      }
    }

    const focused = selectedNeighborIds.size > 0;

    const communities = buildLouvainCommunities(
      model.nodes.map((node) => node.id),
      model.edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
      })),
    );

    const nodes = model.nodes.map((node) => {
      const selected = node.id === effectiveSelectedNodeId;
      const neighbor = selectedNeighborIds.has(node.id);
      const opacity = focused ? (neighbor ? 1 : 0.18) : 1;
      let borderColor = "#64748b";
      let borderWidth = selected ? 3 : 2;
      let borderType: "solid" | "dashed" = "solid";

      if (node.isAdded) {
        borderColor = "#16a34a";
        borderWidth = 3;
      } else if (node.isRejected) {
        borderColor = "#e11d48";
        borderWidth = 3;
      } else if (node.isStatusChanged) {
        borderColor = "#2563eb";
        borderWidth = 3;
      } else if (node.isReused) {
        borderColor = "#0ea5e9";
        borderType = "dashed";
      }

      return {
        id: node.id,
        name: shortText(node.label, 16),
        tooltipTitle: node.label,
        value: node.content,
        category: communities.get(node.id) ?? 0,
        symbol: "circle",
        symbolSize: 20 + (node.isChain ? 6 : 0) + (selected ? 6 : 0),
        itemStyle: {
          color: nodeColorByFamily(node.family),
          borderColor,
          borderWidth,
          borderType,
          opacity,
          shadowBlur: node.isAnchor ? 18 : 0,
          shadowColor: node.isAnchor ? "rgba(2,132,199,0.28)" : "transparent",
        },
        label: {
          show: false,
        },
      };
    });

    const links = model.edges.map((edge) => {
      const edgeSelected =
        effectiveSelectedNodeId &&
        (edge.source === effectiveSelectedNodeId ||
          edge.target === effectiveSelectedNodeId);

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        lineStyle: {
          color: edge.relation === "support" ? "#86d694" : "#f39b76",
          type: edge.relation === "support" ? "solid" : "dashed",
          width: edge.isChain ? 3.2 : edge.isAdded ? 2.8 : edgeSelected ? 3 : 2,
          opacity: edge.isMuted
            ? 0.2
            : effectiveSelectedNodeId
              ? edgeSelected
                ? 1
                : 0.16
              : 0.9,
          curveness: edge.relation === "attack" ? 0.12 : 0.06,
        },
      };
    });

    const option = {
      backgroundColor: "#e5e7eb",
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
            repulsion: 520,
            gravity: 0.06,
            edgeLength: [95, 210],
            friction: 0.12,
          },
          emphasis: {
            focus: "adjacency",
          },
          edgeSymbol: ["none", "none"],
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
      setSelectedNodeId(asString(payload?.id));
    });

    const zr = chart.getZr();
    zr.off("click");

    zr.on("click", (event) => {
      if (event.target) {
        return;
      }

      setSelectedNodeId("");
    });
  }, [effectiveSelectedNodeId, model]);

  return (
    <article className="card wide">
      <h2>{title}</h2>

      <div className="sub-actions">
        <button type="button" onClick={() => setFocusMode("all")}>
          All
        </button>

        <button type="button" onClick={() => setFocusMode("changed")}>
          Changed
        </button>

        <button type="button" onClick={() => setFocusMode("changedNeighbors")}>
          Changed + 1-hop
        </button>

        <button type="button" onClick={() => setFocusMode("rejected")}>
          Rejected
        </button>
      </div>

      {model ? (
        <>
          <div className="graph-toolbar">
            <label className="graph-control">
              Claim Focus
              <select
                value={selectedClaimId}
                onChange={(event) => setSelectedClaimId(event.target.value)}
              >
                <option value="auto">Auto</option>
                <option value="none">None</option>

                {model.claimIds.map((id) => (
                  <option key={id} value={id}>
                    {shortId(id)}
                  </option>
                ))}
              </select>
            </label>

            <label className="graph-control">
              Chain
              <select
                value={chainMode}
                onChange={(event) =>
                  setChainMode(event.target.value as ChainMode)
                }
              >
                <option value="all">All</option>
                <option value="support">Support</option>
                <option value="conflict">Conflict</option>
              </select>
            </label>
          </div>

          <p className="line">
            visible N{model.summary.visibleNodes} E{model.summary.visibleEdges}{" "}
            | +N
            {model.summary.addedNodes} +E{model.summary.addedEdges} | status
            changes {model.summary.statusChanged} | reused{" "}
            {model.summary.reused} | rejected {model.summary.rejected}
            {model.resolvedClaimId
              ? ` | anchor ${shortId(model.resolvedClaimId)}`
              : ""}
            {effectiveSelectedNodeId
              ? ` | selected ${shortId(effectiveSelectedNodeId)}`
              : ""}
          </p>

          <div className="graph-canvas">
            <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
          </div>
        </>
      ) : (
        <p className="hint">Load graph and diff first.</p>
      )}
    </article>
  );
}
