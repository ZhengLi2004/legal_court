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

import type { GraphDiffView, GraphView, TurnArtifact } from "../../compat";
type FocusMode = "all" | "changed" | "changedNeighbors" | "rejected";
type ChainMode = "all" | "support" | "conflict";
type LayoutMode = "distanceFlow" | "typeLanes";
type Lane = "FACT" | "LAW" | "CLAIM" | "OTHER";
const LANE_ORDER: Lane[] = ["FACT", "LAW", "CLAIM", "OTHER"];

const LANE_Y: Record<Lane, number> = {
  FACT: 40,
  LAW: 220,
  CLAIM: 400,
  OTHER: 580,
};

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function normalizeEdgeType(raw: string): string {
  const upper = raw.toUpperCase();

  if (upper === "ATTACK") {
    return "CONFLICT";
  }

  return upper;
}

function laneForType(rawType: string): Lane {
  const upper = rawType.toUpperCase();

  if (upper === "FACT") {
    return "FACT";
  }

  if (upper === "LAW") {
    return "LAW";
  }

  if (upper === "CLAIM") {
    return "CLAIM";
  }

  return "OTHER";
}

function shortId(id: string): string {
  if (!id) {
    return "N/A";
  }

  if (id.length <= 16) {
    return id;
  }

  return `${id.slice(0, 12)}...${id.slice(-3)}`;
}

function truncate(text: string, maxLen = 24): string {
  if (text.length <= maxLen) {
    return text;
  }

  return `${text.slice(0, maxLen)}...`;
}

function gatherRejectedNodeIds(artifacts: TurnArtifact[]): Set<string> {
  const ids = new Set<string>();

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
        const id = asString(item);

        if (id) {
          ids.add(id);
        }
      }
    }
  }

  return ids;
}

function buildUndirectedAdjacency(
  edges: GraphView["edges"],
): Map<string, Set<string>> {
  const map = new Map<string, Set<string>>();

  for (const edge of edges) {
    if (!map.has(edge.source)) {
      map.set(edge.source, new Set<string>());
    }

    if (!map.has(edge.target)) {
      map.set(edge.target, new Set<string>());
    }

    map.get(edge.source)?.add(edge.target);
    map.get(edge.target)?.add(edge.source);
  }

  return map;
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

function buildDistanceMap(
  roots: string[],
  adjacency: Map<string, Set<string>>,
): Map<string, number> {
  const dist = new Map<string, number>();

  if (!roots.length) {
    return dist;
  }

  const queue: string[] = [];

  for (const root of roots) {
    if (!dist.has(root)) {
      dist.set(root, 0);
      queue.push(root);
    }
  }

  while (queue.length > 0) {
    const current = queue.shift() as string;
    const currentDist = dist.get(current) ?? 0;

    for (const neighbor of adjacency.get(current) ?? []) {
      if (!dist.has(neighbor)) {
        dist.set(neighbor, currentDist + 1);
        queue.push(neighbor);
      }
    }
  }

  return dist;
}

function collectClaimChain(
  startClaimId: string,
  edges: GraphView["edges"],
  hops: number,
  chainMode: ChainMode,
): { nodeIds: Set<string>; edgeIds: Set<string> } {
  const nodeIds = new Set<string>([startClaimId]);
  const edgeIds = new Set<string>();
  let frontier = new Set<string>([startClaimId]);

  for (let depth = 0; depth < hops; depth += 1) {
    const next = new Set<string>();

    for (const edge of edges) {
      const edgeType = normalizeEdgeType(edge.type);

      if (chainMode === "support" && edgeType !== "SUPPORT") {
        continue;
      }

      if (chainMode === "conflict" && edgeType !== "CONFLICT") {
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

interface GraphDiffPanelProps {
  currentGraph: GraphView | null;
  baselineGraph: GraphView | null;
  diff: GraphDiffView | null;
  artifacts: TurnArtifact[];
  title?: string;
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
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("distanceFlow");
  const [hopDepth, setHopDepth] = useState<number>(2);
  const [selectedClaimId, setSelectedClaimId] = useState<string>("auto");

  const model = useMemo(() => {
    if (!currentGraph) {
      return null;
    }

    const previousNodes = new Map(
      (baselineGraph?.nodes ?? []).map((node) => [node.id, node]),
    );

    const currentNodeIds = new Set(currentGraph.nodes.map((node) => node.id));

    const previousNodeIds = new Set(
      (baselineGraph?.nodes ?? []).map((node) => node.id),
    );

    const addedNodeIds = new Set(diff?.addedNodeIds ?? []);
    const addedEdgeIds = new Set(diff?.addedEdgeIds ?? []);
    const statusChangedNodeIds = new Set<string>();

    for (const node of currentGraph.nodes) {
      const prev = previousNodes.get(node.id);

      if (prev && (prev.status ?? "") !== (node.status ?? "")) {
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

    const claimIds = currentGraph.nodes
      .filter((node) => laneForType(node.type) === "CLAIM")
      .map((node) => node.id)
      .sort();

    const adjacency = buildUndirectedAdjacency(currentGraph.edges);

    let baseVisibleIds = new Set<string>(currentNodeIds);

    if (focusMode === "changed") {
      baseVisibleIds = new Set(changedNodeIds);
    } else if (focusMode === "changedNeighbors") {
      baseVisibleIds = collectWithinHops(changedNodeIds, adjacency, 1);
    } else if (focusMode === "rejected") {
      baseVisibleIds = new Set(rejectedNodeIds);
    }

    if (!baseVisibleIds.size) {
      if (claimIds.length > 0) {
        baseVisibleIds = new Set(claimIds.slice(0, 12));
      } else {
        baseVisibleIds = new Set(currentNodeIds);
      }
    }

    const autoClaimFromChanged = claimIds.find((id) => changedNodeIds.has(id));

    const resolvedClaimId =
      selectedClaimId === "none"
        ? ""
        : selectedClaimId === "auto"
          ? (autoClaimFromChanged ?? claimIds[0] ?? "")
          : selectedClaimId;

    const chain = resolvedClaimId
      ? collectClaimChain(
          resolvedClaimId,
          currentGraph.edges,
          Math.max(1, hopDepth),
          chainMode,
        )
      : { nodeIds: new Set<string>(), edgeIds: new Set<string>() };

    const visibleNodeIds = new Set<string>(baseVisibleIds);

    if (resolvedClaimId) {
      for (const id of chain.nodeIds) {
        visibleNodeIds.add(id);
      }
    }

    const filteredNodes = currentGraph.nodes.filter((node) =>
      visibleNodeIds.has(node.id),
    );

    const filteredEdges = currentGraph.edges.filter(
      (edge) =>
        visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target),
    );

    const typeFilteredEdges =
      chainMode === "all" || !resolvedClaimId
        ? filteredEdges
        : filteredEdges.filter((edge) => chain.edgeIds.has(edge.id));

    const degree = new Map<string, number>();

    for (const edge of typeFilteredEdges) {
      degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1);
      degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1);
    }

    const distanceRoots = resolvedClaimId
      ? [resolvedClaimId]
      : [...changedNodeIds].slice(0, 6);

    const distances = buildDistanceMap(distanceRoots, adjacency);
    const lanes = new Map<Lane, GraphView["nodes"]>();

    for (const lane of LANE_ORDER) {
      lanes.set(lane, []);
    }

    for (const node of filteredNodes) {
      const lane = laneForType(node.type);
      (lanes.get(lane) ?? []).push(node);
    }

    const positionedNodes: Node[] = [];
    const decorateNode = (
      node: GraphView["nodes"][number],
      x: number,
      y: number,
    ): void => {
      const isAdded = addedNodeIds.has(node.id);
      const isRejected = rejectedNodeIds.has(node.id);
      const isChanged = statusChangedNodeIds.has(node.id);
      const isReused = reusedNodeIds.has(node.id);
      const lane = laneForType(node.type);
      const inChain = chain.nodeIds.has(node.id);
      let borderColor = "#64748b";
      let background = "#f8fafc";
      let borderStyle: "solid" | "dashed" = "solid";

      if (isAdded) {
        borderColor = "#16a34a";
        background = "#dcfce7";
      } else if (isRejected) {
        borderColor = "#e11d48";
        background = "#ffe4e6";
      } else if (isChanged) {
        borderColor = "#2563eb";
        background = "#dbeafe";
      } else if (isReused) {
        borderColor = "#0ea5e9";
        borderStyle = "dashed";
        background = "#ecfeff";
      }

      const tags = [
        isAdded ? "Added" : "",
        isChanged ? "Status" : "",
        isRejected ? "Rejected" : "",
      ]
        .filter(Boolean)
        .join(" • ");

      const densityOpacity =
        resolvedClaimId && !inChain && focusMode !== "all" ? 0.48 : 1;

      positionedNodes.push({
        id: node.id,
        position: { x, y },
        data: {
          label: (
            <div>
              <strong>{shortId(node.id)}</strong>

              <div style={{ fontSize: "0.74rem", marginTop: 2 }}>
                {lane} {node.status ? `• ${node.status}` : ""}
              </div>

              <div style={{ fontSize: "0.72rem", marginTop: 3, opacity: 0.86 }}>
                {truncate(node.label || "")}
              </div>

              {tags ? (
                <div style={{ fontSize: "0.7rem", marginTop: 3 }}>{tags}</div>
              ) : null}
            </div>
          ),
        },
        style: {
          border: `2px ${borderStyle} ${borderColor}`,
          borderRadius: 10,
          padding: 6,
          width: 175,
          background,
          color: "#0f172a",
          opacity: densityOpacity,
        },
      });
    };

    if (layoutMode === "distanceFlow") {
      const laneSlots = new Map<string, number>();

      const maxDistance = Math.max(
        0,
        ...filteredNodes.map((node) => distances.get(node.id) ?? 0),
      );

      const orderedNodes = [...filteredNodes].sort((a, b) => {
        const da = distances.get(a.id) ?? maxDistance + 1;
        const db = distances.get(b.id) ?? maxDistance + 1;

        if (da !== db) {
          return da - db;
        }

        const la = LANE_ORDER.indexOf(laneForType(a.type));
        const lb = LANE_ORDER.indexOf(laneForType(b.type));

        if (la !== lb) {
          return la - lb;
        }

        const ga = degree.get(a.id) ?? 0;
        const gb = degree.get(b.id) ?? 0;

        if (ga !== gb) {
          return gb - ga;
        }

        return a.id.localeCompare(b.id);
      });

      for (const node of orderedNodes) {
        const lane = laneForType(node.type);
        const laneOrder = Math.max(0, LANE_ORDER.indexOf(lane));
        const dist = distances.get(node.id) ?? maxDistance + 1;
        const slotKey = `${dist}:${lane}`;
        const slot = laneSlots.get(slotKey) ?? 0;
        laneSlots.set(slotKey, slot + 1);
        const x = 48 + dist * 240;
        const y = 48 + laneOrder * 150 + slot * 92;
        decorateNode(node, x, y);
      }
    } else {
      for (const lane of LANE_ORDER) {
        const laneNodes = lanes.get(lane) ?? [];

        laneNodes.sort((a, b) => {
          const da = distances.get(a.id) ?? 999;
          const db = distances.get(b.id) ?? 999;

          if (da !== db) {
            return da - db;
          }

          const ga = degree.get(a.id) ?? 0;
          const gb = degree.get(b.id) ?? 0;

          if (ga !== gb) {
            return gb - ga;
          }

          return a.id.localeCompare(b.id);
        });

        laneNodes.forEach((node, index) => {
          decorateNode(node, 40 + index * 200, LANE_Y[lane]);
        });
      }
    }

    const renderedEdges: Edge[] = typeFilteredEdges.map((edge) => {
      const isAdded = addedEdgeIds.has(edge.id);
      const edgeType = normalizeEdgeType(edge.type);
      const inChain = chain.edgeIds.has(edge.id);
      let stroke = "#64748b";

      if (edgeType === "CONFLICT") {
        stroke = isAdded ? "#dc2626" : "#f97316";
      } else if (edgeType === "SUPPORT") {
        stroke = isAdded ? "#0891b2" : "#0ea5e9";
      }

      const edgeOpacity =
        resolvedClaimId && !inChain && focusMode !== "all" ? 0.2 : 0.9;

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: "smoothstep",
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 20,
          height: 20,
          color: stroke,
        },
        style: {
          stroke,
          strokeWidth: inChain ? 2.8 : isAdded ? 2.4 : 1.8,
          opacity: edgeOpacity,
          strokeDasharray: edgeType === "CONFLICT" ? "6 3" : undefined,
        },
        label: inChain ? edgeType : undefined,
        labelStyle: {
          fill: "#334155",
          fontSize: 10,
        },
        animated: isAdded,
      } as Edge;
    });

    return {
      nodes: positionedNodes,
      edges: renderedEdges,
      claimIds,
      resolvedClaimId,
      summary: {
        addedNodes: addedNodeIds.size,
        addedEdges: addedEdgeIds.size,
        statusChanged: statusChangedNodeIds.size,
        reused: reusedNodeIds.size,
        rejected: rejectedNodeIds.size,
        visibleNodes: positionedNodes.length,
        visibleEdges: renderedEdges.length,
      },
    };
  }, [
    artifacts,
    baselineGraph,
    chainMode,
    currentGraph,
    diff,
    focusMode,
    hopDepth,
    layoutMode,
    selectedClaimId,
  ]);

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
              Layout
              <select
                value={layoutMode}
                onChange={(event) =>
                  setLayoutMode(event.target.value as LayoutMode)
                }
              >
                <option value="distanceFlow">Distance Flow</option>
                <option value="typeLanes">Type Lanes</option>
              </select>
            </label>

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

            <label className="graph-control">
              Hop Depth
              <select
                value={hopDepth}
                onChange={(event) => setHopDepth(Number(event.target.value))}
              >
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={3}>3</option>
              </select>
            </label>
          </div>

          <p className="line">
            layout {layoutMode} | visible N{model.summary.visibleNodes} E
            {model.summary.visibleEdges} | +N{model.summary.addedNodes} +E
            {model.summary.addedEdges} | status changes{" "}
            {model.summary.statusChanged} | reused {model.summary.reused} |
            rejected {model.summary.rejected}
            {model.resolvedClaimId
              ? ` | anchor ${shortId(model.resolvedClaimId)}`
              : ""}
          </p>

          <div className="graph-canvas">
            <ReactFlow
              nodes={model.nodes}
              edges={model.edges}
              fitView
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable
              proOptions={{ hideAttribution: true }}
            >
              <MiniMap pannable zoomable />
              <Controls />
              <Background gap={18} size={1} />
            </ReactFlow>
          </div>
        </>
      ) : (
        <p className="hint">Load graph and diff first.</p>
      )}
    </article>
  );
}
