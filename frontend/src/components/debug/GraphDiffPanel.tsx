import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Core } from "cytoscape";
import CytoscapeComponent from "react-cytoscapejs";

import {
  clearNeighborhoodFocus,
  DEBATE_GRAPH_STYLESHEET,
  ensureCytoscapeFcoseRegistered,
  mapDebateGraphToElements,
  runFcoseLayout,
  setNeighborhoodFocus,
  toEdgeRelation,
  toNodeFamily,
} from "../../app/graph/cytoscape/debateGraphCytoscape";

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

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
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
  edges: GraphView["edges"],
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
      const relation = toEdgeRelation(edge.type);

      if (chainMode === "support" && relation !== "support") {
        continue;
      }

      if (chainMode === "conflict" && relation !== "attack") {
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

ensureCytoscapeFcoseRegistered();

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
  const cyRef = useRef<Core | null>(null);

  const model = useMemo(() => {
    if (!currentGraph) {
      return null;
    }

    const previousNodes = new Map(
      (baselineGraph?.nodes ?? []).map((node) => [node.id, node]),
    );

    const currentNodeIds = new Set(currentGraph.nodes.map((node) => node.id));

    const previousNodeIds = new Set(
      (baselineGraph?.nodes ?? []).map((n) => n.id),
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
      .filter((node) => toNodeFamily(node.type) === "CLAIM")
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
      ? collectClaimChain(resolvedClaimId, currentGraph.edges, 3, chainMode)
      : { nodeIds: new Set<string>(), edgeIds: new Set<string>() };

    const visibleNodeIds = new Set<string>(baseVisibleIds);

    for (const nodeId of chain.nodeIds) {
      visibleNodeIds.add(nodeId);
    }

    let visibleEdges = currentGraph.edges.filter(
      (edge) =>
        visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target),
    );

    if (resolvedClaimId && chainMode !== "all") {
      visibleEdges = visibleEdges.filter((edge) => chain.edgeIds.has(edge.id));
    }

    const visibleEdgeIds = new Set(visibleEdges.map((edge) => edge.id));
    const nodeClassById: Record<string, string[]> = {};

    for (const node of currentGraph.nodes) {
      if (!visibleNodeIds.has(node.id)) {
        continue;
      }

      const classes: string[] = [];

      if (addedNodeIds.has(node.id)) {
        classes.push("node-added");
      } else if (rejectedNodeIds.has(node.id)) {
        classes.push("node-rejected");
      } else if (statusChangedNodeIds.has(node.id)) {
        classes.push("node-status-changed");
      } else if (reusedNodeIds.has(node.id)) {
        classes.push("node-reused");
      }

      if (chain.nodeIds.has(node.id)) {
        classes.push("node-chain");
      }

      if (resolvedClaimId && resolvedClaimId === node.id) {
        classes.push("node-anchor");
      }

      nodeClassById[node.id] = classes;
    }

    const edgeClassById: Record<string, string[]> = {};

    for (const edge of visibleEdges) {
      const classes: string[] = [];

      if (addedEdgeIds.has(edge.id)) {
        classes.push("edge-added");
      }

      if (chain.edgeIds.has(edge.id)) {
        classes.push("edge-chain");
      }

      if (
        resolvedClaimId &&
        focusMode !== "all" &&
        !chain.edgeIds.has(edge.id)
      ) {
        classes.push("edge-muted");
      }

      edgeClassById[edge.id] = classes;
    }

    const mapped = mapDebateGraphToElements(
      currentGraph.nodes,
      currentGraph.edges,
      {
        visibleNodeIds,
        visibleEdgeIds,
        nodeClassById,
        edgeClassById,
      },
    );

    return {
      ...mapped,
      claimIds,
      resolvedClaimId,
      summary: {
        addedNodes: addedNodeIds.size,
        addedEdges: addedEdgeIds.size,
        statusChanged: statusChangedNodeIds.size,
        reused: reusedNodeIds.size,
        rejected: rejectedNodeIds.size,
        visibleNodes: mapped.nodes.length,
        visibleEdges: mapped.edges.length,
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
    cy.off("tap.graphDiff");
    cy.off("tap.graphDiff", "node");

    cy.on("tap.graphDiff", "node", (event) => {
      const nodeId = event.target.id();
      setSelectedNodeId(nodeId);
      setNeighborhoodFocus(cy, nodeId);
    });

    cy.on("tap.graphDiff", (event) => {
      if (event.target !== cy) {
        return;
      }

      setSelectedNodeId("");
      clearNeighborhoodFocus(cy);
    });
  }, []);

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
        </>
      ) : (
        <p className="hint">Load graph and diff first.</p>
      )}
    </article>
  );
}
