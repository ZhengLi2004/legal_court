import Graph from "graphology";
import louvain from "graphology-communities-louvain";
export type DebateNodeFamily = "FACT" | "LAW" | "CLAIM" | "OTHER";
export type DebateEdgeRelation = "support" | "attack" | "cite";

export function shortText(value: string, maxLength = 18): string {
  const text = value.trim();

  if (!text) {
    return "";
  }

  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength - 1)}...`;
}

export function toNodeFamily(typeRaw: string): DebateNodeFamily {
  const upper = typeRaw.toUpperCase();

  if (upper.includes("FACT")) {
    return "FACT";
  }

  if (upper.includes("LAW")) {
    return "LAW";
  }

  if (upper.includes("CLAIM")) {
    return "CLAIM";
  }

  return "OTHER";
}

export function toEdgeRelation(typeRaw: string): DebateEdgeRelation {
  const upper = typeRaw.toUpperCase().trim();

  if (
    upper === "SUPPORT" ||
    upper === "EDGETYPE.SUPPORT" ||
    upper.includes("SUPPORT")
  ) {
    return "support";
  }

  if (
    upper === "ATTACK" ||
    upper === "CONFLICT" ||
    upper === "EDGETYPE.CONFLICT" ||
    upper.includes("CONFLICT") ||
    upper.includes("ATTACK")
  ) {
    return "attack";
  }

  if (upper.includes("CITE") || upper.includes("REFERENCE")) {
    return "cite";
  }

  return "support";
}

export function paletteColor(index: number): string {
  const colors = [
    "#14b8a6",
    "#2563eb",
    "#2f855a",
    "#facc15",
    "#e11d48",
    "#0ea5e9",
    "#f97316",
    "#7c3aed",
  ];

  return colors[index % colors.length];
}

export function buildLouvainCommunities(
  nodeIds: string[],
  edges: Array<{ id: string; source: string; target: string }>,
): Map<string, number> {
  const graph = new Graph({ type: "undirected" });

  for (const nodeId of nodeIds) {
    if (!graph.hasNode(nodeId)) {
      graph.addNode(nodeId);
    }
  }

  for (const edge of edges) {
    if (
      !graph.hasNode(edge.source) ||
      !graph.hasNode(edge.target) ||
      edge.source === edge.target
    ) {
      continue;
    }

    const edgeKey = `${edge.source}=>${edge.target}#${edge.id}`;

    if (!graph.hasEdge(edgeKey)) {
      graph.addUndirectedEdgeWithKey(edgeKey, edge.source, edge.target);
    }
  }

  if (graph.order === 0) {
    return new Map<string, number>();
  }

  try {
    const raw = louvain(graph) as Record<string, number | string>;
    const clusterValues = new Set<string>();

    for (const nodeId of nodeIds) {
      clusterValues.add(String(raw[nodeId] ?? "0"));
    }

    const normalized = [...clusterValues].sort();

    const clusterIndex = new Map(
      normalized.map((clusterId, idx) => [clusterId, idx]),
    );

    const result = new Map<string, number>();

    for (const nodeId of nodeIds) {
      const clusterId = String(raw[nodeId] ?? "0");
      result.set(nodeId, clusterIndex.get(clusterId) ?? 0);
    }

    return result;
  } catch {
    const adjacency = new Map<string, Set<string>>();

    for (const nodeId of nodeIds) {
      adjacency.set(nodeId, new Set<string>());
    }

    for (const edge of edges) {
      adjacency.get(edge.source)?.add(edge.target);
      adjacency.get(edge.target)?.add(edge.source);
    }

    const visited = new Set<string>();
    const result = new Map<string, number>();
    let clusterId = 0;

    for (const nodeId of nodeIds) {
      if (visited.has(nodeId)) {
        continue;
      }

      const queue = [nodeId];
      visited.add(nodeId);

      while (queue.length > 0) {
        const current = queue.shift();

        if (!current) {
          continue;
        }

        result.set(current, clusterId);

        for (const neighbor of adjacency.get(current) ?? []) {
          if (visited.has(neighbor)) {
            continue;
          }

          visited.add(neighbor);
          queue.push(neighbor);
        }
      }

      clusterId += 1;
    }

    return result;
  }
}
