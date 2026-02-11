import cytoscape, {
  type Core,
  type ElementDefinition,
  type LayoutOptions,
} from "cytoscape";

import fcose from "cytoscape-fcose";
import type { GraphEdge, GraphNode } from "../../../compat";
let fcoseRegistered = false;
export type DebateNodeFamily = "FACT" | "LAW" | "CLAIM" | "OTHER";
export type DebateEdgeRelation = "support" | "attack" | "cite";
export type DebateEdgeMode = "all" | DebateEdgeRelation;
type NodeStatusFamily = "VALIDATED" | "DEFEATED" | "HYPOTHETICAL";

interface NodeClassMap {
  [nodeId: string]: string[];
}

interface EdgeClassMap {
  [edgeId: string]: string[];
}

export interface DebateGraphMapOptions {
  visibleFamilies?: ReadonlySet<DebateNodeFamily>;
  edgeMode?: DebateEdgeMode;
  onlyClaimNeighborhood?: boolean;
  visibleNodeIds?: ReadonlySet<string>;
  visibleEdgeIds?: ReadonlySet<string>;
  nodeClassById?: NodeClassMap;
  edgeClassById?: EdgeClassMap;
}

export interface DebateNodeView {
  id: string;
  family: DebateNodeFamily;
  statusFamily: NodeStatusFamily;
  label: string;
  content: string;
  agentId: string;
}

export interface DebateEdgeView {
  id: string;
  source: string;
  target: string;
  relation: DebateEdgeRelation;
  rawType: string;
}

export interface DebateGraphElementsModel {
  elements: ElementDefinition[];
  nodes: DebateNodeView[];
  edges: DebateEdgeView[];
}

export function ensureCytoscapeFcoseRegistered(): void {
  if (fcoseRegistered) {
    return;
  }

  cytoscape.use(fcose);
  fcoseRegistered = true;
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

export function toStatusFamily(statusRaw: string): NodeStatusFamily {
  const upper = statusRaw.toUpperCase();

  if (upper === "VALIDATED" || upper === "ACCEPTED" || upper === "SUPPORTED") {
    return "VALIDATED";
  }

  if (upper === "DEFEATED" || upper === "REJECTED" || upper === "INVALID") {
    return "DEFEATED";
  }

  return "HYPOTHETICAL";
}

export function toEdgeRelation(typeRaw: string): DebateEdgeRelation {
  const upper = typeRaw.toUpperCase();

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

  return "cite";
}

function shortText(value: string, maxLength = 36): string {
  const text = value.trim();

  if (!text) {
    return "";
  }

  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength - 1)}...`;
}

function asClassName(classes: string[] | undefined): string {
  return Array.isArray(classes) ? classes.join(" ") : "";
}

export function mapDebateGraphToElements(
  nodes: GraphNode[],
  edges: GraphEdge[],
  options: DebateGraphMapOptions = {},
): DebateGraphElementsModel {
  const mappedNodes: DebateNodeView[] = nodes.map((node) => {
    const content = (node.content ?? node.label ?? node.id).trim();

    return {
      id: node.id,
      family: toNodeFamily(node.type),
      statusFamily: toStatusFamily(node.status ?? "HYPOTHETICAL"),
      label: (node.label ?? node.id).trim() || node.id,
      content,
      agentId: node.agentId ?? "unknown",
    };
  });

  const mappedEdges: DebateEdgeView[] = edges.map((edge, index) => ({
    id: edge.id || `${edge.source}=>${edge.target}#${index}`,
    source: edge.source,
    target: edge.target,
    relation: toEdgeRelation(edge.type),
    rawType: edge.type,
  }));

  const familyFilteredNodes = options.visibleFamilies
    ? mappedNodes.filter((node) => options.visibleFamilies?.has(node.family))
    : mappedNodes;

  let visibleNodeIds = new Set(familyFilteredNodes.map((node) => node.id));

  let visibleEdges = mappedEdges.filter(
    (edge) =>
      visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target),
  );

  if (options.edgeMode && options.edgeMode !== "all") {
    visibleEdges = visibleEdges.filter(
      (edge) => edge.relation === options.edgeMode,
    );
  }

  if (options.onlyClaimNeighborhood) {
    const claimNodeIds = new Set(
      familyFilteredNodes
        .filter((node) => node.family === "CLAIM")
        .map((node) => node.id),
    );

    const claimNeighborhood = new Set<string>(claimNodeIds);

    for (const edge of visibleEdges) {
      if (claimNodeIds.has(edge.source) || claimNodeIds.has(edge.target)) {
        claimNeighborhood.add(edge.source);
        claimNeighborhood.add(edge.target);
      }
    }

    if (claimNeighborhood.size > 0) {
      visibleNodeIds = claimNeighborhood;
    }
  }

  if (options.visibleNodeIds && options.visibleNodeIds.size > 0) {
    visibleNodeIds = new Set(
      [...visibleNodeIds].filter((nodeId) =>
        options.visibleNodeIds?.has(nodeId),
      ),
    );
  }

  const filteredNodes = familyFilteredNodes.filter((node) =>
    visibleNodeIds.has(node.id),
  );

  visibleEdges = visibleEdges.filter(
    (edge) =>
      visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target),
  );

  if (options.visibleEdgeIds && options.visibleEdgeIds.size > 0) {
    visibleEdges = visibleEdges.filter((edge) =>
      options.visibleEdgeIds?.has(edge.id),
    );
  }

  const elements: ElementDefinition[] = [
    ...filteredNodes.map((node) => ({
      data: {
        id: node.id,
        label: shortText(node.label || node.id, 42),
        fullLabel: node.label || node.id,
        content: node.content,
        typeFamily: node.family,
        statusFamily: node.statusFamily,
        agentId: node.agentId,
      },
      classes: asClassName(options.nodeClassById?.[node.id]),
    })),
    ...visibleEdges.map((edge) => ({
      data: {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        rel: edge.relation,
        typeRaw: edge.rawType,
      },
      classes: asClassName(options.edgeClassById?.[edge.id]),
    })),
  ];

  return {
    elements,
    nodes: filteredNodes,
    edges: visibleEdges,
  };
}

const FOCUS_CLASS_LIST =
  "is-dimmed is-focus-node is-focus-neighbor is-focus-edge";

export function clearNeighborhoodFocus(cy: Core): void {
  cy.batch(() => {
    cy.elements().removeClass(FOCUS_CLASS_LIST);
  });
}

export function setNeighborhoodFocus(cy: Core, nodeId: string): void {
  const center = cy.getElementById(nodeId);

  if (center.empty()) {
    clearNeighborhoodFocus(cy);
    return;
  }

  const neighborhood = center.closedNeighborhood();
  const keep = neighborhood.union(center);

  cy.batch(() => {
    cy.elements().removeClass(FOCUS_CLASS_LIST);
    cy.elements().difference(keep).addClass("is-dimmed");
    center.addClass("is-focus-node");
    neighborhood.nodes().difference(center).addClass("is-focus-neighbor");
    neighborhood.edges().addClass("is-focus-edge");
  });
}

export const FCOSE_LAYOUT_CONFIG: LayoutOptions = {
  name: "fcose",
  quality: "default",
  randomize: true,
  animate: false,
  fit: true,
  padding: 40,
  nodeRepulsion: 8800,
  idealEdgeLength: 160,
  edgeElasticity: 0.2,
  nestingFactor: 0.75,
  gravity: 0.28,
  gravityRange: 3.4,
  gravityCompound: 0.7,
  gravityRangeCompound: 1.5,
  numIter: 2600,
  packComponents: true,
  tile: true,
  tilingPaddingVertical: 20,
  tilingPaddingHorizontal: 20,
} as LayoutOptions;

export function runFcoseLayout(
  cy: Core,
  overrides: Partial<LayoutOptions> = {},
): void {
  if (!cy.elements().length) {
    return;
  }

  cy.resize();

  const layout = cy.layout({
    ...FCOSE_LAYOUT_CONFIG,
    ...overrides,
    name: "fcose",
  } as LayoutOptions);

  layout.run();
}

export const DEBATE_GRAPH_STYLESHEET = [
  {
    selector: "node",
    style: {
      shape: "round-rectangle",
      label: "data(label)",
      width: "label",
      height: "label",
      padding: "10px",
      "font-size": "10px",
      "text-wrap": "wrap",
      "text-max-width": "180px",
      color: "#0f172a",
      "text-valign": "center",
      "text-halign": "center",
      "background-color": "#e2e8f0",
      "border-width": 2,
      "border-color": "#64748b",
      "overlay-padding": "8px",
      "overlay-opacity": 0,
      "min-zoomed-font-size": 7,
    },
  },
  {
    selector: 'node[typeFamily = "FACT"]',
    style: {
      "background-color": "#fef3c7",
      "border-color": "#b45309",
    },
  },
  {
    selector: 'node[typeFamily = "LAW"]',
    style: {
      "background-color": "#dbeafe",
      "border-color": "#1d4ed8",
    },
  },
  {
    selector: 'node[typeFamily = "CLAIM"]',
    style: {
      "background-color": "#dcfce7",
      "border-color": "#15803d",
    },
  },
  {
    selector: 'node[typeFamily = "OTHER"]',
    style: {
      "background-color": "#e2e8f0",
      "border-color": "#475569",
    },
  },
  {
    selector: 'node[statusFamily = "VALIDATED"]',
    style: {
      "border-width": 3,
      "border-color": "#15803d",
    },
  },
  {
    selector: 'node[statusFamily = "DEFEATED"]',
    style: {
      "border-width": 3,
      "border-color": "#be123c",
    },
  },
  {
    selector: 'node[statusFamily = "HYPOTHETICAL"]',
    style: {
      "border-width": 2,
      "border-color": "#1d4ed8",
    },
  },
  {
    selector: "edge",
    style: {
      width: 2,
      "curve-style": "bezier",
      "line-color": "#64748b",
      "target-arrow-color": "#64748b",
      "target-arrow-shape": "triangle",
      "arrow-scale": 0.9,
      opacity: 0.86,
    },
  },
  {
    selector: 'edge[rel = "support"]',
    style: {
      "line-color": "#0284c7",
      "target-arrow-color": "#0284c7",
      "line-style": "solid",
    },
  },
  {
    selector: 'edge[rel = "attack"]',
    style: {
      "line-color": "#dc2626",
      "target-arrow-color": "#dc2626",
      "line-style": "dashed",
    },
  },
  {
    selector: 'edge[rel = "cite"]',
    style: {
      "line-color": "#7c3aed",
      "target-arrow-color": "#7c3aed",
      "line-style": "dotted",
    },
  },
  {
    selector: ".is-dimmed",
    style: {
      opacity: 0.14,
      "text-opacity": 0.2,
    },
  },
  {
    selector: ".is-focus-node",
    style: {
      "underlay-color": "#0f172a",
      "underlay-opacity": 0.14,
      "underlay-padding": 8,
      "z-compound-depth": "top",
      "z-index": 999,
    },
  },
  {
    selector: ".is-focus-neighbor",
    style: {
      "underlay-color": "#0284c7",
      "underlay-opacity": 0.08,
      "underlay-padding": 5,
      "z-index": 500,
    },
  },
  {
    selector: ".is-focus-edge",
    style: {
      width: 3,
      opacity: 1,
    },
  },
  {
    selector: ".node-added",
    style: {
      "background-color": "#dcfce7",
      "border-color": "#16a34a",
      "border-width": 3,
    },
  },
  {
    selector: ".node-status-changed",
    style: {
      "background-color": "#dbeafe",
      "border-color": "#2563eb",
      "border-width": 3,
    },
  },
  {
    selector: ".node-rejected",
    style: {
      "background-color": "#ffe4e6",
      "border-color": "#e11d48",
      "border-width": 3,
    },
  },
  {
    selector: ".node-reused",
    style: {
      "background-color": "#ecfeff",
      "border-style": "dashed",
      "border-color": "#0ea5e9",
    },
  },
  {
    selector: ".node-chain",
    style: {
      "border-width": 4,
    },
  },
  {
    selector: ".node-anchor",
    style: {
      "overlay-opacity": 0.18,
      "overlay-color": "#0f172a",
      "overlay-padding": 12,
    },
  },
  {
    selector: ".node-preferred",
    style: {
      "border-width": 4,
      "border-color": "#15803d",
    },
  },
  {
    selector: ".edge-added",
    style: {
      width: 2.8,
      opacity: 1,
    },
  },
  {
    selector: ".edge-chain",
    style: {
      width: 3.2,
      opacity: 1,
    },
  },
  {
    selector: ".edge-muted",
    style: {
      opacity: 0.24,
    },
  },
];
