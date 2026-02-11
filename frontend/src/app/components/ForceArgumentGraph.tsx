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

import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
} from "d3-force";

import { nodeStatusLabel } from "../utils/payload";
import type { GraphEdge, GraphNode, GraphView } from "../../compat";
type NodeFamily = "FACT" | "LAW" | "CLAIM" | "OTHER";
type EdgeMode = "all" | "support" | "conflict";
type EdgeFamily = "SUPPORT" | "CONFLICT" | "OTHER";

interface ForceArgumentGraphProps {
  graph: GraphView | null;
  title?: string;
}

interface GraphNodeView {
  id: string;
  family: NodeFamily;
  status: string;
  label: string;
  content: string;
  agentId: string;
  degree: number;
  metadata?: Record<string, unknown>;
}

interface GraphEdgeView {
  id: string;
  source: string;
  target: string;
  family: EdgeFamily;
}

interface SimNode {
  id: string;
  family: NodeFamily;
  radius: number;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

interface SimLink {
  source: string;
  target: string;
  family: EdgeFamily;
}

const NODE_FAMILIES: NodeFamily[] = ["FACT", "LAW", "CLAIM", "OTHER"];
const LAYOUT_WIDTH = 1700;
const LAYOUT_HEIGHT = 980;

const NODE_FAMILY_LABEL: Record<NodeFamily, string> = {
  FACT: "事实",
  LAW: "法条",
  CLAIM: "主张",
  OTHER: "其他",
};

const NODE_FAMILY_GLYPH: Record<NodeFamily, string> = {
  FACT: "F",
  LAW: "L",
  CLAIM: "C",
  OTHER: "O",
};

const CLUSTER_CENTER: Record<NodeFamily, { x: number; y: number }> = {
  FACT: { x: 460, y: 260 },
  LAW: { x: 1230, y: 260 },
  CLAIM: { x: 860, y: 720 },
  OTHER: { x: 1360, y: 720 },
};

function normalizeNodeFamily(type: string): NodeFamily {
  const upper = type.toUpperCase();

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

function normalizeStatus(status: string): string {
  const upper = status.toUpperCase();

  if (upper === "VALIDATED" || upper === "ACCEPTED" || upper === "SUPPORTED") {
    return "VALIDATED";
  }

  if (upper === "DEFEATED" || upper === "REJECTED") {
    return "DEFEATED";
  }

  return "HYPOTHETICAL";
}

function normalizeEdgeFamily(type: string): EdgeFamily {
  const upper = type.toUpperCase();

  if (
    upper === "CONFLICT" ||
    upper === "ATTACK" ||
    upper === "EDGETYPE.CONFLICT"
  ) {
    return "CONFLICT";
  }

  if (upper === "SUPPORT" || upper === "EDGETYPE.SUPPORT") {
    return "SUPPORT";
  }

  return "OTHER";
}

function nodeTypePalette(family: NodeFamily): { bg: string; text: string } {
  if (family === "FACT") {
    return { bg: "#fef3c7", text: "#78350f" };
  }

  if (family === "LAW") {
    return { bg: "#dbeafe", text: "#1e40af" };
  }

  if (family === "CLAIM") {
    return { bg: "#dcfce7", text: "#166534" };
  }

  return { bg: "#e2e8f0", text: "#334155" };
}

function statusBorder(status: string): string {
  if (status === "VALIDATED") {
    return "#15803d";
  }

  if (status === "DEFEATED") {
    return "#be123c";
  }

  return "#1d4ed8";
}

function edgePalette(family: EdgeFamily): { stroke: string; dash?: string } {
  if (family === "CONFLICT") {
    return { stroke: "#dc2626", dash: "7 4" };
  }

  if (family === "SUPPORT") {
    return { stroke: "#0284c7" };
  }

  return { stroke: "#64748b", dash: "3 3" };
}

function shortText(value: string, max = 20): string {
  const text = value.trim();

  if (!text) {
    return "";
  }

  if (text.length <= max) {
    return text;
  }

  return `${text.slice(0, max - 1)}…`;
}

function compactNodeKey(id: string): string {
  const text = id.trim();

  if (!text) {
    return "node";
  }

  if (text.length <= 10) {
    return text;
  }

  return `${text.slice(0, 4)}…${text.slice(-3)}`;
}

function hashText(text: string): number {
  let hash = 0;

  for (let idx = 0; idx < text.length; idx += 1) {
    hash = (hash * 31 + text.charCodeAt(idx)) >>> 0;
  }

  return hash;
}

function radiusFromDegree(degree: number): number {
  if (degree >= 8) {
    return 52;
  }

  if (degree >= 4) {
    return 46;
  }

  return 42;
}

function toNodeView(row: GraphNode): GraphNodeView {
  const content = (row.content ?? row.label ?? row.id).trim();

  return {
    id: row.id,
    family: normalizeNodeFamily(row.type),
    status: normalizeStatus(row.status ?? "HYPOTHETICAL"),
    label: shortText(row.label || row.id || "node", 22),
    content,
    agentId: row.agentId ?? "unknown",
    degree: 0,
    metadata: row.metadata,
  };
}

function toEdgeView(row: GraphEdge): GraphEdgeView {
  return {
    id: row.id,
    source: row.source,
    target: row.target,
    family: normalizeEdgeFamily(row.type),
  };
}

export function ForceArgumentGraph({
  graph,
  title = "论证图谱（力导布局）",
}: ForceArgumentGraphProps) {
  const [visibleFamilies, setVisibleFamilies] = useState<
    Record<NodeFamily, boolean>
  >({
    FACT: true,
    LAW: true,
    CLAIM: true,
    OTHER: true,
  });

  const [edgeMode, setEdgeMode] = useState<EdgeMode>("all");
  const [onlyClaimNeighborhood, setOnlyClaimNeighborhood] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string>("");

  const model = useMemo(() => {
    if (!graph) {
      return null;
    }

    const allNodes = graph.nodes.map(toNodeView);
    const allEdges = graph.edges.map(toEdgeView);

    const familyFilteredNodes = allNodes.filter(
      (node) => visibleFamilies[node.family],
    );

    const visibleNodeIds = new Set(familyFilteredNodes.map((node) => node.id));

    const edgeModeFiltered = allEdges.filter((edge) => {
      if (
        !visibleNodeIds.has(edge.source) ||
        !visibleNodeIds.has(edge.target)
      ) {
        return false;
      }

      if (edgeMode === "support") {
        return edge.family === "SUPPORT";
      }

      if (edgeMode === "conflict") {
        return edge.family === "CONFLICT";
      }

      return true;
    });

    let displayedNodes = familyFilteredNodes;
    let displayedEdges = edgeModeFiltered;

    if (onlyClaimNeighborhood) {
      const claimIds = new Set(
        displayedNodes
          .filter((node) => node.family === "CLAIM")
          .map((node) => node.id),
      );

      const neighborhood = new Set<string>(claimIds);

      for (const edge of displayedEdges) {
        if (claimIds.has(edge.source) || claimIds.has(edge.target)) {
          neighborhood.add(edge.source);
          neighborhood.add(edge.target);
        }
      }

      displayedNodes = displayedNodes.filter((node) =>
        neighborhood.has(node.id),
      );

      const neighborhoodIds = new Set(displayedNodes.map((node) => node.id));

      displayedEdges = displayedEdges.filter(
        (edge) =>
          neighborhoodIds.has(edge.source) && neighborhoodIds.has(edge.target),
      );
    }

    const degreeById = new Map<string, number>();

    for (const edge of displayedEdges) {
      degreeById.set(edge.source, (degreeById.get(edge.source) ?? 0) + 1);
      degreeById.set(edge.target, (degreeById.get(edge.target) ?? 0) + 1);
    }

    const enrichedNodes = displayedNodes.map((node) => ({
      ...node,
      degree: degreeById.get(node.id) ?? 0,
    }));

    const simNodes: SimNode[] = enrichedNodes.map((node, index) => {
      const angle = (Math.PI * 2 * index) / Math.max(enrichedNodes.length, 1);
      const center = CLUSTER_CENTER[node.family];
      const seed = hashText(node.id);
      const jitterRadius = 80 + (seed % 120);

      return {
        id: node.id,
        family: node.family,
        radius: radiusFromDegree(node.degree),
        x: center.x + Math.cos(angle + seed * 0.001) * jitterRadius,
        y: center.y + Math.sin(angle + seed * 0.001) * jitterRadius,
      };
    });

    const simLinks: SimLink[] = displayedEdges.map((edge) => ({
      source: edge.source,
      target: edge.target,
      family: edge.family,
    }));

    if (simNodes.length > 0) {
      const linkForce = forceLink<SimNode, SimLink>(simLinks)
        .id((d: SimNode) => d.id)
        .distance((link: SimLink) => {
          const family = link.family;
          return family === "CONFLICT" ? 170 : family === "SUPPORT" ? 130 : 145;
        })
        .strength((link: SimLink) =>
          link.family === "CONFLICT" ? 0.34 : 0.22,
        );

      const chargeStrength = Math.max(-540, -280 - simNodes.length * 1.7);

      const simulation = forceSimulation(simNodes)
        .force("charge", forceManyBody<SimNode>().strength(chargeStrength))
        .force("link", linkForce)
        .force(
          "collide",
          forceCollide<SimNode>()
            .radius((node) => node.radius + 16)
            .iterations(2),
        )
        .force("center", forceCenter(LAYOUT_WIDTH / 2, LAYOUT_HEIGHT / 2))
        .force(
          "x",
          forceX<SimNode>((node) => CLUSTER_CENTER[node.family].x).strength(
            0.085,
          ),
        )
        .force(
          "y",
          forceY<SimNode>((node) => CLUSTER_CENTER[node.family].y).strength(
            0.085,
          ),
        )
        .stop();

      const ticks = Math.min(780, Math.max(360, simNodes.length * 11));

      for (let idx = 0; idx < ticks; idx += 1) {
        simulation.tick();
      }
    }

    const positionMap = new Map<
      string,
      { x: number; y: number; radius: number }
    >(
      simNodes.map((node) => [
        node.id,
        { x: node.x ?? 0, y: node.y ?? 0, radius: node.radius },
      ]),
    );

    const reactNodes: Node[] = enrichedNodes.map((node) => {
      const palette = nodeTypePalette(node.family);
      const position = positionMap.get(node.id) ?? { x: 0, y: 0, radius: 42 };
      const nodeWidth = Math.max(84, Math.round(position.radius * 1.9));
      const badge = NODE_FAMILY_GLYPH[node.family];

      return {
        id: node.id,
        position,
        data: {
          label: (
            <div className="ux-force-node">
              <span
                className={`ux-force-node-badge ux-force-node-badge-${node.family.toLowerCase()}`}
              >
                {badge}
              </span>
              <strong className="ux-force-node-title">
                {compactNodeKey(node.id)}
              </strong>
            </div>
          ),
        },
        style: {
          width: nodeWidth,
          border: `1.35px solid ${statusBorder(node.status)}`,
          borderRadius: 12,
          background: palette.bg,
          color: palette.text,
          padding: "4px 6px",
          fontSize: 9.5,
          boxShadow: "0 1px 2px rgb(15 23 42 / 0.15)",
        },
      } as Node;
    });

    const reactEdges: Edge[] = displayedEdges.map((edge) => {
      const palette = edgePalette(edge.family);

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: "smoothstep",
        style: {
          stroke: palette.stroke,
          strokeWidth: edge.family === "CONFLICT" ? 1.65 : 1.5,
          strokeDasharray: palette.dash,
          opacity: edge.family === "OTHER" ? 0.55 : 0.82,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 16,
          height: 16,
          color: palette.stroke,
        },
      } as Edge;
    });

    const nodeById = new Map<string, GraphNodeView>(
      enrichedNodes.map((node) => [node.id, node]),
    );

    return {
      nodes: reactNodes,
      edges: reactEdges,
      nodeById,
      edgesByNode: displayedEdges,
      stats: {
        nodes: displayedNodes.length,
        edges: displayedEdges.length,
      },
    };
  }, [edgeMode, graph, onlyClaimNeighborhood, visibleFamilies]);

  const selectedNode =
    selectedNodeId && model?.nodeById.has(selectedNodeId)
      ? model.nodeById.get(selectedNodeId)
      : undefined;

  const selectedEdges = selectedNode
    ? (model?.edgesByNode.filter(
        (edge) =>
          edge.source === selectedNode.id || edge.target === selectedNode.id,
      ) ?? [])
    : [];

  const toggleFamily = (family: NodeFamily): void => {
    setVisibleFamilies((prev) => ({
      ...prev,
      [family]: !prev[family],
    }));
  };

  return (
    <article className="ux-card">
      <h2>{title}</h2>

      <p className="ux-muted">
        自动力导布局：节点按 FACT / LAW / CLAIM / OTHER
        分组聚类，蓝线为支持、红虚线为冲突。点击节点查看完整内容。
      </p>

      <div className="ux-graph-toolbar">
        <div className="ux-chip-row">
          {NODE_FAMILIES.map((family) => (
            <button
              className={`ux-chip ${visibleFamilies[family] ? "ux-chip-active" : ""}`}
              key={family}
              onClick={() => toggleFamily(family)}
              type="button"
            >
              {NODE_FAMILY_LABEL[family]}
            </button>
          ))}
        </div>

        <label className="ux-field ux-inline-field">
          关系筛选
          <select
            onChange={(event) => setEdgeMode(event.target.value as EdgeMode)}
            value={edgeMode}
          >
            <option value="all">全部关系</option>
            <option value="support">仅支持链</option>
            <option value="conflict">仅冲突链</option>
          </select>
        </label>

        <label className="ux-check">
          <input
            checked={onlyClaimNeighborhood}
            onChange={(event) => setOnlyClaimNeighborhood(event.target.checked)}
            type="checkbox"
          />
          仅显示主张邻域
        </label>
      </div>

      <div className="ux-graph-legend">
        <span>节点类型：黄=事实，浅蓝=法条，浅绿=主张，灰=其他</span>
        <span>节点边框状态：绿=被采纳，蓝=待验证，红=被驳回</span>
        <span>关系：蓝实线=SUPPORT，红虚线=CONFLICT</span>

        <span>
          当前可见：{model?.stats.nodes ?? 0} 节点 / {model?.stats.edges ?? 0}{" "}
          条边
        </span>
      </div>

      {model ? (
        <div className="ux-graph-layout">
          <div className="ux-graph-canvas">
            <ReactFlow
              edges={model.edges}
              fitView
              fitViewOptions={{ padding: 0.22, maxZoom: 1.35 }}
              minZoom={0.08}
              maxZoom={1.7}
              nodes={model.nodes}
              nodesConnectable={false}
              onNodeClick={(_, node) => setSelectedNodeId(node.id)}
              proOptions={{ hideAttribution: true }}
            >
              <MiniMap pannable zoomable />
              <Controls />
              <Background gap={18} size={1} />
            </ReactFlow>
          </div>

          <aside className="ux-node-inspector">
            <h3>节点详情</h3>

            {selectedNode ? (
              <div className="ux-kv">
                <p>
                  <span>ID</span>
                  <strong>{selectedNode.id}</strong>
                </p>

                <p>
                  <span>类型</span>
                  <strong>{NODE_FAMILY_LABEL[selectedNode.family]}</strong>
                </p>

                <p>
                  <span>状态</span>
                  <strong>{nodeStatusLabel(selectedNode.status)}</strong>
                </p>

                <p>
                  <span>来源</span>
                  <strong>{selectedNode.agentId}</strong>
                </p>

                <p>
                  <span>关联边</span>
                  <strong>{selectedEdges.length}</strong>
                </p>

                <p>
                  <span>节点连接度</span>
                  <strong>{selectedNode.degree}</strong>
                </p>

                <div className="ux-inspector-text">
                  <strong>完整内容</strong>
                  <p>{selectedNode.content || selectedNode.label || "无"}</p>
                </div>

                {selectedNode.metadata ? (
                  <div className="ux-inspector-text">
                    <strong>元数据</strong>
                    <pre>{JSON.stringify(selectedNode.metadata, null, 2)}</pre>
                  </div>
                ) : null}
              </div>
            ) : (
              <p className="ux-empty">点击图中的节点查看详情。</p>
            )}
          </aside>
        </div>
      ) : (
        <p className="ux-empty">当前暂无图谱数据。</p>
      )}
    </article>
  );
}
