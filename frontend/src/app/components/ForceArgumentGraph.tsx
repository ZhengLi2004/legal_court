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
  metadata?: Record<string, unknown>;
}

interface GraphEdgeView {
  id: string;
  source: string;
  target: string;
  family: EdgeFamily;
  rawType: string;
}

interface SimNode {
  id: string;
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

const NODE_FAMILY_LABEL: Record<NodeFamily, string> = {
  FACT: "事实",
  LAW: "法条",
  CLAIM: "主张",
  OTHER: "其他",
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

function statusPalette(status: string): { border: string; bg: string } {
  if (status === "VALIDATED") {
    return { border: "#15803d", bg: "#dcfce7" };
  }

  if (status === "DEFEATED") {
    return { border: "#be123c", bg: "#ffe4e6" };
  }

  return { border: "#1d4ed8", bg: "#dbeafe" };
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

function shortText(value: string, max = 64): string {
  const text = value.trim();

  if (!text) {
    return "";
  }

  if (text.length <= max) {
    return text;
  }

  return `${text.slice(0, max - 1)}…`;
}

function toNodeView(row: GraphNode): GraphNodeView {
  const content = (row.content ?? row.label ?? row.id).trim();

  return {
    id: row.id,
    family: normalizeNodeFamily(row.type),
    status: normalizeStatus(row.status ?? "HYPOTHETICAL"),
    label: shortText(row.label || row.id, 60),
    content,
    agentId: row.agentId ?? "unknown",
    metadata: row.metadata,
  };
}

function toEdgeView(row: GraphEdge): GraphEdgeView {
  return {
    id: row.id,
    source: row.source,
    target: row.target,
    family: normalizeEdgeFamily(row.type),
    rawType: row.type,
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

    const simNodes: SimNode[] = displayedNodes.map((node, index) => {
      const angle = (Math.PI * 2 * index) / Math.max(displayedNodes.length, 1);

      return {
        id: node.id,
        x: 760 + Math.cos(angle) * 180,
        y: 430 + Math.sin(angle) * 180,
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
          return family === "CONFLICT" ? 190 : family === "SUPPORT" ? 140 : 170;
        })
        .strength((link: SimLink) => (link.family === "CONFLICT" ? 0.45 : 0.3));

      const simulation = forceSimulation(simNodes)
        .force("charge", forceManyBody().strength(-420))
        .force("link", linkForce)
        .force("collide", forceCollide(88))
        .force("center", forceCenter(760, 430))
        .force("x", forceX(760).strength(0.015))
        .force("y", forceY(430).strength(0.015))
        .stop();

      const ticks = Math.min(460, Math.max(220, simNodes.length * 9));

      for (let idx = 0; idx < ticks; idx += 1) {
        simulation.tick();
      }
    }

    const positionMap = new Map<string, { x: number; y: number }>(
      simNodes.map((node) => [node.id, { x: node.x ?? 0, y: node.y ?? 0 }]),
    );

    const reactNodes: Node[] = displayedNodes.map((node) => {
      const palette = statusPalette(node.status);
      const position = positionMap.get(node.id) ?? { x: 0, y: 0 };

      return {
        id: node.id,
        position,
        data: {
          label: (
            <div className="ux-force-node">
              <strong className="ux-force-node-title">
                {node.label || node.id}
              </strong>

              <div className="ux-force-node-meta">
                {NODE_FAMILY_LABEL[node.family]} ·{" "}
                {nodeStatusLabel(node.status)}
              </div>
            </div>
          ),
        },
        style: {
          width: 228,
          border: `2px solid ${palette.border}`,
          borderRadius: 12,
          background: palette.bg,
          color: "#0f172a",
          padding: 8,
        },
      } as Node;
    });

    const reactEdges: Edge[] = displayedEdges.map((edge) => {
      const palette = edgePalette(edge.family);
      const label = edge.family === "OTHER" ? edge.rawType : edge.family;

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: "straight",
        label,
        style: {
          stroke: palette.stroke,
          strokeWidth: edge.family === "CONFLICT" ? 2.3 : 2,
          strokeDasharray: palette.dash,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 18,
          height: 18,
          color: palette.stroke,
        },
        labelStyle: {
          fill: "#334155",
          fontSize: 10,
        },
      } as Edge;
    });

    const nodeById = new Map<string, GraphNodeView>(
      displayedNodes.map((node) => [node.id, node]),
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
        自动力导布局：蓝线为支持、红虚线为冲突。点击节点可查看完整内容与关联关系。
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
        <span>节点状态：绿色=被采纳，蓝色=待验证，红色=被驳回</span>
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
              fitViewOptions={{ padding: 0.15, maxZoom: 1.3 }}
              minZoom={0.1}
              maxZoom={1.8}
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
