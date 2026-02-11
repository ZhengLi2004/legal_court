import { useEffect, useMemo, useRef } from "react";
import * as echarts from "echarts";
import type { EChartsOption, EChartsType } from "echarts";
import type { TurnArtifact } from "../../compat";
type StepStatus = "done" | "active" | "pending" | "warning";

function hasValue(value: unknown): boolean {
  if (value === null || value === undefined) {
    return false;
  }

  if (typeof value === "string") {
    return value.trim().length > 0;
  }

  if (Array.isArray(value)) {
    return value.length > 0;
  }

  if (typeof value === "object") {
    return Object.keys(value as Record<string, unknown>).length > 0;
  }

  return true;
}

function deriveStatuses(
  artifact: TurnArtifact | null,
): Record<string, StepStatus> {
  if (!artifact) {
    return {
      ASSESS: "pending",
      DELEGATE: "pending",
      WAIT: "pending",
      DECIDE: "pending",
      RETRY: "pending",
      DONE: "pending",
    };
  }

  const execution = artifact.executionLogs.toLowerCase();

  const hasRetry =
    Array.isArray(artifact.retryHistory) && artifact.retryHistory.length > 0;

  const failed =
    execution.includes("reject") ||
    execution.includes("failed") ||
    execution.includes("error") ||
    execution.includes("rollback");

  const assessDone = hasValue(artifact.controllerAssessment);
  const delegateDone = hasValue(artifact.batchInstructions);
  const waitDone = hasValue(artifact.workerReports);

  const decideDone =
    hasValue(artifact.decisionRaw) || hasValue(artifact.parsedActions);

  return {
    ASSESS: assessDone ? "done" : "active",
    DELEGATE: delegateDone ? "done" : assessDone ? "active" : "pending",
    WAIT: waitDone ? "done" : delegateDone ? "active" : "pending",
    DECIDE: decideDone ? "done" : waitDone ? "active" : "pending",
    RETRY: hasRetry ? "warning" : failed ? "warning" : "pending",
    DONE: failed ? "warning" : decideDone ? "done" : "pending",
  };
}

function colorByStatus(status: StepStatus): {
  fill: string;
  border: string;
  text: string;
} {
  if (status === "done") {
    return { fill: "#dcfce7", border: "#15803d", text: "#14532d" };
  }

  if (status === "active") {
    return { fill: "#dbeafe", border: "#1d4ed8", text: "#1e3a8a" };
  }

  if (status === "warning") {
    return { fill: "#fee2e2", border: "#dc2626", text: "#7f1d1d" };
  }

  return { fill: "#f1f5f9", border: "#64748b", text: "#334155" };
}

interface ControllerPipelineGraphProps {
  artifact: TurnArtifact | null;
}

export function ControllerPipelineGraph({
  artifact,
}: ControllerPipelineGraphProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<EChartsType | null>(null);
  const status = deriveStatuses(artifact);

  const elements = useMemo(
    () => [
      {
        id: "ASSESS",
        label: "ASSESS",
        status: status.ASSESS,
        x: 80,
        y: 120,
      },
      {
        id: "DELEGATE",
        label: "DELEGATE",
        status: status.DELEGATE,
        x: 250,
        y: 120,
      },
      {
        id: "WAIT",
        label: "WAIT",
        status: status.WAIT,
        x: 430,
        y: 120,
      },
      {
        id: "DECIDE",
        label: "DECIDE",
        status: status.DECIDE,
        x: 600,
        y: 120,
      },
      {
        id: "DONE",
        label: "DONE",
        status: status.DONE,
        x: 770,
        y: 120,
      },
      {
        id: "RETRY",
        label: "RETRY",
        status: status.RETRY,
        x: 600,
        y: 260,
      },
    ],
    [status],
  );

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

    if (!chart) {
      return;
    }

    const nodes = elements.map((node) => {
      const color = colorByStatus(node.status);

      return {
        id: node.id,
        name: node.label,
        x: node.x,
        y: node.y,
        symbol: "roundRect",
        symbolSize: [126, 52],
        itemStyle: {
          color: color.fill,
          borderColor: color.border,
          borderWidth: 2,
        },
        label: {
          show: true,
          color: color.text,
          fontSize: 11,
          fontWeight: "bold",
        },
        draggable: false,
      };
    });

    const links = [
      { source: "ASSESS", target: "DELEGATE", retry: false },
      { source: "DELEGATE", target: "WAIT", retry: false },
      { source: "WAIT", target: "DECIDE", retry: false },
      { source: "DECIDE", target: "DONE", retry: false },
      { source: "DECIDE", target: "RETRY", retry: true },
      { source: "RETRY", target: "DELEGATE", retry: true },
    ].map((edge, idx) => ({
      id: `${edge.source}-${edge.target}-${idx}`,
      source: edge.source,
      target: edge.target,
      lineStyle: {
        color: edge.retry ? "#dc2626" : "#0284c7",
        width: 2,
        type: edge.retry ? "dashed" : "solid",
      },
    }));

    const option = {
      backgroundColor: "#f8fafc",
      series: [
        {
          type: "graph",
          layout: "none",
          coordinateSystem: null,
          roam: false,
          data: nodes,
          links,
          edgeSymbol: ["none", "arrow"],
          edgeSymbolSize: 9,
          lineStyle: {
            curveness: 0.05,
          },
        },
      ],
    } as EChartsOption;

    chart.setOption(option, true);
  }, [elements]);

  return (
    <div className="ux-graph-canvas ux-graph-canvas-compact">
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
    </div>
  );
}
