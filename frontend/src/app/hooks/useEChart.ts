import { useEffect, useRef, type MutableRefObject } from "react";
import * as echarts from "echarts";
import type { EChartsOption, EChartsType } from "echarts";

interface UseEChartParams {
  option: EChartsOption | null;
  enabled: boolean;
  notMerge?: boolean;
}

interface UseEChartResult {
  containerRef: MutableRefObject<HTMLDivElement | null>;
  chartRef: MutableRefObject<EChartsType | null>;
}

export function useEChart({
  option,
  enabled,
  notMerge = true,
}: UseEChartParams): UseEChartResult {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<EChartsType | null>(null);

  useEffect(() => {
    if (!enabled) {
      if (chartRef.current) {
        chartRef.current.dispose();
        chartRef.current = null;
      }

      return;
    }

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
  }, [enabled]);

  useEffect(() => {
    const chart = chartRef.current;

    if (!chart) {
      return;
    }

    if (!option) {
      chart.clear();
      return;
    }

    chart.setOption(option, notMerge);
  }, [notMerge, option]);

  return { containerRef, chartRef };
}
