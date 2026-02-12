import { useId, useMemo } from "react";

interface ConvergenceSparklineProps {
  history: number[];
  deltaPhi: number;
  sma: number;
  epsilon: number;
}

function formatValue(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : "-";
}

export function ConvergenceSparkline({
  history,
  deltaPhi,
  sma,
  epsilon,
}: ConvergenceSparklineProps) {
  const gradientId = useId().replace(/:/g, "");
  const width = 520;
  const height = 180;
  const paddingX = 18;
  const paddingY = 18;
  const source = history.length > 0 ? history : [deltaPhi];
  const points = source.slice(-64);

  const shape = useMemo(() => {
    if (points.length === 0) {
      return {
        polylinePoints: "",
        areaPath: "",
        epsilonY: height / 2,
        smaY: height / 2,
        lastX: width / 2,
        lastY: height / 2,
        first: 0,
        last: 0,
        min: 0,
        max: 0,
        xTicks: [] as number[],
        yTicks: [] as number[],
      };
    }

    const min = Math.min(...points, epsilon, sma);
    const max = Math.max(...points, epsilon, sma);
    const span = Math.max(max - min, 1e-9);
    const innerWidth = width - paddingX * 2;
    const innerHeight = height - paddingY * 2;
    const step = points.length > 1 ? innerWidth / (points.length - 1) : 0;
    const toX = (idx: number): number => paddingX + idx * step;

    const toY = (value: number): number =>
      paddingY + ((max - value) / span) * innerHeight;

    const polylinePoints = points
      .map((value, idx) => `${toX(idx)},${toY(value)}`)
      .join(" ");

    const firstX = toX(0);
    const firstY = toY(points[0]);
    const lastX = toX(points.length - 1);
    const lastY = toY(points[points.length - 1]);
    const baselineY = height - paddingY;

    const areaPath = `M ${firstX},${baselineY} L ${firstX},${firstY} ${points
      .map((value, idx) => `L ${toX(idx)},${toY(value)}`)
      .join(" ")} L ${lastX},${baselineY} Z`;

    return {
      polylinePoints,
      areaPath,
      epsilonY: toY(epsilon),
      smaY: toY(sma),
      lastX,
      lastY,
      first: points[0],
      last: points[points.length - 1],
      min,
      max,
      xTicks: Array.from(
        { length: 5 },
        (_, idx) => paddingX + (innerWidth * idx) / 4,
      ),
      yTicks: Array.from(
        { length: 5 },
        (_, idx) => paddingY + (innerHeight * idx) / 4,
      ),
    };
  }, [epsilon, height, points, sma, width]);

  const trend = shape.last - shape.first;

  const trendLabel =
    trend > 0
      ? `上升 ${formatValue(trend)}`
      : trend < 0
        ? `下降 ${formatValue(Math.abs(trend))}`
        : "持平";

  return (
    <div className="ux-convergence-chart">
      <svg
        aria-label="收敛轨迹图"
        className="ux-convergence-svg"
        viewBox={`0 0 ${width} ${height}`}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="rgb(37 99 235 / 0.36)" />
            <stop offset="100%" stopColor="rgb(37 99 235 / 0.02)" />
          </linearGradient>
        </defs>

        {shape.xTicks.map((x) => (
          <line
            className="ux-convergence-grid"
            key={`x-${x}`}
            x1={x}
            x2={x}
            y1={paddingY}
            y2={height - paddingY}
          />
        ))}

        {shape.yTicks.map((y) => (
          <line
            className="ux-convergence-grid"
            key={`y-${y}`}
            x1={paddingX}
            x2={width - paddingX}
            y1={y}
            y2={y}
          />
        ))}

        <line
          className="ux-convergence-axis"
          x1={paddingX}
          x2={width - paddingX}
          y1={height - paddingY}
          y2={height - paddingY}
        />

        <path
          className="ux-convergence-area"
          d={shape.areaPath}
          fill={`url(#${gradientId})`}
        />

        <line
          className="ux-convergence-sma"
          x1={paddingX}
          x2={width - paddingX}
          y1={shape.smaY}
          y2={shape.smaY}
        />

        <line
          className="ux-convergence-epsilon"
          x1={paddingX}
          x2={width - paddingX}
          y1={shape.epsilonY}
          y2={shape.epsilonY}
        />

        <polyline
          className="ux-convergence-line"
          fill="none"
          points={shape.polylinePoints}
        />

        <circle
          className="ux-convergence-point"
          cx={shape.lastX}
          cy={shape.lastY}
          r={3.4}
        />

        <text
          className="ux-convergence-label"
          x={paddingX + 4}
          y={paddingY + 12}
        >
          max {formatValue(shape.max)}
        </text>

        <text
          className="ux-convergence-label"
          x={paddingX + 4}
          y={height - paddingY - 4}
        >
          min {formatValue(shape.min)}
        </text>

        <text
          className="ux-convergence-label ux-convergence-label-alert"
          x={width - paddingX - 108}
          y={Math.max(shape.epsilonY - 6, paddingY + 12)}
        >
          ε {formatValue(epsilon)}
        </text>
      </svg>

      <div className="ux-convergence-meta">
        <span>ΔΦ {formatValue(deltaPhi)}</span>
        <span>SMA {formatValue(sma)}</span>
        <span>样本 {points.length}</span>
        <span>{trendLabel}</span>
      </div>
    </div>
  );
}
