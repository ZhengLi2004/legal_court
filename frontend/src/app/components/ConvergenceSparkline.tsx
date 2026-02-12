import { useMemo } from "react";

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
  const width = 360;
  const height = 120;
  const paddingX = 10;
  const paddingY = 14;
  const source = history.length > 0 ? history : [deltaPhi];
  const points = source.slice(-36);

  const shape = useMemo(() => {
    if (points.length === 0) {
      return {
        polylinePoints: "",
        epsilonY: height / 2,
        lastX: width / 2,
        lastY: height / 2,
      };
    }

    const min = Math.min(...points, epsilon);
    const max = Math.max(...points, epsilon);
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

    return {
      polylinePoints,
      epsilonY: toY(epsilon),
      lastX: toX(points.length - 1),
      lastY: toY(points[points.length - 1]),
    };
  }, [epsilon, height, points, width]);

  return (
    <div className="ux-convergence-chart">
      <svg
        aria-label="收敛轨迹图"
        className="ux-convergence-svg"
        viewBox={`0 0 ${width} ${height}`}
      >
        <line
          className="ux-convergence-axis"
          x1={paddingX}
          x2={width - paddingX}
          y1={height - paddingY}
          y2={height - paddingY}
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
      </svg>

      <div className="ux-convergence-meta">
        <span>ΔΦ {formatValue(deltaPhi)}</span>
        <span>SMA {formatValue(sma)}</span>
        <span>ε {formatValue(epsilon)}</span>
      </div>
    </div>
  );
}
