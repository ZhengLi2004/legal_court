import { useMemo, useState } from "react";
import type { TimelineEvent } from "../../compat";

interface TimelinePanelProps {
  timeline: TimelineEvent[];
  selectedSeq?: number;
  onSelectEvent?: (event: TimelineEvent) => void;
}

export function TimelinePanel({
  timeline,
  selectedSeq,
  onSelectEvent,
}: TimelinePanelProps) {
  const [sourceFilter, setSourceFilter] = useState<string>("all");

  const sources = useMemo(() => {
    const rows = new Set<string>();
    timeline.forEach((item) => rows.add(item.source));
    return ["all", ...Array.from(rows)];
  }, [timeline]);

  const filtered = useMemo(() => {
    return timeline.filter((item) =>
      sourceFilter === "all" ? true : item.source === sourceFilter,
    );
  }, [sourceFilter, timeline]);

  return (
    <article className="card">
      <h2>Timeline Stream</h2>

      <label className="filter-label">
        source filter
        <select
          value={sourceFilter}
          onChange={(event) => setSourceFilter(event.target.value)}
        >
          {sources.map((source) => (
            <option key={source} value={source}>
              {source}
            </option>
          ))}
        </select>
      </label>

      <div className="scrollbox">
        {filtered.length ? (
          [...filtered]
            .slice(-30)
            .reverse()
            .map((row) => {
              const selected = row.seq === selectedSeq;

              return (
                <button
                  className={`timeline-row ${selected ? "timeline-row-selected" : ""}`}
                  key={`${row.seq}-${row.event}`}
                  onClick={() => onSelectEvent?.(row)}
                  type="button"
                >
                  <span>#{row.seq}</span>
                  <span>{row.event}</span>
                  <span>{row.roundIdx ?? "-"}</span>
                </button>
              );
            })
        ) : (
          <p className="hint">No timeline events.</p>
        )}
      </div>
    </article>
  );
}
