import { useCallback, useEffect, useRef, useState } from "react";
import type { EngineAdapter, TimelineEvent } from "../../compat";
import type { StreamStatus } from "../types";

function sortedTimeline(rows: TimelineEvent[]): TimelineEvent[] {
  return [...rows].sort((a, b) => a.seq - b.seq || a.ts - b.ts).slice(-180);
}

interface UseTimelineStreamParams {
  adapter: EngineAdapter;
  sessionId: string;
  loadGraph: () => Promise<boolean>;
  reportError: (err: unknown) => void;
}

interface UseTimelineStreamResult {
  timeline: TimelineEvent[];
  streamStatus: StreamStatus;
  loadTimeline: (limit?: number) => Promise<boolean>;
  clearTimelineState: () => void;
}

export function useTimelineStream({
  adapter,
  sessionId,
  loadGraph,
  reportError,
}: UseTimelineStreamParams): UseTimelineStreamResult {
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [streamStatus, setStreamStatus] = useState<StreamStatus>("idle");
  const sessionIdRef = useRef<string>(sessionId);
  const lastSeqRef = useRef<number>(0);
  const wsLiveRef = useRef<boolean>(false);
  const timelineInFlightRef = useRef<Promise<boolean> | null>(null);
  const timelineInFlightSessionRef = useRef<string>("");

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  const clearTimelineState = useCallback(() => {
    setTimeline([]);
    setStreamStatus("idle");
    lastSeqRef.current = 0;
    wsLiveRef.current = false;
    timelineInFlightRef.current = null;
    timelineInFlightSessionRef.current = "";
  }, []);

  const replaceTimeline = useCallback((rows: TimelineEvent[]): void => {
    const sorted = sortedTimeline(rows);
    const latest = sorted.length > 0 ? sorted[sorted.length - 1] : null;
    lastSeqRef.current = latest?.seq ?? 0;
    setTimeline(sorted);
  }, []);

  const mergeTimeline = useCallback((rows: TimelineEvent[]): void => {
    if (!rows.length) {
      return;
    }

    setTimeline((prev) => {
      const bucket = new Map<number, TimelineEvent>();

      for (const row of prev) {
        bucket.set(row.seq, row);
      }

      for (const row of rows) {
        bucket.set(row.seq, row);
      }

      const merged = sortedTimeline([...bucket.values()]);
      const latest = merged.length > 0 ? merged[merged.length - 1] : null;
      lastSeqRef.current = latest?.seq ?? lastSeqRef.current;
      return merged;
    });
  }, []);

  const loadTimeline = useCallback(
    async (limit = 80): Promise<boolean> => {
      const targetSessionId = sessionId;

      if (!targetSessionId) {
        return false;
      }

      if (
        timelineInFlightRef.current &&
        timelineInFlightSessionRef.current === targetSessionId
      ) {
        return timelineInFlightRef.current;
      }

      const task = (async (): Promise<boolean> => {
        try {
          const rows = await adapter.insight.getTimeline(
            targetSessionId,
            limit,
          );

          if (sessionIdRef.current !== targetSessionId) {
            return false;
          }

          replaceTimeline(rows);
          return true;
        } catch (err) {
          reportError(err);
          return false;
        }
      })();

      const inFlight = task.finally(() => {
        if (timelineInFlightRef.current === inFlight) {
          timelineInFlightRef.current = null;
          timelineInFlightSessionRef.current = "";
        }
      });

      timelineInFlightSessionRef.current = targetSessionId;
      timelineInFlightRef.current = inFlight;
      return inFlight;
    },
    [adapter, replaceTimeline, reportError, sessionId],
  );

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    let alive = true;
    wsLiveRef.current = false;

    const pullTimeline = async (): Promise<void> => {
      const ok = await loadTimeline(80);

      if (!alive) {
        return;
      }

      if (!wsLiveRef.current) {
        setStreamStatus("poll");
      }

      if (!ok) {
        setStreamStatus("poll");
      }
    };

    void pullTimeline();

    const stopStreaming = adapter.capabilities.supportsStreaming
      ? adapter.insight.subscribeTimeline(
          sessionId,
          (event) => {
            if (!alive) {
              return;
            }

            wsLiveRef.current = true;
            setStreamStatus("ws");
            mergeTimeline([event]);
          },
          {
            fromSeq:
              lastSeqRef.current > 0 ? lastSeqRef.current + 1 : undefined,
            onError: () => {
              if (!alive) {
                return;
              }

              wsLiveRef.current = false;
              setStreamStatus("poll");
            },
          },
        )
      : () => {};

    let timerId: number | null = null;

    const schedulePoll = (): void => {
      timerId = window.setTimeout(() => {
        if (!alive) {
          return;
        }

        const pageVisible =
          typeof document === "undefined" ||
          document.visibilityState === "visible";

        if (!wsLiveRef.current && pageVisible) {
          void pullTimeline();
        }

        schedulePoll();
      }, 3500);
    };

    schedulePoll();

    const handleVisibilityChange = (): void => {
      if (
        alive &&
        !wsLiveRef.current &&
        typeof document !== "undefined" &&
        document.visibilityState === "visible"
      ) {
        void pullTimeline();
        void loadGraph();
      }
    };

    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", handleVisibilityChange);
    }

    return () => {
      alive = false;
      stopStreaming();

      if (timerId !== null) {
        window.clearTimeout(timerId);
      }

      if (typeof document !== "undefined") {
        document.removeEventListener(
          "visibilitychange",
          handleVisibilityChange,
        );
      }
    };
  }, [adapter, loadGraph, loadTimeline, mergeTimeline, sessionId]);

  return {
    timeline,
    streamStatus,
    loadTimeline,
    clearTimelineState,
  };
}
