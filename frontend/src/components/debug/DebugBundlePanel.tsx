import type { DebugBundleView } from "../../compat";

function toPretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

interface DebugBundlePanelProps {
  bundle: DebugBundleView | null;
  loading: boolean;
  onLoad: () => Promise<void>;
}

export function DebugBundlePanel({
  bundle,
  loading,
  onLoad,
}: DebugBundlePanelProps) {
  const copyBundle = async (): Promise<void> => {
    if (!bundle) {
      return;
    }

    const content = JSON.stringify(bundle.raw ?? bundle, null, 2);
    await navigator.clipboard.writeText(content);
  };

  const downloadBundle = (): void => {
    if (!bundle) {
      return;
    }

    const content = JSON.stringify(bundle.raw ?? bundle, null, 2);
    const blob = new Blob([content], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${bundle.sessionId}-debug-bundle.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <article className="card wide">
      <h2>Debug Bundle</h2>

      <div className="sub-actions">
        <button disabled={loading} onClick={() => void onLoad()} type="button">
          {loading ? "Loading..." : "Refresh Bundle"}
        </button>

        <button
          disabled={!bundle || loading}
          onClick={() => void copyBundle()}
          type="button"
        >
          Copy JSON
        </button>

        <button
          disabled={!bundle || loading}
          onClick={downloadBundle}
          type="button"
        >
          Download JSON
        </button>
      </div>

      {bundle ? (
        <>
          <p className="line">
            session={bundle.sessionId} round={bundle.round} turn=
            {bundle.turnUid || "-"}
          </p>

          <p className="line">
            summary: nodes={bundle.snapshotSummary.nodeCount}, edges=
            {bundle.snapshotSummary.edgeCount}, claims=
            {bundle.snapshotSummary.claimCount}, conflicts=
            {bundle.snapshotSummary.conflictCount}
          </p>

          <pre className="json-block">{toPretty(bundle.raw ?? bundle)}</pre>
        </>
      ) : (
        <p className="hint">No debug bundle loaded.</p>
      )}
    </article>
  );
}
