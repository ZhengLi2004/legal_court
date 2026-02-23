import { useState, type ChangeEvent } from "react";
import { useDebate } from "../../../app/state/useDebate";
import { phaseLabel } from "../../../shared/lib/payload";

interface LaunchPageProps {
  onGoLive: () => void;
}

export function LaunchPage({ onGoLive }: LaunchPageProps) {
  const {
    sessions,
    sessionId,
    snapshot,
    frontendSnapshots,
    createSession,
    selectSession,
    saveFrontendSnapshot,
    importFrontendSnapshotBundle,
    loadFrontendSnapshot,
    busyAction,
  } = useDebate();

  const [snapshotLabel, setSnapshotLabel] = useState("");
  const [snapshotMessage, setSnapshotMessage] = useState("");

  const handleCreate = async (): Promise<void> => {
    const ok = await createSession();

    if (ok) {
      onGoLive();
    }
  };

  const handleContinue = async (sessionId: string): Promise<void> => {
    const ok = await selectSession(sessionId);

    if (ok) {
      onGoLive();
    }
  };

  const handleSaveSnapshot = async (): Promise<void> => {
    if (!sessionId) {
      setSnapshotMessage("请先创建或选择会话后再保存。");
      return;
    }

    const ok = await saveFrontendSnapshot(snapshotLabel, {
      route: "/app/live",
    });

    if (ok) {
      setSnapshotMessage("会话已保存到服务器磁盘。");
      setSnapshotLabel("");
    }
  };

  const handleImportFile = async (
    event: ChangeEvent<HTMLInputElement>,
  ): Promise<void> => {
    const file = event.target.files?.[0];
    event.target.value = "";

    if (!file) {
      return;
    }

    try {
      const content = await file.text();
      const parsed = JSON.parse(content) as Record<string, unknown>;

      const label =
        snapshotLabel.trim() || file.name.replace(/\.[^.]+$/, "") || "imported";

      const ok = await importFrontendSnapshotBundle(parsed, label, {
        route: "/app/live",
      });

      if (ok) {
        setSnapshotMessage(`导入成功：${file.name}`);
      } else {
        setSnapshotMessage(`导入失败：${file.name}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setSnapshotMessage(`解析失败：${message}`);
    }
  };

  const handleLoadSnapshot = async (snapshotId: string): Promise<void> => {
    const ok = await loadFrontendSnapshot(snapshotId);

    if (ok) {
      onGoLive();
    } else {
      setSnapshotMessage("恢复失败，请稍后重试。");
    }
  };

  const formatCreatedAt = (value: string): string => {
    const ts = Date.parse(value);
    return Number.isFinite(ts) ? new Date(ts).toLocaleString() : value;
  };

  return (
    <section className="ux-grid ux-grid-2">
      <article className="ux-card">
        <h2>开始一次新庭审</h2>

        <button
          disabled={Boolean(busyAction)}
          onClick={() => {
            void handleCreate();
          }}
          type="button"
        >
          开始庭审
        </button>
      </article>

      <article className="ux-card">
        <h2>当前会话概览</h2>
        {snapshot ? (
          <div className="ux-kv">
            <p>
              <span>会话</span>
              <strong>{snapshot.sessionId}</strong>
            </p>

            <p>
              <span>阶段</span>
              <strong>{phaseLabel(snapshot.phase)}</strong>
            </p>

            <p>
              <span>当前回合</span>
              <strong>{snapshot.round}</strong>
            </p>
          </div>
        ) : (
          <p className="ux-empty">尚未创建会话。</p>
        )}
      </article>

      <article className="ux-card ux-card-full">
        <h2>历史会话</h2>

        {sessions.length > 0 ? (
          <div className="ux-list">
            {sessions.map((item) => (
              <button
                className="ux-list-row"
                key={item.sessionId}
                onClick={() => {
                  void handleContinue(item.sessionId);
                }}
                type="button"
              >
                <span>{item.sessionId}</span>
                <span>r{item.round}</span>
                <span>{phaseLabel(item.phase)}</span>
              </button>
            ))}
          </div>
        ) : (
          <p className="ux-empty">暂无历史会话。</p>
        )}
      </article>

      <article className="ux-card ux-card-full">
        <h2>会话存档与恢复</h2>

        <label className="ux-field">
          存档标签（可选）
          <input
            onChange={(event) => setSnapshotLabel(event.target.value)}
            placeholder="例如：庭审-阶段A"
            type="text"
            value={snapshotLabel}
          />
        </label>

        <div className="ux-row">
          <button
            disabled={!sessionId || Boolean(busyAction)}
            onClick={() => {
              void handleSaveSnapshot();
            }}
            type="button"
          >
            保存当前会话到磁盘
          </button>
        </div>

        <label className="ux-field">
          手动导入会话文件（JSON）
          <input
            accept=".json,application/json"
            className="ux-file-input"
            disabled={Boolean(busyAction)}
            onChange={(event) => {
              void handleImportFile(event);
            }}
            type="file"
          />
        </label>

        {snapshotMessage ? <p className="ux-note">{snapshotMessage}</p> : null}

        {frontendSnapshots.length > 0 ? (
          <div className="ux-list">
            {frontendSnapshots.map((item) => (
              <button
                className="ux-list-row"
                disabled={Boolean(busyAction)}
                key={item.snapshotId}
                onClick={() => {
                  void handleLoadSnapshot(item.snapshotId);
                }}
                type="button"
              >
                <span>{item.label || item.snapshotId}</span>
                <span>{formatCreatedAt(item.createdAt)}</span>

                <span>
                  e{item.eventCount} / a{item.artifactCount} / s
                  {item.snapshotCount}
                </span>
              </button>
            ))}
          </div>
        ) : (
          <p className="ux-empty">暂无会话存档。</p>
        )}
      </article>
    </section>
  );
}
