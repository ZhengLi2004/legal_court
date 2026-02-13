import type { TeamFlowMessage, TeamFlowTurn } from "../../compat";

interface TeamFlowConversationProps {
  turn: TeamFlowTurn | null;
}

function formatTs(ts?: number): string {
  if (typeof ts !== "number" || !Number.isFinite(ts) || ts <= 0) {
    return "";
  }

  return new Date(ts).toLocaleTimeString();
}

function roleClass(role: TeamFlowMessage["role"]): string {
  if (role === "controller") {
    return "ux-teamflow-message-controller";
  }

  if (role === "worker") {
    return "ux-teamflow-message-worker";
  }

  if (role === "narrator") {
    return "ux-teamflow-message-narrator";
  }

  return "ux-teamflow-message-system";
}

export function TeamFlowConversation({ turn }: TeamFlowConversationProps) {
  if (!turn) {
    return <p className="ux-empty">暂无可展示的协作对话流。</p>;
  }

  return (
    <div className="ux-teamflow-conversation">
      {turn.messages.map((message, index) => {
        const metaEntries = Object.entries(message.meta ?? {});
        const tsText = formatTs(message.ts);

        return (
          <article
            className={`ux-teamflow-message ${roleClass(message.role)}`}
            key={`${turn.turnUid}-${message.id || index}`}
          >
            <header className="ux-teamflow-message-head">
              <span className="ux-teamflow-phase">{message.phase}</span>
              <strong>{message.title}</strong>
              <span>{message.actor}</span>
              {tsText ? <span>{tsText}</span> : null}
            </header>

            <p className="ux-teamflow-message-body">{message.content}</p>

            {metaEntries.length > 0 ? (
              <div className="ux-teamflow-message-meta">
                {metaEntries.map(([key, value]) => (
                  <span key={`${message.id}-${key}`}>
                    {key}: {String(value)}
                  </span>
                ))}
              </div>
            ) : null}
          </article>
        );
      })}
    </div>
  );
}
