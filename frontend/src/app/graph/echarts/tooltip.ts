function escapeHtml(raw: string): string {
  return raw
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function asHtmlWithBreaks(raw: string): string {
  return escapeHtml(raw).replace(/\r?\n/g, "<br/>");
}

export function renderScrollableNodeTooltip(
  titleRaw: string,
  contentRaw: string,
): string {
  const title = titleRaw.trim() || "节点详情";
  const content = contentRaw.trim();

  return [
    '<div class="ux-node-tooltip">',
    `<div class="ux-node-tooltip-title">${asHtmlWithBreaks(title)}</div>`,
    `<div class="ux-node-tooltip-body">${asHtmlWithBreaks(content || "（空内容）")}</div>`,
    "</div>",
  ].join("");
}
