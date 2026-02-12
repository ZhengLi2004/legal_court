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

function normalizeForCompare(raw: string): string {
  return raw
    .trim()
    .toLowerCase()
    .replace(/[\s\u3000:：,，.。!！?？、】【（）()"'“”‘’—_-]|\[|\]/g, "");
}

export function renderScrollableNodeTooltip(
  titleRaw: string,
  contentRaw: string,
): string {
  const title = titleRaw.trim() || "节点详情";
  const content = contentRaw.trim();
  const normalizedTitle = normalizeForCompare(title);
  let body = content;

  if (body) {
    const lines = body.split(/\r?\n/).map((line) => line.trim());
    const firstLine = lines[0] ?? "";

    if (normalizeForCompare(body) === normalizedTitle) {
      body = "";
    } else if (
      lines.length > 1 &&
      normalizeForCompare(firstLine) === normalizedTitle
    ) {
      body = lines.slice(1).join("\n").trim();
    }
  }

  const bodyBlock = body
    ? `<div class="ux-node-tooltip-body">${asHtmlWithBreaks(body)}</div>`
    : content
      ? ""
      : `<div class="ux-node-tooltip-body">${asHtmlWithBreaks("（空内容）")}</div>`;

  return [
    '<div class="ux-node-tooltip">',
    `<div class="ux-node-tooltip-title">${asHtmlWithBreaks(title)}</div>`,
    bodyBlock,
    "</div>",
  ].join("");
}
