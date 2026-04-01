"""Token-based text helpers shared by runtime components."""

from __future__ import annotations

from functools import lru_cache

try:
    import tiktoken

except Exception:
    tiktoken = None


@lru_cache(maxsize=1)
def _get_default_encoding():
    """Return default tokenizer encoding used for cross-model token budgeting."""
    if tiktoken is None:
        return None

    try:
        return tiktoken.get_encoding("cl100k_base")

    except Exception:
        return None


def truncate_by_tokens(text: str, max_tokens: int, ellipsis: str = "...") -> str:
    """Truncate text by token count with a robust fallback.

    Notes:
        - Uses `cl100k_base` as a model-agnostic proxy tokenizer.
        - Falls back to char-length truncation when tokenizer is unavailable.

    Args:
        text: Raw text to truncate.
        max_tokens: Maximum token budget proxy.
        ellipsis: Suffix appended after truncation.

    Returns:
        Token-truncated text that fits within the requested budget proxy.
    """
    if not text:
        return ""

    normalized = str(text).strip()

    if max_tokens <= 0:
        return ""

    encoding = _get_default_encoding()

    if encoding is None:
        if len(normalized) <= max_tokens:
            return normalized

        keep = max(0, max_tokens - len(ellipsis))
        return f"{normalized[:keep].rstrip()}{ellipsis}"

    token_ids = encoding.encode(normalized)

    if len(token_ids) <= max_tokens:
        return normalized

    truncated = encoding.decode(token_ids[:max_tokens]).rstrip()

    if not truncated:
        return ellipsis

    return f"{truncated}{ellipsis}"
