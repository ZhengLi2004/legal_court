"""HTTP error helpers for API route modules."""

from __future__ import annotations

from typing import Sequence, Tuple, Type

from fastapi import HTTPException

ErrorMapping = Tuple[Type[BaseException], int]


def raise_as_http(
    exc: Exception,
    *,
    action: str,
    mappings: Sequence[ErrorMapping] = (),
) -> HTTPException:
    """Convert domain exceptions into API-friendly `HTTPException`.

    Args:
        exc: Original exception.
        action: Action label used in fallback error message.
        mappings: Ordered `(exception_type, status_code)` mapping rules.

    Returns:
        `HTTPException` mapped from `exc`, or generic 500 fallback.
    """
    for error_type, status_code in mappings:
        if isinstance(exc, error_type):
            return HTTPException(status_code=status_code, detail=str(exc))

    return HTTPException(status_code=500, detail=f"{action} failed: {exc}")
