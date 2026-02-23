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
    """Convert domain errors into consistent `HTTPException`s."""
    for error_type, status_code in mappings:
        if isinstance(exc, error_type):
            return HTTPException(status_code=status_code, detail=str(exc))

    return HTTPException(status_code=500, detail=f"{action} failed: {exc}")

