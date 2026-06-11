"""Access control for engineer-only (debug) endpoints."""

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status

from ragproject.config import get_settings


def require_debug_key(
    x_debug_key: Annotated[str | None, Header()] = None,
) -> None:
    """Allow the request only if a valid debug key is presented.

    * If no ``DEBUG_API_KEY`` is configured, debug endpoints are disabled and
      respond ``404`` (we don't even reveal that they exist).
    * If a key is configured, the caller must send a matching ``X-Debug-Key``
      header, or the request is rejected with ``401``.
    """
    expected = get_settings().debug_api_key
    if not expected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if x_debug_key is None or not secrets.compare_digest(x_debug_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing debug key",
        )
