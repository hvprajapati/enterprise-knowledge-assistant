"""Authentication dependency.

Provides API-key-based authentication that can be enabled/disabled
via ``Settings.require_auth``.  When disabled (the default), all
requests pass through unauthenticated.
"""

from __future__ import annotations

import logging
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config.settings import settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    api_key: Annotated[str | None, Security(API_KEY_HEADER)],
) -> bool:
    """Validate the ``X-API-Key`` header against the configured API key.

    Returns ``True`` when:
    - ``Settings.require_auth`` is ``False`` (auth disabled), or
    - The provided key matches ``Settings.api_key``.

    Raises ``HTTPException(401)`` when auth is required but the key
    is missing or incorrect.
    """
    # Auth is off by default — opt-in via settings
    if not settings.require_auth:
        return True

    if not settings.api_key:
        logger.warning(
            "require_auth=True but api_key is empty — "
            "all requests will be rejected.  Set API_KEY in .env"
        )
        raise HTTPException(status_code=500, detail="Server misconfiguration")

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return True
