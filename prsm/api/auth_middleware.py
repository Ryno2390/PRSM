"""
API Authentication Middleware
=============================

Lightweight API key authentication for node management endpoints.
Protects settler, content economy, and other sensitive routes.
"""

import hashlib
import logging
import os
import secrets
from typing import Optional, Set

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Endpoints that don't require authentication
PUBLIC_ENDPOINTS: Set[str] = {
    "/", "/health", "/status", "/docs", "/openapi.json", "/redoc",
    "/peers", "/node/info",
}

# Endpoints that require auth
PROTECTED_PREFIXES = [
    "/settler/",
    "/content/upload",
    "/compute/forge",
]


def generate_api_key() -> str:
    """Generate a new API key."""
    return f"prsm_{secrets.token_urlsafe(32)}"


def hash_api_key(key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


class NodeAuthMiddleware(BaseHTTPMiddleware):
    """API key authentication for the node management API.

    Checks for 'Authorization: Bearer <api_key>' or 'X-API-Key: <key>' header.
    If PRSM_NODE_API_KEY env var is set, that key is required for protected endpoints.
    If not set, all endpoints are open (development mode).
    """

    def __init__(self, app, api_key_hash: str = ""):
        super().__init__(app)
        self._api_key_hash = api_key_hash

    @property
    def auth_enabled(self) -> bool:
        return bool(self._api_key_hash)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Public endpoints always allowed
        if path in PUBLIC_ENDPOINTS:
            return await call_next(request)

        # Check if this is a protected endpoint
        is_protected = any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES)

        if is_protected and self.auth_enabled:
            # Extract API key from headers
            api_key = self._extract_key(request)
            if not api_key:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required. Provide 'Authorization: Bearer <key>' or 'X-API-Key: <key>' header.",
                )

            if hash_api_key(api_key) != self._api_key_hash:
                raise HTTPException(
                    status_code=403,
                    detail="Invalid API key.",
                )

        return await call_next(request)

    @staticmethod
    def _extract_key(request: Request) -> Optional[str]:
        """Extract API key from Authorization or X-API-Key header."""
        # Try Bearer token
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()

        # Try X-API-Key
        return request.headers.get("x-api-key", "").strip() or None


def get_node_auth_middleware(app) -> Optional[NodeAuthMiddleware]:
    """Create auth middleware if PRSM_NODE_API_KEY is configured."""
    env_key = os.environ.get("PRSM_NODE_API_KEY", "")
    if env_key:
        key_hash = hash_api_key(env_key)
        middleware = NodeAuthMiddleware(app, api_key_hash=key_hash)
        logger.info("Node API authentication enabled (PRSM_NODE_API_KEY set)")
        return middleware
    else:
        logger.info("Node API authentication disabled (set PRSM_NODE_API_KEY to enable)")
        return None
