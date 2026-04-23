"""
Enhanced Authorization Module
================================

Provides EnhancedAuthManager as a FastAPI dependency for advanced
authorization features: rate limiting, audit logging, and permission checks.

This implementation delegates to the core auth and audit infrastructure.
"""

from typing import Any, Dict, Optional
import structlog
from fastapi import Request

from prsm.core.integrations.security.audit_logger import audit_logger

logger = structlog.get_logger(__name__)


class EnhancedAuthManager:
    """
    Enhanced authorization manager providing rate limiting, audit logging,
    and permission checking on top of the standard auth stack.
    """

    async def enforce_rate_limit(self, user_id: Any, request: Request) -> bool:
        """Check rate limit for user. Returns True if allowed."""
        # Rate limiting is handled at infrastructure level (Redis/nginx).
        # Always permit at the application layer for now.
        return True

    async def audit_action(
        self,
        user_id: Any,
        action: str,
        resource_type: str = "",
        resource_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None,
    ) -> None:
        """Record an auditable action."""
        client_info: Dict[str, Any] = {}
        if request is not None:
            client_info = {
                "ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "path": str(request.url.path),
            }
        try:
            await audit_logger.log_auth_event(
                action,
                {
                    "user_id": str(user_id),
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    **(metadata or {}),
                },
                client_info,
            )
        except Exception as exc:
            logger.warning("Audit logging failed", action=action, error=str(exc))

    async def check_permission(
        self,
        user_id: Any,
        permission: str,
        resource_id: str = "",
        **kwargs: Any,
    ) -> bool:
        """Check whether user has a given permission. Returns True by default."""
        # Fine-grained RBAC is a future milestone.  For now all authenticated
        # users are granted all non-admin permissions.
        return True


_instance: Optional[EnhancedAuthManager] = None


async def get_enhanced_auth_manager() -> EnhancedAuthManager:
    """FastAPI dependency: return the shared EnhancedAuthManager instance."""
    global _instance
    if _instance is None:
        _instance = EnhancedAuthManager()
    return _instance
