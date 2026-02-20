"""
PRSM Interface Authentication Module

This module provides FastAPI authentication dependencies for the API layer.
It re-exports authentication functions from the core auth module for convenient
importing in API route handlers.

Usage:
    from prsm.interface.auth import get_current_user, require_auth
    
    @router.get("/protected")
    async def protected_route(user: User = Depends(get_current_user)):
        return {"user_id": user.user_id}
"""

from prsm.core.auth.auth_manager import (
    get_current_user,
    require_auth,
    require_permission,
    require_role,
    auth_manager,
    AuthenticationError,
    AuthorizationError,
)
from prsm.core.auth.models import User, UserRole, Permission

__all__ = [
    'get_current_user',
    'require_auth',
    'require_permission',
    'require_role',
    'auth_manager',
    'AuthenticationError',
    'AuthorizationError',
    'User',
    'UserRole',
    'Permission',
]
