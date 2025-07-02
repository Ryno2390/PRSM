"""
Standardized Dependencies for PRSM API
Provides consistent authentication, authorization, and request handling
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import Depends, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from prsm.auth.models import User, Permission
from prsm.core.database import get_db
from .exceptions import (
    raise_unauthorized,
    raise_forbidden,
    raise_rate_limit,
    RateLimitException
)
from .standards import APIConfig


logger = logging.getLogger(__name__)
security = HTTPBearer()

# In-memory rate limiting storage (in production, use Redis)
_rate_limit_storage: Dict[str, Dict[str, Any]] = {}


async def add_request_metadata(
    request: Request,
    x_request_id: Optional[str] = Header(None),
    x_trace_id: Optional[str] = Header(None),
    user_agent: Optional[str] = Header(None)
):
    """Add standard metadata to request state"""
    
    # Generate IDs if not provided
    request_id = x_request_id or str(uuid4())
    trace_id = x_trace_id or str(uuid4())
    
    # Store in request state for use throughout request lifecycle
    request.state.request_id = request_id
    request.state.trace_id = trace_id
    request.state.user_agent = user_agent
    request.state.start_time = datetime.now(timezone.utc)
    
    # Add to response headers
    request.state.response_headers = {
        "X-Request-ID": request_id,
        "X-Trace-ID": trace_id
    }
    
    return {
        "request_id": request_id,
        "trace_id": trace_id,
        "user_agent": user_agent
    }


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> str:
    """Verify API key and return user ID"""
    
    if not credentials or not credentials.credentials:
        raise_unauthorized("Missing or invalid API key")
    
    # Extract token
    token = credentials.credentials
    
    # Verify token and get user (implementation depends on your auth system)
    try:
        # This would be replaced with actual token verification logic
        from prsm.auth.jwt_handler import verify_jwt_token
        
        payload = verify_jwt_token(token)
        user_id = payload.get("user_id")
        
        if not user_id:
            raise_unauthorized("Invalid token payload")
            
        return user_id
        
    except Exception as e:
        logger.warning(f"Token verification failed: {str(e)}")
        raise_unauthorized("Invalid or expired token")


async def get_current_user(
    user_id: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise_unauthorized("User not found")
    
    if not user.is_active:
        raise_unauthorized("User account is disabled")
    
    return user


async def require_permissions(
    required_permissions: List[Permission],
    user: User = Depends(get_current_user)
) -> User:
    """Require specific permissions for endpoint access"""
    
    if user.is_superuser:
        return user
    
    user_permissions = user.get_permissions()
    
    for permission in required_permissions:
        if permission not in user_permissions:
            raise_forbidden(f"Missing required permission: {permission.value}")
    
    return user


def create_permission_dependency(permissions: List[Permission]):
    """Create a permission dependency for specific permissions"""
    
    async def permission_dependency(
        user: User = Depends(get_current_user)
    ) -> User:
        return await require_permissions(permissions, user)
    
    return permission_dependency


async def check_rate_limit(
    request: Request,
    user: User = Depends(get_current_user)
) -> User:
    """Check rate limits for user"""
    
    # Get user's rate limit configuration
    user_tier = user.role.value if user.role else "user"
    rate_config = APIConfig.RATE_LIMITS.get(user_tier, APIConfig.RATE_LIMITS["user"])
    
    # Create rate limit key
    rate_key = f"{user.id}:{request.url.path}:{user_tier}"
    current_time = datetime.now(timezone.utc)
    
    # Get or create rate limit record
    if rate_key not in _rate_limit_storage:
        _rate_limit_storage[rate_key] = {
            "requests": [],
            "last_reset": current_time
        }
    
    rate_data = _rate_limit_storage[rate_key]
    
    # Clean old requests outside the window
    window_start = current_time.timestamp() - rate_config["window"]
    rate_data["requests"] = [
        req_time for req_time in rate_data["requests"]
        if req_time > window_start
    ]
    
    # Check if limit exceeded
    if len(rate_data["requests"]) >= rate_config["requests"]:
        retry_after = rate_config["window"] - (current_time.timestamp() - min(rate_data["requests"]))
        raise_rate_limit(
            retry_after=int(retry_after) + 1,
            limit=rate_config["requests"],
            window=rate_config["window"]
        )
    
    # Add current request
    rate_data["requests"].append(current_time.timestamp())
    
    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise"""
    
    if not credentials or not credentials.credentials:
        return None
    
    try:
        user_id = await verify_api_key(credentials, db)
        return await get_current_user(user_id, db)
    except:
        return None


async def require_admin(user: User = Depends(get_current_user)) -> User:
    """Require admin role"""
    
    if not user.is_superuser and user.role.value != "admin":
        raise_forbidden("Admin access required")
    
    return user


async def require_enterprise(user: User = Depends(get_current_user)) -> User:
    """Require enterprise tier or higher"""
    
    if user.is_superuser:
        return user
    
    enterprise_roles = ["admin", "enterprise", "researcher", "developer"]
    if user.role.value not in enterprise_roles:
        raise_forbidden("Enterprise access required")
    
    return user


def get_pagination_params(
    page: int = 1,
    page_size: int = 20,
    sort_by: Optional[str] = None,
    sort_order: str = "asc"
) -> Dict[str, Any]:
    """Get standardized pagination parameters"""
    
    # Validate parameters
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = min(max(page_size, 1), 100)
    if sort_order not in ["asc", "desc"]:
        sort_order = "asc"
    
    return {
        "page": page,
        "page_size": page_size,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "offset": (page - 1) * page_size
    }


async def log_request(
    request: Request,
    user: Optional[User] = Depends(get_optional_user)
):
    """Log API request for monitoring and analytics"""
    
    request_data = {
        "method": request.method,
        "url": str(request.url),
        "user_id": user.id if user else None,
        "user_role": user.role.value if user else None,
        "user_agent": getattr(request.state, 'user_agent', None),
        "request_id": getattr(request.state, 'request_id', None),
        "trace_id": getattr(request.state, 'trace_id', None),
        "timestamp": datetime.now(timezone.utc)
    }
    
    # Log the request (in production, send to monitoring service)
    logger.info(
        f"API Request: {request.method} {request.url.path}",
        extra=request_data
    )
    
    return request_data


# Pre-configured permission dependencies for common use cases
RequireModelCreate = create_permission_dependency([Permission.MODEL_CREATE])
RequireModelRead = create_permission_dependency([Permission.MODEL_READ])
RequireModelExecute = create_permission_dependency([Permission.MODEL_EXECUTE])
RequireAgentCreate = create_permission_dependency([Permission.AGENT_CREATE])
RequireAgentExecute = create_permission_dependency([Permission.AGENT_EXECUTE])
RequireDataCreate = create_permission_dependency([Permission.DATA_CREATE])
RequireDataRead = create_permission_dependency([Permission.DATA_READ])
RequireTokenTransfer = create_permission_dependency([Permission.TOKEN_TRANSFER])
RequireSystemAdmin = create_permission_dependency([Permission.SYSTEM_ADMIN])


# Common dependency combinations
StandardAuth = Depends(get_current_user)
RateLimitedAuth = Depends(check_rate_limit)
AdminAuth = Depends(require_admin)
EnterpriseAuth = Depends(require_enterprise)
OptionalAuth = Depends(get_optional_user)


def create_endpoint_dependencies(
    require_auth: bool = True,
    required_permissions: Optional[List[Permission]] = None,
    rate_limited: bool = True,
    min_role: Optional[str] = None
) -> List[Depends]:
    """Create a standardized set of dependencies for an endpoint"""
    
    dependencies = [Depends(add_request_metadata)]
    
    if require_auth:
        if rate_limited:
            dependencies.append(RateLimitedAuth)
        else:
            dependencies.append(StandardAuth)
        
        if required_permissions:
            dependencies.append(create_permission_dependency(required_permissions))
        
        if min_role == "admin":
            dependencies.append(AdminAuth)
        elif min_role == "enterprise":
            dependencies.append(EnterpriseAuth)
    
    dependencies.append(Depends(log_request))
    
    return dependencies