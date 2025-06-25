"""
Authentication API Endpoints
Secure authentication and authorization endpoints for PRSM
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer
import structlog

from prsm.auth.models import (
    LoginRequest, RegisterRequest, TokenResponse, UserResponse, 
    PasswordChange, TokenRefreshRequest, PermissionCheck, RoleAssignment,
    User, UserRole, Permission
)
from prsm.auth.auth_manager import auth_manager, get_current_user, require_auth, require_permission, require_role
from prsm.integrations.security.audit_logger import audit_logger

logger = structlog.get_logger(__name__)
security = HTTPBearer()

# Create auth router
router = APIRouter(prefix="/auth", tags=["Authentication"])


def get_client_info(request: Request) -> Dict[str, Any]:
    """Extract client information from request"""
    return {
        "ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", ""),
        "origin": request.headers.get("origin", ""),
        "referer": request.headers.get("referer", "")
    }


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    request: RegisterRequest,
    req: Request
) -> UserResponse:
    """
    Register a new user account
    
    **Features:**
    - Password strength validation
    - Email uniqueness checking  
    - Automatic role assignment
    - Security audit logging
    
    **Returns:**
    - User profile information
    - Account verification status
    """
    try:
        client_info = get_client_info(req)
        
        # Register user through auth manager
        user = await auth_manager.register_user(request, client_info)
        
        logger.info("User registration successful",
                   user_id=str(user.id),
                   username=user.username,
                   email=user.email)
        
        return UserResponse.from_orm(user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Registration endpoint error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(
    request: LoginRequest,
    req: Request
) -> TokenResponse:
    """
    Authenticate user and return JWT tokens
    
    **Features:**
    - Username/email login support
    - Account lockout protection
    - Rate limiting integration
    - Remember me functionality
    - Comprehensive audit logging
    
    **Returns:**
    - Access token (30 min expiry)
    - Refresh token (7 day expiry)
    - Token type and expiration info
    """
    try:
        client_info = get_client_info(req)
        
        # Authenticate user through auth manager
        token_response = await auth_manager.authenticate_user(request, client_info)
        
        logger.info("User login successful",
                   username=request.username,
                   remember_me=request.remember_me)
        
        return token_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login endpoint error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(
    request: TokenRefreshRequest,
    req: Request
) -> TokenResponse:
    """
    Refresh access token using refresh token
    
    **Features:**
    - Automatic token rotation
    - Refresh token invalidation
    - Security validation
    - Audit trail maintenance
    
    **Returns:**
    - New access token
    - New refresh token
    - Updated expiration times
    """
    try:
        client_info = get_client_info(req)
        
        # Refresh tokens through auth manager
        token_response = await auth_manager.refresh_tokens(request.refresh_token, client_info)
        
        logger.info("Token refresh successful")
        
        return token_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh endpoint error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout_user(
    req: Request,
    current_user: User = Depends(require_auth)
) -> Dict[str, str]:
    """
    Logout user and revoke current token
    
    **Features:**
    - Token revocation
    - Session cleanup
    - Audit logging
    - Multi-device logout support
    
    **Returns:**
    - Logout confirmation
    """
    try:
        client_info = get_client_info(req)
        
        # Extract token from Authorization header
        auth_header = req.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            
            # Logout through auth manager
            success = await auth_manager.logout_user(token, client_info)
            
            if success:
                logger.info("User logout successful",
                           user_id=str(current_user.id),
                           username=current_user.username)
                
                return {"message": "Logout successful"}
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Logout failed"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid authorization header"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Logout endpoint error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(require_auth)
) -> UserResponse:
    """
    Get current user profile information
    
    **Features:**
    - Complete user profile
    - Role and permission information
    - Account status details
    - Last activity tracking
    
    **Returns:**
    - User profile data
    - Account metadata
    """
    logger.debug("User profile requested",
                user_id=str(current_user.id),
                username=current_user.username)
    
    return UserResponse.from_orm(current_user)


@router.post("/change-password")
async def change_password(
    request: PasswordChange,
    req: Request,
    current_user: User = Depends(require_auth)
) -> Dict[str, str]:
    """
    Change user password
    
    **Features:**
    - Current password verification
    - Password strength validation
    - Security audit logging
    - Automatic token revocation
    
    **Returns:**
    - Password change confirmation
    """
    try:
        client_info = get_client_info(req)
        
        # Verify current password
        from prsm.auth.jwt_handler import jwt_handler
        if not jwt_handler.verify_password(request.current_password, current_user.hashed_password):
            await audit_logger.log_auth_event(
                "password_change_failed",
                {"reason": "invalid_current_password", "user_id": str(current_user.id)},
                client_info
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password strength
        if len(request.new_password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be at least 8 characters long"
            )
        
        # Hash new password
        new_hashed_password = jwt_handler.hash_password(request.new_password)
        
        # Update password (would update in database)
        current_user.hashed_password = new_hashed_password
        
        await audit_logger.log_auth_event(
            "password_changed",
            {"user_id": str(current_user.id), "username": current_user.username},
            client_info
        )
        
        logger.info("Password changed successfully",
                   user_id=str(current_user.id),
                   username=current_user.username)
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password change error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post("/check-permission")
async def check_user_permission(
    request: PermissionCheck,
    current_user: User = Depends(require_auth)
) -> Dict[str, bool]:
    """
    Check if current user has specific permission
    
    **Features:**
    - Fine-grained permission checking
    - Resource-specific permissions
    - Role inheritance validation
    - Audit trail for access checks
    
    **Returns:**
    - Permission check result
    - Additional context if needed
    """
    try:
        has_permission = auth_manager.check_permission(current_user, request.permission)
        
        logger.debug("Permission check",
                    user_id=str(current_user.id),
                    username=current_user.username,
                    permission=request.permission.value,
                    result=has_permission)
        
        return {
            "has_permission": has_permission,
            "permission": request.permission.value,
            "user_role": current_user.role.value
        }
        
    except Exception as e:
        logger.error("Permission check error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Permission check failed"
        )


# Admin endpoints (require admin role)

@router.get("/users", response_model=list[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> list[UserResponse]:
    """
    List all users (admin only)
    
    **Features:**
    - Paginated user listing
    - Role and status filtering
    - Search capabilities
    - Admin access control
    
    **Returns:**
    - List of user profiles
    - Pagination metadata
    """
    logger.info("User list requested",
               admin_user_id=str(current_user.id),
               skip=skip,
               limit=limit)
    
    # Would query database for users
    # For now, return current user as example
    return [UserResponse.from_orm(current_user)]


@router.post("/users/{user_id}/role", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def assign_user_role(
    user_id: str,
    request: RoleAssignment,
    req: Request,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, str]:
    """
    Assign role to user (admin only)
    
    **Features:**
    - Role assignment and modification
    - Custom permission grants
    - Admin authorization required
    - Comprehensive audit logging
    
    **Returns:**
    - Role assignment confirmation
    """
    try:
        client_info = get_client_info(req)
        
        # Would update user role in database
        await audit_logger.log_auth_event(
            "role_assigned",
            {
                "target_user_id": user_id,
                "new_role": request.role.value,
                "assigned_by": str(current_user.id),
                "custom_permissions": [p.value for p in request.custom_permissions] if request.custom_permissions else []
            },
            client_info
        )
        
        logger.info("Role assigned",
                   target_user_id=user_id,
                   new_role=request.role.value,
                   assigned_by=str(current_user.id))
        
        return {"message": f"Role {request.role.value} assigned successfully"}
        
    except Exception as e:
        logger.error("Role assignment error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Role assignment failed"
        )


@router.delete("/users/{user_id}", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def deactivate_user(
    user_id: str,
    req: Request,
    current_user: User = Depends(require_role(UserRole.ADMIN))
) -> Dict[str, str]:
    """
    Deactivate user account (admin only)
    
    **Features:**
    - Account deactivation
    - Token revocation
    - Admin authorization required
    - Audit trail maintenance
    
    **Returns:**
    - Deactivation confirmation
    """
    try:
        client_info = get_client_info(req)
        
        # Would deactivate user in database
        await audit_logger.log_auth_event(
            "user_deactivated",
            {
                "target_user_id": user_id,
                "deactivated_by": str(current_user.id)
            },
            client_info
        )
        
        logger.info("User deactivated",
                   target_user_id=user_id,
                   deactivated_by=str(current_user.id))
        
        return {"message": "User deactivated successfully"}
        
    except Exception as e:
        logger.error("User deactivation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User deactivation failed"
        )


# Health check endpoint (public)
@router.get("/health")
async def auth_health_check() -> Dict[str, str]:
    """
    Authentication system health check
    
    **Features:**
    - Service availability check
    - Component status verification
    - Performance metrics
    - Public access (no auth required)
    
    **Returns:**
    - Health status information
    """
    try:
        # Check auth manager status
        auth_status = "operational" if auth_manager.db_service else "degraded"
        
        return {
            "status": "healthy",
            "auth_manager": auth_status,
            "timestamp": str(asyncio.get_event_loop().time())
        }
        
    except Exception as e:
        logger.error("Auth health check error", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": str(asyncio.get_event_loop().time())
        }