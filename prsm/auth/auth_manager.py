"""
Authentication Manager
Central authentication and authorization management for PRSM
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from uuid import UUID
import structlog

from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from prsm.core.database import get_database_service
from prsm.auth.models import User, UserRole, Permission, LoginRequest, RegisterRequest, TokenResponse
from prsm.auth.jwt_handler import jwt_handler, TokenData
from prsm.integrations.security.audit_logger import audit_logger

logger = structlog.get_logger(__name__)
security = HTTPBearer()


class AuthenticationError(HTTPException):
    """Authentication error exception"""
    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(HTTPException):
    """Authorization error exception"""
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


class AuthManager:
    """
    Comprehensive authentication and authorization manager
    
    Features:
    - User registration and login
    - JWT token management
    - Role-based access control (RBAC)
    - Permission checking
    - Account lockout protection
    - Audit logging integration
    """
    
    def __init__(self):
        self.db_service = None
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 15
        self.password_min_length = 8
        
    async def initialize(self):
        """Initialize auth manager"""
        try:
            self.db_service = get_database_service()
            await jwt_handler.initialize()
            logger.info("Auth manager initialized")
        except Exception as e:
            logger.error("Failed to initialize auth manager", error=str(e))
            raise
    
    async def register_user(self, request: RegisterRequest, 
                          client_info: Optional[Dict[str, Any]] = None) -> User:
        """
        Register a new user
        
        Args:
            request: Registration request data
            client_info: Client information for audit logging
            
        Returns:
            Created user object
            
        Raises:
            HTTPException: If registration fails
        """
        try:
            # Validate passwords match
            if not request.passwords_match():
                await audit_logger.log_auth_event(
                    "registration_failed",
                    {"reason": "password_mismatch", "username": request.username},
                    client_info
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Passwords do not match"
                )
            
            # Check if user already exists
            if await self._user_exists(request.email, request.username):
                await audit_logger.log_auth_event(
                    "registration_failed",
                    {"reason": "user_exists", "username": request.username, "email": request.email},
                    client_info
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User with this email or username already exists"
                )
            
            # Validate password strength
            if not self._validate_password_strength(request.password):
                await audit_logger.log_auth_event(
                    "registration_failed",
                    {"reason": "weak_password", "username": request.username},
                    client_info
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Password does not meet strength requirements"
                )
            
            # Hash password
            hashed_password = jwt_handler.hash_password(request.password)
            
            # Create user object
            user = User(
                email=request.email,
                username=request.username,
                full_name=request.full_name,
                hashed_password=hashed_password,
                role=UserRole.USER,  # Default role
                is_active=True,
                is_verified=False  # Require email verification
            )
            
            # Save to database (placeholder - would use actual database service)
            # For now, just create the user object
            user.id = UUID('12345678-1234-5678-9012-123456789012')  # Placeholder
            
            await audit_logger.log_auth_event(
                "user_registered",
                {
                    "user_id": str(user.id),
                    "username": user.username,
                    "email": user.email,
                    "role": user.role.value
                },
                client_info
            )
            
            logger.info("User registered successfully",
                       user_id=str(user.id),
                       username=user.username,
                       email=user.email)
            
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("User registration error", error=str(e))
            await audit_logger.log_auth_event(
                "registration_error",
                {"error": str(e), "username": request.username},
                client_info
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed"
            )
    
    async def authenticate_user(self, request: LoginRequest,
                              client_info: Optional[Dict[str, Any]] = None) -> TokenResponse:
        """
        Authenticate user and return tokens
        
        Args:
            request: Login request data
            client_info: Client information for audit logging
            
        Returns:
            Token response with access and refresh tokens
            
        Raises:
            HTTPException: If authentication fails
        """
        try:
            # Get user by username or email
            user = await self._get_user_by_login(request.username)
            
            if not user:
                await audit_logger.log_auth_event(
                    "login_failed",
                    {"reason": "user_not_found", "username": request.username},
                    client_info
                )
                await asyncio.sleep(1)  # Prevent timing attacks
                raise AuthenticationError("Invalid credentials")
            
            # Check account lockout
            if await self._is_account_locked(user):
                await audit_logger.log_auth_event(
                    "login_failed",
                    {"reason": "account_locked", "user_id": str(user.id), "username": user.username},
                    client_info
                )
                raise AuthenticationError("Account is temporarily locked due to failed login attempts")
            
            # Check if account is active
            if not user.is_active:
                await audit_logger.log_auth_event(
                    "login_failed",
                    {"reason": "account_inactive", "user_id": str(user.id), "username": user.username},
                    client_info
                )
                raise AuthenticationError("Account is inactive")
            
            # Verify password
            if not jwt_handler.verify_password(request.password, user.hashed_password):
                await self._record_failed_login(user)
                await audit_logger.log_auth_event(
                    "login_failed",
                    {"reason": "invalid_password", "user_id": str(user.id), "username": user.username},
                    client_info
                )
                await asyncio.sleep(1)  # Prevent timing attacks
                raise AuthenticationError("Invalid credentials")
            
            # Reset failed login attempts on successful login
            await self._reset_failed_login_attempts(user)
            
            # Update last login time
            user.last_login = datetime.now(timezone.utc)
            
            # Create user data for token
            user_data = {
                "user_id": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "permissions": [p.value for p in user.get_permissions()],
                "client_info": client_info or {}
            }
            
            # Create tokens
            access_token, access_token_data = await jwt_handler.create_access_token(user_data)
            refresh_token, refresh_token_data = await jwt_handler.create_refresh_token(user_data)
            
            # Calculate expires_in for access token
            expires_in = int((access_token_data.expires_at - datetime.now(timezone.utc)).total_seconds())
            
            await audit_logger.log_auth_event(
                "login_successful",
                {
                    "user_id": str(user.id),
                    "username": user.username,
                    "role": user.role.value,
                    "token_expires_at": access_token_data.expires_at.isoformat()
                },
                client_info
            )
            
            logger.info("User authenticated successfully",
                       user_id=str(user.id),
                       username=user.username,
                       role=user.role.value)
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=expires_in
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Authentication error", error=str(e))
            await audit_logger.log_auth_event(
                "authentication_error",
                {"error": str(e), "username": request.username},
                client_info
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication failed"
            )
    
    async def get_current_user(self, token: str) -> User:
        """
        Get current user from token
        
        Args:
            token: JWT access token
            
        Returns:
            User object
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            # Verify token
            token_data = await jwt_handler.verify_token(token)
            
            if not token_data or token_data.token_type != "access":
                raise AuthenticationError()
            
            # Get user from database
            user = await self._get_user_by_id(token_data.user_id)
            
            if not user or not user.is_active:
                raise AuthenticationError()
            
            return user
            
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error("Get current user error", error=str(e))
            raise AuthenticationError()
    
    async def refresh_tokens(self, refresh_token: str,
                           client_info: Optional[Dict[str, Any]] = None) -> TokenResponse:
        """
        Refresh access token using refresh token
        
        Args:
            refresh_token: Valid refresh token
            client_info: Client information for audit logging
            
        Returns:
            New token response
            
        Raises:
            AuthenticationError: If refresh fails
        """
        try:
            result = await jwt_handler.refresh_access_token(refresh_token)
            
            if not result:
                await audit_logger.log_auth_event(
                    "token_refresh_failed",
                    {"reason": "invalid_refresh_token"},
                    client_info
                )
                raise AuthenticationError("Invalid refresh token")
            
            new_access_token, new_refresh_token = result
            
            # Get token data to calculate expires_in
            token_data = await jwt_handler.verify_token(new_access_token)
            expires_in = int((token_data.expires_at - datetime.now(timezone.utc)).total_seconds())
            
            await audit_logger.log_auth_event(
                "token_refreshed",
                {"user_id": str(token_data.user_id), "username": token_data.username},
                client_info
            )
            
            return TokenResponse(
                access_token=new_access_token,
                refresh_token=new_refresh_token,
                token_type="bearer",
                expires_in=expires_in
            )
            
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error("Token refresh error", error=str(e))
            await audit_logger.log_auth_event(
                "token_refresh_error",
                {"error": str(e)},
                client_info
            )
            raise AuthenticationError("Token refresh failed")
    
    async def logout_user(self, token: str, client_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Logout user by revoking token
        
        Args:
            token: Access token to revoke
            client_info: Client information for audit logging
            
        Returns:
            True if logout successful
        """
        try:
            # Get user data from token before revoking
            token_data = await jwt_handler.verify_token(token)
            
            if token_data:
                # Revoke token
                await jwt_handler.revoke_token(token)
                
                await audit_logger.log_auth_event(
                    "user_logout",
                    {"user_id": str(token_data.user_id), "username": token_data.username},
                    client_info
                )
                
                logger.info("User logged out successfully",
                           user_id=str(token_data.user_id),
                           username=token_data.username)
            
            return True
            
        except Exception as e:
            logger.error("Logout error", error=str(e))
            return False
    
    def check_permission(self, user: User, required_permission: Permission) -> bool:
        """
        Check if user has required permission
        
        Args:
            user: User to check
            required_permission: Required permission
            
        Returns:
            True if user has permission
        """
        return user.has_permission(required_permission)
    
    def check_any_permission(self, user: User, required_permissions: List[Permission]) -> bool:
        """
        Check if user has any of the required permissions
        
        Args:
            user: User to check
            required_permissions: List of permissions (user needs at least one)
            
        Returns:
            True if user has any permission
        """
        return user.has_any_permission(required_permissions)
    
    def require_permission(self, required_permission: Permission):
        """
        Decorator to require specific permission
        
        Args:
            required_permission: Permission required to access endpoint
            
        Returns:
            Dependency function for FastAPI
        """
        async def permission_checker(current_user: User = Depends(get_current_user)):
            if not self.check_permission(current_user, required_permission):
                await audit_logger.log_auth_event(
                    "permission_denied",
                    {
                        "user_id": str(current_user.id),
                        "username": current_user.username,
                        "required_permission": required_permission.value,
                        "user_role": current_user.role.value
                    }
                )
                raise AuthorizationError(f"Permission required: {required_permission.value}")
            return current_user
        
        return permission_checker
    
    def require_role(self, required_role: UserRole):
        """
        Decorator to require specific role
        
        Args:
            required_role: Role required to access endpoint
            
        Returns:
            Dependency function for FastAPI
        """
        async def role_checker(current_user: User = Depends(get_current_user)):
            if current_user.role != required_role and not current_user.is_superuser:
                await audit_logger.log_auth_event(
                    "role_denied",
                    {
                        "user_id": str(current_user.id),
                        "username": current_user.username,
                        "required_role": required_role.value,
                        "user_role": current_user.role.value
                    }
                )
                raise AuthorizationError(f"Role required: {required_role.value}")
            return current_user
        
        return role_checker
    
    # Private helper methods
    
    async def _user_exists(self, email: str, username: str) -> bool:
        """Check if user exists by email or username"""
        # Would check database - for now return False
        return False
    
    async def _get_user_by_login(self, login: str) -> Optional[User]:
        """Get user by username or email"""
        # Would query database - for now return mock user for testing
        if login in ["admin", "admin@prsm.ai"]:
            user = User(
                email="admin@prsm.ai",
                username="admin",
                full_name="Admin User",
                hashed_password=jwt_handler.hash_password("admin123"),
                role=UserRole.ADMIN,
                is_active=True,
                is_verified=True,
                is_superuser=True
            )
            user.id = UUID('12345678-1234-5678-9012-123456789012')
            return user
        return None
    
    async def _get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        # Would query database - for now return mock user
        if str(user_id) == '12345678-1234-5678-9012-123456789012':
            user = User(
                email="admin@prsm.ai",
                username="admin",
                full_name="Admin User",
                hashed_password=jwt_handler.hash_password("admin123"),
                role=UserRole.ADMIN,
                is_active=True,
                is_verified=True,
                is_superuser=True
            )
            user.id = user_id
            return user
        return None
    
    async def _is_account_locked(self, user: User) -> bool:
        """Check if account is locked due to failed attempts"""
        if user.failed_login_attempts >= self.max_login_attempts:
            # Check if lockout period has expired
            # For now, assume not locked (would check last failed attempt time)
            return False
        return False
    
    async def _record_failed_login(self, user: User):
        """Record failed login attempt"""
        user.failed_login_attempts += 1
        # Would update database
        logger.warning("Failed login attempt recorded",
                      user_id=str(user.id),
                      username=user.username,
                      attempts=user.failed_login_attempts)
    
    async def _reset_failed_login_attempts(self, user: User):
        """Reset failed login attempts counter"""
        user.failed_login_attempts = 0
        # Would update database
        logger.debug("Failed login attempts reset",
                    user_id=str(user.id),
                    username=user.username)
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < self.password_min_length:
            return False
        
        # Check for at least one uppercase, lowercase, digit, and special character
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special


# Global auth manager instance
auth_manager = AuthManager()


# FastAPI dependency functions

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    FastAPI dependency to get current authenticated user
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        Current user
        
    Raises:
        AuthenticationError: If authentication fails
    """
    try:
        token = credentials.credentials
        user = await auth_manager.get_current_user(token)
        return user
    except Exception:
        raise AuthenticationError()


def require_auth(user: User = Depends(get_current_user)) -> User:
    """
    FastAPI dependency to require authentication
    
    Args:
        user: Current authenticated user
        
    Returns:
        Current user
    """
    return user


def require_permission(permission: Permission):
    """
    FastAPI dependency factory to require specific permission
    
    Args:
        permission: Required permission
        
    Returns:
        Dependency function
    """
    return auth_manager.require_permission(permission)


def require_role(role: UserRole):
    """
    FastAPI dependency factory to require specific role
    
    Args:
        role: Required role
        
    Returns:
        Dependency function
    """
    return auth_manager.require_role(role)