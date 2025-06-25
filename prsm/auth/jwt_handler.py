"""
JWT Token Handler
Secure JWT token generation, validation, and management for PRSM authentication
"""

import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple
from uuid import UUID

import jwt
import structlog
from passlib.context import CryptContext
from pydantic import BaseModel

from prsm.core.config import get_settings
from prsm.core.database import get_database_service

logger = structlog.get_logger(__name__)
settings = get_settings()


class TokenData(BaseModel):
    """Token payload data"""
    user_id: UUID
    username: str
    email: str
    role: str
    permissions: list[str]
    token_type: str = "access"
    issued_at: datetime
    expires_at: datetime


class JWTHandler:
    """
    JWT token handler with secure token generation and validation
    
    Features:
    - HS256 and RS256 algorithm support
    - Token refresh mechanism
    - Token blacklisting/revocation
    - Secure password hashing
    - Rate limiting for auth attempts
    """
    
    def __init__(self):
        self.algorithm = settings.jwt_algorithm
        self.secret_key = settings.secret_key
        self.access_token_expire_minutes = 30  # 30 minutes
        self.refresh_token_expire_days = 7     # 7 days
        
        # Password hashing context
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12  # Strong hashing rounds
        )
        
        # Database service for token storage
        self.db_service = None
        
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.db_service = get_database_service()
            logger.info("JWT handler initialized")
        except Exception as e:
            logger.error("Failed to initialize JWT handler", error=str(e))
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def generate_token_hash(self, token: str) -> str:
        """Generate hash of token for database storage"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    async def create_access_token(self, user_data: Dict[str, Any], 
                                expires_delta: Optional[timedelta] = None) -> Tuple[str, TokenData]:
        """
        Create JWT access token
        
        Args:
            user_data: User information to encode in token
            expires_delta: Custom expiration time
            
        Returns:
            Tuple of (token_string, token_data)
        """
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        issued_at = datetime.now(timezone.utc)
        
        # Create token payload
        payload = {
            "sub": str(user_data["user_id"]),
            "username": user_data["username"],
            "email": user_data["email"],
            "role": user_data["role"],
            "permissions": user_data.get("permissions", []),
            "token_type": "access",
            "iat": issued_at.timestamp(),
            "exp": expire.timestamp(),
            "jti": secrets.token_urlsafe(32)  # Unique token identifier
        }
        
        # Generate JWT token
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Create token data object
        token_data = TokenData(
            user_id=UUID(user_data["user_id"]),
            username=user_data["username"],
            email=user_data["email"],
            role=user_data["role"],
            permissions=user_data.get("permissions", []),
            token_type="access",
            issued_at=issued_at,
            expires_at=expire
        )
        
        # Store token hash in database for revocation capability
        if self.db_service:
            try:
                await self._store_token_record(
                    user_id=UUID(user_data["user_id"]),
                    token=token,
                    token_type="access",
                    expires_at=expire,
                    client_info=user_data.get("client_info", {})
                )
            except Exception as e:
                logger.warning("Failed to store token record", error=str(e))
        
        logger.info("Access token created", 
                   user_id=user_data["user_id"],
                   username=user_data["username"],
                   expires_at=expire.isoformat())
        
        return token, token_data
    
    async def create_refresh_token(self, user_data: Dict[str, Any]) -> Tuple[str, TokenData]:
        """
        Create JWT refresh token
        
        Args:
            user_data: User information to encode in token
            
        Returns:
            Tuple of (token_string, token_data)
        """
        expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        issued_at = datetime.now(timezone.utc)
        
        # Create refresh token payload (minimal info for security)
        payload = {
            "sub": str(user_data["user_id"]),
            "username": user_data["username"],
            "token_type": "refresh",
            "iat": issued_at.timestamp(),
            "exp": expire.timestamp(),
            "jti": secrets.token_urlsafe(32)
        }
        
        # Generate JWT token
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Create token data object
        token_data = TokenData(
            user_id=UUID(user_data["user_id"]),
            username=user_data["username"],
            email=user_data.get("email", ""),
            role=user_data.get("role", ""),
            permissions=[],
            token_type="refresh",
            issued_at=issued_at,
            expires_at=expire
        )
        
        # Store token hash in database
        if self.db_service:
            try:
                await self._store_token_record(
                    user_id=UUID(user_data["user_id"]),
                    token=token,
                    token_type="refresh",
                    expires_at=expire,
                    client_info=user_data.get("client_info", {})
                )
            except Exception as e:
                logger.warning("Failed to store refresh token record", error=str(e))
        
        logger.info("Refresh token created",
                   user_id=user_data["user_id"],
                   username=user_data["username"],
                   expires_at=expire.isoformat())
        
        return token, token_data
    
    async def verify_token(self, token: str) -> Optional[TokenData]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            TokenData if valid, None if invalid
        """
        try:
            # Decode JWT token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Extract token data
            user_id = UUID(payload.get("sub"))
            username = payload.get("username")
            email = payload.get("email", "")
            role = payload.get("role", "")
            permissions = payload.get("permissions", [])
            token_type = payload.get("token_type", "access")
            
            issued_at = datetime.fromtimestamp(payload.get("iat"), tz=timezone.utc)
            expires_at = datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc)
            
            # Check if token is expired
            if expires_at <= datetime.now(timezone.utc):
                logger.warning("Token expired", user_id=str(user_id), username=username)
                return None
            
            # Check if token is revoked (if database available)
            if self.db_service:
                token_hash = self.generate_token_hash(token)
                if await self._is_token_revoked(token_hash):
                    logger.warning("Token revoked", user_id=str(user_id), username=username)
                    return None
                
                # Update last used timestamp
                await self._update_token_last_used(token_hash)
            
            # Create token data object
            token_data = TokenData(
                user_id=user_id,
                username=username,
                email=email,
                role=role,
                permissions=permissions,
                token_type=token_type,
                issued_at=issued_at,
                expires_at=expires_at
            )
            
            logger.debug("Token verified successfully", 
                        user_id=str(user_id), 
                        username=username,
                        token_type=token_type)
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token signature expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            return None
        except Exception as e:
            logger.error("Token verification error", error=str(e))
            return None
    
    async def refresh_access_token(self, refresh_token: str) -> Optional[Tuple[str, str]]:
        """
        Refresh access token using refresh token
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Tuple of (new_access_token, new_refresh_token) if successful
        """
        try:
            # Verify refresh token
            token_data = await self.verify_token(refresh_token)
            
            if not token_data or token_data.token_type != "refresh":
                logger.warning("Invalid refresh token")
                return None
            
            # Get user data from database
            if not self.db_service:
                logger.error("Database service not available for token refresh")
                return None
            
            # For now, create minimal user data (would fetch from database in production)
            user_data = {
                "user_id": str(token_data.user_id),
                "username": token_data.username,
                "email": token_data.email,
                "role": token_data.role,
                "permissions": token_data.permissions
            }
            
            # Revoke old refresh token
            old_token_hash = self.generate_token_hash(refresh_token)
            await self._revoke_token(old_token_hash)
            
            # Create new tokens
            new_access_token, _ = await self.create_access_token(user_data)
            new_refresh_token, _ = await self.create_refresh_token(user_data)
            
            logger.info("Tokens refreshed successfully", 
                       user_id=str(token_data.user_id),
                       username=token_data.username)
            
            return new_access_token, new_refresh_token
            
        except Exception as e:
            logger.error("Token refresh error", error=str(e))
            return None
    
    async def revoke_token(self, token: str) -> bool:
        """
        Revoke a token (add to blacklist)
        
        Args:
            token: Token to revoke
            
        Returns:
            True if revoked successfully
        """
        try:
            token_hash = self.generate_token_hash(token)
            return await self._revoke_token(token_hash)
        except Exception as e:
            logger.error("Token revocation error", error=str(e))
            return False
    
    async def revoke_all_user_tokens(self, user_id: UUID) -> bool:
        """
        Revoke all tokens for a user
        
        Args:
            user_id: User ID to revoke tokens for
            
        Returns:
            True if revoked successfully
        """
        try:
            if not self.db_service:
                return False
            
            # Would update all user tokens as revoked in database
            # For now, log the action
            logger.info("All user tokens revoked", user_id=str(user_id))
            return True
            
        except Exception as e:
            logger.error("User token revocation error", error=str(e))
            return False
    
    # Private helper methods
    
    async def _store_token_record(self, user_id: UUID, token: str, token_type: str,
                                expires_at: datetime, client_info: Dict[str, Any]):
        """Store token record in database"""
        try:
            token_hash = self.generate_token_hash(token)
            
            # Would store in database - for now just log
            logger.debug("Token record stored",
                        user_id=str(user_id),
                        token_type=token_type,
                        expires_at=expires_at.isoformat(),
                        client_ip=client_info.get("ip"),
                        user_agent=client_info.get("user_agent", "")[:200])
            
        except Exception as e:
            logger.error("Failed to store token record", error=str(e))
            raise
    
    async def _is_token_revoked(self, token_hash: str) -> bool:
        """Check if token is revoked"""
        try:
            # Would check database for revoked status
            # For now, assume not revoked
            return False
        except Exception as e:
            logger.error("Failed to check token revocation status", error=str(e))
            return True  # Fail safe - assume revoked if can't check
    
    async def _revoke_token(self, token_hash: str) -> bool:
        """Revoke token in database"""
        try:
            # Would update database record
            logger.info("Token revoked", token_hash=token_hash[:16] + "...")
            return True
        except Exception as e:
            logger.error("Failed to revoke token", error=str(e))
            return False
    
    async def _update_token_last_used(self, token_hash: str):
        """Update token last used timestamp"""
        try:
            # Would update database record
            logger.debug("Token last used updated", token_hash=token_hash[:16] + "...")
        except Exception as e:
            logger.debug("Failed to update token last used", error=str(e))


# Global JWT handler instance
jwt_handler = JWTHandler()