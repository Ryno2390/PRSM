"""
Production RBAC System with Database Integration
==============================================

Production-ready Role-Based Access Control system that addresses Gemini's 
audit findings by implementing:
- Database-backed user roles and permissions
- Distributed Redis-based rate limiting 
- Persistent audit logging
- Real user management (not prototype defaults)

This replaces the in-memory prototype with production-grade security.
"""

import asyncio
import time
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from uuid import UUID, uuid4
from functools import wraps
import structlog

import asyncpg
import redis.asyncio as redis
from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import select, insert, update, delete, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession

from prsm.core.database_service import get_database_service
from prsm.core.models import UserRole, User
from prsm.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class UserPermission(BaseModel):
    """Database model for user permissions"""
    id: Optional[str] = None
    user_id: str
    resource_type: str
    resource_id: Optional[str] = None
    action: str
    granted_by: str
    granted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    is_active: bool = True


class RateLimitEntry(BaseModel):
    """Redis-based rate limit tracking"""
    user_id: str
    ip_address: str
    endpoint: str
    timestamp: datetime
    request_count: int


class AuditLog(BaseModel):
    """Persistent audit log entry"""
    id: Optional[str] = None
    user_id: str
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProductionRBACManager:
    """
    Production-ready RBAC manager with database and Redis integration.
    
    Addresses Gemini's audit findings:
    - Replaces in-memory storage with persistent database
    - Implements distributed rate limiting with Redis
    - Provides real user role management
    - Comprehensive audit logging for compliance
    """
    
    def __init__(self):
        self.database_service = get_database_service()
        self.redis_client: Optional[redis.Redis] = None
        self.rate_limit_prefix = "prsm:rate_limit"
        self.blocked_ips_key = "prsm:blocked_ips"
        
        # Initialize Redis connection
        asyncio.create_task(self._init_redis())
        
        # Core permission matrix for role-based access
        self.base_permissions = {
            UserRole.ADMIN: {
                "*": ["create", "read", "update", "delete", "admin"]
            },
            UserRole.ENTERPRISE: {
                "ai_model": ["create", "read", "update", "delete"],
                "dataset": ["create", "read", "update", "delete"],
                "agent_workflow": ["create", "read", "update", "delete"],
                "tool": ["create", "read", "update", "delete"],
                "compute_resource": ["create", "read", "update"],
                "knowledge_base": ["create", "read", "update", "delete"],
                "evaluation_metric": ["create", "read", "update"],
                "training_dataset": ["create", "read", "update", "delete"],
                "safety_dataset": ["create", "read", "update", "delete"],
                "marketplace": ["create", "read", "update", "delete"],
                "financial": ["read", "update"]
            },
            UserRole.DEVELOPER: {
                "ai_model": ["create", "read", "update"],
                "dataset": ["read", "update"],
                "agent_workflow": ["create", "read", "update"],
                "tool": ["create", "read", "update"],
                "compute_resource": ["read"],
                "knowledge_base": ["read", "update"],
                "evaluation_metric": ["read"],
                "training_dataset": ["read"],
                "safety_dataset": ["read"]
            },
            UserRole.RESEARCHER: {
                "ai_model": ["read"],
                "dataset": ["read", "create"],
                "agent_workflow": ["read"],
                "tool": ["read"],
                "knowledge_base": ["read", "create"],
                "evaluation_metric": ["read", "create"],
                "training_dataset": ["read", "create"],
                "safety_dataset": ["read", "create"]
            },
            UserRole.USER: {
                "ai_model": ["read"],
                "dataset": ["read"],
                "agent_workflow": ["read"],
                "tool": ["read"],
                "knowledge_base": ["read"],
                "evaluation_metric": ["read"]
            }
        }

    async def _init_redis(self):
        """Initialize Redis connection for distributed rate limiting"""
        try:
            redis_url = settings.REDIS_URL or "redis://localhost:6379/0"
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("✅ Redis connection established for distributed rate limiting")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            logger.warning("⚠️ Falling back to database-only rate limiting")
            self.redis_client = None

    async def get_user_role(self, user_id: str) -> Optional[UserRole]:
        """
        Get user role from database (not hardcoded defaults).
        Addresses Gemini's finding about defaulting to lowest privilege.
        """
        try:
            async with self.database_service.get_session() as session:
                query = select(User.role).where(User.id == user_id)
                result = await session.execute(query)
                user_role = result.scalar_one_or_none()
                
                if user_role:
                    return UserRole(user_role)
                else:
                    logger.warning(f"User {user_id} not found in database")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get user role for {user_id}: {e}")
            return None

    async def check_permission(
        self,
        user_id: str,
        resource_type: str,
        action: str,
        resource_id: Optional[str] = None,
        resource_owner_id: Optional[str] = None
    ) -> bool:
        """
        Production permission check with database integration.
        
        Process:
        1. Get user role from database
        2. Check base role permissions
        3. Check specific user permissions
        4. Validate resource ownership
        5. Log access attempt
        """
        try:
            # Get user role from database
            user_role = await self.get_user_role(user_id)
            if not user_role:
                await self._audit_access_denied(user_id, resource_type, action, "User not found")
                return False
            
            # Admin bypass
            if user_role == UserRole.ADMIN:
                await self._audit_access_granted(user_id, resource_type, action, "Admin access")
                return True
            
            # Check base role permissions
            has_base_permission = await self._check_base_permission(user_role, resource_type, action)
            
            # Check specific user permissions from database
            has_specific_permission = await self._check_specific_permission(user_id, resource_type, action, resource_id)
            
            if not (has_base_permission or has_specific_permission):
                await self._audit_access_denied(user_id, resource_type, action, "Insufficient permissions")
                return False
            
            # Ownership validation for update/delete operations
            if action in ["update", "delete"] and resource_owner_id:
                if user_id != resource_owner_id:
                    # Check if user has admin permission for this resource type
                    if not await self._check_admin_override(user_id, resource_type):
                        await self._audit_access_denied(user_id, resource_type, action, "Ownership check failed")
                        return False
            
            await self._audit_access_granted(user_id, resource_type, action, "Permission granted")
            return True
            
        except Exception as e:
            logger.error(f"Permission check failed for user {user_id}: {e}")
            await self._audit_access_denied(user_id, resource_type, action, f"System error: {str(e)}")
            return False

    async def _check_base_permission(self, user_role: UserRole, resource_type: str, action: str) -> bool:
        """Check base role permissions from permission matrix"""
        role_permissions = self.base_permissions.get(user_role, {})
        
        # Check wildcard permissions
        if "*" in role_permissions and action in role_permissions["*"]:
            return True
        
        # Check resource-specific permissions
        if resource_type in role_permissions and action in role_permissions[resource_type]:
            return True
        
        return False

    async def _check_specific_permission(
        self, 
        user_id: str, 
        resource_type: str, 
        action: str, 
        resource_id: Optional[str] = None
    ) -> bool:
        """Check specific user permissions from database"""
        try:
            async with self.database_service.get_session() as session:
                # Build query for specific permissions
                query = text("""
                    SELECT COUNT(*) FROM user_permissions 
                    WHERE user_id = :user_id 
                    AND resource_type = :resource_type 
                    AND action = :action
                    AND is_active = true
                    AND (expires_at IS NULL OR expires_at > NOW())
                    AND (resource_id IS NULL OR resource_id = :resource_id)
                """)
                
                result = await session.execute(query, {
                    "user_id": user_id,
                    "resource_type": resource_type,
                    "action": action,
                    "resource_id": resource_id
                })
                
                count = result.scalar()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to check specific permissions: {e}")
            return False

    async def _check_admin_override(self, user_id: str, resource_type: str) -> bool:
        """Check if user has admin override for resource type"""
        try:
            async with self.database_service.get_session() as session:
                query = text("""
                    SELECT COUNT(*) FROM user_permissions 
                    WHERE user_id = :user_id 
                    AND resource_type = :resource_type 
                    AND action = 'admin'
                    AND is_active = true
                    AND (expires_at IS NULL OR expires_at > NOW())
                """)
                
                result = await session.execute(query, {
                    "user_id": user_id,
                    "resource_type": resource_type
                })
                
                count = result.scalar()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to check admin override: {e}")
            return False

    async def enforce_rate_limit(
        self, 
        user_id: str, 
        request: Request,
        endpoint: str = "default"
    ) -> bool:
        """
        Production rate limiting using Redis for distributed enforcement.
        Addresses Gemini's finding about in-memory rate limiting.
        """
        if not self.redis_client:
            # Fallback to database rate limiting
            return await self._database_rate_limit(user_id, request, endpoint)
        
        try:
            current_time = int(time.time())
            ip_address = request.client.host
            
            # Check blocked IPs
            is_blocked = await self.redis_client.sismember(self.blocked_ips_key, ip_address)
            if is_blocked:
                logger.warning(f"Request from blocked IP: {ip_address}")
                return False
            
            # Rate limit keys
            minute_key = f"{self.rate_limit_prefix}:{user_id}:{endpoint}:{current_time // 60}"
            hour_key = f"{self.rate_limit_prefix}:{user_id}:{endpoint}:{current_time // 3600}"
            day_key = f"{self.rate_limit_prefix}:{user_id}:{endpoint}:{current_time // 86400}"
            
            # Get current counts
            minute_count = await self.redis_client.get(minute_key) or 0
            hour_count = await self.redis_client.get(hour_key) or 0
            day_count = await self.redis_client.get(day_key) or 0
            
            minute_count = int(minute_count)
            hour_count = int(hour_count)
            day_count = int(day_count)
            
            # Check limits
            if minute_count >= 60:  # 60 requests per minute
                logger.warning(f"Rate limit exceeded (minute): {user_id}")
                return False
            
            if hour_count >= 1000:  # 1000 requests per hour
                logger.warning(f"Rate limit exceeded (hour): {user_id}")
                return False
            
            if day_count >= 10000:  # 10000 requests per day
                logger.warning(f"Rate limit exceeded (day): {user_id}")
                return False
            
            # Increment counters with expiry
            pipe = self.redis_client.pipeline()
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)
            pipe.incr(hour_key)
            pipe.expire(hour_key, 3600)
            pipe.incr(day_key)
            pipe.expire(day_key, 86400)
            await pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Redis rate limiting failed: {e}")
            # Fallback to database
            return await self._database_rate_limit(user_id, request, endpoint)

    async def _database_rate_limit(self, user_id: str, request: Request, endpoint: str) -> bool:
        """Fallback database-based rate limiting"""
        try:
            current_time = datetime.now(timezone.utc)
            one_minute_ago = current_time - timedelta(minutes=1)
            
            async with self.database_service.get_session() as session:
                # Count recent requests
                query = text("""
                    SELECT COUNT(*) FROM rate_limit_log 
                    WHERE user_id = :user_id 
                    AND endpoint = :endpoint 
                    AND timestamp > :one_minute_ago
                """)
                
                result = await session.execute(query, {
                    "user_id": user_id,
                    "endpoint": endpoint,
                    "one_minute_ago": one_minute_ago
                })
                
                recent_count = result.scalar()
                
                if recent_count >= 60:  # 60 per minute limit
                    return False
                
                # Log this request
                insert_query = text("""
                    INSERT INTO rate_limit_log (user_id, endpoint, timestamp, ip_address)
                    VALUES (:user_id, :endpoint, :timestamp, :ip_address)
                """)
                
                await session.execute(insert_query, {
                    "user_id": user_id,
                    "endpoint": endpoint,
                    "timestamp": current_time,
                    "ip_address": request.client.host
                })
                
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Database rate limiting failed: {e}")
            # Fail open for availability
            return True

    async def grant_permission(
        self,
        user_id: str,
        resource_type: str,
        action: str,
        granted_by: str,
        resource_id: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> str:
        """Grant specific permission to user"""
        try:
            permission_id = str(uuid4())
            
            async with self.database_service.get_session() as session:
                query = text("""
                    INSERT INTO user_permissions 
                    (id, user_id, resource_type, resource_id, action, granted_by, granted_at, expires_at, is_active)
                    VALUES (:id, :user_id, :resource_type, :resource_id, :action, :granted_by, :granted_at, :expires_at, :is_active)
                """)
                
                await session.execute(query, {
                    "id": permission_id,
                    "user_id": user_id,
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "action": action,
                    "granted_by": granted_by,
                    "granted_at": datetime.now(timezone.utc),
                    "expires_at": expires_at,
                    "is_active": True
                })
                
                await session.commit()
                
                logger.info(f"Permission granted: {user_id} -> {resource_type}:{action}")
                return permission_id
                
        except Exception as e:
            logger.error(f"Failed to grant permission: {e}")
            raise

    async def revoke_permission(self, permission_id: str, revoked_by: str) -> bool:
        """Revoke specific permission"""
        try:
            async with self.database_service.get_session() as session:
                query = text("""
                    UPDATE user_permissions 
                    SET is_active = false, revoked_by = :revoked_by, revoked_at = :revoked_at
                    WHERE id = :permission_id
                """)
                
                await session.execute(query, {
                    "permission_id": permission_id,
                    "revoked_by": revoked_by,
                    "revoked_at": datetime.now(timezone.utc)
                })
                
                await session.commit()
                
                logger.info(f"Permission revoked: {permission_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to revoke permission: {e}")
            return False

    async def _audit_access_granted(
        self, 
        user_id: str, 
        resource_type: str, 
        action: str, 
        reason: str
    ):
        """Log successful access for audit trail"""
        await self._create_audit_log(user_id, resource_type, action, True, {"reason": reason})

    async def _audit_access_denied(
        self, 
        user_id: str, 
        resource_type: str, 
        action: str, 
        reason: str
    ):
        """Log denied access for security monitoring"""
        await self._create_audit_log(user_id, resource_type, action, False, {"reason": reason})

    async def _create_audit_log(
        self,
        user_id: str,
        resource_type: str,
        action: str,
        success: bool,
        metadata: Dict[str, Any]
    ):
        """Create persistent audit log entry"""
        try:
            async with self.database_service.get_session() as session:
                query = text("""
                    INSERT INTO audit_logs 
                    (id, user_id, resource_type, action, success, metadata, timestamp)
                    VALUES (:id, :user_id, :resource_type, :action, :success, :metadata, :timestamp)
                """)
                
                await session.execute(query, {
                    "id": str(uuid4()),
                    "user_id": user_id,
                    "resource_type": resource_type,
                    "action": action,
                    "success": success,
                    "metadata": json.dumps(metadata),
                    "timestamp": datetime.now(timezone.utc)
                })
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")

    async def block_ip(self, ip_address: str, reason: str) -> bool:
        """Block IP address for security"""
        try:
            if self.redis_client:
                await self.redis_client.sadd(self.blocked_ips_key, ip_address)
            
            # Also store in database for persistence
            async with self.database_service.get_session() as session:
                query = text("""
                    INSERT INTO blocked_ips (ip_address, reason, blocked_at)
                    VALUES (:ip_address, :reason, :blocked_at)
                    ON CONFLICT (ip_address) DO NOTHING
                """)
                
                await session.execute(query, {
                    "ip_address": ip_address,
                    "reason": reason,
                    "blocked_at": datetime.now(timezone.utc)
                })
                
                await session.commit()
                
            logger.warning(f"IP blocked: {ip_address} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to block IP {ip_address}: {e}")
            return False

    async def get_user_permissions(self, user_id: str) -> List[UserPermission]:
        """Get all active permissions for user"""
        try:
            async with self.database_service.get_session() as session:
                query = text("""
                    SELECT * FROM user_permissions 
                    WHERE user_id = :user_id 
                    AND is_active = true 
                    AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY granted_at DESC
                """)
                
                result = await session.execute(query, {"user_id": user_id})
                rows = result.fetchall()
                
                permissions = []
                for row in rows:
                    permissions.append(UserPermission(
                        id=row.id,
                        user_id=row.user_id,
                        resource_type=row.resource_type,
                        resource_id=row.resource_id,
                        action=row.action,
                        granted_by=row.granted_by,
                        granted_at=row.granted_at,
                        expires_at=row.expires_at,
                        is_active=row.is_active
                    ))
                
                return permissions
                
        except Exception as e:
            logger.error(f"Failed to get user permissions: {e}")
            return []

    async def cleanup_expired_permissions(self) -> int:
        """Clean up expired permissions (run periodically)"""
        try:
            async with self.database_service.get_session() as session:
                query = text("""
                    UPDATE user_permissions 
                    SET is_active = false 
                    WHERE expires_at < NOW() AND is_active = true
                """)
                
                result = await session.execute(query)
                await session.commit()
                
                cleaned_count = result.rowcount
                logger.info(f"Cleaned up {cleaned_count} expired permissions")
                return cleaned_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired permissions: {e}")
            return 0


# Global instance
_rbac_manager = None

async def get_rbac_manager() -> ProductionRBACManager:
    """Get the global RBAC manager instance"""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = ProductionRBACManager()
    return _rbac_manager


# Database schema creation SQL (run during deployment)
RBAC_SCHEMA_SQL = """
-- User permissions table
CREATE TABLE IF NOT EXISTS user_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    action VARCHAR(50) NOT NULL,
    granted_by UUID NOT NULL REFERENCES users(id),
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    revoked_by UUID REFERENCES users(id),
    revoked_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    
    INDEX idx_user_permissions_user_id (user_id),
    INDEX idx_user_permissions_resource (resource_type, resource_id),
    INDEX idx_user_permissions_active (is_active, expires_at)
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    action VARCHAR(50) NOT NULL,
    success BOOLEAN NOT NULL,
    ip_address INET,
    user_agent TEXT,
    metadata JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_audit_logs_user_id (user_id),
    INDEX idx_audit_logs_timestamp (timestamp),
    INDEX idx_audit_logs_resource (resource_type, resource_id)
);

-- Rate limiting log table (fallback when Redis unavailable)
CREATE TABLE IF NOT EXISTS rate_limit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    ip_address INET,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_rate_limit_user_endpoint (user_id, endpoint),
    INDEX idx_rate_limit_timestamp (timestamp)
);

-- Blocked IPs table
CREATE TABLE IF NOT EXISTS blocked_ips (
    ip_address INET PRIMARY KEY,
    reason TEXT NOT NULL,
    blocked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    blocked_by UUID REFERENCES users(id),
    
    INDEX idx_blocked_ips_blocked_at (blocked_at)
);

-- Cleanup function for rate limit logs (keep only last 24 hours)
CREATE OR REPLACE FUNCTION cleanup_rate_limit_logs()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM rate_limit_log WHERE timestamp < NOW() - INTERVAL '24 hours';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
"""